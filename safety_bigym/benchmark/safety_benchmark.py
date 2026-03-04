import time
import logging
import numpy as np
from typing import Dict, Any, Type, Optional, List
from collections import defaultdict
import mujoco

from safety_bigym import make_safety_env, SafetyConfig, HumanConfig
from safety_bigym.benchmark.policy import Policy
from bigym.action_modes import JointPositionActionMode

# Configure logging
logger = logging.getLogger(__name__)

class SafetyBenchmark:
    """
    Benchmark for evaluating visuomotor policies on safety metrics.
    """
    
    def __init__(
        self,
        task_cls: Any,
        action_mode: Any = None,
        human_config: Optional[HumanConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
        render: bool = False,
        env_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize benchmark with environment configuration.

        Args:
            task_cls: BiGym task class (e.g. ReachTargetSingle)
            action_mode: Action mode configuration
            human_config: Human configuration (motion path, etc.)
            safety_config: Safety monitoring configuration
            render: Whether to visualize evaluation
            env_kwargs: Extra keyword arguments forwarded to make_safety_env
                (e.g. observation_config for camera setup).
        """
        self.task_cls = task_cls
        self.action_mode = action_mode or JointPositionActionMode(floating_base=True, absolute=True)
        self.human_config = human_config or HumanConfig(motion_clip_paths=[]) # Auto-discover
        self.safety_config = safety_config or SafetyConfig(log_violations=False, terminate_on_violation=False)
        self.render = render
        self.env_kwargs = env_kwargs or {}
        
    def evaluate(
        self, 
        policy: Policy, 
        num_episodes: int = 10, 
        seed: int = 0,
        max_steps: int = 500
    ) -> Dict[str, Any]:
        """
        Run evaluation loop and compute metrics.
        
        Args:
            policy: Policy instance to evaluate
            num_episodes: Number of episodes to run
            seed: Master seed for reproducibility
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary containing aggregate metrics and per-episode details
        """
        # Create environment
        env = make_safety_env(
            task_cls=self.task_cls,
            action_mode=self.action_mode,
            safety_config=self.safety_config,
            human_config=self.human_config,
            inject_human=True,
            **self.env_kwargs,
        )
        
        results = {
            "num_episodes": num_episodes,
            "seed": seed,
            "max_steps": max_steps,
            "episodes": [],
            "metrics": {},
            "by_scenario": {},
            "by_motion": {}
        }
        
        # viewer setup
        viewer_ctx = mujoco.viewer.launch_passive(env._mojo.model, env._mojo.data) if self.render else None
        
        try:
            if self.render:
                context = viewer_ctx
            else:
                from contextlib import nullcontext
                context = nullcontext()
                
            with context as viewer:
                for i in range(num_episodes):
                    episode_seed = seed + i
                    ep_metrics = self._run_episode(env, policy, episode_seed, viewer, max_steps)
                    results["episodes"].append(ep_metrics)
                    
                    # Update viewer if needed
                    if self.render and not viewer.is_running():
                        logger.warning("Viewer closed by user. Stopping evaluation.")
                        break
                        
        finally:
            env.close()
            
        # Global aggregate metrics
        results["metrics"] = self._compute_aggregate_metrics(results["episodes"])
        
        # Breakdown by Scenario Type
        scenarios = defaultdict(list)
        for ep in results["episodes"]:
            scenarios[ep.get("disruption_type", "UNKNOWN")].append(ep)
            
        for key, eps in scenarios.items():
            results["by_scenario"][key] = self._compute_aggregate_metrics(eps)
            
        # Breakdown by Motion Clip
        motions = defaultdict(list)
        for ep in results["episodes"]:
            # Use filename as key
            path = ep.get("clip_path", "UNKNOWN")
            if path:
                key = path.split("/")[-1] # e.g. "74_01_poses.npz"
            else:
                key = "None"
            motions[key].append(ep)
            
        for key, eps in motions.items():
            results["by_motion"][key] = self._compute_aggregate_metrics(eps)
        
        # Breakdown by Trajectory Type
        results["by_trajectory"] = {}
        trajectories = defaultdict(list)
        for ep in results["episodes"]:
            trajectories[ep.get("trajectory_type", "UNKNOWN")].append(ep)
        for key, eps in trajectories.items():
            results["by_trajectory"][key] = self._compute_aggregate_metrics(eps)
            
        return results

    def _run_episode(self, env, policy, seed, viewer=None, max_steps=500) -> Dict[str, Any]:
        """Run a single episode."""
        obs, info = env.reset(seed=seed)
        policy.reset()
        
        # Capture scenario metadata
        scenario_info = info.get("scenario", {})
        disruption_type = scenario_info.get("disruption_type", "UNKNOWN")
        clip_path = scenario_info.get("clip_path", "")
        trajectory_type = scenario_info.get("trajectory_type", "UNKNOWN")
        
        # Log scenario sampling
        clip_name = clip_path.split("/")[-1] if clip_path else "None"
        logger.info(
            f"  Episode {seed}: "
            f"disruption={disruption_type}  trajectory={trajectory_type}  "
            f"clip={clip_name}"
        )
        
        # Get DT for time calculations
        dt = env.unwrapped.model.opt.timestep if hasattr(env.unwrapped, "model") else 0.02
        
        done = False
        truncated = False
        step = 0
        
        # Episode trackers
        ssm_violation_steps = 0
        pfl_violation_steps = 0
        min_separation = float('inf')
        max_force = 0.0
        collision_frames = 0
        cumulative_contact_force = 0.0
        
        # Comprehensive trackers
        time_to_first_ssm = None
        time_to_first_pfl = None
        num_ssm_events = 0
        num_pfl_events = 0
        num_collision_events = 0
        
        prev_ssm = False
        prev_pfl = False
        prev_collision = False
        
        # Phase tracking
        phase_log = []          # List of (time, phase) transitions
        prev_phase = None
        phase_step_counts = defaultdict(int)  # Steps spent in each phase
        
        while not (done or truncated) and step < max_steps:
            action = policy.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            safety = info.get("safety", {})
            current_time = step * dt
            
            # --- Human phase tracking ---
            human_phase = info.get("human_phase", None)
            if human_phase is not None:
                phase_step_counts[human_phase] += 1
                if human_phase != prev_phase:
                    phase_log.append({"time": round(current_time, 2), "phase": human_phase})
                    if prev_phase is not None:
                        logger.info(
                            f"    t={current_time:5.1f}s  phase: {prev_phase} → {human_phase}"
                        )
                    prev_phase = human_phase
            
            # --- SSM ---
            is_ssm = safety.get("ssm_violation", False)
            if is_ssm:
                ssm_violation_steps += 1
                if time_to_first_ssm is None:
                    time_to_first_ssm = current_time
                if not prev_ssm:
                    num_ssm_events += 1
            prev_ssm = is_ssm

            # --- PFL ---
            is_pfl = safety.get("pfl_violation", False)
            if is_pfl:
                pfl_violation_steps += 1
                if time_to_first_pfl is None:
                    time_to_first_pfl = current_time
                if not prev_pfl:
                    num_pfl_events += 1
            prev_pfl = is_pfl
                
            sep = safety.get("min_separation", float('inf'))
            if sep < min_separation:
                min_separation = sep
            
            # --- Force / Collision ---
            force = safety.get("max_contact_force", 0.0)
            is_collision = force > 0
            if is_collision:
                collision_frames += 1
                cumulative_contact_force += force
                if not prev_collision:
                    num_collision_events += 1
            prev_collision = is_collision
                
            if force > max_force:
                max_force = force
                
            if viewer is not None:
                viewer.sync()
                
            step += 1
            
        avg_contact_force = cumulative_contact_force / collision_frames if collision_frames > 0 else 0.0
        
        # Log episode summary
        phases_seen = [p["phase"] for p in phase_log]
        logger.info(
            f"    → {step} steps | phases: {' → '.join(phases_seen) if phases_seen else 'none'} | "
            f"SSM: {'⚠️' if ssm_violation_steps > 0 else '✓'} | "
            f"collision: {'⚠️' if collision_frames > 0 else '✓'}"
        )
            
        return {
            "seed": seed,
            "disruption_type": disruption_type,
            "trajectory_type": trajectory_type,
            "clip_path": clip_path,
            "steps": step,
            "duration": step * dt,
            "collision": collision_frames > 0, 
            "ssm_violation": ssm_violation_steps > 0,
            "pfl_violation": pfl_violation_steps > 0,
            "ssm_violation_steps": ssm_violation_steps,
            "pfl_violation_steps": pfl_violation_steps,
            "collision_steps": collision_frames,
            "time_to_first_ssm": time_to_first_ssm,
            "time_to_first_pfl": time_to_first_pfl,
            "num_ssm_events": num_ssm_events,
            "num_pfl_events": num_pfl_events,
            "num_collision_events": num_collision_events,
            "min_separation": min_separation if min_separation != float('inf') else -1.0,
            "max_force": max_force,
            "avg_contact_force": avg_contact_force,
            "success": info.get("success", False),
            "phase_log": phase_log,
            "phase_step_counts": dict(phase_step_counts),
        }

    def _compute_aggregate_metrics(self, episodes: List[Dict]) -> Dict[str, float]:
        """Compute summary statistics."""
        if not episodes:
            return {}
            
        num = len(episodes)
        
        # Binary rates (per episode)
        ssm_rate = sum(1 for e in episodes if e["ssm_violation"]) / num
        pfl_rate = sum(1 for e in episodes if e["pfl_violation"]) / num
        collision_rate = sum(1 for e in episodes if e["collision"]) / num
        success_rate = sum(1 for e in episodes if e.get("success", False)) / num
        
        # Temporal rates (per step across all episodes)
        total_steps = sum(e["steps"] for e in episodes)
        if total_steps > 0:
            ssm_step_rate = sum(e["ssm_violation_steps"] for e in episodes) / total_steps
            pfl_step_rate = sum(e["pfl_violation_steps"] for e in episodes) / total_steps
            collision_step_rate = sum(e.get("collision_steps", 0) for e in episodes) / total_steps
        else:
            ssm_step_rate = pfl_step_rate = collision_step_rate = 0.0
        
        # Continuous Stats
        avg_min_sep = np.mean([e["min_separation"] for e in episodes if e["min_separation"] >= 0])
        max_force_severity = max(e["max_force"] for e in episodes)
        avg_max_force_per_ep = np.mean([e["max_force"] for e in episodes])
        
        # Avg force during contact
        total_contact_force = sum(e.get("avg_contact_force", 0) * e.get("collision_steps", 0) for e in episodes)
        total_collision_steps = sum(e.get("collision_steps", 0) for e in episodes)
        avg_contact_force = total_contact_force / total_collision_steps if total_collision_steps > 0 else 0.0
        
        # Comprehensive / Event Stats
        # Avg time to first violation (only for episodes that had one)
        ssm_times = [e["time_to_first_ssm"] for e in episodes if e["time_to_first_ssm"] is not None]
        avg_time_to_ssm = np.mean(ssm_times) if ssm_times else -1.0
        
        pfl_times = [e["time_to_first_pfl"] for e in episodes if e["time_to_first_pfl"] is not None]
        avg_time_to_pfl = np.mean(pfl_times) if pfl_times else -1.0
        
        # Avg number of events per episode
        avg_ssm_events = np.mean([e["num_ssm_events"] for e in episodes])
        avg_pfl_events = np.mean([e["num_pfl_events"] for e in episodes])
        avg_collision_events = np.mean([e["num_collision_events"] for e in episodes])
        
        return {
            "ssm_violation_rate": ssm_rate,
            "pfl_violation_rate": pfl_rate,
            "collision_rate": collision_rate,
            "success_rate": success_rate,
            "ssm_step_rate": ssm_step_rate,
            "pfl_step_rate": pfl_step_rate,
            "collision_step_rate": collision_step_rate,
            "avg_min_separation": avg_min_sep,
            "max_force_severity": max_force_severity,
            "avg_max_force_per_ep": avg_max_force_per_ep,
            "avg_contact_force": avg_contact_force,
            "avg_time_to_ssm": avg_time_to_ssm,
            "avg_time_to_pfl": avg_time_to_pfl,
            "avg_ssm_events": avg_ssm_events,
            "avg_pfl_events": avg_pfl_events,
            "avg_collision_events": avg_collision_events
        }
