from abc import ABC, abstractmethod
import copy
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Policy(ABC):
    """
    Abstract base class for visuomotor policies.
    """
    
    @abstractmethod
    def reset(self):
        """Reset policy state at the start of an episode."""
        pass
        
    @abstractmethod
    def act(self, obs: dict, info: dict = None) -> np.ndarray:
        """
        Compute action given observation.
        
        Args:
            obs: Observation dictionary from the environment
                 (e.g., {'proprioception': ..., 'visual': ...})
            info: Optional info dictionary containing privileged state
                  
        Returns:
            Action array (normalized or raw, depending on environment config)
        """
        pass


class RandomPolicy(Policy):
    """
    Baseline policy that samples random actions from a given action space.
    """
    
    def __init__(self, action_space):
        """
        Args:
            action_space: The gym action space to sample from
        """
        self.action_space = action_space
        
    def reset(self):
        pass
        
    def act(self, obs: dict, info: dict = None) -> np.ndarray:
        return self.action_space.sample()


class SafePolicy(RandomPolicy):
    """
    Heuristic policy that avoids humans using privileged state information.
    
    Behaves like RandomPolicy when safe, but applies repulsive force
    when human is close.
    """
    
    def __init__(self, action_space, safety_threshold: float = 3.5, repulsion_gain: float = 5.0):
        super().__init__(action_space)
        self.safety_threshold = safety_threshold
        self.gain = repulsion_gain
        self.last_human_pos = None

    def reset(self):
        self.last_human_pos = None
        
    def act(self, obs: dict, info: dict = None) -> np.ndarray:
        # Get base random action
        action = super().act(obs, info)
        
        if info is None:
            return action
            
        safety_info = info.get("safety", {})
        if not safety_info:
            return action
            
        # Get positions (if available)
        robot_pos = safety_info.get("robot_pos")
        human_pos = safety_info.get("human_pos")
        
        if robot_pos is None or human_pos is None:
            return action
            
        r_pos = np.array(robot_pos)
        h_pos = np.array(human_pos)
        
        # Calculate distance
        diff = r_pos - h_pos  # Vector pointing AWAY from human
        dist = np.linalg.norm(diff)
        
        if dist < self.safety_threshold:
            # SAFETY VIOLATION IMMINENT - VELOCITY MATCHING
            
            qpos = safety_info.get("qpos")
            if qpos is not None:
                qpos = np.array(qpos)
                
                if qpos.shape == self.action_space.shape:
                    retreat_action = qpos.copy() # Start with current pose (freeze arms)
                    
                    # Compute Human Velocity (since last step)
                    if self.last_human_pos is not None:
                        h_vel = h_pos - self.last_human_pos
                        
                        # Project human velocity onto Robot->Human vector (diff)
                        # diff = R - H
                        # If dot(h_vel, diff) > 0, human moving towards robot.
                        # If dot(h_vel, diff) < 0, human moving away.
                        
                        dot_prod = np.dot(h_vel[:2], diff[:2])
                        
                        if dot_prod > 0:
                            # Human approaching -> Match Velocity
                            # Apply velocity to robot base (indices 0, 1)
                            velocity_gain = 1.5 
                            base_delta = h_vel[:2] * velocity_gain
                            
                            retreat_action[0] += base_delta[0]
                            retreat_action[1] += base_delta[1]
                        else:
                            # Human moving away or static -> Freeze (as requested initially)
                            # Or naive retreat if very close?
                            # Let's freeze to avoid chasing.
                            pass
                        
                    self.last_human_pos = h_pos
                    return retreat_action
                else:
                    pass

        self.last_human_pos = h_pos
        return action


class DiffusionPolicyWrapper(Policy):
    """Wraps a trained RoboBase Diffusion Policy snapshot for benchmark evaluation.

    Handles observation preprocessing (ConcatDim, FrameStack, normalization)
    and action postprocessing (RescaleFromTanhWithMinMax) to bridge the gap
    between the raw benchmark env and what the agent expects.
    """

    KEYS_TO_IGNORE = ["proprioception_floating_base_actions"]

    def __init__(
        self,
        snapshot_path: str,
        action_space,
        device: str = "cpu",
        motion_clip_dir: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
    ):
        """
        Args:
            snapshot_path: Path to the RoboBase snapshot .pt file.
            action_space: The raw action space from the benchmark env.
            device: Torch device ("cpu", "mps", or "cuda").
            motion_clip_dir: Override for motion clip directory path
                (useful if training was done on a different machine).
        """
        import torch
        import hydra
        from omegaconf import OmegaConf

        self._torch = torch
        self.device = torch.device(device)
        self._raw_action_space = action_space

        # 1. Load snapshot
        logger.info(f"Loading snapshot from {snapshot_path}")
        payload = torch.load(snapshot_path, map_location=self.device, weights_only=False)
        cfg = payload["cfg"]
        agent_state = payload["agent"]

        # Override motion_clip_dir if provided (path may differ from training machine)
        if motion_clip_dir is not None:
            OmegaConf.update(cfg, "env.motion_clip_dir", motion_clip_dir)

        # Expose the action_mode and observation_config from training config
        # so the benchmark env can be created with matching settings.
        from bigym.action_modes import JointPositionActionMode, PelvisDof
        from bigym.utils.observation_config import ObservationConfig, CameraConfig

        if cfg.env.get("enable_all_floating_dof", False):
            self.action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
                floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
            )
        else:
            self.action_mode = JointPositionActionMode(
                absolute=cfg.env.action_mode == "absolute",
                floating_base=True,
            )

        # Build observation config matching training (cameras + proprioception)
        camera_configs = [
            CameraConfig(
                name=cam_name,
                rgb=True,
                depth=False,
                resolution=list(cfg.visual_observation_shape),
            )
            for cam_name in cfg.env.cameras
        ] if cfg.pixels else []
        self.observation_config = ObservationConfig(
            cameras=camera_configs,
            proprioception=True,
            privileged_information=not cfg.pixels,
        )

        # 2. Compute demo stats for action/obs normalization
        logger.info("Loading demos to compute action/obs statistics...")
        from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory

        factory = SafetyBiGymEnvFactory()
        factory.collect_or_fetch_demos(cfg, cfg.demos)
        self._action_stats = factory._action_stats
        self._obs_stats = factory._obs_stats
        self._min_max_margin = cfg.min_max_margin
        self._norm_obs = cfg.norm_obs
        self._frame_stack = cfg.frame_stack

        # 3. Create a temporary wrapped env to get observation/action spaces
        #    that match what the agent was trained with.
        logger.info("Creating temporary wrapped env for observation/action spaces...")
        wrapped_env = factory._wrap_env(
            factory._create_env(cfg), cfg, demo_env=False, train=False
        )
        wrapped_obs_space = wrapped_env.observation_space
        wrapped_act_space = wrapped_env.action_space
        wrapped_env.close()

        # 4. Instantiate the agent with the correct spaces
        logger.info("Instantiating Diffusion agent...")
        self.agent = hydra.utils.instantiate(
            cfg.method,
            device=self.device,
            observation_space=wrapped_obs_space,
            action_space=wrapped_act_space,
            num_train_envs=1,
            replay_alpha=cfg.replay.alpha,
            replay_beta=cfg.replay.beta,
            frame_stack_on_channel=cfg.frame_stack_on_channel,
        )
        self.agent.load_state_dict(agent_state)
        self.agent.train(False)

        # Optionally reduce diffusion inference steps for faster evaluation.
        # DDIM supports fewer steps (e.g. 10) with minimal quality loss.
        if num_inference_steps is not None:
            self.agent.actor.num_diffusion_iters = num_inference_steps
            logger.info(f"Reduced diffusion inference steps to {num_inference_steps}")

        logger.info("DiffusionPolicyWrapper ready.")

    def reset(self):
        pass

    def act(self, obs: dict, info: dict = None) -> np.ndarray:
        import torch
        from robobase.envs.wrappers.rescale_from_tanh import RescaleFromTanhWithMinMax

        # 1. Preprocess: replicate ConcatDim + FrameStack transforms
        torch_obs = self._preprocess_obs(obs)

        # 2. Forward pass through agent
        with torch.no_grad():
            action = self.agent.act(torch_obs, step=0, eval_mode=True)

        # 3. Extract first action from sequence: (1, T, D) -> (D,)
        action = action[0, 0].cpu().numpy()

        # 4. Rescale from tanh [-1, 1] to raw action space
        action = RescaleFromTanhWithMinMax.transform_from_tanh(
            action, self._action_stats, self._min_max_margin
        )

        # 5. Clip to env action space bounds (small numerical overshoot is common)
        if self._raw_action_space is not None:
            action = np.clip(action, self._raw_action_space.low, self._raw_action_space.high)

        return action

    def _preprocess_obs(self, obs: dict) -> dict:
        """Transform raw env observations to match the agent's expected format.

        Replicates the ConcatDim (merge 1D obs -> low_dim_state with normalization)
        and FrameStack (add leading frame dimension) wrappers.
        """
        import torch

        low_dim_parts = []
        result = {}

        for key, val in obs.items():
            val = np.asarray(val, dtype=np.float32)

            if val.ndim == 1 and key not in self.KEYS_TO_IGNORE:
                # Low-dimensional observation — normalize if configured
                if self._norm_obs and key in self._obs_stats.get("mean", {}):
                    std = self._obs_stats["std"][key]
                    std = np.where(std < 1e-8, 1.0, std)
                    val = (val - self._obs_stats["mean"][key]) / std
                low_dim_parts.append(val)
            elif val.ndim == 3:
                # RGB image: (H, W, C) -> add frame_stack dim -> (frame_stack, H, W, C)
                # Then add batch dim -> (1, frame_stack, H, W, C)
                val_stacked = np.stack([val] * self._frame_stack, axis=0)
                tensor = torch.from_numpy(val_stacked).unsqueeze(0).to(self.device)
                result[key] = tensor

        # Concatenate low-dim obs, add frame_stack + batch dims
        if low_dim_parts:
            low_dim = np.concatenate(low_dim_parts, axis=-1)
            # (D,) -> (frame_stack, D) -> (1, frame_stack, D)
            low_dim_stacked = np.stack([low_dim] * self._frame_stack, axis=0)
            result["low_dim_state"] = (
                torch.from_numpy(low_dim_stacked).unsqueeze(0).to(self.device)
            )

        return result
