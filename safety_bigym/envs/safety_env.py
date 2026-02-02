"""
SafetyBiGymEnv - BiGym environment with safety-aware human.

This module provides a wrapper around BiGym environments that adds:
- SMPL-H human with PD-controlled motion
- ISO 15066 safety monitoring (SSM + PFL)
- Scenario-based human behavior sampling
- Safety-aware observations and rewards
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import mujoco
import numpy as np
from gymnasium import spaces

# BiGym imports
from bigym.bigym_env import BiGymEnv, PHYSICS_DT
from bigym.action_modes import ActionMode
from bigym.robots.robot import Robot
from bigym.utils.observation_config import ObservationConfig

# Safety BiGym imports
from safety_bigym.safety import ISO15066Wrapper, SSMConfig, SafetyInfo
from safety_bigym.human.human_controller import HumanController
from safety_bigym.human.pd_controller import PDGains
from safety_bigym.scenarios.scenario_sampler import ScenarioSampler, ParameterSpace, ScenarioParams

logger = logging.getLogger(__name__)


# Default spawn positions around workspace perimeter
DEFAULT_SPAWN_POSITIONS = [
    {"pos": [2.0, 0.0, 0.0], "yaw": np.pi},        # Behind robot
    {"pos": [0.0, 2.0, 0.0], "yaw": -np.pi/2},     # Robot's left
    {"pos": [0.0, -2.0, 0.0], "yaw": np.pi/2},     # Robot's right
    {"pos": [1.5, 1.5, 0.0], "yaw": -3*np.pi/4},   # Behind-left
    {"pos": [1.5, -1.5, 0.0], "yaw": 3*np.pi/4},   # Behind-right
]


@dataclass
class SafetyConfig:
    """Configuration for safety monitoring."""
    
    # SSM parameters
    T_r: float = 0.1        # Robot reaction time
    T_s: float = 0.05       # System response time
    a_max: float = 5.0      # Max robot deceleration
    C: float = 0.1          # Safety margin constant
    v_h_max: float = 1.6    # Max human velocity (walking)
    
    # PFL parameters
    use_pfl: bool = True
    
    # Reward shaping
    ssm_penalty_scale: float = 0.1      # Penalty per meter of SSM margin violation
    pfl_penalty_scale: float = 0.01     # Penalty per % over force limit
    terminate_on_violation: bool = False


@dataclass
class HumanConfig:
    """Configuration for human behavior."""
    
    # SMPL-H asset
    smplh_xml_path: Optional[Path] = None
    
    # Motion
    amass_clip_path: Optional[Path] = None
    use_ik_reaching: bool = True
    
    # PD gains
    kp: float = 200.0
    kd: float = 20.0
    
    # Spawn positions (task-specific)
    spawn_positions: List[Dict] = field(default_factory=lambda: DEFAULT_SPAWN_POSITIONS.copy())


class SafetyBiGymEnv(BiGymEnv):
    """
    BiGym environment with integrated safety-aware human.
    
    Extends BiGym's step loop to:
    1. Update human PD targets every sub-step
    2. Capture peak contact forces every sub-step
    3. Compute safety metrics (SSM, PFL) for observations/rewards
    
    Usage:
        env = SafetyBiGymEnv(
            action_mode=JointPositionActionMode(),
            safety_config=SafetyConfig(),
            human_config=HumanConfig(),
        )
    """
    
    def __init__(
        self,
        action_mode: ActionMode,
        observation_config: ObservationConfig = ObservationConfig(),
        render_mode: Optional[str] = None,
        start_seed: Optional[int] = None,
        control_frequency: int = 500,
        robot_cls: Optional[Type[Robot]] = None,
        safety_config: Optional[SafetyConfig] = None,
        human_config: Optional[HumanConfig] = None,
        scenario_config: Optional[ParameterSpace] = None,
    ):
        """
        Initialize SafetyBiGymEnv.
        
        Args:
            action_mode: Robot action mode (from BiGym)
            observation_config: Observation configuration (from BiGym)
            render_mode: Render mode
            start_seed: Random seed
            control_frequency: Control loop frequency
            robot_cls: Robot class override
            safety_config: Safety monitoring configuration
            human_config: Human behavior configuration
            scenario_config: Scenario sampling configuration
        """
        # Store configs before super().__init__ (which calls _initialize_env)
        self._safety_config = safety_config or SafetyConfig()
        self._human_config = human_config or HumanConfig()
        self._scenario_config = scenario_config
        
        # Will be initialized in _initialize_env
        self._safety_wrapper: Optional[ISO15066Wrapper] = None
        self._human_controller: Optional[HumanController] = None
        self._scenario_sampler: Optional[ScenarioSampler] = None
        self._current_scenario = None
        
        # Safety state tracking
        self._step_safety_info: Optional[SafetyInfo] = None
        self._episode_safety_stats = {
            "ssm_violations": 0,
            "pfl_violations": 0,
            "max_force": 0.0,
            "min_ssm_margin": float('inf'),
        }
        
        # Create merged model with human
        self._temp_model_file = self._create_merged_model()
        self._MODEL_PATH = Path(self._temp_model_file.name)
        
        # Call parent init
        super().__init__(
            action_mode=action_mode,
            observation_config=observation_config,
            render_mode=render_mode,
            start_seed=start_seed,
            control_frequency=control_frequency,
            robot_cls=robot_cls,
        )
    
    def _create_merged_model(self) -> Any:
        """
        Create a temporary XML file that merges the world model and human model.
        Returns the temporary file object (must be kept alive).
        """
        import tempfile
        import xml.etree.ElementTree as ET
        from bigym.const import WORLD_MODEL
        
        human_xml = self._human_config.smplh_xml_path
        if human_xml is None:
            # Default to packaged asset
            import safety_bigym
            asset_dir = Path(safety_bigym.__file__).parent / "assets"
            human_xml = asset_dir / "smplh_human.xml"
        
        if not human_xml.exists():
            raise FileNotFoundError(f"Human XML not found at {human_xml}")
        
        # Parse both files
        world_tree = ET.parse(WORLD_MODEL)
        world_root = world_tree.getroot()
        
        human_tree = ET.parse(human_xml)
        human_root = human_tree.getroot()
        
        # Merge human into world
        # 1. Merge attributes of root option/compiler/statistic/visual/size
        for tag in ['option', 'compiler', 'statistic', 'visual', 'size']:
            human_elem = human_root.find(tag)
            if human_elem is None:
                continue
            
            # Find destination in world (or create so we can filter attributes)
            world_elem = world_root.find(tag)
            if world_elem is None:
                world_elem = ET.SubElement(world_root, tag)
            
            # Merge attributes (WORLD PRESERVED in conflict)
            for k, v in human_elem.attrib.items():
                if k not in world_elem.attrib:
                    # Filter dangerous flags that might break robot physics
                    if tag == 'compiler' and k == 'inertiafromgeom':
                        continue
                    if tag == 'compiler' and k == 'angle':
                        continue # H1 might use degrees default
                    world_elem.set(k, v)
            
            # Merge children intelligently
            for h_child in human_elem:
                # Check if this tag exists in world_elem
                w_child = world_elem.find(h_child.tag)
                if w_child is not None:
                    # Merge attributes (WORLD PRESERVED)
                    for k, v in h_child.attrib.items():
                        if k not in w_child.attrib:
                            w_child.set(k, v)
                else:
                    world_elem.append(h_child)

        # 2. Append children for collection tags (asset, worldbody, default, actuator, sensor, equality)
        for tag in ['asset', 'worldbody', 'default', 'actuator', 'sensor', 'equality']:
            human_elems = human_root.findall(tag)
            world_elem = world_root.find(tag)
            
            if world_elem is None and human_elems:
                # If world doesn't have this section, create it
                world_elem = ET.SubElement(world_root, tag)
            
            if world_elem is not None:
                for h_elem in human_elems:
                     for child in h_elem:
                        # Filter out environmental elements from human XML to avoid duplicates
                        if tag == 'worldbody':
                             # Skip lights and floor geoms (assume world provides them)
                             if child.tag == 'light': continue
                             if child.tag == 'geom' and child.get('name') in ['floor', 'plane']: continue
                        
                        if tag == 'asset':
                             # Skip skybox and common environment materials
                             if child.tag == 'texture' and child.get('type') == 'skybox': continue
                             if child.get('name') in ['grid', 'grid_mat']: continue
                        
                        world_elem.append(child)

        # 3. Handle other top-level extension or custom tags
        # (Skip for now, assuming standard MJCF structure)
        
        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".xml", mode='w+')
        world_tree.write(temp_file, encoding='unicode')
        temp_file.flush()
        
        return temp_file
    
    def _initialize_env(self):
        """Initialize environment including human and safety wrapper."""
        super()._initialize_env()
        
        # Initialize safety wrapper
        ssm_config = SSMConfig(
            T_r=self._safety_config.T_r,
            T_s=self._safety_config.T_s,
            a_max=self._safety_config.a_max,
            C=self._safety_config.C,
            v_h_max=self._safety_config.v_h_max,
        )
        
        self._safety_wrapper = ISO15066Wrapper(
            model=self._mojo.model,
            data=self._mojo.data,
            ssm_config=ssm_config,
        )
        
        # Add robot geoms to safety wrapper
        self._register_robot_geoms()
        
        # Initialize human controller
        pd_gains = PDGains(
            kp=self._human_config.kp,
            kd=self._human_config.kd
        )
        self._human_controller = HumanController(
            model=self._mojo.model,
            data=self._mojo.data,
            gains=pd_gains
        )
        
        # Initialize scenario sampler if config provided
        if self._scenario_config is not None:
            self._scenario_sampler = ScenarioSampler(self._scenario_config)
        
        logger.info("SafetyBiGymEnv initialized with safety wrapper and human controller")
    
    def _register_robot_geoms(self):
        """Register robot collision geoms with safety wrapper."""
        # Find all geoms attached to robot bodies
        for i in range(self._mojo.model.ngeom):
            geom_name = mujoco.mj_id2name(
                self._mojo.model, mujoco.mjtObj.mjOBJ_GEOM, i
            )
            if geom_name is None:
                continue
            
            # Heuristic: robot geoms typically have specific naming patterns
            # This should be customized per robot model
            if any(pattern in geom_name.lower() for pattern in 
                   ['robot', 'gripper', 'hand', 'arm', 'link', 'ee']):
                self._safety_wrapper.add_robot_geom(geom_name)
    
    def _step_mujoco_simulation(self, action):
        """
        Override BiGym's simulation step to integrate human and safety.
        
        This is the key integration point. We take control of the sub-step
        loop to:
        1. Apply robot controls (first sub-step, as BiGym does)
        2. Update human PD targets (every sub-step)
        3. Capture peak forces (every sub-step)
        """
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Action shape mismatch: "
                f"expected {self.action_space.shape}, got {action.shape}."
            )
        
        # Clip action to bounds
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Reset peak force tracking for this step
        self._safety_wrapper._peak_forces.clear()
        self._safety_wrapper._peak_contact_info.clear()
        
        with self._env_health.track():
            for i in range(self._sub_steps_count):
                # === Robot Control ===
                # Apply robot action on first sub-step only (as BiGym does)
                if i == 0:
                    self.action_mode.step(action)
                else:
                    self._mojo.step()
                
                # === Human Control ===
                # Update human targets and apply controls (advances controller time by dt)
                if self._human_controller is not None:
                    self._human_controller.step(PHYSICS_DT)
                
                # === Safety Monitoring ===
                # Capture peak forces THIS sub-step
                self._safety_wrapper.check_safety_substep()
                
                # Post-constraint computation (for contact forces)
                mujoco.mj_rnePostConstraint(self._mojo.model, self._mojo.data)
        
        # Compute final safety info for this step
        self._compute_step_safety()
    
    def _compute_step_safety(self):
        """Compute safety metrics for the completed step."""
        # Get robot end-effector position (customize per robot)
        robot_pos = self._get_robot_ee_position()
        robot_vel = self._get_robot_ee_velocity()
        
        # Get human position (pelvis)
        human_pos = self._get_human_position()
        human_vel = self._get_human_velocity()
        
        # Compute SSM
        if robot_pos is not None and human_pos is not None:
            ssm_violation, ssm_margin, min_sep = self._safety_wrapper.compute_ssm(
                robot_pos, robot_vel, human_pos, human_vel
            )
        else:
            ssm_violation, ssm_margin, min_sep = False, float('inf'), float('inf')
        
        # Build safety info from peak forces
        self._step_safety_info = SafetyInfo(
            ssm_violation=ssm_violation,
            ssm_margin=ssm_margin,
            min_separation=min_sep,
        )
        
        # Add PFL info from peak contacts
        max_force = 0.0
        max_ratio = 0.0
        max_region = ""
        max_type = ""
        pfl_violation = False
        
        for region, contact_info in self._safety_wrapper._peak_contact_info.items():
            if contact_info.force > max_force:
                max_force = contact_info.force
                max_region = region
                max_type = contact_info.contact_type
            if contact_info.force_ratio > max_ratio:
                max_ratio = contact_info.force_ratio
            if contact_info.is_violation:
                pfl_violation = True
        
        self._step_safety_info.pfl_violation = pfl_violation
        self._step_safety_info.pfl_force_ratio = max_ratio
        self._step_safety_info.max_contact_force = max_force
        self._step_safety_info.contact_region = max_region
        self._step_safety_info.contact_type = max_type
        
        # Update episode stats
        if ssm_violation:
            self._episode_safety_stats["ssm_violations"] += 1
        if pfl_violation:
            self._episode_safety_stats["pfl_violations"] += 1
        self._episode_safety_stats["max_force"] = max(
            self._episode_safety_stats["max_force"], max_force
        )
        self._episode_safety_stats["min_ssm_margin"] = min(
            self._episode_safety_stats["min_ssm_margin"], ssm_margin
        )
    
    def _get_robot_ee_position(self) -> Optional[np.ndarray]:
        """Get robot end-effector position. Override per robot."""
        # Default: try to find gripper site or body
        try:
            # Try left gripper first
            site_id = mujoco.mj_name2id(
                self._mojo.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_site"
            )
            if site_id >= 0:
                return self._mojo.data.site_xpos[site_id].copy()
        except Exception:
            pass
        
        # Fallback: use robot pelvis
        return self._robot.pelvis.get_position()
    
    def _get_robot_ee_velocity(self) -> float:
        """Get robot end-effector velocity magnitude."""
        # Approximation from joint velocities
        return float(np.linalg.norm(self._robot.qvel[:6]))
    
    def _get_human_position(self) -> Optional[np.ndarray]:
        """Get human position (pelvis/root)."""
        if self._human_controller is None:
            return None
        
        # Find Pelvis body
        pelvis_id = mujoco.mj_name2id(
            self._mojo.model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis"
        )
        if pelvis_id >= 0:
            return self._mojo.data.xpos[pelvis_id].copy()
        return None
    
    def _get_human_velocity(self) -> float:
        """Get human velocity magnitude."""
        if self._human_controller is None:
            return 0.0
        
        # Find Pelvis body velocity
        pelvis_id = mujoco.mj_name2id(
            self._mojo.model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis"
        )
        if pelvis_id >= 0:
            return float(np.linalg.norm(self._mojo.data.cvel[pelvis_id, 3:6]))
        return 0.0
    
    def _on_reset(self):
        """Reset human position and sample new scenario."""
        super()._on_reset()
        
        # Reset safety stats
        self._episode_safety_stats = {
            "ssm_violations": 0,
            "pfl_violations": 0,
            "max_force": 0.0,
            "min_ssm_margin": float('inf'),
        }
        self._step_safety_info = None
        
        # Reset safety wrapper
        if self._safety_wrapper is not None:
            self._safety_wrapper.reset()
        
        # Sample new scenario if sampler exists
        if self._scenario_sampler is not None:
            # Use random seed for variation (can be deterministic if env seeded)
            # Use a large random int derived from internal state if possible, or just np.random
            seed = np.random.randint(0, 100000)
            self._current_scenario = self._scenario_sampler.sample_scenario(seed)
            self._spawn_human_scenario()
    
    def _spawn_human_scenario(self):
        """Spawn human according to sampled scenario."""
        if self._current_scenario is None:
            return
        
        # Get spawn position from scenario
        spawn = self._current_scenario
        
        # Find valid spawn position from config
        spawn_positions = self._human_config.spawn_positions
        if spawn_positions:
            idx = np.random.randint(len(spawn_positions))
            spawn_config = spawn_positions[idx]
            
            # Apply scenario's start_distance and approach_angle
            # This overrides the preset position if scenario specifies
            if hasattr(spawn, 'start_distance') and hasattr(spawn, 'approach_angle'):
                x = spawn.start_distance * np.cos(spawn.approach_angle)
                y = spawn.start_distance * np.sin(spawn.approach_angle)
                spawn_config = {
                    "pos": [x, y, 0.0],
                    "yaw": spawn.approach_angle + np.pi  # Face robot
                }
            
            self._set_human_position(spawn_config)
            
            # Update controller with new scenario
            if self._human_controller is not None:
                self._human_controller.set_scenario(self._current_scenario)
    
    def _set_human_position(self, spawn_config: Dict):
        """Set human position and orientation."""
        pos = spawn_config.get("pos", [0, 0, 0])
        yaw = spawn_config.get("yaw", 0)
        
        if self._human_controller is not None:
            self._human_controller.set_base_transform(np.array(pos), yaw)
            # Apply immediately to initial frame
            self._human_controller.reset()
    
    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to quaternion (w, x, y, z)."""
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        
        return np.array([w, x, y, z])
    
    def get_observation_space(self) -> spaces.Space:
        """Extend observation space with safety info."""
        obs_space = super().get_observation_space()
        
        # Add safety observations
        safety_obs = {
            "safety_ssm_margin": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "safety_pfl_ratio": spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "safety_human_distance": spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            ),
        }
        
        # Merge with existing observation space
        if isinstance(obs_space, spaces.Dict):
            return spaces.Dict({**obs_space.spaces, **safety_obs})
        else:
            return obs_space
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Extend observations with safety info."""
        obs = super().get_observation()
        
        # Add safety observations
        if self._step_safety_info is not None:
            info = self._step_safety_info
            obs["safety_ssm_margin"] = np.array([info.ssm_margin], dtype=np.float32)
            obs["safety_pfl_ratio"] = np.array([info.pfl_force_ratio], dtype=np.float32)
            obs["safety_human_distance"] = np.array([info.min_separation], dtype=np.float32)
        else:
            obs["safety_ssm_margin"] = np.array([float('inf')], dtype=np.float32)
            obs["safety_pfl_ratio"] = np.array([0.0], dtype=np.float32)
            obs["safety_human_distance"] = np.array([float('inf')], dtype=np.float32)
        
        return obs
    
    def _reward(self) -> float:
        """Compute reward with safety penalties."""
        base_reward = super()._reward()
        
        if self._step_safety_info is None:
            return base_reward
        
        safety_penalty = 0.0
        info = self._step_safety_info
        
        # SSM penalty: penalize negative margin (violation)
        if info.ssm_margin < 0:
            safety_penalty += self._safety_config.ssm_penalty_scale * abs(info.ssm_margin)
        
        # PFL penalty: penalize force ratio over 1.0
        if info.pfl_force_ratio > 1.0:
            excess = (info.pfl_force_ratio - 1.0) * 100  # Convert to percentage
            safety_penalty += self._safety_config.pfl_penalty_scale * excess
        
        return base_reward - safety_penalty
    
    def _fail(self) -> bool:
        """Check for failure including safety violations."""
        base_fail = super()._fail()
        
        if self._safety_config.terminate_on_violation:
            if self._step_safety_info is not None:
                if self._step_safety_info.pfl_violation:
                    return True
        
        return base_fail
    
    def get_info(self) -> Dict[str, Any]:
        """Extend info dict with safety stats."""
        info = super().get_info()
        
        # Add safety info
        if self._step_safety_info is not None:
            si = self._step_safety_info
            info["safety"] = {
                "ssm_violation": si.ssm_violation,
                "ssm_margin": si.ssm_margin,
                "pfl_violation": si.pfl_violation,
                "max_contact_force": si.max_contact_force,
                "pfl_force_ratio": si.pfl_force_ratio,
                "contact_region": si.contact_region,
            }
        
        # Add episode stats
        info["safety_episode"] = self._episode_safety_stats.copy()
        
        return info
    
    @property
    def safety_info(self) -> Optional[SafetyInfo]:
        """Get current step safety info."""
        return self._step_safety_info
    
    @property
    def episode_safety_stats(self) -> Dict[str, Any]:
        """Get episode-level safety statistics."""
        return self._episode_safety_stats.copy()
