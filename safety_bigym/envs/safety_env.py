"""
SafetyBiGymEnv - BiGym Environment with ISO 15066 Safety Monitoring

Wraps any BiGym task to add:
- SMPL-H human that follows AMASS motion with IK disruption
- ISO 15066 safety monitoring (SSM + PFL)
- Per-sub-step force capture for accurate PFL checking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Type, List, Dict
from dataclasses import asdict

import numpy as np
import mujoco
from lxml import etree

from bigym.bigym_env import BiGymEnv, PHYSICS_DT
from bigym.action_modes import ActionMode
from bigym.robots.robot import Robot

from safety_bigym.config import (
    SafetyConfig,
    HumanConfig,
    SSMConfig,
    get_spawn_positions,
)
from safety_bigym.safety import (
    ISO15066Wrapper,
    SafetyInfo,
    ContactInfo,
)
from safety_bigym.human import (
    HumanController,
    PDController,
    PDGains,
    TrajectoryPlanner,
    TrajectoryConfig,
    TrajectoryType,
)
from safety_bigym.scenarios import (
    ScenarioSampler,
    ScenarioParams,
    ParameterSpace,
)

logger = logging.getLogger(__name__)


class SafetyBiGymEnv(BiGymEnv):
    """
    BiGym environment with ISO 15066 safety monitoring.
    
    Extends BiGym to add:
    - SMPL-H human spawned in the scene
    - Human motion from AMASS with IK-based disruption
    - SSM (Speed and Separation Monitoring)
    - PFL (Power and Force Limiting) with per-sub-step capture
    
    The safety information is returned in info["safety"] dict containing:
    - ssm_violation: bool
    - pfl_violation: bool
    - ssm_margin: float (positive = safe, negative = violation)
    - max_contact_force: float
    - contact_region: str (body part if contact)
    - contact_type: str ("transient" or "quasi_static")
    """
    
    # Path to SMPL-H human asset
    HUMAN_ASSET_PATH = Path(__file__).parent.parent / "assets" / "smplh_human.xml"
    HUMAN_BODY_PATH = Path(__file__).parent.parent / "assets" / "smplh_human_body.xml"
    
    def __init__(
        self,
        action_mode: ActionMode,
        safety_config: Optional[SafetyConfig] = None,
        human_config: Optional[HumanConfig] = None,
        scenario_sampler: Optional[ScenarioSampler] = None,
        inject_human: bool = True,
        **kwargs
    ):
        """
        Initialize SafetyBiGymEnv.
        
        Args:
            action_mode: BiGym action mode for robot control
            safety_config: ISO 15066 safety parameters
            human_config: Human spawn and motion configuration
            scenario_sampler: Sampler for diverse scenarios (uses default if None)
            inject_human: Whether to inject human into scene (set False for testing)
            **kwargs: Additional arguments passed to BiGymEnv
        """
        self.safety_config = safety_config or SafetyConfig()
        self.human_config = human_config or HumanConfig()
        self._inject_human = inject_human
        
        # Human body IDs (set during injection)
        self._human_pelvis_id: Optional[int] = None
        self._human_root_qpos_start: Optional[int] = None
        
        # If injecting human, we need to do it BEFORE parent __init__ creates robot
        # because robot binds to physics and changing model after breaks bindings.
        # We'll use a temporary custom model path.
        if inject_human and self.HUMAN_BODY_PATH.exists():
            # Create merged world XML
            self._merged_model_path = self._create_merged_world()
            # Temporarily override class model path
            original_model_path = self._MODEL_PATH
            self.__class__._MODEL_PATH = Path(self._merged_model_path)
            try:
                super().__init__(action_mode=action_mode, **kwargs)
            finally:
                self.__class__._MODEL_PATH = original_model_path
            self._setup_human_indices()
        else:
            # Normal init without human
            super().__init__(action_mode=action_mode, **kwargs)
        
        # Scenario sampler
        if scenario_sampler is None:
            param_space = ParameterSpace(
                clip_paths=self.human_config.motion_clip_paths,
            )
            self.scenario_sampler = ScenarioSampler(
                parameter_space=param_space,
                motion_dir=self.human_config.motion_clip_dir,
            )
        else:
            self.scenario_sampler = scenario_sampler
        
        # Current scenario (set on reset)
        self._current_scenario: Optional[ScenarioParams] = None
        
        # Human components (initialized after human injection)
        self.human_controller: Optional[HumanController] = None
        self.safety_wrapper: Optional[ISO15066Wrapper] = None
        
        # Per-step safety tracking
        self._step_contacts: List[ContactInfo] = []
        self._step_safety_info: Optional[SafetyInfo] = None
        self._prev_human_pos: Optional[np.ndarray] = None
        
        # Spawn positions for this task
        self._spawn_positions = get_spawn_positions(self.task_name)
        if self.human_config.spawn_positions:
            self._spawn_positions = self.human_config.spawn_positions
        
        # Robot geom names for collision detection
        self._robot_geom_names: List[str] = []
        self._collect_robot_geoms()

        # Wire up the human<->{robot, floor} collision channel (bit 1). The
        # human MJCF uses contype/conaffinity=2 so scene props (dishwasher,
        # cabinets) on the default bit-0 channel never see the human.
        if self._inject_human:
            self._configure_collision_bits()

        # Initialize safety wrapper and human controller
        if self._inject_human:
            self._init_safety_wrapper()
            self._init_human_controller()
    
    def _create_merged_world(self) -> str:
        """
        Create a merged world XML that includes the human body.
        Returns path to the temporary merged XML file.
        """
        import tempfile
        from lxml import etree
        
        # Read world XML
        world_xml = etree.parse(str(self._MODEL_PATH))
        world_root = world_xml.getroot()
        
        # Read human body XML
        human_xml = etree.parse(str(self.HUMAN_BODY_PATH))
        human_root = human_xml.getroot()
        
        # Find worldbody in both
        world_worldbody = world_root.find(".//worldbody")
        human_worldbody = human_root.find(".//worldbody")
        
        # Copy human body hierarchy to world
        import copy
        for body in human_worldbody:
            world_worldbody.append(copy.deepcopy(body))
        
        # Copy human defaults (under class prefix to avoid conflicts)
        human_default = human_root.find(".//default")
        if human_default is not None:
            world_default = world_root.find("default")
            if world_default is None:
                world_default = etree.SubElement(world_root, "default")
            for child in human_default:
                world_default.append(copy.deepcopy(child))
        
        # Copy human actuators
        human_actuators = human_root.find(".//actuator")
        if human_actuators is not None:
            world_actuator = world_root.find("actuator")
            if world_actuator is None:
                world_actuator = etree.SubElement(world_root, "actuator")
            for act in human_actuators:
                world_actuator.append(copy.deepcopy(act))
        
        # Write merged XML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(etree.tostring(world_root, encoding="unicode"))
            return f.name
    
    def _setup_human_indices(self):
        """Find human body/joint indices after model is loaded."""
        model = self._mojo.model
        
        # Find Pelvis body
        self._human_pelvis_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis"
        )
        
        # Find root freejoint
        root_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        if root_joint_id >= 0:
            self._human_root_qpos_start = model.jnt_qposadr[root_joint_id]
        
        self._setup_human_ssm_bodies()

        logger.info(
            f"Human indices: pelvis_id={self._human_pelvis_id}, "
            f"root_qpos_start={self._human_root_qpos_start}, "
            f"ssm_bodies={len(self._human_body_ids)}"
        )

    _HUMAN_SSM_BODY_NAMES = [
        "Pelvis", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
        "Spine1", "Spine2", "Spine3", "Neck", "Head",
        "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist",
    ]

    def _setup_human_ssm_bodies(self):
        """Collect SMPL body IDs used for closest-joint SSM (T0.2)."""
        model = self._mojo.model
        self._human_body_ids: List[int] = []
        self._human_body_names: List[str] = []
        for name in self._HUMAN_SSM_BODY_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._human_body_ids.append(bid)
                self._human_body_names.append(name)
    
    def _collect_robot_geoms(self):
        """Collect robot geom names for safety wrapper."""
        # Get all geoms from robot
        model = self._mojo.model
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and self._is_robot_geom(name):
                self._robot_geom_names.append(name)
        logger.debug(f"Collected {len(self._robot_geom_names)} robot geoms")
    
    def _is_robot_geom(self, name: str) -> bool:
        """Check if geom belongs to robot."""
        # Exclude human collision geoms (they end with _col)
        if name.endswith("_col"):
            return False

        # Robot geoms typically have patterns like "h1/*", etc.
        # Only match h1/ prefix to avoid matching human body parts
        robot_patterns = ["h1/", "robotiq"]
        name_lower = name.lower()
        return any(p in name_lower for p in robot_patterns)

    _HUMAN_CHANNEL_BIT = 0b10  # bit 1 — the human<->{robot,floor} collision channel

    def _configure_collision_bits(self) -> None:
        """OR bit 1 into contype/conaffinity for every geom that must still
        collide with the human: robot collision geoms and the floor.

        Scene geoms (dishwasher, cabinets, walls) are intentionally left on the
        default bit-0 channel so the human passes through them. The human XML
        uses contype=conaffinity=2 (bit 1 only), so an object must be promoted
        into bit 1 here to physically interact with the human.

        Safety semantics are preserved: SSM is geometric (distance between
        tracked body centers, not contact-based), so it keeps working; PFL
        still observes human<->robot contact forces because that pair is on
        the shared bit-1 channel.
        """
        model = self._mojo.model
        bit = self._HUMAN_CHANNEL_BIT
        promoted = 0
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if not name:
                continue
            # Skip geoms that are disabled from collision entirely (visual-only).
            if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
                continue
            is_robot = self._is_robot_geom(name)
            is_floor = name == "floor" or name.endswith("/floor") or "ground" in name.lower()
            if is_robot or is_floor:
                model.geom_contype[gid] = int(model.geom_contype[gid]) | bit
                model.geom_conaffinity[gid] = int(model.geom_conaffinity[gid]) | bit
                promoted += 1
        logger.info(
            f"Promoted {promoted} geoms onto human collision channel "
            f"(bit={bit}); scene geoms remain isolated from the human."
        )
    

    
    def _init_safety_wrapper(self):
        """Initialize ISO 15066 safety wrapper."""
        self.safety_wrapper = ISO15066Wrapper(
            model=self._mojo.model,
            data=self._mojo.data,
            ssm_config=self.safety_config.ssm,
            human_geom_suffix="_col",
        )
        
        # Register robot geoms
        for geom_name in self._robot_geom_names:
            try:
                self.safety_wrapper.add_robot_geom(geom_name)
            except ValueError:
                pass  # Geom not found in merged model
        
        # Register fixture geoms (floor, walls, tables, etc.)
        fixture_patterns = ["floor", "wall", "table", "counter", "cabinet"]
        model = self._mojo.model
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name:
                name_lower = name.lower()
                if any(p in name_lower for p in fixture_patterns):
                    try:
                        self.safety_wrapper.add_fixture_geom(name)
                    except ValueError:
                        pass
        
        logger.info("Safety wrapper initialized")
    
    def _init_human_controller(self):
        """Initialize human motion controller."""
        gains = PDGains(
            kp=self.human_config.pd_kp,
            kd=self.human_config.pd_kd,
        )
        
        self.human_controller = HumanController(
            model=self._mojo.model,
            data=self._mojo.data,
            gains=gains,
        )
        
        logger.info("Human controller initialized")
    
    def _position_human(self, spawn_config: Dict[str, Any]):
        """
        Position human at spawn location.
        
        Args:
            spawn_config: Dict with 'pos' and 'yaw' keys
        """
        if self._human_root_qpos_start is None:
            return
        
        pos = np.array(spawn_config.get("pos", [2.0, 0.0, 0.0]))
        yaw = spawn_config.get("yaw", 0.0)
        
        # Get correct Z position from AMASS clip (pelvis height when standing)
        initial_z = 1.0  # Default standing pelvis height
        if self.human_controller is not None and self.human_controller.clip is not None:
            _, root_trans, _ = self.human_controller.clip.get_frame(0)
            initial_z = root_trans[2]  # Z from AMASS first frame
        
        # Set root offset in human controller (shifts AMASS motion to spawn pos)
        if self.human_controller is not None:
            self.human_controller.set_root_offset(pos)
        
        # Set root position (freejoint: 3 pos + 4 quat)
        # Use XY from spawn, Z from AMASS (correct pelvis height)
        qpos_start = self._human_root_qpos_start
        self._mojo.data.qpos[qpos_start:qpos_start + 3] = [pos[0], pos[1], initial_z]
        
        # Set orientation (quaternion from yaw)
        # Quaternion for rotation around Z axis
        quat = np.array([
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2)
        ])
        self._mojo.data.qpos[qpos_start + 3:qpos_start + 7] = quat
        
        # Forward kinematics to update body positions
        mujoco.mj_forward(self._mojo.model, self._mojo.data)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment with new scenario."""
        # Parent reset (resets robot and world)
        obs, info = super().reset(seed=seed, options=options)
        
        if not self._inject_human:
            return obs, info
        
        # Sample new scenario
        scenario_seed = seed if seed is not None else np.random.randint(2**31)
        self._current_scenario = self.scenario_sampler.sample_scenario(scenario_seed)
        
        # Reset human controller
        if self.human_controller is not None:
            self.human_controller.reset()
            
            # Load motion clip
            if self._current_scenario.clip_path:
                # Build full path from motion_clip_dir + relative clip path
                if self.human_config.motion_clip_dir:
                    full_clip_path = str(Path(self.human_config.motion_clip_dir) / self._current_scenario.clip_path)
                else:
                    full_clip_path = self._current_scenario.clip_path
                    
                try:
                    self.human_controller.load_clip(full_clip_path)
                except Exception as e:
                    logger.warning(f"Failed to load clip: {e}")
            
            # Set scenario parameters (ScenarioParams is now unified)
            self.human_controller.set_scenario(self._current_scenario)
        
        # Compute spawn position from scenario parameters
        # approach_angle: 0° = in front (+X), 90° = left (+Y), etc.
        angle_rad = np.deg2rad(self._current_scenario.approach_angle)
        dist = self._current_scenario.spawn_distance
        
        # Robot is at approximately (0, 0) — spawn on a circle around it
        spawn_x = dist * np.cos(angle_rad)
        spawn_y = dist * np.sin(angle_rad)
        
        # Yaw to face toward robot (opposite of spawn direction)
        face_robot_yaw = np.arctan2(-spawn_y, -spawn_x)
        
        spawn_config = {
            "pos": [spawn_x, spawn_y, 0.0],
            "yaw": face_robot_yaw,
        }
        self._position_human(spawn_config)
        
        # --- Create trajectory planner from scenario params ---
        trajectory_type_map = {
            "PASS_BY": TrajectoryType.PASS_BY,
            "APPROACH_LOITER_DEPART": TrajectoryType.APPROACH_LOITER_DEPART,
            "ARC": TrajectoryType.ARC,
        }
        traj_type_str = getattr(self._current_scenario, 'trajectory_type', 'PASS_BY')
        traj_type = trajectory_type_map.get(traj_type_str, TrajectoryType.PASS_BY)
        
        traj_config = TrajectoryConfig(
            trajectory_type=traj_type,
            robot_pos=np.array([0.0, 0.0]),  # Robot at origin
            spawn_pos=np.array([spawn_x, spawn_y]),
            approach_yaw=face_robot_yaw,
            pass_by_offset=getattr(self._current_scenario, 'pass_by_offset', 1.0),
            pass_by_side=getattr(self._current_scenario, 'pass_by_side', 1),
            closest_approach=getattr(self._current_scenario, 'closest_approach', 1.0),
            loiter_duration=getattr(self._current_scenario, 'loiter_duration', 2.0),
            departure_angle=getattr(self._current_scenario, 'departure_angle', 150.0),
            arc_radius=getattr(self._current_scenario, 'arc_radius', 1.5),
            arc_extent=getattr(self._current_scenario, 'arc_extent', 120.0),
            walk_speed=getattr(self._current_scenario, 'walk_speed', 1.2),
        )
        
        trajectory_planner = TrajectoryPlanner(traj_config)
        
        if self.human_controller is not None:
            self.human_controller.set_trajectory_planner(trajectory_planner)
        
        # Orient AMASS motion direction toward robot
        if self.human_controller is not None:
            self.human_controller.set_root_yaw(face_robot_yaw)
        
        # Reset safety wrapper
        if self.safety_wrapper is not None:
            self.safety_wrapper.reset()
        
        # Clear per-step tracking
        self._step_contacts = []
        self._step_safety_info = None
        self._prev_human_pos = None
        
        # Add scenario info
        info["scenario"] = {
            "disruption_type": self._current_scenario.disruption_type.name,
            "trigger_time": self._current_scenario.trigger_time,
            "clip_path": self._current_scenario.clip_path,
            "trajectory_type": traj_type_str,
        }
        
        return obs, info
    
    def _step_mujoco_simulation(self, action):
        """
        Step physics with human control and safety monitoring.
        
        This overrides BiGym's method to:
        1. Update human PD targets before each sub-step
        2. Capture contact forces after each sub-step
        """
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Action shape mismatch: "
                f"expected {self.action_space.shape}, but got {action.shape}."
            )
        if np.any(action < self.action_space.low) or np.any(
            action > self.action_space.high
        ):
            clipped_action = np.clip(
                action, self.action_space.low, self.action_space.high
            )
            raise ValueError(
                f"Action {action} is out of the action space bounds. "
                f"Overhead: {action - clipped_action}"
            )
        
        # Clear per-step contacts
        self._step_contacts = []
        
        # Get robot state for IK (if human controller needs it)
        robot_state = self._get_robot_state()
        
        with self._env_health.track():
            for i in range(self._sub_steps_count):
                # 1. Update human controls BEFORE physics step
                if self.human_controller is not None:
                    self.human_controller.step(PHYSICS_DT, robot_state=robot_state)
                
                # 2. Apply robot action (BiGym's original logic)
                if i == 0:
                    self.action_mode.step(action)
                else:
                    self._mojo.step()
                
                mujoco.mj_rnePostConstraint(self._mojo.model, self._mojo.data)
                
                # 3. Capture forces AFTER physics step
                if self.safety_wrapper is not None:
                    substep_contacts = self.safety_wrapper.check_safety_substep()
                    self._step_contacts.extend(substep_contacts)
        
        # Aggregate safety info for this step
        self._aggregate_safety_info()
    
    def _get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state for IK computation."""
        state = {}
        
        # Robot pelvis position
        try:
            pelvis_pos = self._robot.pelvis.get_position()
            state["robot_pos"] = pelvis_pos
        except:
            state["robot_pos"] = np.zeros(3)
        
        # End effector positions (if available)
        try:
            # This depends on the robot type
            state["ee_pos"] = self._robot.get_ee_position()
        except:
            pass
        
        return state
    
    def _aggregate_safety_info(self):
        """Aggregate sub-step contacts into step-level safety info.

        Closest-joint SSM (T0.2) + shared build_safety_info (T0.1)."""
        if self.safety_wrapper is None:
            self._step_safety_info = SafetyInfo()
            return

        robot_positions, robot_names, robot_vel = self._robot_ssm_state()
        human_positions, human_names, human_vel = self._human_ssm_state()

        self._step_safety_info = self.safety_wrapper.build_safety_info(
            contacts=self._step_contacts,
            robot_positions=robot_positions,
            robot_vel=robot_vel,
            human_positions=human_positions,
            human_vel=human_vel,
            human_names=human_names,
            robot_names=robot_names,
        )
        return

    def _robot_ssm_state(self):
        """Positions (Nr,3), names, and max link speed for every robot geom
        registered on the safety wrapper. Falls back to pelvis-only."""
        model = self._mojo.model
        data = self._mojo.data
        names = []
        positions = []
        max_vel = 0.0
        for gid in sorted(self.safety_wrapper.robot_geoms):
            positions.append(data.geom_xpos[gid].copy())
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            names.append(gname or f"geom_{gid}")
            body_id = int(model.geom_bodyid[gid])
            if body_id >= 0:
                linvel = float(np.linalg.norm(data.cvel[body_id, 3:6]))
                if linvel > max_vel:
                    max_vel = linvel
        if not positions:
            try:
                pos = np.asarray(self._robot.pelvis.get_position(), dtype=float)
                positions = [pos]
                names = ["h1/pelvis"]
                pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "h1/pelvis")
                if pid >= 0:
                    max_vel = float(np.linalg.norm(data.cvel[pid, 3:6]))
            except Exception:
                positions = [np.zeros(3)]
                names = ["unknown"]
        return np.stack(positions), names, max_vel

    def _human_ssm_state(self):
        """Positions (Nh,3), names, and max body speed for every SMPL body we
        track. Max-over-bodies speed is ISO 15066\'s conservative reading."""
        data = self._mojo.data
        body_ids = getattr(self, "_human_body_ids", [])
        if not body_ids:
            if self._human_pelvis_id is not None:
                pos = data.xpos[self._human_pelvis_id].copy()
                return np.stack([pos]), ["Pelvis"], 0.0
            return np.zeros((1, 3)), ["unknown"], 0.0
        positions = [data.xpos[bid].copy() for bid in body_ids]
        names = list(self._human_body_names)
        max_vel = 0.0
        for bid in body_ids:
            linvel = float(np.linalg.norm(data.cvel[bid, 3:6]))
            if linvel > max_vel:
                max_vel = linvel
        if self._human_pelvis_id is not None:
            self._prev_human_pos = data.xpos[self._human_pelvis_id].copy()
            self._prev_sim_time = data.time
        return np.stack(positions), names, max_vel

    def _on_step(self):
        """Called after step - log violations if configured."""
        super()._on_step()
        
        if self._step_safety_info is None:
            return
        
        if self.safety_config.log_violations:
            if self._step_safety_info.ssm_violation:
                required_sep = self._step_safety_info.min_separation - self._step_safety_info.ssm_margin
                logger.warning(
                    f"SSM Violation! Distance: {self._step_safety_info.min_separation:.2f}m, "
                    f"Required: {required_sep:.2f}m, Margin: {self._step_safety_info.ssm_margin:.3f}m"
                )
            if self._step_safety_info.pfl_violation:
                logger.warning(
                    f"PFL Violation! Force: {self._step_safety_info.max_contact_force:.1f}N "
                    f"on {self._step_safety_info.contact_region}"
                )
    
    def _reward(self) -> float:
        """Get reward with optional violation penalty."""
        base_reward = super()._reward()
        
        if self.safety_config.add_violation_penalty and self._step_safety_info:
            if self._step_safety_info.ssm_violation or self._step_safety_info.pfl_violation:
                base_reward -= self.safety_config.violation_penalty
                
        return base_reward

    def step(self, action):
        """Step environment and add privileged info."""
        obs, reward, done, truncated, info = super().step(action)
        
        # Add human motion phase to info
        if self.human_controller is not None:
            info["human_phase"] = self.human_controller.current_phase
        
        # Add privileged info for SafePolicy
        try:
            # Get joint positions for freezing/retreat behavior
            # Need to use ACTUATED joint positions to match action space
            if hasattr(self._robot, "qpos_actuated"):
                qpos = self._robot.qpos_actuated
            elif hasattr(self._robot, "get_joint_positions"):
                qpos = self._robot.get_joint_positions()
            elif hasattr(self._robot, "qpos"):
                qpos = self._robot.qpos
            else:
                qpos = None
                
            if qpos is not None:
                if "safety" not in info:
                    info["safety"] = {}
                info["safety"]["qpos"] = qpos
        except Exception:
            pass
            
        return obs, reward, done, truncated, info
    
    @property
    def terminate(self) -> bool:
        """Check termination including safety violations."""
        base_terminate = super().terminate
        
        if self.safety_config.terminate_on_violation and self._step_safety_info:
            if self._step_safety_info.ssm_violation or self._step_safety_info.pfl_violation:
                return True
        
        return base_terminate
    
    def get_info(self) -> Dict[str, Any]:
        """Get info dict including safety information."""
        info = super().get_info()
        
        if self._step_safety_info is not None:
            info["safety"] = self._step_safety_info.to_dict()
        else:
            info["safety"] = {}
        
        # Add current scenario info
        if self._current_scenario is not None:
            info["scenario"] = {
                "disruption_type": self._current_scenario.disruption_type.name,
                "trigger_time": self._current_scenario.trigger_time,
            }
        
        return info


def make_safety_env(
    task_cls: type,
    action_mode: ActionMode,
    safety_config: Optional[SafetyConfig] = None,
    human_config: Optional[HumanConfig] = None,
    scenario_sampler: Optional[ScenarioSampler] = None,
    inject_human: bool = True,
    **kwargs,
) -> SafetyBiGymEnv:
    """
    Create a SafetyBiGymEnv for any BiGym task.

    Dynamically creates a class that inherits from both SafetyBiGymEnv and
    the given BiGym task class, so you get the task's scene + safety monitoring.

    Example::

        from bigym.envs.reach_target import ReachTargetSingle
        from bigym.envs.pick_and_place import PickBox

        env = make_safety_env(ReachTargetSingle, action_mode=..., human_config=...)
        env = make_safety_env(PickBox, action_mode=..., inject_human=True)

    Args:
        task_cls: A BiGym task class (e.g. ReachTargetSingle, PickBox).
                  Pass BiGymEnv for the default empty scene.
        action_mode: BiGym action mode for robot control.
        safety_config: ISO 15066 safety parameters.
        human_config: Human spawn and motion configuration.
        scenario_sampler: Sampler for diverse scenarios.
        inject_human: Whether to inject human into scene.
        **kwargs: Extra arguments forwarded to the task class __init__.

    Returns:
        An environment instance with both task behaviour and safety monitoring.
    """
    # Create a combined class: SafetyBiGymEnv first (overrides step/reset),
    # then task_cls (provides scene setup and reward).
    cls_name = f"Safety{task_cls.__name__}"
    combined_cls = type(cls_name, (SafetyBiGymEnv, task_cls), {})

    return combined_cls(
        action_mode=action_mode,
        safety_config=safety_config,
        human_config=human_config,
        scenario_sampler=scenario_sampler,
        inject_human=inject_human,
        **kwargs,
    )
