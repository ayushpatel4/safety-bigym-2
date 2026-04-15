"""
ISO 15066 Safety Wrapper

Implements ISO/TS 15066:2016 safety monitoring for collaborative robots:
- SSM (Speed and Separation Monitoring)
- PFL (Power and Force Limiting)
- Sub-step contact force capture
- Quasi-static vs transient contact classification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
import mujoco

from safety_bigym.config import SSMConfig
from safety_bigym.safety.pfl_limits import (
    PFL_LIMITS,
    GEOM_TO_REGION,
    get_region_for_geom,
    get_limits_for_geom,
    BodyRegionLimits,
)


@dataclass
class ContactInfo:
    """Information about a single contact."""
    
    geom1_name: str
    geom2_name: str
    force: float
    contact_type: str  # 'quasi_static' or 'transient'
    body_region: Optional[str]
    is_human_robot: bool  # True if human-robot contact
    
    # Violation info
    is_violation: bool = False
    force_ratio: float = 0.0
    force_limit: float = 0.0


@dataclass
class SafetyInfo:
    """
    Safety monitoring output.
    
    This is the interface that will be used in Phase 8 for training.
    """
    
    # Binary flags
    ssm_violation: bool = False
    pfl_violation: bool = False
    
    # Proportional signals for reward shaping
    ssm_margin: float = float('inf')  # d_min - S_p (negative = violation)
    pfl_force_ratio: float = 0.0      # max(F_actual / F_limit) across regions
    
    # Logging and analysis
    min_separation: float = float('inf')
    max_contact_force: float = 0.0
    contact_region: str = ""
    contact_type: str = ""
    
    # Detailed contact info
    contacts: List[ContactInfo] = field(default_factory=list)
    
    # Per-region violation counts
    violations_by_region: Dict[str, int] = field(default_factory=dict)
    
    # State info for privileged policies
    robot_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    human_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Names of the closest human joint / robot link that drove ssm_margin.
    # Populated by build_safety_info when pairwise SSM is used. Empty strings
    # when SSM was skipped or only a single point was passed with no name list.
    closest_human_joint: str = ""
    closest_robot_link: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for info['safety']. Cast numpy scalars to
        Python primitives so the dict stays JSON-serialisable (W&B, pickle)."""
        return {
            'ssm_violation': bool(self.ssm_violation),
            'pfl_violation': bool(self.pfl_violation),
            'ssm_margin': float(self.ssm_margin),
            'pfl_force_ratio': float(self.pfl_force_ratio),
            'min_separation': float(self.min_separation),
            'max_contact_force': float(self.max_contact_force),
            'contact_region': self.contact_region,
            'contact_type': self.contact_type,
            'violations_by_region': self.violations_by_region.copy(),
            'robot_pos': list(self.robot_pos),
            'human_pos': list(self.human_pos),
            'closest_human_joint': self.closest_human_joint,
            'closest_robot_link': self.closest_robot_link,
        }


class ISO15066Wrapper:
    """
    ISO 15066 safety monitoring wrapper.
    
    Monitors robot-human interactions for:
    - SSM: Speed and Separation Monitoring
    - PFL: Power and Force Limiting
    
    Captures peak forces at sub-step level and classifies contacts
    as quasi-static (clamping) or transient (impact).
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ssm_config: Optional[SSMConfig] = None,
        human_geom_prefix: str = "",
        human_geom_suffix: str = "_col",
        robot_geom_names: Optional[Set[str]] = None,
        fixture_geom_names: Optional[Set[str]] = None,
    ):
        """
        Initialize the safety wrapper.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            ssm_config: SSM configuration (uses defaults if None)
            human_geom_prefix: Prefix for human collision geoms
            human_geom_suffix: Suffix for human collision geoms
            robot_geom_names: Set of robot geom names (auto-detected if None)
            fixture_geom_names: Set of fixture/environment geom names
        """
        self.model = model
        self.data = data
        self.ssm_config = ssm_config or SSMConfig()
        
        # Build geom sets (IDs)
        self.human_geoms: Set[int] = set()
        self.robot_geoms: Set[int] = set()
        self.fixture_geoms: Set[int] = set()
        
        # Also track fixture names for quasi-static detection
        self.fixture_geom_names: Set[str] = set()
        
        self._build_geom_sets(
            human_geom_prefix, 
            human_geom_suffix,
            robot_geom_names,
            fixture_geom_names,
        )
        
        # Geom ID to name cache
        self._geom_id_to_name: Dict[int, str] = {}
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name:
                self._geom_id_to_name[i] = name
        
        # Per-step tracking for quasi-static detection
        self._human_contacts_this_step: Dict[str, Set[str]] = defaultdict(set)

        # Peak force tracking for sub-step capture
        self._peak_forces: Dict[str, float] = {}
        self._peak_contact_info: Dict[str, ContactInfo] = {}

        # Indices of the (human, robot) point pair with minimum separation
        # from the most recent compute_ssm call. -1 = not computed yet.
        self.last_closest_human_idx: int = -1
        self.last_closest_robot_idx: int = -1
    
    def _build_geom_sets(
        self,
        human_prefix: str,
        human_suffix: str,
        robot_names: Optional[Set[str]],
        fixture_names: Optional[Set[str]],
    ):
        """Build sets of geom IDs by category."""
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if not name:
                continue
            
            # Human geoms (by suffix)
            if name.endswith(human_suffix):
                self.human_geoms.add(i)
            
            # Robot geoms (by name or user-specified)
            if robot_names and name in robot_names:
                self.robot_geoms.add(i)
            
            # Fixture geoms
            if fixture_names and name in fixture_names:
                self.fixture_geoms.add(i)
                self.fixture_geom_names.add(name)
    
    def add_robot_geom(self, name: str):
        """Add a robot geom by name."""
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            self.robot_geoms.add(gid)
    
    def add_fixture_geom(self, name: str):
        """Add a fixture geom by name."""
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            self.fixture_geoms.add(gid)
            self.fixture_geom_names.add(name)
    
    def reset(self):
        """Reset per-episode tracking."""
        self._human_contacts_this_step.clear()
        self._peak_forces.clear()
        self._peak_contact_info.clear()
    
    def _get_geom_name(self, geom_id: int) -> str:
        """Get geom name from ID."""
        return self._geom_id_to_name.get(geom_id, f"geom_{geom_id}")
    
    def _classify_contact_type(
        self,
        human_geom_id: int,
        other_geom_id: int,
    ) -> str:
        """
        Classify contact as quasi-static or transient.
        
        Quasi-static: Human part contacts BOTH robot AND fixture (clamping)
        Transient: All other contacts
        
        Args:
            human_geom_id: Human body part geom ID
            other_geom_id: The other geom in this contact
            
        Returns:
            'quasi_static' or 'transient'
        """
        human_name = self._get_geom_name(human_geom_id)
        
        # Check what else this human part is contacting
        contacts_robot = other_geom_id in self.robot_geoms
        
        # Check if human part is also touching a fixture (by name)
        human_other_contacts = self._human_contacts_this_step.get(human_name, set())
        contacts_fixture = len(human_other_contacts & self.fixture_geom_names) > 0
        
        # If robot contact AND already contacting fixture = clamping
        if contacts_robot and contacts_fixture:
            return 'quasi_static'
        
        return 'transient'
    
    def _process_contact(
        self,
        contact_idx: int,
    ) -> Optional[ContactInfo]:
        """
        Process a single contact and return info if human-robot.
        
        Args:
            contact_idx: Index in data.contact
            
        Returns:
            ContactInfo if human-robot contact, None otherwise
        """
        contact = self.data.contact[contact_idx]
        geom1 = contact.geom1
        geom2 = contact.geom2
        
        # Identify which is human and which is robot
        human_geom = None
        robot_geom = None
        
        if geom1 in self.human_geoms and geom2 in self.robot_geoms:
            human_geom, robot_geom = geom1, geom2
        elif geom2 in self.human_geoms and geom1 in self.robot_geoms:
            human_geom, robot_geom = geom2, geom1
        elif geom1 in self.human_geoms:
            # Human-fixture or human-other contact
            human_name = self._get_geom_name(geom1)
            other_name = self._get_geom_name(geom2)
            self._human_contacts_this_step[human_name].add(other_name)
            return None
        elif geom2 in self.human_geoms:
            human_name = self._get_geom_name(geom2)
            other_name = self._get_geom_name(geom1)
            self._human_contacts_this_step[human_name].add(other_name)
            return None
        else:
            return None  # Not a human contact
        
        # Get contact force
        force_vec = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, contact_idx, force_vec)
        force_magnitude = np.linalg.norm(force_vec[:3])
        
        # Get geom names
        human_name = self._get_geom_name(human_geom)
        robot_name = self._get_geom_name(robot_geom)
        
        # Track contact for quasi-static detection
        self._human_contacts_this_step[human_name].add(robot_name)
        
        # Classify contact type
        contact_type = self._classify_contact_type(human_geom, robot_geom)
        
        # Get body region and limits
        body_region = get_region_for_geom(human_name)
        limits = get_limits_for_geom(human_name)
        
        # Check for violation
        is_violation = False
        force_ratio = 0.0
        force_limit = 0.0
        
        if limits is not None:
            force_limit = limits.get_force_limit(contact_type)
            is_violation, force_ratio = limits.check_violation(
                force_magnitude, 
                contact_type
            )
        
        return ContactInfo(
            geom1_name=human_name,
            geom2_name=robot_name,
            force=force_magnitude,
            contact_type=contact_type,
            body_region=body_region,
            is_human_robot=True,
            is_violation=is_violation,
            force_ratio=force_ratio,
            force_limit=force_limit,
        )
    
    def check_safety_substep(self) -> List[ContactInfo]:
        """
        Check safety at current sub-step.
        
        Call this after each mj_step() to capture peak forces.
        
        Returns:
            List of ContactInfo for human-robot contacts
        """
        contacts = []
        
        # First pass: track all human contacts for quasi-static detection
        self._human_contacts_this_step.clear()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            
            # Track human-fixture contacts
            if geom1 in self.human_geoms and geom2 in self.fixture_geoms:
                human_name = self._get_geom_name(geom1)
                fixture_name = self._get_geom_name(geom2)
                self._human_contacts_this_step[human_name].add(fixture_name)
            elif geom2 in self.human_geoms and geom1 in self.fixture_geoms:
                human_name = self._get_geom_name(geom2)
                fixture_name = self._get_geom_name(geom1)
                self._human_contacts_this_step[human_name].add(fixture_name)
        
        # Second pass: process human-robot contacts
        for i in range(self.data.ncon):
            info = self._process_contact(i)
            if info is not None:
                contacts.append(info)
                
                # Track peak force for this region
                region = info.body_region or "unknown"
                if region not in self._peak_forces or info.force > self._peak_forces[region]:
                    self._peak_forces[region] = info.force
                    self._peak_contact_info[region] = info
        
        return contacts
    
    def compute_ssm(
        self,
        robot_pos: np.ndarray,
        robot_vel: float,
        human_pos: np.ndarray,
        human_vel: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """Compute SSM violation and margin using the closest-joint distance.

        Accepts either (3,) single points or (N,3) arrays per agent. When
        arrays are given, d_min is the minimum distance across all pairs, and
        the indices of the closest pair are cached as last_closest_human_idx
        and last_closest_robot_idx so the caller can resolve them to names.
        """
        human_arr = np.atleast_2d(np.asarray(human_pos, dtype=float))
        robot_arr = np.atleast_2d(np.asarray(robot_pos, dtype=float))

        if human_arr.shape[-1] != 3 or robot_arr.shape[-1] != 3:
            raise ValueError(
                f"positions must end with dim=3, got human {human_arr.shape} "
                f"robot {robot_arr.shape}"
            )

        diff = human_arr[:, None, :] - robot_arr[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        flat_idx = int(dists.argmin())
        h_idx, r_idx = np.unravel_index(flat_idx, dists.shape)
        d_min = float(dists[h_idx, r_idx])

        self.last_closest_human_idx = int(h_idx)
        self.last_closest_robot_idx = int(r_idx)

        S_p = self.ssm_config.compute_separation_distance(robot_vel, human_vel)
        margin = d_min - S_p
        return bool(margin < 0), float(margin), d_min

    def build_safety_info(
        self,
        contacts,
        robot_positions=None,
        robot_vel=0.0,
        human_positions=None,
        human_vel=None,
        human_names=None,
        robot_names=None,
    ):
        """Aggregate contacts + SSM state into one SafetyInfo. Shared by
        step(), check_safety_no_step(), and env._aggregate_safety_info."""
        info = SafetyInfo()
        if robot_positions is not None and human_positions is not None:
            self._ssm_into(info, robot_positions, robot_vel,
                           human_positions, human_vel,
                           human_names, robot_names)
        self._pfl_into(info, contacts)
        return info

    def _ssm_into(self, info, robot_positions, robot_vel,
                  human_positions, human_vel, human_names, robot_names):
        is_v, margin, d_min = self.compute_ssm(
            robot_positions, robot_vel, human_positions, human_vel
        )
        info.ssm_violation = is_v
        info.ssm_margin = margin
        info.min_separation = d_min
        h_idx = self.last_closest_human_idx
        r_idx = self.last_closest_robot_idx
        if human_names is not None and 0 <= h_idx < len(human_names):
            info.closest_human_joint = human_names[h_idx]
        if robot_names is not None and 0 <= r_idx < len(robot_names):
            info.closest_robot_link = robot_names[r_idx]
        human_arr = np.atleast_2d(np.asarray(human_positions, dtype=float))
        robot_arr = np.atleast_2d(np.asarray(robot_positions, dtype=float))
        if 0 <= h_idx < len(human_arr):
            info.human_pos = human_arr[h_idx].tolist()
        if 0 <= r_idx < len(robot_arr):
            info.robot_pos = robot_arr[r_idx].tolist()

    def _pfl_into(self, info, contacts):
        max_force = 0.0
        max_region = ""
        max_type = ""
        max_ratio = 0.0
        violations_by_region = defaultdict(int)
        pfl_violation = False
        for c in contacts:
            if c.force > max_force:
                max_force = c.force
                max_region = c.body_region or ""
                max_type = c.contact_type
            if c.force_ratio > max_ratio:
                max_ratio = c.force_ratio
            if c.is_violation:
                pfl_violation = True
                region = c.body_region or "unknown"
                violations_by_region[region] += 1
        info.pfl_violation = pfl_violation
        info.pfl_force_ratio = max_ratio
        info.max_contact_force = max_force
        info.contact_region = max_region
        info.contact_type = max_type
        info.contacts = list(contacts)
        info.violations_by_region = dict(violations_by_region)

    def step(
        self,
        n_substeps: int = 1,
        robot_pos: Optional[np.ndarray] = None,
        robot_vel: float = 0.0,
        human_pos: Optional[np.ndarray] = None,
        human_vel: Optional[float] = None,
    ) -> SafetyInfo:
        """
        Run physics sub-steps and collect safety information.
        
        This method calls mj_step internally to capture peak forces.
        
        Args:
            n_substeps: Number of physics sub-steps
            robot_pos: Robot position for SSM (skips SSM if None)
            robot_vel: Robot velocity magnitude
            human_pos: Human position for SSM
            human_vel: Human velocity (uses config default if None)
            
        Returns:
            SafetyInfo with all safety data
        """
        # Reset peak tracking for this step
        self._peak_forces.clear()
        self._peak_contact_info.clear()
        
        all_contacts = []
        
        # Run sub-steps
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
            contacts = self.check_safety_substep()
            all_contacts.extend(contacts)
        
        # Build safety info
        info = SafetyInfo()
        
        # SSM check
        if robot_pos is not None and human_pos is not None:
            ssm_violation, ssm_margin, min_sep = self.compute_ssm(
                robot_pos, robot_vel, human_pos, human_vel
            )
            info.ssm_violation = ssm_violation
            info.ssm_margin = ssm_margin
            info.min_separation = min_sep
        
        # PFL check from peak forces
        max_ratio = 0.0
        max_force = 0.0
        max_region = ""
        max_type = ""
        
        violations_by_region: Dict[str, int] = defaultdict(int)
        
        for region, contact_info in self._peak_contact_info.items():
            if contact_info.force > max_force:
                max_force = contact_info.force
                max_region = region
                max_type = contact_info.contact_type
            
            if contact_info.force_ratio > max_ratio:
                max_ratio = contact_info.force_ratio
            
            if contact_info.is_violation:
                info.pfl_violation = True
                violations_by_region[region] += 1
        
        info.pfl_force_ratio = max_ratio
        info.max_contact_force = max_force
        info.contact_region = max_region
        info.contact_type = max_type
        info.contacts = all_contacts
        info.violations_by_region = dict(violations_by_region)
        
        return info
    
    def check_safety_no_step(
        self,
        robot_pos: Optional[np.ndarray] = None,
        robot_vel: float = 0.0,
        human_pos: Optional[np.ndarray] = None,
        human_vel: Optional[float] = None,
    ) -> SafetyInfo:
        """
        Check safety at current state without stepping physics.
        
        Use this when the physics step is done externally.
        
        Args:
            robot_pos: Robot position for SSM
            robot_vel: Robot velocity magnitude
            human_pos: Human position for SSM
            human_vel: Human velocity
            
        Returns:
            SafetyInfo
        """
        contacts = self.check_safety_substep()
        
        info = SafetyInfo()
        
        # SSM
        if robot_pos is not None and human_pos is not None:
            ssm_violation, ssm_margin, min_sep = self.compute_ssm(
                robot_pos, robot_vel, human_pos, human_vel
            )
            info.ssm_violation = ssm_violation
            info.ssm_margin = ssm_margin
            info.min_separation = min_sep
        
        # PFL
        max_ratio = 0.0
        max_force = 0.0
        max_region = ""
        max_type = ""
        
        for contact in contacts:
            if contact.force > max_force:
                max_force = contact.force
                max_region = contact.body_region or ""
                max_type = contact.contact_type
            
            if contact.force_ratio > max_ratio:
                max_ratio = contact.force_ratio
            
            if contact.is_violation:
                info.pfl_violation = True
        
        info.pfl_force_ratio = max_ratio
        info.max_contact_force = max_force
        info.contact_region = max_region
        info.contact_type = max_type
        info.contacts = contacts
        
        return info
