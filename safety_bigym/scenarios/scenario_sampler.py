"""
Parameterised Scenario Sampler

Samples diverse human behaviour scenarios for safety filter evaluation.
Each scenario defines:
- Motion clip selection
- Disruption type and timing
- Speed/height variations
- Approach geometry
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

from safety_bigym.scenarios.disruption_types import (
    DisruptionType,
    DisruptionConfig,
    DEFAULT_CONFIGS,
)


@dataclass
class ScenarioParams:
    """Complete parameters for a single scenario."""
    
    # Motion clip
    clip_path: str
    
    # Disruption configuration
    disruption_type: DisruptionType = DisruptionType.INCIDENTAL
    disruption_config: Optional[DisruptionConfig] = None
    
    # Timing
    trigger_time: float = 2.0  # When disruption starts (seconds)
    blend_duration: float = 0.4  # AMASS -> IK blend time
    
    # Motion modifiers
    speed_multiplier: float = 1.0  # 0.5 - 2.0 range
    
    # Human configuration
    human_height_percentile: float = 0.5  # 0.05 - 0.95
    
    # Spatial configuration
    approach_angle: float = 0.0  # degrees, 0 = front
    spawn_distance: float = 2.0  # meters from robot
    
    # Target body part for IK (when applicable)
    reaching_arm: str = "right_arm"
    
    # --- Trajectory parameters ---
    trajectory_type: str = "PASS_BY"      # PASS_BY | APPROACH_LOITER_DEPART | ARC
    pass_by_offset: float = 1.0           # Lateral offset from robot (meters)
    closest_approach: float = 1.0         # How close before stopping (meters)
    loiter_duration: float = 2.0          # Time near robot (seconds)
    departure_angle: float = 150.0        # Relative departure direction (degrees)
    walk_speed: float = 1.2               # Walking speed (m/s)
    arc_radius: float = 1.5               # Arc radius for ARC type (meters)
    arc_extent: float = 120.0             # Arc angular extent (degrees)
    pass_by_side: int = 1                 # +1 left, -1 right (which side to pass)

    # --- COWORKER_PATROL parameters (unused for non-patrol types) ---
    patrol_near_loiter: float = 8.0          # seconds at the near position
    patrol_away_loiter: float = 3.5          # seconds at the away position
    patrol_away_distance: float = 2.5        # how far the away position is from robot
    patrol_excursions: int = 2               # number of near->away->near cycles
    patrol_near_distance_std: float = 0.12   # per-visit NEAR distance stdev (meters)
    patrol_near_distance_clip: float = 0.25  # max abs jitter on NEAR distance (meters)

    # Reproducibility
    seed: int = 0
    
    def __post_init__(self):
        """Set default disruption config if not provided."""
        if self.disruption_config is None:
            self.disruption_config = DEFAULT_CONFIGS.get(
                self.disruption_type,
                DisruptionConfig(disruption_type=self.disruption_type)
            )


@dataclass
class ParameterSpace:
    """Defines the ranges for scenario parameter sampling."""
    
    # Motion clips (paths to .npz files)
    clip_paths: List[str] = field(default_factory=list)
    
    # Disruption type probabilities
    disruption_weights: Dict[DisruptionType, float] = field(default_factory=lambda: {
        DisruptionType.INCIDENTAL: 0.13,
        DisruptionType.SHARED_GOAL: 0.13,
        DisruptionType.DIRECT: 0.17,
        DisruptionType.OBSTRUCTION: 0.17,
        DisruptionType.RANDOM_PERTURBED: 0.10,
        DisruptionType.CONTACT: 0.15,
        DisruptionType.COWORKER: 0.15,
    })

    # Timing ranges
    trigger_time_range: tuple = (0.5, 5.0)  # seconds
    blend_duration_range: tuple = (0.2, 0.6)  # seconds

    # Motion modifiers
    speed_range: tuple = (0.5, 2.0)

    # Human anthropometry
    height_percentile_range: tuple = (0.05, 0.95)

    # Spatial configuration
    approach_angle_range: tuple = (0.0, 360.0)  # degrees
    spawn_distance_range: tuple = (0.8, 1.8)  # meters

    # --- Trajectory parameter ranges ---
    # Tightened to push the human into the contact regime in a fraction
    # of episodes. Loiter is bounded so the human always eventually
    # departs and lets the robot continue its task.
    pass_by_offset_range: tuple = (0.05, 1.2)      # Lateral offset (meters)
    closest_approach_range: tuple = (0.0, 0.8)     # Stop distance (meters)
    loiter_duration_range: tuple = (4.0, 10.0)     # Near-robot time (seconds)
    departure_angle_range: tuple = (120.0, 240.0)  # Departure angle (degrees)
    walk_speed_range: tuple = (1.0, 2.0)           # Walk speed (m/s)
    arc_radius_range: tuple = (0.5, 1.8)           # Arc radius (meters)
    arc_extent_range: tuple = (90.0, 180.0)        # Arc extent (degrees)
    embed_distance_range: tuple = (0.0, 0.05)      # CONTACT: meters past surface

    # --- COWORKER continuous knobs ---
    # The five axes the user can tune for train/eval distribution
    # shaping. Each is sampled per-episode when disruption_type ==
    # COWORKER. Defaults match the historical hand-tuned settings; the
    # ``make_coworker_train_space`` / ``make_coworker_eval_space``
    # factories below provide moderate vs. wider preset bands for
    # train-on-narrow / eval-on-wider experiments.
    coworker_closest_approach_range: tuple = (0.9, 1.4)   # NEAR distance (m)
    coworker_reach_period_range: tuple = (4.5, 6.5)       # seconds per cycle (1/freq)
    coworker_target_mix_p_ee_range: tuple = (0.4, 0.6)    # P(reach EE); P(task)=1-p
    coworker_near_loiter_range: tuple = (7.0, 11.0)       # dwell at NEAR (s)
    coworker_walk_speed_range: tuple = (1.0, 1.6)         # walk speed (m/s)


# --- COWORKER train/eval distribution presets ---------------------------
#
# Five-axis continuous parameterization for the COWORKER disruption:
#   1. closest-approach distance       (coworker_closest_approach_range)
#   2. reach period  = 1 / frequency   (coworker_reach_period_range)
#   3. P(reach EE)   (vs task object)  (coworker_target_mix_p_ee_range)
#   4. dwell time at NEAR              (coworker_near_loiter_range)
#   5. walk speed                      (coworker_walk_speed_range)
#
# ``train`` is the moderate band the policy/filter is trained on.
# ``eval`` is a strict superset on every axis so the user can measure
# how performance degrades as the human's behaviour moves outside the
# training support (generalisation test).
#
# Keep these two factories in lock-step: every train range must lie
# inside the corresponding eval range. The test suite enforces this
# invariant.

_COWORKER_TRAIN_RANGES: Dict[str, tuple] = {
    "coworker_closest_approach_range": (0.9, 1.4),
    "coworker_reach_period_range": (4.5, 6.5),
    "coworker_target_mix_p_ee_range": (0.4, 0.6),
    "coworker_near_loiter_range": (7.0, 11.0),
    "coworker_walk_speed_range": (1.0, 1.6),
}

_COWORKER_EVAL_RANGES: Dict[str, tuple] = {
    "coworker_closest_approach_range": (0.6, 1.8),   # closer + further
    "coworker_reach_period_range": (3.0, 9.0),       # ~2x more / less frequent
    "coworker_target_mix_p_ee_range": (0.1, 0.9),    # near-pure EE / near-pure task
    "coworker_near_loiter_range": (4.0, 16.0),       # short + long dwells
    "coworker_walk_speed_range": (0.6, 2.2),         # shuffle + brisk walk
}


def _coworker_space(ranges: Dict[str, tuple], **overrides) -> "ParameterSpace":
    """Build a ParameterSpace with COWORKER pinned to the given ranges.

    Forces ``disruption_weights = {COWORKER: 1.0}`` so callers don't
    have to do it themselves — the train/eval split is about generalising
    on the COWORKER axes, so other types being mixed in only adds noise.
    Override any single field by keyword.
    """
    kwargs = dict(ranges)
    kwargs["disruption_weights"] = {DisruptionType.COWORKER: 1.0}
    kwargs.update(overrides)
    return ParameterSpace(**kwargs)


def make_coworker_train_space(**overrides) -> "ParameterSpace":
    """Moderate COWORKER distribution for training the safety filter /
    policy. All five continuous knobs sit in their *narrow* bands."""
    return _coworker_space(_COWORKER_TRAIN_RANGES, **overrides)


def make_coworker_eval_space(**overrides) -> "ParameterSpace":
    """Wider COWORKER distribution for evaluation. Each knob's range
    strictly contains the training range, so eval episodes exercise
    both in-distribution and out-of-distribution conditions."""
    return _coworker_space(_COWORKER_EVAL_RANGES, **overrides)


class ScenarioSampler:
    """
    Samples diverse human behaviour scenarios.
    
    Each call to sample_scenario() with a seed produces a reproducible
    scenario that can be used for evaluation and debugging.
    """
    
    def __init__(
        self,
        parameter_space: Optional[ParameterSpace] = None,
        motion_dir: Optional[Path] = None,
    ):
        """
        Initialize the sampler.
        
        Args:
            parameter_space: Custom parameter ranges (uses defaults if None)
            motion_dir: Directory containing AMASS .npz files
        """
        self.params = parameter_space or ParameterSpace()
        self.motion_dir = motion_dir
        
        # Auto-discover motion clips if directory provided
        if motion_dir and not self.params.clip_paths:
            self._discover_clips(motion_dir)
    
    def _discover_clips(self, motion_dir: Path):
        """Discover all .npz motion clips in directory."""
        motion_dir = Path(motion_dir)
        if motion_dir.exists():
            clips = list(motion_dir.rglob("*.npz"))
            self.params.clip_paths = [str(p) for p in clips]
    
    def sample_scenario(self, seed: int) -> ScenarioParams:
        """
        Sample a complete scenario with the given seed.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            ScenarioParams with all scenario configuration
        """
        rng = np.random.default_rng(seed)
        
        # Sample clip
        if self.params.clip_paths:
            clip_idx = rng.integers(0, len(self.params.clip_paths))
            clip_path = self.params.clip_paths[clip_idx]
        else:
            clip_path = ""  # Will need to be set manually
        
        # Sample disruption type
        disruption_type = self._sample_disruption_type(rng)
        
        # Sample timing
        trigger_time = rng.uniform(*self.params.trigger_time_range)
        blend_duration = rng.uniform(*self.params.blend_duration_range)
        
        # Sample motion modifiers
        speed = rng.uniform(*self.params.speed_range)
        
        # Sample anthropometry
        height_pct = rng.uniform(*self.params.height_percentile_range)
        
        # Sample spatial
        angle = rng.uniform(*self.params.approach_angle_range)
        distance = rng.uniform(*self.params.spawn_distance_range)
        
        # Select arm (based on angle - right arm for right-side approach)
        reaching_arm = "right_arm" if 270 < angle or angle < 90 else "left_arm"
        
        # --- Trajectory type selection (based on disruption type) ---
        trajectory_type = self._select_trajectory_type(disruption_type, rng)
        
        # Sample trajectory parameters
        pass_by_offset = rng.uniform(*self.params.pass_by_offset_range)
        closest_approach = rng.uniform(*self.params.closest_approach_range)
        loiter_duration = rng.uniform(*self.params.loiter_duration_range)
        departure_angle = rng.uniform(*self.params.departure_angle_range)
        walk_speed = rng.uniform(*self.params.walk_speed_range)
        arc_radius = rng.uniform(*self.params.arc_radius_range)
        arc_extent = rng.uniform(*self.params.arc_extent_range)
        pass_by_side = rng.choice([-1, 1])  # Random side

        # Patrol params default to the ScenarioParams defaults; the
        # COWORKER branch below overrides them with task-tuned values.
        patrol_near_loiter = 8.0
        patrol_away_loiter = 3.5
        patrol_away_distance = 2.5
        patrol_excursions = 2
        patrol_near_distance_std = 0.12
        patrol_near_distance_clip = 0.25

        # Create disruption config with sampled noise values
        base_config = DEFAULT_CONFIGS.get(
            disruption_type,
            DisruptionConfig(disruption_type=disruption_type)
        )

        if disruption_type == DisruptionType.OBSTRUCTION:
            # Passive intrusion: human plants inside the robot's task
            # workspace, loiters there long enough to provoke a safety
            # event, then walks away so the robot can continue its task.
            obstruction_target = np.array([
                0.45 + rng.uniform(-0.10, 0.15),
                rng.uniform(-0.15, 0.15),
                0.75 + rng.uniform(-0.10, 0.15),
            ])
            closest_approach = rng.uniform(0.2, 0.5)
            disruption_config = DisruptionConfig(
                disruption_type=disruption_type,
                obstruction_target=obstruction_target,
                hold_duration=loiter_duration,
            )
        elif disruption_type == DisruptionType.COWORKER:
            # Sustained co-working with five continuous knobs sampled
            # per-episode from ParameterSpace ranges:
            #   - closest-approach distance
            #   - reach period (= 1 / reach frequency)
            #   - reach target probability (arm vs. task object)
            #   - dwell time at NEAR (per visit)
            #   - walk speed
            # The default ranges match the moderate "training"
            # distribution. Use make_coworker_eval_space() (or YAML
            # overrides) to widen them for evaluation generalization.
            closest_approach = float(rng.uniform(
                *self.params.coworker_closest_approach_range
            ))
            coworker_reach_period = float(rng.uniform(
                *self.params.coworker_reach_period_range
            ))
            p_ee = float(rng.uniform(
                *self.params.coworker_target_mix_p_ee_range
            ))
            patrol_near_loiter = float(rng.uniform(
                *self.params.coworker_near_loiter_range
            ))
            # Walk speed for COWORKER trajectories overrides the
            # general walk_speed pulled from walk_speed_range above.
            walk_speed = float(rng.uniform(
                *self.params.coworker_walk_speed_range
            ))

            # Loiter duration covers most of an episode so the human
            # stays around. APPROACH_LOITER_DEPART uses this directly;
            # STATIONARY just respects it. COWORKER_PATROL uses
            # patrol_near_loiter (sampled separately above) as its
            # per-visit dwell instead.
            loiter_duration = max(self.params.loiter_duration_range[1], 30.0)

            # When the trajectory walks the human in, the visible walk
            # length is (spawn_distance - closest_approach). Force a
            # ~1.5-2.5 m gap so the approach is clearly a walk, not a
            # teleport. (For STATIONARY this just sets where the
            # planner *would* have spawned; my _build_stationary
            # projects to closest_approach regardless.)
            distance = closest_approach + float(rng.uniform(1.5, 2.5))
            active_arm = str(rng.choice(["left_arm", "right_arm"]))
            disruption_config = DisruptionConfig(
                disruption_type=disruption_type,
                target_noise_std=0.0,
                coworker_reach_period=coworker_reach_period,
                coworker_reach_fraction=0.15,
                coworker_hold_fraction=0.20,
                coworker_retract_fraction=0.15,
                coworker_target_mix=(p_ee, 1.0 - p_ee),
                coworker_active_arm=active_arm,
            )
            # Patrol-specific knobs. Only consumed when the chosen
            # trajectory_type is COWORKER_PATROL — harmless for the
            # other two COWORKER trajectories. patrol_near_loiter was
            # sampled above (it's the COWORKER dwell-time knob).
            patrol_away_loiter = float(rng.uniform(3.0, 5.0))
            patrol_away_distance = float(rng.uniform(2.0, 3.0))
            patrol_excursions = int(rng.integers(1, 3))  # 1 or 2
        elif disruption_type == DisruptionType.CONTACT:
            # Active reach: human walks into the robot, presses for the
            # loiter window, then departs. embed_distance puts the IK
            # target slightly past the link surface so reaching is
            # unambiguously a press while the human is in the loiter
            # phase.
            target_part = rng.choice(
                ["ee", "left_forearm", "right_forearm", "torso"],
                p=[0.40, 0.25, 0.25, 0.10],
            )
            embed_distance = rng.uniform(*self.params.embed_distance_range)
            # CONTACT is the most aggressive type — pin closest_approach low.
            closest_approach = rng.uniform(0.0, 0.2)
            disruption_config = DisruptionConfig(
                disruption_type=disruption_type,
                target_noise_std=0.0,
                contact_target_part=str(target_part),
                embed_distance=float(embed_distance),
            )
        else:
            disruption_config = base_config
        
        return ScenarioParams(
            clip_path=clip_path,
            disruption_type=disruption_type,
            disruption_config=disruption_config,
            trigger_time=trigger_time,
            blend_duration=blend_duration,
            speed_multiplier=speed,
            human_height_percentile=height_pct,
            approach_angle=angle,
            spawn_distance=distance,
            reaching_arm=reaching_arm,
            trajectory_type=trajectory_type,
            pass_by_offset=pass_by_offset,
            closest_approach=closest_approach,
            loiter_duration=loiter_duration,
            departure_angle=departure_angle,
            walk_speed=walk_speed,
            arc_radius=arc_radius,
            arc_extent=arc_extent,
            pass_by_side=pass_by_side,
            patrol_near_loiter=patrol_near_loiter,
            patrol_away_loiter=patrol_away_loiter,
            patrol_away_distance=patrol_away_distance,
            patrol_excursions=patrol_excursions,
            patrol_near_distance_std=patrol_near_distance_std,
            patrol_near_distance_clip=patrol_near_distance_clip,
            seed=seed,
        )
    
    def _sample_disruption_type(self, rng: np.random.Generator) -> DisruptionType:
        """Sample a disruption type based on weights."""
        types = list(self.params.disruption_weights.keys())
        weights = [self.params.disruption_weights[t] for t in types]
        
        # Normalize weights
        total = sum(weights)
        probs = [w / total for w in weights]
        
        idx = rng.choice(len(types), p=probs)
        return types[idx]
    
    @staticmethod
    def _select_trajectory_type(
        disruption_type: DisruptionType, rng: np.random.Generator
    ) -> str:
        """Choose trajectory type based on disruption type."""
        if disruption_type in {
            DisruptionType.SHARED_GOAL,
            DisruptionType.DIRECT,
            DisruptionType.OBSTRUCTION,
            DisruptionType.CONTACT,
        }:
            # These need the human to stop near the robot, then depart.
            return "APPROACH_LOITER_DEPART"
        elif disruption_type == DisruptionType.COWORKER:
            # COWORKER picks among three patterns: walk in and stay,
            # spawn already in place, or walk in then occasionally move
            # away and return to a different angle.
            return str(rng.choice(
                ["APPROACH_LOITER_DEPART", "STATIONARY", "COWORKER_PATROL"]
            ))
        elif disruption_type == DisruptionType.INCIDENTAL:
            # Incidental: walk past (PASS_BY or ARC)
            return rng.choice(["PASS_BY", "ARC"])
        elif disruption_type == DisruptionType.RANDOM_PERTURBED:
            # Similar to incidental but with noise
            return "PASS_BY"
        else:
            return "PASS_BY"
    
    def sample_batch(self, n: int, base_seed: int = 0) -> List[ScenarioParams]:
        """
        Sample a batch of scenarios.
        
        Args:
            n: Number of scenarios
            base_seed: Starting seed
            
        Returns:
            List of ScenarioParams
        """
        return [self.sample_scenario(base_seed + i) for i in range(n)]
    
    def get_stratified_sample(
        self,
        n_per_type: int = 10,
        base_seed: int = 0,
    ) -> Dict[DisruptionType, List[ScenarioParams]]:
        """
        Sample scenarios stratified by disruption type.
        
        Args:
            n_per_type: Number of scenarios per disruption type
            base_seed: Starting seed
            
        Returns:
            Dict mapping disruption type to list of scenarios
        """
        result = {}
        seed = base_seed
        
        for dtype in DisruptionType:
            scenarios = []
            while len(scenarios) < n_per_type:
                scenario = self.sample_scenario(seed)
                if scenario.disruption_type == dtype:
                    scenarios.append(scenario)
                seed += 1
            result[dtype] = scenarios
        
        return result
