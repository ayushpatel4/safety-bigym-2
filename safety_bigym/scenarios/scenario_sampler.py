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
        DisruptionType.INCIDENTAL: 0.3,
        DisruptionType.SHARED_GOAL: 0.2,
        DisruptionType.DIRECT: 0.2,
        DisruptionType.OBSTRUCTION: 0.15,
        DisruptionType.RANDOM_PERTURBED: 0.15,
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
    spawn_distance_range: tuple = (1.0, 2.0)  # meters


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
        
        # Create disruption config with sampled noise values
        base_config = DEFAULT_CONFIGS.get(
            disruption_type,
            DisruptionConfig(disruption_type=disruption_type)
        )
        
        # For OBSTRUCTION, sample a fixed target position
        if disruption_type == DisruptionType.OBSTRUCTION:
            # Random point in robot workspace (will be refined in integration)
            obstruction_target = np.array([
                0.3 + rng.uniform(-0.2, 0.2),
                rng.uniform(-0.3, 0.3),
                0.8 + rng.uniform(-0.2, 0.2),
            ])
            disruption_config = DisruptionConfig(
                disruption_type=disruption_type,
                obstruction_target=obstruction_target,
                hold_duration=rng.uniform(1.0, 3.0),
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
