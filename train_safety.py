#!/usr/bin/env python
"""
Training entry point for Diffusion Policy on SafetyBiGym via RoboBase.

Usage:
    python train_safety.py launch=dp_pixel_safety_bigym env=safety_bigym/reach_target_single

    # Override parameters:
    python train_safety.py launch=dp_pixel_safety_bigym env=safety_bigym/reach_target_single \
        demos=30 num_pretrain_steps=100000 batch_size=256

    # With W&B logging:
    python train_safety.py launch=dp_pixel_safety_bigym env=safety_bigym/reach_target_single \
        wandb.use=true wandb.name=dp_safety_reach
"""
from pathlib import Path

import hydra


@hydra.main(
    config_path="cfgs", config_name="safety_config", version_base=None
)
def main(cfg):
    from robobase.workspace import Workspace
    from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory

    root_dir = Path.cwd()

    workspace = Workspace(cfg, env_factory=SafetyBiGymEnvFactory())

    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"Resuming from: {snapshot}")
        workspace.load_snapshot()

    workspace.train()


if __name__ == "__main__":
    main()
