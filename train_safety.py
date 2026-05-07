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

    # Explicit snapshot override (e.g. baseline_sweep.py eval runs):
    #   +snapshot_path=/abs/path/to/100000_snapshot.pt
    # Workspace.__init__ already read the snapshot eagerly (to seed env stats
    # before wrappers captured them by reference). Calling load_snapshot() here
    # finishes the restore — it reuses the stashed payload instead of re-reading.
    if cfg.get("snapshot_path", None):
        print(f"Loading snapshot from override: {cfg.snapshot_path}")
        workspace.load_snapshot()
    else:
        snapshot = root_dir / "snapshot.pt"
        if snapshot.exists():
            print(f"Resuming from: {snapshot}")
            workspace.load_snapshot()

    if cfg.num_train_frames == 0 and getattr(cfg, "num_pretrain_steps", 0) == 0:
        print("Pure eval mode detected. Running evaluation only...")
        eval_metrics = workspace._eval()
        eval_metrics.update(workspace._get_common_metrics())
        workspace.logger.log_metrics(eval_metrics, workspace.global_env_steps, prefix="eval")
        
        explicit_out = getattr(cfg, "eval_output_path", None)
        if explicit_out:
            import json
            import numpy as np
            import torch
            def safe_convert(o):
                if isinstance(o, np.generic): return o.item()
                if isinstance(o, torch.Tensor): return o.item()
                return str(o)
            with open(explicit_out, "w") as f:
                json.dump(eval_metrics, f, default=safe_convert)
    else:
        workspace.train()


if __name__ == "__main__":
    main()
