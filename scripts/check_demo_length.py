"""Print per-demo length for a task at the eval control frequency."""
import numpy as np
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from bigym.envs.reach_target import ReachTargetDual
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

DEMO_DOWN_SAMPLE = 10   # from cfgs/env/safety_bigym/dishwasher_close.yaml
TASK_CLS = ReachTargetDual

action_mode = JointPositionActionMode(
    floating_base=True, absolute=True,
    floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
)
env = TASK_CLS(action_mode=action_mode,
               control_frequency=CONTROL_FREQUENCY_MAX // DEMO_DOWN_SAMPLE)

demos = DemoStore().get_demos(
    Metadata.from_env(env),
    amount=-1,
    frequency=CONTROL_FREQUENCY_MAX // DEMO_DOWN_SAMPLE,
)
lens = np.array([len(d.timesteps) for d in demos])
print(f"n={len(lens)}  min={lens.min()}  median={int(np.median(lens))} "
      f"mean={lens.mean():.1f}  max={lens.max()}  p95={int(np.percentile(lens, 95))}")
env.close()
