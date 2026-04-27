import sys
import imageio
from bigym.action_modes import JointPositionActionMode
from bigym.envs.cupboards import DrawersAllOpen
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_player import DemoPlayer
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

# Use a high control frequency for playback to be smoother
control_frequency = 50

# Setup the environment
env = DrawersAllOpen(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    control_frequency=control_frequency,
    observation_config=ObservationConfig(
        cameras=[
            CameraConfig("head", resolution=(256, 256)),
            CameraConfig("left_wrist", resolution=(256, 256)),
            CameraConfig("right_wrist", resolution=(256, 256)),
        ]
    ),
    render_mode="rgb_array",
)
metadata = Metadata.from_env(env)

# Get demonstrations from DemoStore
print("Downloading/fetching demo...")
demo_store = DemoStore()
demos = demo_store.get_demos(metadata, amount=1, frequency=control_frequency)

demo = demos[0]
print(f"Replaying demo {demo.seed}...")

# Manually replay to capture frames
timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, control_frequency)
env.reset(seed=demo.seed)

frames = []
for i, step in enumerate(timesteps):
    action = step.executed_action
    env.step(action, fast=True)
    if i % 2 == 0:  # Save every 2nd frame to speed up encoding
        frame = env.render()
        if frame is not None:
            frames.append(frame)

env.close()

output_path = sys.argv[1] if len(sys.argv) > 1 else "demo.mp4"
print(f"Saving video to {output_path}...")
# Save using imageio
imageio.mimsave(output_path, frames, fps=25)
print("Done!")

