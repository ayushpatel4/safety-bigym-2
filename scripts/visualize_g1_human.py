"""Standalone MuJoCo viewer for the G1 *view* asset.

Training loads ``assets/g1_human_body.xml`` (2683b67 capsule layout). This
script loads ``assets/g1_human_body_view.xml`` (connected hips/shoulders) so
you can inspect a nicer silhouette without changing what the CNN sees.

Generate the view asset::

    python scripts/build_g1_human_body.py --view

Run from the repo root::

    cd safety_bigym
    venv/bin/mjpython scripts/visualize_g1_human.py

To preview the training asset instead::

    G1_VIEW_ASSET=training venv/bin/mjpython scripts/visualize_g1_human.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "safety_bigym" / "assets"
G1_BODY_TRAINING = ASSETS / "g1_human_body.xml"
G1_BODY_VIEW = ASSETS / "g1_human_body_view.xml"

STANDING_Z = 0.95  # raise the pelvis so the feet clear the floor.

WRAPPER_TEMPLATE = """<?xml version='1.0' encoding='utf-8'?>
<mujoco model='g1_human_viewer'>
  <option timestep='0.002' gravity='0 0 -9.81'/>
  <visual>
    <headlight diffuse='0.6 0.6 0.6' ambient='0.3 0.3 0.3' specular='0 0 0'/>
    <rgba haze='0.15 0.25 0.35 1'/>
    <global azimuth='120' elevation='-20'/>
  </visual>
  <asset>
    <texture type='skybox' builtin='gradient' rgb1='0.3 0.5 0.7'
             rgb2='0 0 0' width='512' height='3072'/>
    <texture type='2d' name='groundplane' builtin='checker'
             mark='edge' rgb1='0.2 0.3 0.4' rgb2='0.1 0.2 0.3'
             markrgb='0.8 0.8 0.8' width='300' height='300'/>
    <material name='groundplane' texture='groundplane' texuniform='true'
              texrepeat='5 5' reflectance='0.2'/>
  </asset>
  <worldbody>
    <light name='top' pos='0 0 3' dir='0 0 -1' diffuse='0.7 0.7 0.7'/>
    <geom name='floor' size='5 5 0.05' type='plane' material='groundplane'/>
    <camera name='front' pos='2.0 0 1.2' xyaxes='0 -1 0  0 0 1'/>
  </worldbody>
  <include file='{include}'/>
</mujoco>
"""


def _resolve_body_path() -> Path:
    mode = os.environ.get("G1_VIEW_ASSET", "view").strip().lower()
    if mode in ("training", "train"):
        return G1_BODY_TRAINING
    return G1_BODY_VIEW


def main() -> int:
    g1_body = _resolve_body_path()
    if not g1_body.is_file():
        hint = (
            "python scripts/build_g1_human_body.py"
            if g1_body == G1_BODY_TRAINING
            else "python scripts/build_g1_human_body.py --view"
        )
        print(f"ERROR: {g1_body} not found. Run {hint} first.", file=sys.stderr)
        return 1

    print(f"Loading {g1_body.name} ({os.environ.get('G1_VIEW_ASSET', 'view')} preset)")

    body_text = g1_body.read_text()
    body_text = body_text.replace(
        'name="Pelvis" pos="0 0 0"',
        f'name="Pelvis" pos="0 0 {STANDING_Z}"',
        1,
    )
    mesh_root = (g1_body.parent / "g1" / "assets").resolve()
    body_text = body_text.replace(
        'file="g1/assets/',
        f'file="{mesh_root}/',
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="g1_viewer_", dir=str(g1_body.parent)))
    try:
        include_path = tmp_dir / "g1_human_body_viewer.xml"
        include_path.write_text(body_text)

        wrapper_path = tmp_dir / "world.xml"
        wrapper_path.write_text(
            WRAPPER_TEMPLATE.format(include=include_path.name)
        )

        import mujoco
        import mujoco.viewer

        model = mujoco.MjModel.from_xml_path(str(wrapper_path))
        data = mujoco.MjData(model)
        # Place the mocap pelvis at the standing height so it stays put.
        for mocap_id in range(model.nmocap):
            data.mocap_pos[mocap_id] = [0.0, 0.0, STANDING_Z]
            data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        print(f"Loaded model with {model.ngeom} geoms, {model.nbody} bodies.")
        print("Launching viewer (close the window or press ESC to exit).")

        # launch_passive is what every other script in this repo uses.
        # launch() spins a separate physics thread via _Simulate() and
        # crashes on some macOS/mjpython builds.
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -10
            viewer.cam.azimuth = 120
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]

            while viewer.is_running():
                # Keep the mocap root pinned; forward only (no physics step)
                # so the capsule pose stays stable for inspection.
                for mocap_id in range(model.nmocap):
                    data.mocap_pos[mocap_id] = [0.0, 0.0, STANDING_Z]
                    data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.01)
    finally:
        for p in tmp_dir.glob("*"):
            p.unlink()
        tmp_dir.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
