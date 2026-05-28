"""Derive G1 human MJCF fragments from upstream ``assets/g1/g1.xml``.

Two presets share the same kinematic tree and PD defaults; only the
hand-authored ``_col`` capsule layout differs:

- **training** (default) — commit ``2683b67`` layout, 18 geoms. Written to
  ``assets/g1_human_body.xml``. Loaded by ``SafetyBiGymEnv`` when
  ``human_model=g1`` (CQN-AS curriculum, eval, safety metrics).
- **view** — connected hip/thorax bridges + upper arms on ``shoulder_yaw_link``,
  22 geoms. Written to ``assets/g1_human_body_view.xml``. **Not** used by the
  env; only for ``scripts/visualize_g1_human.py`` and manual inspection.

Run from the repo root::

    cd safety_bigym
    python scripts/build_g1_human_body.py              # training only
    python scripts/build_g1_human_body.py --view       # view only
    python scripts/build_g1_human_body.py --all        # both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "safety_bigym" / "assets"
UPSTREAM_PATH = ASSETS_DIR / "g1" / "g1.xml"

OUTPUT_TRAINING = ASSETS_DIR / "g1_human_body.xml"
OUTPUT_VIEW = ASSETS_DIR / "g1_human_body_view.xml"

import importlib.util  # noqa: E402

_SPEC_PATH = REPO_ROOT / "safety_bigym" / "human" / "g1_human_spec.py"
_spec = importlib.util.spec_from_file_location("g1_human_spec", _SPEC_PATH)
g1_human_spec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1_human_spec)
BODY_JOINT_NAMES = g1_human_spec.BODY_JOINT_NAMES

CapsuleSpec = tuple[str, str, dict[str, str]]

# 2683b67 — curriculum-validated training silhouette (18 _col geoms).
COLLISION_CAPSULES_TRAINING: list[CapsuleSpec] = [
    ("Pelvis", "Pelvis_col", {"fromto": "0 -0.10 0  0 0.10 0", "size": "0.12"}),
    ("waist_yaw_link", "Spine_col", {"fromto": "0 0 0  0 0 0.10", "size": "0.08"}),
    ("torso_link", "Chest_col", {"fromto": "0 0 0.05  0 0 0.30", "size": "0.13"}),
    ("torso_link", "Head_col", {"pos": "0 0 0.48", "size": "0.10", "type": "sphere"}),
    ("left_hip_yaw_link", "L_Thigh_col",
     {"fromto": "0 0 0  -0.078 0.002 -0.177", "size": "0.07"}),
    ("left_knee_link", "L_Shin_col",
     {"fromto": "0 0 0  0 0 -0.30", "size": "0.055"}),
    ("left_ankle_roll_link", "L_Foot_col",
     {"fromto": "-0.04 0 -0.025  0.12 0 -0.025", "size": "0.04"}),
    ("right_hip_yaw_link", "R_Thigh_col",
     {"fromto": "0 0 0  -0.078 -0.002 -0.177", "size": "0.07"}),
    ("right_knee_link", "R_Shin_col",
     {"fromto": "0 0 0  0 0 -0.30", "size": "0.055"}),
    ("right_ankle_roll_link", "R_Foot_col",
     {"fromto": "-0.04 0 -0.025  0.12 0 -0.025", "size": "0.04"}),
    ("left_shoulder_pitch_link", "L_Shoulder_col",
     {"fromto": "0 0 0  0 0.04 -0.02", "size": "0.06"}),
    ("left_elbow_link", "L_Elbow_col",
     {"fromto": "0 0 0  0.10 0 -0.01", "size": "0.045"}),
    ("left_wrist_roll_link", "L_Wrist_col",
     {"fromto": "0 0 0  0.084 0 0", "size": "0.035"}),
    ("left_wrist_yaw_link", "L_Hand_col",
     {"fromto": "0 0 0  0.09 0 0", "size": "0.040"}),
    ("right_shoulder_pitch_link", "R_Shoulder_col",
     {"fromto": "0 0 0  0 -0.04 -0.02", "size": "0.06"}),
    ("right_elbow_link", "R_Elbow_col",
     {"fromto": "0 0 0  0.10 0 -0.01", "size": "0.045"}),
    ("right_wrist_roll_link", "R_Wrist_col",
     {"fromto": "0 0 0  0.084 0 0", "size": "0.035"}),
    ("right_wrist_yaw_link", "R_Hand_col",
     {"fromto": "0 0 0  0.09 0 0", "size": "0.040"}),
]

# Viewer-only — same skin tone; extra bridges for a connected silhouette.
COLLISION_CAPSULES_VIEW: list[CapsuleSpec] = [
    ("Pelvis", "Pelvis_col", {"fromto": "0 0 0.02  0 0 -0.10", "size": "0.10"}),
    ("waist_yaw_link", "Spine_col", {"fromto": "0 0 0  0 0 0.10", "size": "0.08"}),
    ("torso_link", "Chest_col", {"fromto": "0 0 0.05  0 0 0.30", "size": "0.13"}),
    ("torso_link", "Head_col", {"pos": "0 0 0.48", "size": "0.10", "type": "sphere"}),
    ("left_hip_pitch_link", "L_Hip_col",
     {"fromto": "0 0 0.02  0 0 -0.08", "size": "0.075"}),
    ("right_hip_pitch_link", "R_Hip_col",
     {"fromto": "0 0 0.02  0 0 -0.08", "size": "0.075"}),
    ("left_hip_yaw_link", "L_Thigh_col",
     {"fromto": "0 0 0  -0.078 0.002 -0.177", "size": "0.075"}),
    ("left_knee_link", "L_Shin_col",
     {"fromto": "0 0 0  0 0 -0.30", "size": "0.06"}),
    ("left_ankle_roll_link", "L_Foot_col",
     {"fromto": "-0.04 0 -0.025  0.12 0 -0.025", "size": "0.04"}),
    ("right_hip_yaw_link", "R_Thigh_col",
     {"fromto": "0 0 0  -0.078 -0.002 -0.177", "size": "0.075"}),
    ("right_knee_link", "R_Shin_col",
     {"fromto": "0 0 0  0 0 -0.30", "size": "0.06"}),
    ("right_ankle_roll_link", "R_Foot_col",
     {"fromto": "-0.04 0 -0.025  0.12 0 -0.025", "size": "0.04"}),
    ("torso_link", "L_Thorax_col",
     {"fromto": "0 0.00 0.28  0.004 0.10 0.248", "size": "0.065"}),
    ("torso_link", "R_Thorax_col",
     {"fromto": "0 0.00 0.28  0.004 -0.10 0.248", "size": "0.065"}),
    ("left_shoulder_yaw_link", "L_Shoulder_col",
     {"fromto": "0 0 0.04  0.016 0 -0.081", "size": "0.055"}),
    ("left_elbow_link", "L_Elbow_col",
     {"fromto": "0 0 0  0.10 0 -0.01", "size": "0.045"}),
    ("left_wrist_roll_link", "L_Wrist_col",
     {"fromto": "0 0 0  0.084 0 0", "size": "0.035"}),
    ("left_wrist_yaw_link", "L_Hand_col",
     {"fromto": "0 0 0  0.09 0 0", "size": "0.040"}),
    ("right_shoulder_yaw_link", "R_Shoulder_col",
     {"fromto": "0 0 0.04  0.016 0 -0.081", "size": "0.055"}),
    ("right_elbow_link", "R_Elbow_col",
     {"fromto": "0 0 0  0.10 0 -0.01", "size": "0.045"}),
    ("right_wrist_roll_link", "R_Wrist_col",
     {"fromto": "0 0 0  0.084 0 0", "size": "0.035"}),
    ("right_wrist_yaw_link", "R_Hand_col",
     {"fromto": "0 0 0  0.09 0 0", "size": "0.040"}),
]

PRESETS: dict[str, tuple[Path, str, list[CapsuleSpec]]] = {
    "training": (OUTPUT_TRAINING, "g1_human_body", COLLISION_CAPSULES_TRAINING),
    "view": (OUTPUT_VIEW, "g1_human_body_view", COLLISION_CAPSULES_VIEW),
}


def build_defaults() -> etree._Element:
    defaults = etree.Element("default")
    human = etree.SubElement(defaults, "default", {"class": "human"})
    etree.SubElement(human, "joint", {"damping": "50", "armature": "0.01"})
    etree.SubElement(human, "geom", {
        "type": "capsule", "condim": "3",
        "friction": "1 0.5 0.001", "density": "1000",
    })
    coll = etree.SubElement(defaults, "default", {"class": "human_collision"})
    etree.SubElement(coll, "geom", {
        "type": "capsule",
        "solref": "0.02 1.0", "solimp": "0.9 0.95 0.001",
        "group": "0", "contype": "2", "conaffinity": "4",
        "rgba": "0.8 0.6 0.5 1.0",
    })
    pos = etree.SubElement(defaults, "default", {"class": "position_actuator"})
    etree.SubElement(pos, "position", {"kp": "200", "kv": "20"})
    return defaults


def transform_pelvis(pelvis: etree._Element) -> None:
    pelvis.set("name", "Pelvis")
    pelvis.set("pos", "0 0 0")
    pelvis.set("mocap", "true")
    if "childclass" in pelvis.attrib:
        del pelvis.attrib["childclass"]
    for fj in list(pelvis.findall("freejoint")):
        pelvis.remove(fj)


def strip_geoms_and_sites(body: etree._Element) -> None:
    for tag in ("geom", "site"):
        for el in list(body.findall(tag)):
            body.remove(el)
    for child in body.findall("body"):
        strip_geoms_and_sites(child)


def stamp_joint_class(body: etree._Element) -> None:
    for joint in body.findall("joint"):
        joint.set("class", "human")
    for child in body.findall("body"):
        stamp_joint_class(child)


def insert_collision_capsules(
    pelvis: etree._Element, capsules: Sequence[CapsuleSpec]
) -> None:
    by_name: dict[str, etree._Element] = {}

    def collect(b: etree._Element) -> None:
        name = b.get("name")
        if name:
            by_name[name] = b
        for child in b.findall("body"):
            collect(child)

    collect(pelvis)
    for carrier_name, geom_name, attrs in capsules:
        body = by_name.get(carrier_name)
        if body is None:
            raise RuntimeError(
                f"build_g1_human_body: carrier body '{carrier_name}' "
                f"not found — capsule '{geom_name}' cannot be placed."
            )
        attrib = {"name": geom_name, "class": "human_collision"}
        attrib.update(attrs)
        body.append(etree.Element("geom", attrib))


def build_actuator_block() -> etree._Element:
    actuator = etree.Element("actuator")
    for joint_name in BODY_JOINT_NAMES:
        etree.SubElement(actuator, "position", {
            "name": f"act_{joint_name}",
            "joint": joint_name,
            "class": "position_actuator",
        })
    return actuator


def build_xml(model_name: str, capsules: Sequence[CapsuleSpec]) -> str:
    upstream = etree.parse(str(UPSTREAM_PATH))
    root = upstream.getroot()
    new_root = etree.Element("mujoco", {"model": model_name})
    new_root.append(build_defaults())
    new_world = etree.SubElement(new_root, "worldbody")
    upstream_world = root.find("worldbody")
    if upstream_world is None:
        raise RuntimeError("Upstream g1.xml has no <worldbody>")
    pelvis = upstream_world.find("body[@name='pelvis']")
    if pelvis is None:
        raise RuntimeError("Upstream g1.xml has no body named 'pelvis'")
    transform_pelvis(pelvis)
    strip_geoms_and_sites(pelvis)
    stamp_joint_class(pelvis)
    insert_collision_capsules(pelvis, capsules)
    new_world.append(pelvis)
    new_root.append(build_actuator_block())
    return etree.tostring(
        new_root, pretty_print=True, xml_declaration=True, encoding="utf-8"
    ).decode("utf-8")


def write_preset(name: str) -> Path:
    out_path, model_name, capsules = PRESETS[name]
    xml = build_xml(model_name, capsules)
    out_path.write_text(xml)
    print(f"Wrote {out_path} ({len(xml)} bytes) [{name}]")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--view", action="store_true",
        help="Build viewer asset only (g1_human_body_view.xml).",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Build training + view assets.",
    )
    args = parser.parse_args()

    if not UPSTREAM_PATH.exists():
        print(f"ERROR: upstream g1.xml not found at {UPSTREAM_PATH}", file=sys.stderr)
        return 1

    if args.view:
        write_preset("view")
    elif args.all:
        write_preset("training")
        write_preset("view")
    else:
        write_preset("training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
