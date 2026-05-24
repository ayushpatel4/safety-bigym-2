"""Generate ``assets/g1_human_body.xml`` from the vendored mujoco_menagerie
Unitree G1 model (``assets/g1/g1.xml``).

The output is a drop-in replacement for ``smplh_human_body.xml`` that slots
into ``SafetyBiGymEnv``'s injection contract:

* the ``pelvis`` root is converted from a ``<freejoint>`` to ``mocap="true"``
  so the trajectory planner can teleport it via ``data.mocap_pos/quat``;
* every collision geom is given a unique name ending in ``_col`` and a
  non-zero contype/conaffinity (so ``_configure_collision_bits`` re-homes it
  onto the cross-paired human<->robot channel rather than skipping it);
* visual geoms keep ``contype=0 conaffinity=0`` so they never collide;
* each position actuator is renamed ``act_<joint>`` so ``PDController`` (which
  filters on the ``act_`` prefix) drives it;
* mesh/material names are prefixed ``g1_`` to avoid colliding with BiGym scene
  asset names when merged;
* ``<compiler meshdir="g1/assets">`` stays *relative* so the committed file is
  portable; ``SafetyBiGymEnv._create_merged_world`` rewrites mesh paths to
  absolute at merge time.

Run from the repo root: ``python scripts/build_g1_human_body.py``.
"""

from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

ASSETS = Path(__file__).resolve().parent.parent / "safety_bigym" / "assets"
SRC = ASSETS / "g1" / "g1.xml"
OUT = ASSETS / "g1_human_body.xml"

# ISO-region-friendly collision geom names keyed by the source mesh name.
# Non-mesh collision geoms (shoulder cylinders, foot spheres) are named after
# their parent body. Names must end in "_col".
MESH_COL_NAME = {
    "pelvis_contour_link": "pelvis",
    "torso_link": "torso",
    "logo_link": "torso_logo",
    "head_link": "head",
}

# Material recolor (2026-05-24). The vendored Unitree mujoco_menagerie model
# renders the G1 as dark/metallic (`black` rgba 0.2/0.2/0.2, `metal` rgba
# 0.7/0.7/0.7). On the saucepan_to_hob G1 base-curriculum the high-contrast
# dark silhouette disrupted the CQN-AS CNN encoder enough to break task
# learning (robot retreated from workspace); MASK_PIXELS=1 confirmed the CNN
# was in the failure path. Recoloring to warm skin-tones moves G1 closer to
# the kitchen-background distribution the CNN learned from coworker-free
# demos, so it can extract task features without the encoder being dominated
# by the G1 blob. Keyed by ORIGINAL menagerie material name (the script
# prefixes ``g1_`` after the recolor map is applied).
MATERIAL_RECOLOR = {
    "black": "0.90 0.78 0.65 1",   # warm light skin (was 0.2 0.2 0.2)
    "metal": "0.78 0.66 0.55 1",   # slightly darker accent (was 0.7 0.7 0.7)
}


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def main() -> None:
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(SRC), parser)
    root = tree.getroot()
    root.set("model", "g1_human_body")

    # --- compiler: keep meshdir relative to the assets dir ---------------
    compiler = root.find("compiler")
    if compiler is None:
        compiler = etree.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", "g1/assets")

    # --- drop runtime-only blocks we don't want in the merged world ------
    for tag in ("option", "sensor", "keyframe"):
        el = root.find(tag)
        if el is not None:
            root.remove(el)

    # --- prefix mesh + material names; record rename maps ----------------
    asset = root.find("asset")
    mesh_rename: dict[str, str] = {}
    mat_rename: dict[str, str] = {}
    for m in asset.findall("mesh"):
        # derive the implicit name from file when no explicit name is set
        name = m.get("name")
        if name is None:
            name = Path(m.get("file")).stem
            m.set("name", name)
        new = f"g1_{name}"
        mesh_rename[name] = new
        m.set("name", new)
    for mat in asset.findall("material"):
        name = mat.get("name")
        # Recolor BEFORE prefixing so the lookup key matches the vendored
        # menagerie name (see MATERIAL_RECOLOR for the why).
        if name in MATERIAL_RECOLOR:
            mat.set("rgba", MATERIAL_RECOLOR[name])
        new = f"g1_{name}"
        mat_rename[name] = new
        mat.set("name", new)

    # --- defaults: force non-zero collision bits, keep visual at 0/0 -----
    # Find the nested <default class="collision"><geom .../> and set bits.
    for d in root.iter("default"):
        if d.get("class") == "collision":
            g = d.find("geom")
            if g is not None:
                g.set("contype", "2")
                g.set("conaffinity", "4")
                g.set("group", "0")
        # The menagerie position actuator uses dampratio/inheritrange, which
        # require MuJoCo >= 3.2. Rewrite to explicit kp/kv so the model loads
        # on the pinned 3.1.x runtime. PD is done by the position actuator
        # internally (PDController only writes ctrl=target).
        if d.get("class") == "g1":
            pos = d.find("position")
            if pos is not None:
                for attr in ("dampratio", "inheritrange"):
                    if attr in pos.attrib:
                        del pos.attrib[attr]
                pos.set("kp", "500")
                pos.set("kv", "50")

    # --- rewrite mesh/material references inside <default> geoms ----------
    # (body-geom refs are rewritten in the worldbody pass below, which also
    # needs the *original* mesh name to build _col names.)
    default_root = root.find("default")
    if default_root is not None:
        for g in default_root.iter("geom"):
            mesh = g.get("mesh")
            if mesh in mesh_rename:
                g.set("mesh", mesh_rename[mesh])
            mat = g.get("material")
            if mat in mat_rename:
                g.set("material", mat_rename[mat])

    worldbody = root.find("worldbody")

    # --- pelvis: freejoint -> mocap --------------------------------------
    pelvis = worldbody.find(".//body[@name='pelvis']")
    fj = pelvis.find("freejoint")
    if fj is not None:
        pelvis.remove(fj)
    pelvis.set("mocap", "true")
    # mocap bodies are positioned at the world origin by the controller; the
    # source pos="0 0 0.793" is irrelevant once mocap-driven, so zero it.
    pelvis.set("pos", "0 0 0")

    # --- walk every geom: rename refs, name + bit collision geoms --------
    col_counts: dict[str, int] = {}
    for body in worldbody.iter("body"):
        bname = body.get("name")
        for g in body.findall("geom"):
            cls = g.get("class")
            mesh = g.get("mesh")
            if mesh in mesh_rename:
                g.set("mesh", mesh_rename[mesh])
            mat = g.get("material")
            if mat in mat_rename:
                g.set("material", mat_rename[mat])

            is_collision = cls in ("collision", "foot")
            if not is_collision:
                continue

            # Build a stable, unique, region-friendly name ending in _col.
            if mesh is not None:
                stem = mesh  # original mesh name (pre-prefix lookup)
                base = MESH_COL_NAME.get(stem, stem)
            else:
                base = bname
            key = f"{base}_col"
            n = col_counts.get(key, 0)
            col_counts[key] = n + 1
            g.set("name", key if n == 0 else f"{key}{n}")

    # --- actuators: rename to act_<joint> --------------------------------
    actuator = root.find("actuator")
    for pos in actuator.findall("position"):
        joint = pos.get("joint")
        pos.set("name", f"act_{joint}")

    # --- header comment --------------------------------------------------
    header = etree.Comment(
        " AUTOGENERATED by scripts/build_g1_human_body.py from assets/g1/g1.xml. "
        "Do not edit by hand; edit the generator and regenerate. "
        "Drop-in for smplh_human_body.xml: mocap pelvis, act_<joint> actuators, "
        "*_col collision geoms on the cross-paired human<->robot channel. "
    )
    root.insert(0, header)

    OUT.write_bytes(
        etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="utf-8")
    )

    # --- report ----------------------------------------------------------
    col_names = sorted(col_counts)
    print(f"wrote {OUT}")
    print(f"{len(col_names)} distinct _col geom names:")
    for c in col_names:
        suffix = f" (x{col_counts[c]})" if col_counts[c] > 1 else ""
        print(f"  {c}{suffix}")


if __name__ == "__main__":
    main()
