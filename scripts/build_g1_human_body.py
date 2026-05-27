"""Derive ``assets/g1_human_body.xml`` from upstream ``assets/g1/g1.xml``.

The output is the merge-into-world fragment loaded by ``SafetyBiGymEnv``
when ``human_model == "g1"``. The script is deterministic and idempotent:
re-running on the same upstream input produces a byte-identical output.

Transformations applied:

1. Rename root body ``pelvis`` → ``Pelvis``, drop the ``childclass="g1"``
   attribute, replace ``<freejoint>`` with ``mocap="true"``, set
   ``pos="0 0 0"`` (the controller writes runtime position to mocap_pos).
2. **Keep** every ``class="visual"`` geom so the silhouette remains a
   real G1, but remap all visual geoms to one matte skin-toned material.
   This preserves the robot shape while reducing the high-contrast
   black/metal visual shift that can destabilise the pixel encoder. Strip
   the upstream ``class="collision"`` mesh geoms and ``class="foot"``
   proxies — they are replaced by the hand-authored ``_col`` capsule
   primitives.
3. **Keep** the ``<asset>`` block (mesh refs are needed by the visual
   geoms). Rewrite each ``<mesh file="...">`` to a path relative to the
   output XML; ``SafetyBiGymEnv._create_merged_world`` absolutises those
   paths when copying the asset block into the temp merged XML.
4. Strip the upstream ``<keyframe>`` (it referenced the now-removed
   freejoint) and the upstream ``<actuator>`` / ``<sensor>`` blocks.
   Regenerate ``<actuator>`` with one ``class="position_actuator"``
   entry per body joint in ``g1_human_spec.BODY_JOINT_NAMES``.
5. Insert ``<default>`` blocks for ``human`` / ``human_collision`` /
   ``position_actuator`` matching ``assets/smplh_human_body.xml`` so the
   existing env wrappers (collision-bits, PFL geom suffix) work unchanged.
   Also declare ``g1_matte_skin``, a low-specular material used by every
   visual mesh geom.
6. Stamp ``class="human"`` on every joint so they inherit human damping /
   armature defaults.
7. Insert hand-authored collision capsules on the chosen carrier bodies,
   each named ``<Region>_col`` so ``ISO15066Wrapper`` and
   ``_configure_collision_bits`` find them by suffix.

Trade-off: mesh shape still introduces more visual detail than the all-capsule
strategy α, but the material is deliberately low-contrast to keep the pixel
distribution closer to the SMPL-H baseline.

Run from the repo root:
    cd safety_bigym && python scripts/build_g1_human_body.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_PATH = REPO_ROOT / "safety_bigym" / "assets" / "g1" / "g1.xml"
OUTPUT_PATH = REPO_ROOT / "safety_bigym" / "assets" / "g1_human_body.xml"

# Load the spec module directly to avoid triggering ``safety_bigym.human``'s
# transitive imports (mujoco/mojo) at build time.
import importlib.util  # noqa: E402

_SPEC_PATH = REPO_ROOT / "safety_bigym" / "human" / "g1_human_spec.py"
_spec = importlib.util.spec_from_file_location("g1_human_spec", _SPEC_PATH)
g1_human_spec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1_human_spec)
BODY_JOINT_NAMES = g1_human_spec.BODY_JOINT_NAMES
VISUAL_MATERIAL_NAME = "g1_matte_skin"


# (carrier_body, geom_name, geom_attrs) — appended in document order on the
# matching body. Sizes / fromto coords are body-LOCAL.
COLLISION_CAPSULES: list[tuple[str, str, dict[str, str]]] = [
    # Root + trunk (mocap-driven; matches SMPL-H pelvis Pelvis_col).
    ("Pelvis", "Pelvis_col", {"fromto": "0 -0.10 0  0 0.10 0", "size": "0.12"}),
    ("waist_yaw_link", "Spine_col", {"fromto": "0 0 0  0 0 0.10", "size": "0.08"}),
    ("torso_link", "Chest_col", {"fromto": "0 0 0.05  0 0 0.30", "size": "0.13"}),
    # Head as a sphere on torso_link (no separate head body in upstream g1).
    ("torso_link", "Head_col", {"pos": "0 0 0.48", "size": "0.10", "type": "sphere"}),
    # Left leg.
    ("left_hip_yaw_link", "L_Thigh_col",
     {"fromto": "0 0 0  -0.078 0.002 -0.177", "size": "0.07"}),
    ("left_knee_link", "L_Shin_col",
     {"fromto": "0 0 0  0 0 -0.30", "size": "0.055"}),
    ("left_ankle_roll_link", "L_Foot_col",
     {"fromto": "-0.04 0 -0.025  0.12 0 -0.025", "size": "0.04"}),
    # Right leg.
    ("right_hip_yaw_link", "R_Thigh_col",
     {"fromto": "0 0 0  -0.078 -0.002 -0.177", "size": "0.07"}),
    ("right_knee_link", "R_Shin_col",
     {"fromto": "0 0 0  0 0 -0.30", "size": "0.055"}),
    ("right_ankle_roll_link", "R_Foot_col",
     {"fromto": "-0.04 0 -0.025  0.12 0 -0.025", "size": "0.04"}),
    # Left arm.
    ("left_shoulder_pitch_link", "L_Shoulder_col",
     {"fromto": "0 0 0  0 0.04 -0.02", "size": "0.06"}),
    ("left_elbow_link", "L_Elbow_col",
     {"fromto": "0 0 0  0.10 0 -0.01", "size": "0.045"}),
    ("left_wrist_roll_link", "L_Wrist_col",
     {"fromto": "0 0 0  0.084 0 0", "size": "0.035"}),
    ("left_wrist_yaw_link", "L_Hand_col",
     {"fromto": "0 0 0  0.09 0 0", "size": "0.040"}),
    # Right arm.
    ("right_shoulder_pitch_link", "R_Shoulder_col",
     {"fromto": "0 0 0  0 -0.04 -0.02", "size": "0.06"}),
    ("right_elbow_link", "R_Elbow_col",
     {"fromto": "0 0 0  0.10 0 -0.01", "size": "0.045"}),
    ("right_wrist_roll_link", "R_Wrist_col",
     {"fromto": "0 0 0  0.084 0 0", "size": "0.035"}),
    ("right_wrist_yaw_link", "R_Hand_col",
     {"fromto": "0 0 0  0.09 0 0", "size": "0.040"}),
]


def build_defaults() -> etree._Element:
    """Default classes the merged-into-world XML needs.

    Three from the SMPL-H contract (``human`` / ``human_collision`` /
    ``position_actuator``) plus an upstream-G1 ``visual`` class so the
    kept mesh geoms render with the right group / contact bits / texture.
    """
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
        # Collision capsules stay invisible (group 3) — render is now
        # carried by the upstream G1 visual meshes. We still need the
        # capsules for SSM / PFL / collision-channel wiring, but we
        # don't want them showing up over the rendered mesh.
        "group": "3", "contype": "2", "conaffinity": "4",
        "rgba": "0.8 0.6 0.5 0.0",
    })

    pos = etree.SubElement(defaults, "default", {"class": "position_actuator"})
    etree.SubElement(pos, "position", {"kp": "200", "kv": "20"})

    # Keep the real G1 mesh silhouette but make the render closer to the
    # SMPL-H training distribution: low-contrast, matte, skin-toned, and no
    # black/metal material split for the CNN encoder to latch onto.
    vis = etree.SubElement(defaults, "default", {"class": "visual"})
    etree.SubElement(vis, "geom", {
        "group": "2", "type": "mesh",
        "contype": "0", "conaffinity": "0",
        "density": "0", "material": VISUAL_MATERIAL_NAME,
    })

    return defaults


def build_asset_block(upstream_root: etree._Element) -> etree._Element:
    """Copy the upstream ``<asset>`` block with mesh paths rewritten as
    paths **relative to** the output ``g1_human_body.xml`` location.

    The output XML lives at ``safety_bigym/assets/g1_human_body.xml`` and
    the upstream STL files live at ``safety_bigym/assets/g1/assets/*.STL``,
    so the relative reference is ``g1/assets/<file>``. The env's
    ``_create_merged_world`` resolves these to absolute paths at runtime
    (because the merged-into-world XML is written to a temp dir where the
    relative path would otherwise break). This keeps the checked-in XML
    portable across machines (no absolute paths baked in).
    """
    upstream_asset = upstream_root.find("asset")
    if upstream_asset is None:
        return etree.Element("asset")

    out = etree.Element("asset")
    etree.SubElement(out, "material", {
        "name": VISUAL_MATERIAL_NAME,
        "rgba": "0.75 0.58 0.50 1",
        "specular": "0.05",
        "shininess": "0.05",
        "reflectance": "0",
    })

    for child in upstream_asset:
        clone = etree.fromstring(etree.tostring(child))
        if clone.tag == "mesh":
            file_attr = clone.get("file")
            if file_attr is not None:
                # Upstream g1.xml uses ``meshdir="assets"`` so file paths are
                # bare filenames. The output XML lives one directory above,
                # so prefix ``g1/assets/`` to get a path relative to the
                # output XML's parent.
                clone.set("file", f"g1/assets/{file_attr}")
        out.append(clone)
    return out


def normalize_visual_materials(body: etree._Element) -> None:
    """Assign every kept visual mesh the same low-contrast matte material."""
    for geom in body.findall("geom"):
        if geom.get("class") == "visual":
            geom.set("material", VISUAL_MATERIAL_NAME)
    for child in body.findall("body"):
        normalize_visual_materials(child)


def transform_pelvis(pelvis: etree._Element) -> None:
    """In-place: rename body, drop childclass, replace freejoint with mocap."""
    pelvis.set("name", "Pelvis")
    pelvis.set("pos", "0 0 0")
    pelvis.set("mocap", "true")
    if "childclass" in pelvis.attrib:
        del pelvis.attrib["childclass"]

    # Remove freejoint child (mocap bodies don't have qpos).
    for fj in list(pelvis.findall("freejoint")):
        pelvis.remove(fj)


def strip_collision_geoms_and_sites(body: etree._Element) -> None:
    """Remove only the upstream collision/foot geoms and all sites.

    Visual mesh geoms are kept so the rendered G1 looks like the real
    robot. Collision is supplied by hand-authored ``_col`` capsules added
    later by :func:`insert_collision_capsules`. Sites are stripped because
    the upstream IMU sites have no role in safety_bigym (and the upstream
    ``<sensor>`` block they fed is dropped).
    """
    for geom in list(body.findall("geom")):
        cls = geom.get("class")
        if cls in ("collision", "foot"):
            body.remove(geom)
    for site in list(body.findall("site")):
        body.remove(site)
    for child in body.findall("body"):
        strip_collision_geoms_and_sites(child)


def stamp_joint_class(body: etree._Element) -> None:
    """Set class='human' on every <joint> in the body subtree."""
    for joint in body.findall("joint"):
        joint.set("class", "human")
    for child in body.findall("body"):
        stamp_joint_class(child)


def insert_collision_capsules(pelvis: etree._Element) -> None:
    """Walk the body tree, append the configured _col geom on each carrier."""
    by_name: dict[str, etree._Element] = {}

    def collect(b: etree._Element) -> None:
        name = b.get("name")
        if name:
            by_name[name] = b
        for child in b.findall("body"):
            collect(child)

    collect(pelvis)

    for carrier_name, geom_name, attrs in COLLISION_CAPSULES:
        body = by_name.get(carrier_name)
        if body is None:
            raise RuntimeError(
                f"build_g1_human_body: carrier body '{carrier_name}' "
                f"not found in upstream g1.xml — capsule '{geom_name}' "
                "cannot be placed."
            )
        attrib = {"name": geom_name, "class": "human_collision"}
        attrib.update(attrs)
        geom = etree.Element("geom", attrib)
        body.append(geom)


def build_actuator_block() -> etree._Element:
    """One <position class='position_actuator'> per body joint."""
    actuator = etree.Element("actuator")
    for joint_name in BODY_JOINT_NAMES:
        etree.SubElement(actuator, "position", {
            "name": f"act_{joint_name}",
            "joint": joint_name,
            "class": "position_actuator",
        })
    return actuator


def build() -> str:
    upstream = etree.parse(str(UPSTREAM_PATH))
    root = upstream.getroot()

    # Rebuild root with the model name we want and our minimal structure.
    new_root = etree.Element("mujoco", {"model": "g1_human_body"})

    new_root.append(build_defaults())
    new_root.append(build_asset_block(root))

    # Worldbody: keep the upstream body tree (including visual mesh geoms),
    # transform the pelvis, strip only collision proxies + sites.
    new_world = etree.SubElement(new_root, "worldbody")
    upstream_world = root.find("worldbody")
    if upstream_world is None:
        raise RuntimeError("Upstream g1.xml has no <worldbody>")

    pelvis = upstream_world.find("body[@name='pelvis']")
    if pelvis is None:
        raise RuntimeError("Upstream g1.xml has no body named 'pelvis'")

    transform_pelvis(pelvis)
    strip_collision_geoms_and_sites(pelvis)
    normalize_visual_materials(pelvis)
    stamp_joint_class(pelvis)
    insert_collision_capsules(pelvis)

    new_world.append(pelvis)

    new_root.append(build_actuator_block())

    serialized = etree.tostring(
        new_root, pretty_print=True, xml_declaration=True, encoding="utf-8"
    )
    return serialized.decode("utf-8")


def main() -> int:
    if not UPSTREAM_PATH.exists():
        print(f"ERROR: upstream g1.xml not found at {UPSTREAM_PATH}", file=sys.stderr)
        print(
            "Run: cd safety_bigym && "
            "git checkout safety-critic/g1-coworker -- safety_bigym/assets/g1/",
            file=sys.stderr,
        )
        return 1

    xml = build()
    OUTPUT_PATH.write_text(xml)
    print(f"Wrote {OUTPUT_PATH} ({len(xml)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
