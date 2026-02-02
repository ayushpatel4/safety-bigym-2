
import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path

def test_xml_merge():
    # 1. Create dummy world.xml
    world_xml = Path("dummy_world.xml")
    with open(world_xml, "w") as f:
        f.write("""
<mujoco model="world">
  <option integrator="implicitfast">
    <flag multiccd="enable"/>
  </option>
  <statistic center="0 0 1"/>
  <worldbody>
    <body name="floor">
        <geom name="floor_geom"/>
    </body>
  </worldbody>
  <asset>
    <texture name="sky"/>
  </asset>
</mujoco>
""")

    # 2. Create dummy human.xml
    human_xml = Path("dummy_human.xml")
    with open(human_xml, "w") as f:
        f.write("""
<mujoco model="smplh">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
     <global azimuth="100"/>
  </visual>
  <worldbody>
    <body name="pelvis">
        <geom name="pelvis_geom"/>
    </body>
  </worldbody>
  <asset>
    <texture name="skin"/>
  </asset>
  <actuator>
    <motor name="action"/>
  </actuator>
</mujoco>
""")

    try:
        # 3. Run merge logic (copied from SafetyBiGymEnv)
        world_tree = ET.parse(world_xml)
        world_root = world_tree.getroot()
        
        human_tree = ET.parse(human_xml)
        human_root = human_tree.getroot()
        
        # Merge human into world
        for tag in ['option', 'compiler', 'statistic', 'visual', 'size']:
            human_elem = human_root.find(tag)
            world_elem = world_root.find(tag)
            
            if human_elem is not None:
                if world_elem is None:
                    world_root.append(human_elem)
                else:
                    for k, v in human_elem.attrib.items():
                        world_elem.set(k, v)
                    for child in human_elem:
                        world_elem.append(child)
        
        for tag in ['asset', 'worldbody', 'default', 'actuator', 'sensor', 'equality']:
            human_elems = human_root.findall(tag)
            world_elem = world_root.find(tag)
            
            for h_elem in human_elems:
                if world_elem is None:
                    world_root.append(h_elem)
                    world_elem = h_elem
                else:
                    for child in h_elem:
                        world_elem.append(child)
        
        # 4. Write output and verify
        out_file = Path("merged.xml")
        world_tree.write(out_file)
        
        with open(out_file, "r") as f:
            content = f.read()
            print("Merged Content:\n", content)
            
        # Checks
        tree = ET.parse(out_file)
        root = tree.getroot()
        
        # Check option storage
        opt = root.find("option")
        assert opt.get("integrator") == "implicitfast" # from world
        assert opt.get("gravity") == "0 0 -9.81"       # from human (merged)
        
        # Check worldbody
        wb = root.find("worldbody")
        bodies = [b.get("name") for b in wb.findall("body")]
        assert "floor" in bodies
        assert "pelvis" in bodies
        
        # Check asset
        asset = root.find("asset")
        textures = [t.get("name") for t in asset.findall("texture")]
        assert "sky" in textures
        assert "skin" in textures
        
        # Check actuator
        act = root.find("actuator")
        assert act is not None
        assert act.find("motor").get("name") == "action"
        
        print("Verification SUCCESS")
        
    finally:
        if world_xml.exists(): world_xml.unlink()
        if human_xml.exists(): human_xml.unlink()
        if Path("merged.xml").exists(): Path("merged.xml").unlink()

if __name__ == "__main__":
    test_xml_merge()
