"""
SMPL-H to MuJoCo MJCF Generator (Standalone)

Uses smplx library directly to read SMPL-H pickle files and generate MuJoCo XML.
Does not require SMPLSim - only requires smplx and torch.

Usage:
    python smplh_generator.py --model_path /path/to/smplh --gender male --output smplh_human.xml
"""

import argparse
import numpy as np
from pathlib import Path
import torch
from smplx import SMPLH

# SMPL-H joint names (22 body + 30 hand = 52 total)
SMPLH_BONE_ORDER_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
    "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck",
    "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    # Left hand
    "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    # Right hand  
    "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]

# ISO 15066 body region mapping for PFL limits
ISO_BODY_REGIONS = {
    "Pelvis": "pelvis", "L_Hip": "thigh", "R_Hip": "thigh", 
    "Torso": "abdomen", "L_Knee": "shin", "R_Knee": "shin",
    "Spine": "chest", "L_Ankle": "foot", "R_Ankle": "foot",
    "Chest": "chest", "L_Toe": "foot", "R_Toe": "foot",
    "Neck": "neck", "L_Thorax": "back_shoulders", "R_Thorax": "back_shoulders",
    "Head": "skull", "L_Shoulder": "upper_arm", "R_Shoulder": "upper_arm",
    "L_Elbow": "forearm", "R_Elbow": "forearm",
    "L_Wrist": "hand_palm", "R_Wrist": "hand_palm",
}

# Capsule radii for each body region (meters)
CAPSULE_RADII = {
    "pelvis": 0.10, "thigh": 0.065, "shin": 0.045, "foot": 0.035,
    "abdomen": 0.12, "chest": 0.13, "back_shoulders": 0.08,
    "neck": 0.04, "skull": 0.10, "upper_arm": 0.04, "forearm": 0.035,
    "hand_palm": 0.025, "hand_finger": 0.008,
}


class SMPLHMuJoCoGenerator:
    """Generate MuJoCo MJCF from SMPL-H model using smplx library."""
    
    def __init__(self, model_path: str, gender: str = "male", 
                 include_hands: bool = False):
        self.model_path = Path(model_path)
        self.gender = gender
        self.include_hands = include_hands
        self.num_body_joints = 22
        self._load_smplh()
    
    def _load_smplh(self):
        """Load SMPL-H model using smplx."""
        self.model = SMPLH(
            model_path=str(self.model_path),
            gender=self.gender,
            use_pca=False,
            flat_hand_mean=True,
        )
        
        # Get T-pose joint positions
        with torch.no_grad():
            output = self.model(
                body_pose=torch.zeros(1, 63),  # 21 body joints * 3
                global_orient=torch.zeros(1, 3),
                left_hand_pose=torch.zeros(1, 45),  # 15 hand joints * 3
                right_hand_pose=torch.zeros(1, 45),
                betas=torch.zeros(1, 10),
            )
            self.joint_pos = output.joints[0, :52].numpy()  # (52, 3)
        
        # Parent indices from SMPL-H kinematic tree
        self.parent_indices = self.model.parents[:52].cpu().numpy()
        
        # Compute joint offsets
        self.joint_names = SMPLH_BONE_ORDER_NAMES[:52]
        self.joint_offsets = {}
        for i, name in enumerate(self.joint_names):
            parent_idx = self.parent_indices[i]
            if parent_idx < 0:
                self.joint_offsets[name] = self.joint_pos[i]
            else:
                self.joint_offsets[name] = self.joint_pos[i] - self.joint_pos[parent_idx]
        
        print(f"Loaded SMPL-H model: {len(self.joint_names)} joints")
        # SMPL uses Y-up
        height = self.joint_pos[:, 1].max() - self.joint_pos[:, 1].min()
        print(f"T-pose height: {height:.3f}m")
    
    def _get_capsule_params(self, joint_name: str, joint_idx: int) -> dict:
        """Compute capsule parameters for a body segment."""
        region = ISO_BODY_REGIONS.get(joint_name, "chest")
        radius = CAPSULE_RADII.get(region, 0.05)
        
        # Find children
        children = [i for i, p in enumerate(self.parent_indices) if p == joint_idx]
        
        if children and joint_idx < len(self.joint_pos):
            child_idx = children[0]
            if child_idx < len(self.joint_pos):
                bone_vec = self.joint_pos[child_idx] - self.joint_pos[joint_idx]
                bone_length = np.linalg.norm(bone_vec)
                
                if bone_length > 0.01:
                    # Convert SMPL Y-up to MuJoCo Z-up: x->x, y->z, z->-y
                    fromto = np.array([
                        0, 0, 0,
                        bone_vec[0], -bone_vec[2], bone_vec[1]
                    ])
                    return {"fromto": fromto, "size": radius}
        
        return {"pos": np.zeros(3), "size": radius, "type": "sphere"}
    
    def generate_mjcf(self, output_path: str = None) -> str:
        """Generate MuJoCo MJCF XML string."""
        if self.include_hands:
            joints_to_use = list(range(len(self.joint_names)))
        else:
            joints_to_use = list(range(self.num_body_joints))
        
        xml_parts = [
            self._generate_header(),
            self._generate_defaults(),
            self._generate_assets(),
            self._generate_worldbody(joints_to_use),
            self._generate_actuators(joints_to_use),
            "</mujoco>"
        ]
        
        xml_str = "\n".join(xml_parts)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(xml_str)
            print(f"Wrote MJCF to {output_path}")
        
        return xml_str
    
    def _generate_header(self) -> str:
        return '''<?xml version="1.0" encoding="utf-8"?>
<!--
  SMPL-H Human Skeleton for Safety BiGym
  Generated from SMPL-H model using smplh_generator.py
  
  Features:
  - Joint hierarchy from SMPL-H kintree_table
  - Bone lengths computed from T-pose
  - Collision capsules with soft contact
  - ISO 15066 body region annotations for PFL
  - Position actuators for motion tracking
-->
<mujoco model="smplh_human">
  <compiler angle="radian" autolimits="true" inertiafromgeom="true"/>
  
  <option gravity="0 0 -9.81" timestep="0.002"/>'''
    
    def _generate_defaults(self) -> str:
        return '''
  <default>
    <default class="human">
      <joint damping="50" armature="0.01"/>
      <geom type="capsule" condim="3" friction="1 0.5 0.001" density="1000"/>
    </default>
    
    <default class="human_collision">
      <geom type="capsule" 
            solref="0.02 1.0" 
            solimp="0.9 0.95 0.001"
            group="2"
            contype="1"
            conaffinity="1"
            rgba="0.8 0.6 0.5 0.5"/>
    </default>
    
    <default class="position_actuator">
      <position kp="200" kv="20"/>
    </default>
  </default>'''
    
    def _generate_assets(self) -> str:
        return '''
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.1 0.1 0.1" rgb2="0.2 0.2 0.2"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
  </asset>'''
    
    def _generate_worldbody(self, joints_to_use: list) -> str:
        """Generate worldbody with recursive body hierarchy."""
        lines = ["\n  <worldbody>"]
        lines.append('    <geom name="floor" type="plane" size="10 10 0.1" material="grid_mat" conaffinity="1" contype="1"/>')
        
        # Build tree structure
        children_map = {i: [] for i in range(-1, len(self.joint_names))}
        for i in joints_to_use:
            parent_idx = int(self.parent_indices[i])
            if parent_idx < 0 or parent_idx not in joints_to_use:
                parent_idx = -1
            children_map[parent_idx].append(i)
        
        # Generate bodies recursively
        for root_idx in children_map[-1]:
            self._generate_body_recursive(lines, root_idx, children_map, joints_to_use, indent=2)
        
        lines.append("  </worldbody>")
        return "\n".join(lines)
    
    def _generate_body_recursive(self, lines: list, joint_idx: int, 
                                  children_map: dict, joints_to_use: list,
                                  indent: int = 2):
        """Recursively generate body elements."""
        ind = "  " * indent
        joint_name = self.joint_names[joint_idx]
        
        # Get position offset (convert SMPL Y-up to MuJoCo Z-up)
        offset = self.joint_offsets[joint_name]
        pos_mj = np.array([offset[0], -offset[2], offset[1]])
        pos_str = f"{pos_mj[0]:.4f} {pos_mj[1]:.4f} {pos_mj[2]:.4f}"
        
        lines.append(f'{ind}<body name="{joint_name}" pos="{pos_str}">')
        
        # Root freejoint or regular hinge joints
        if joint_idx == 0:
            lines.append(f'{ind}  <freejoint name="root"/>')
        else:
            for axis_idx, axis_name in enumerate(["x", "y", "z"]):
                axis_vec = ["1 0 0", "0 1 0", "0 0 1"][axis_idx]
                lines.append(f'{ind}  <joint name="{joint_name}_{axis_name}" type="hinge" '
                           f'axis="{axis_vec}" range="-3.14 3.14" class="human"/>')
        
        # Collision geometry
        capsule = self._get_capsule_params(joint_name, joint_idx)
        
        if "fromto" in capsule:
            fromto_str = " ".join(f"{v:.4f}" for v in capsule["fromto"])
            lines.append(f'{ind}  <geom name="{joint_name}_col" class="human_collision" '
                        f'fromto="{fromto_str}" size="{capsule["size"]:.4f}"/>')
        else:
            pos_str = " ".join(f"{v:.4f}" for v in capsule.get("pos", np.zeros(3)))
            lines.append(f'{ind}  <geom name="{joint_name}_col" class="human_collision" '
                        f'type="sphere" pos="{pos_str}" size="{capsule["size"]:.4f}"/>')
        
        # Recurse for children
        for child_idx in children_map.get(joint_idx, []):
            if child_idx in joints_to_use:
                self._generate_body_recursive(lines, child_idx, children_map, 
                                             joints_to_use, indent + 1)
        
        lines.append(f'{ind}</body>')
    
    def _generate_actuators(self, joints_to_use: list) -> str:
        """Generate position actuators for all joints."""
        lines = ["\n  <actuator>"]
        
        for joint_idx in joints_to_use:
            if joint_idx == 0:
                continue
            
            joint_name = self.joint_names[joint_idx]
            for axis in ["x", "y", "z"]:
                lines.append(f'    <position name="act_{joint_name}_{axis}" '
                            f'joint="{joint_name}_{axis}" class="position_actuator"/>')
        
        lines.append("  </actuator>")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate MuJoCo MJCF from SMPL-H")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to SMPL-H model directory")
    parser.add_argument("--gender", type=str, default="male",
                       choices=["male", "female", "neutral"],
                       help="Model gender")
    parser.add_argument("--output", type=str, default="smplh_human.xml",
                       help="Output MJCF file path")
    parser.add_argument("--include_hands", action="store_true",
                       help="Include 30 hand joints")
    
    args = parser.parse_args()
    
    generator = SMPLHMuJoCoGenerator(
        model_path=args.model_path,
        gender=args.gender,
        include_hands=args.include_hands,
    )
    
    generator.generate_mjcf(args.output)


if __name__ == "__main__":
    main()
