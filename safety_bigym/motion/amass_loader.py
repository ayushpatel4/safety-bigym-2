"""
AMASS Motion Data Loader

Loads AMASS motion capture data and converts it to MuJoCo-compatible format.

AMASS data format:
- poses: (N, 156) - 52 joints × 3 axis-angle per frame
- trans: (N, 3) - root translation (SMPL Y-up coordinates)
- mocap_framerate: original frame rate

Conversion:
- SMPL uses Y-up, MuJoCo uses Z-up
- SMPL uses axis-angle, we convert to euler angles for MuJoCo hinge joints
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation


@dataclass
class MotionClip:
    """Represents a motion capture clip ready for MuJoCo playback."""
    
    # Joint angles: (num_frames, num_joints, 3) - euler angles per joint
    joint_angles: np.ndarray
    
    # Root translation: (num_frames, 3) - in MuJoCo Z-up coordinates
    root_translation: np.ndarray
    
    # Root orientation: (num_frames, 4) - quaternion (w, x, y, z)
    root_orientation: np.ndarray
    
    # Timing
    fps: float
    duration: float
    
    # Metadata
    source_file: str
    num_frames: int
    num_joints: int
    
    @property
    def dt(self) -> float:
        """Time step between frames."""
        return 1.0 / self.fps
    
    def get_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get joint angles, translation, and orientation for a single frame."""
        return (
            self.joint_angles[frame_idx],
            self.root_translation[frame_idx],
            self.root_orientation[frame_idx],
        )
    
    def get_time_frame(self, time: float) -> int:
        """Get frame index for a given time."""
        return int(min(time * self.fps, self.num_frames - 1))


class AMASSLoader:
    """Load and convert AMASS motion data for MuJoCo playback."""
    
    # SMPL-H has 52 joints, but first 22 are body joints
    NUM_BODY_JOINTS = 22
    NUM_HAND_JOINTS = 30
    NUM_TOTAL_JOINTS = 52
    
    def __init__(self, include_hands: bool = False):
        """
        Initialize loader.
        
        Args:
            include_hands: Whether to include hand joint data
        """
        self.include_hands = include_hands
        self.num_joints = self.NUM_TOTAL_JOINTS if include_hands else self.NUM_BODY_JOINTS
    
    def load(self, npz_path: str) -> MotionClip:
        """
        Load an AMASS .npz file and convert to MuJoCo format.
        
        Args:
            npz_path: Path to AMASS .npz file
            
        Returns:
            MotionClip ready for MuJoCo playback
        """
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Motion file not found: {npz_path}")
        
        data = np.load(npz_path)
        
        # Extract data
        poses = data['poses']  # (N, 156) - 52 joints × 3 axis-angle
        trans = data['trans']  # (N, 3) - root translation
        fps = float(data['mocap_framerate'])
        
        num_frames = poses.shape[0]
        
        # Reshape poses to (N, 52, 3)
        poses_reshaped = poses.reshape(num_frames, 52, 3)
        
        # Limit to body joints if not including hands
        if not self.include_hands:
            poses_reshaped = poses_reshaped[:, :self.NUM_BODY_JOINTS, :]
        
        # Convert axis-angle to euler angles for each joint
        joint_angles = self._axis_angle_to_euler(poses_reshaped)
        
        # Convert root orientation (joint 0) to quaternion
        root_axis_angle = poses_reshaped[:, 0, :]  # (N, 3)
        root_orientation = self._axis_angle_to_quat_mujoco(root_axis_angle)
        
        # Convert translation from SMPL Y-up to MuJoCo Z-up
        root_translation = self._convert_translation(trans)
        
        # Convert joint angles from SMPL to MuJoCo coordinates
        joint_angles = self._convert_joint_angles(joint_angles)
        
        duration = num_frames / fps
        
        return MotionClip(
            joint_angles=joint_angles,
            root_translation=root_translation,
            root_orientation=root_orientation,
            fps=fps,
            duration=duration,
            source_file=str(path),
            num_frames=num_frames,
            num_joints=self.num_joints,
        )
    
    def _axis_angle_to_euler(self, axis_angles: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle rotations to euler angles.
        
        Args:
            axis_angles: (N, num_joints, 3) axis-angle representations
            
        Returns:
            (N, num_joints, 3) euler angles (XYZ order)
        """
        N, num_joints, _ = axis_angles.shape
        euler_angles = np.zeros_like(axis_angles)
        
        for i in range(N):
            for j in range(num_joints):
                aa = axis_angles[i, j]
                angle = np.linalg.norm(aa)
                
                if angle < 1e-8:
                    euler_angles[i, j] = np.zeros(3)
                else:
                    axis = aa / angle
                    rot = Rotation.from_rotvec(aa)
                    # Use XYZ euler order to match MuJoCo hinge joints
                    euler_angles[i, j] = rot.as_euler('xyz')
        
        return euler_angles
    
    def _axis_angle_to_quat_mujoco(self, axis_angles: np.ndarray) -> np.ndarray:
        """
        Convert axis-angle to quaternion in MuJoCo format (w, x, y, z).
        
        For root orientation, we need to:
        1. Convert the rotation from SMPL Y-up to MuJoCo Z-up coordinate system
        2. Apply a 90° rotation around X to account for the skeleton being built
           in a different default orientation
        
        Args:
            axis_angles: (N, 3) axis-angle representations in SMPL coords
            
        Returns:
            (N, 4) quaternions in (w, x, y, z) format
        """
        N = axis_angles.shape[0]
        quats = np.zeros((N, 4))
        
        # Rotation to bring character right-side up
        # This is a -90° rotation around the X axis
        R_standup = Rotation.from_euler('x', -90, degrees=True)
        
        for i in range(N):
            aa = axis_angles[i]
            
            # Convert axis-angle from SMPL Y-up to MuJoCo Z-up
            # SMPL: x, y, z where y is up
            # MuJoCo: x, y, z where z is up
            # Transform: x -> x, y -> z, z -> -y
            aa_mujoco = np.array([aa[0], -aa[2], aa[1]])
            
            angle = np.linalg.norm(aa_mujoco)
            
            if angle < 1e-8:
                rot = Rotation.identity()
            else:
                rot = Rotation.from_rotvec(aa_mujoco)
            
            # Apply standup rotation FIRST, then pose rotation
            rot_final = R_standup * rot
            
            # Convert to quaternion (scipy uses x, y, z, w)
            q_scipy = rot_final.as_quat()  # (x, y, z, w)
            
            # MuJoCo uses (w, x, y, z)
            quats[i] = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        
        return quats
    
    def _convert_translation(self, trans: np.ndarray) -> np.ndarray:
        """
        Convert translation from SMPL Y-up to MuJoCo Z-up.
        
        Translation and orientation are independent world-coordinate systems.
        We just need to swap the vertical axis:
        - SMPL: Y is up, Z is forward
        - MuJoCo: Z is up, Y is forward (negated for direction)
        
        Also compensates for the MJCF pelvis body offset.
        
        Args:
            trans: (N, 3) translations in SMPL coordinates
            
        Returns:
            (N, 3) translations in MuJoCo coordinates
        """
        # MJCF pelvis Z offset (compensate for pelvis position below freejoint)
        PELVIS_Z_OFFSET = -0.2408
        
        # Coordinate mapping adjusted for correct motion direction
        # SMPL: x=left/right, y=up, z=forward
        # MuJoCo: x=left/right, y=forward, z=up
        trans_mujoco = np.zeros_like(trans)
        trans_mujoco[:, 0] = trans[:, 0]                     # x -> x (left/right)
        trans_mujoco[:, 1] = trans[:, 2]                     # z -> y (forward)
        trans_mujoco[:, 2] = trans[:, 1] - PELVIS_Z_OFFSET   # y -> z (up, with offset)
        
        return trans_mujoco
    
    def _convert_joint_angles(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert joint angles from SMPL to MuJoCo coordinate system.
        
        The euler angles need to be reordered to match the MuJoCo
        hinge joint axes (x, y, z in body-local frame).
        
        Args:
            euler_angles: (N, num_joints, 3) in SMPL coordinates
            
        Returns:
            (N, num_joints, 3) in MuJoCo coordinates
        """
        # For now, we apply a simple coordinate transform
        # This may need refinement based on motion playback results
        converted = np.zeros_like(euler_angles)
        converted[:, :, 0] = euler_angles[:, :, 0]   # x rotation
        converted[:, :, 1] = -euler_angles[:, :, 2]  # z -> -y rotation
        converted[:, :, 2] = euler_angles[:, :, 1]   # y -> z rotation
        return converted


def load_amass_clip(npz_path: str, include_hands: bool = False) -> MotionClip:
    """
    Convenience function to load an AMASS motion clip.
    
    Args:
        npz_path: Path to AMASS .npz file
        include_hands: Whether to include hand joint data
        
    Returns:
        MotionClip ready for MuJoCo playback
    """
    loader = AMASSLoader(include_hands=include_hands)
    return loader.load(npz_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and inspect AMASS motion clip")
    parser.add_argument("npz_path", type=str, help="Path to AMASS .npz file")
    parser.add_argument("--include_hands", action="store_true", help="Include hand joints")
    
    args = parser.parse_args()
    
    clip = load_amass_clip(args.npz_path, args.include_hands)
    
    print(f"Loaded: {clip.source_file}")
    print(f"  Frames: {clip.num_frames}")
    print(f"  Duration: {clip.duration:.2f}s")
    print(f"  FPS: {clip.fps}")
    print(f"  Joints: {clip.num_joints}")
    print(f"  Joint angles shape: {clip.joint_angles.shape}")
    print(f"  Root translation range:")
    print(f"    X: [{clip.root_translation[:, 0].min():.2f}, {clip.root_translation[:, 0].max():.2f}]")
    print(f"    Y: [{clip.root_translation[:, 1].min():.2f}, {clip.root_translation[:, 1].max():.2f}]")
    print(f"    Z: [{clip.root_translation[:, 2].min():.2f}, {clip.root_translation[:, 2].max():.2f}]")
