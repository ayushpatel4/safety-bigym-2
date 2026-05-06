"""
Trajectory Planner for Human Root Motion

Generates smooth 2D root trajectories (x, y, yaw) that control WHERE
the human walks, independently of body joint angles (which come from AMASS).

Three trajectory types:
- PASS_BY: Walk past the robot with a lateral offset
- APPROACH_LOITER_DEPART: Walk to robot area, pause, walk away
- ARC: Curved arc past robot workspace

The planner only controls root position and yaw.
Body joint angles are NEVER modified here — they come from AMASS or IK.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List


class TrajectoryType(Enum):
    """Types of root trajectories."""
    PASS_BY = auto()
    APPROACH_LOITER_DEPART = auto()
    ARC = auto()


@dataclass
class TrajectoryConfig:
    """Configuration for a trajectory."""
    
    trajectory_type: TrajectoryType = TrajectoryType.PASS_BY
    
    # Where the robot is (trajectory is shaped relative to this)
    robot_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    
    # Human spawn position (XY)
    spawn_pos: np.ndarray = field(default_factory=lambda: np.array([2.0, 0.0]))
    
    # Approach direction (yaw in radians, pointing toward robot)
    approach_yaw: float = np.pi  # Default: facing -X (toward robot at origin)
    
    # --- PASS_BY parameters ---
    pass_by_offset: float = 1.0         # Lateral offset from robot (meters)
    pass_by_side: int = 1               # +1 = pass on left, -1 = pass on right
    
    # --- APPROACH_LOITER_DEPART parameters ---
    closest_approach: float = 1.0       # How close to robot before stopping (meters)
    loiter_duration: float = 2.0        # Time spent near robot (seconds)
    departure_angle: float = 150.0      # Relative departure direction (degrees)
    
    # --- ARC parameters ---
    arc_radius: float = 1.5             # Radius of arc past robot (meters)
    arc_extent: float = 120.0           # Angular extent of arc (degrees)
    
    # --- Speed ---
    walk_speed: float = 1.2             # Walking speed (m/s)
    
    # --- Total trajectory duration ---
    # Computed automatically from path length and speed
    
    def __post_init__(self):
        self.robot_pos = np.asarray(self.robot_pos, dtype=np.float64)
        self.spawn_pos = np.asarray(self.spawn_pos, dtype=np.float64)


@dataclass
class TrajectoryWaypoint:
    """A single waypoint on the trajectory."""
    position: np.ndarray    # (2,) XY position
    yaw: float              # Facing direction (radians)
    time: float             # Time at this waypoint (seconds)
    phase: str = "walk"     # "approach" | "loiter" | "depart" | "walk"


class TrajectoryPlanner:
    """
    Generates smooth 2D root trajectories for the human.
    
    Usage:
        config = TrajectoryConfig(trajectory_type=TrajectoryType.PASS_BY, ...)
        planner = TrajectoryPlanner(config)
        
        # At each timestep:
        x, y, yaw, phase = planner.get_pose(t)
    """
    
    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self._waypoints: List[TrajectoryWaypoint] = []
        self._total_duration: float = 0.0
        
        # Build waypoints based on trajectory type
        if config.trajectory_type == TrajectoryType.PASS_BY:
            self._build_pass_by()
        elif config.trajectory_type == TrajectoryType.APPROACH_LOITER_DEPART:
            self._build_approach_loiter_depart()
        elif config.trajectory_type == TrajectoryType.ARC:
            self._build_arc()
        else:
            raise ValueError(f"Unknown trajectory type: {config.trajectory_type}")
    
    @property
    def duration(self) -> float:
        """Total trajectory duration in seconds."""
        return self._total_duration
    
    @property
    def waypoints(self) -> List[TrajectoryWaypoint]:
        """List of trajectory waypoints."""
        return self._waypoints
    
    def get_pose(self, t: float) -> Tuple[float, float, float, str]:
        """
        Get human root pose at time t.
        
        Args:
            t: Time in seconds since episode start
            
        Returns:
            (x, y, yaw, phase) where phase is "approach"/"loiter"/"depart"/"walk"
        """
        if not self._waypoints:
            return (self.config.spawn_pos[0], self.config.spawn_pos[1], 
                    self.config.approach_yaw, "walk")
        
        # Clamp time
        if t <= self._waypoints[0].time:
            wp = self._waypoints[0]
            return (wp.position[0], wp.position[1], wp.yaw, wp.phase)
        
        if t >= self._waypoints[-1].time:
            wp = self._waypoints[-1]
            return (wp.position[0], wp.position[1], wp.yaw, wp.phase)
        
        # Find surrounding waypoints
        for i in range(len(self._waypoints) - 1):
            wp0 = self._waypoints[i]
            wp1 = self._waypoints[i + 1]
            
            if wp0.time <= t < wp1.time:
                # Interpolate between waypoints
                dt = wp1.time - wp0.time
                if dt < 1e-6:
                    alpha = 1.0
                else:
                    alpha = (t - wp0.time) / dt
                
                # Smooth interpolation (cubic ease in-out)
                alpha_smooth = self._smooth_step(alpha)
                
                x = wp0.position[0] + alpha_smooth * (wp1.position[0] - wp0.position[0])
                y = wp0.position[1] + alpha_smooth * (wp1.position[1] - wp0.position[1])
                yaw = self._lerp_angle(wp0.yaw, wp1.yaw, alpha_smooth)
                
                return (x, y, yaw, wp0.phase)
        
        # Fallback
        wp = self._waypoints[-1]
        return (wp.position[0], wp.position[1], wp.yaw, wp.phase)
    
    @staticmethod
    def _smooth_step(t: float) -> float:
        """Cubic ease in-out for smooth transitions."""
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3 - 2 * t)
    
    @staticmethod
    def _lerp_angle(a: float, b: float, t: float) -> float:
        """Linearly interpolate between two angles, handling wraparound."""
        diff = b - a
        # Wrap to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return a + t * diff
    
    # ----------------------------------------------------------------
    # Trajectory builders
    # ----------------------------------------------------------------
    
    def _build_pass_by(self):
        """
        Build PASS_BY trajectory.
        
        Human walks from spawn in a straight line that passes the robot
        with a lateral offset. The human does NOT walk into the robot.
        
        Geometry:
            spawn -----> (pass near robot with offset) ------> exit
        """
        cfg = self.config
        
        # Direction from spawn toward robot
        to_robot = cfg.robot_pos - cfg.spawn_pos
        dist_to_robot = np.linalg.norm(to_robot)
        
        if dist_to_robot < 0.01:
            # Degenerate case — just walk forward
            forward = np.array([np.cos(cfg.approach_yaw), np.sin(cfg.approach_yaw)])
        else:
            forward = to_robot / dist_to_robot
        
        # Perpendicular direction (for offset)
        perp = np.array([-forward[1], forward[0]]) * cfg.pass_by_side
        
        # The pass-by path is a straight line offset from robot
        # Start: spawn position
        # Middle: closest point (robot + offset perpendicular)
        # End: continue past robot same distance
        
        # Closest approach point
        closest_point = cfg.robot_pos + perp * cfg.pass_by_offset
        
        # End point: mirror of spawn around closest point (continue past)
        # Distance from spawn to closest = projection along forward
        approach_dist = np.dot(closest_point - cfg.spawn_pos, forward)
        if approach_dist < 0:
            approach_dist = dist_to_robot  # fallback
        
        end_point = closest_point + forward * approach_dist
        
        # Compute yaw: face direction of travel
        travel_dir = end_point - cfg.spawn_pos
        travel_yaw = np.arctan2(travel_dir[1], travel_dir[0])
        
        # Time from distances
        total_dist = np.linalg.norm(end_point - cfg.spawn_pos)
        total_time = total_dist / max(cfg.walk_speed, 0.1)
        mid_time = total_time * (approach_dist / max(total_dist, 0.01))
        
        self._waypoints = [
            TrajectoryWaypoint(
                position=cfg.spawn_pos.copy(),
                yaw=travel_yaw,
                time=0.0,
                phase="walk",
            ),
            TrajectoryWaypoint(
                position=closest_point.copy(),
                yaw=travel_yaw,
                time=mid_time,
                phase="walk",
            ),
            TrajectoryWaypoint(
                position=end_point.copy(),
                yaw=travel_yaw,
                time=total_time,
                phase="walk",
            ),
        ]
        
        self._total_duration = total_time
    
    def _build_approach_loiter_depart(self):
        """
        Build APPROACH_LOITER_DEPART trajectory.
        
        Human walks toward robot, stops at closest_approach distance,
        loiters for loiter_duration, then departs at departure_angle.
        
        Geometry:
            spawn ---approach---> (stop near robot) ---loiter--- ---depart--->
        """
        cfg = self.config
        
        # Direction from spawn toward robot
        to_robot = cfg.robot_pos - cfg.spawn_pos
        dist_to_robot = np.linalg.norm(to_robot)
        
        if dist_to_robot < 0.01:
            forward = np.array([np.cos(cfg.approach_yaw), np.sin(cfg.approach_yaw)])
        else:
            forward = to_robot / dist_to_robot
        
        # Approach yaw: face toward robot
        approach_yaw = np.arctan2(forward[1], forward[0])
        
        # Loiter position: closest_approach meters from robot, along approach line
        approach_dist = max(dist_to_robot - cfg.closest_approach, 0.5)
        loiter_pos = cfg.spawn_pos + forward * approach_dist
        
        # Approach phase timing
        approach_time = approach_dist / max(cfg.walk_speed, 0.1)
        
        # Loiter phase timing
        loiter_end_time = approach_time + cfg.loiter_duration
        
        # Departure direction
        # departure_angle is relative to approach direction (in degrees)
        # 180° = go back the way you came; 90° = turn left
        depart_angle_rad = np.deg2rad(cfg.departure_angle)
        depart_dir = np.array([
            np.cos(approach_yaw + depart_angle_rad),
            np.sin(approach_yaw + depart_angle_rad),
        ])
        
        # Departure distance: walk far enough to clear the scene
        depart_dist = dist_to_robot + 1.0  # Walk past original distance
        depart_pos = loiter_pos + depart_dir * depart_dist
        depart_yaw = np.arctan2(depart_dir[1], depart_dir[0])
        
        # Departure phase timing
        depart_time = depart_dist / max(cfg.walk_speed, 0.1)
        total_time = loiter_end_time + depart_time
        
        self._waypoints = [
            # Start at spawn
            TrajectoryWaypoint(
                position=cfg.spawn_pos.copy(),
                yaw=approach_yaw,
                time=0.0,
                phase="approach",
            ),
            # Arrive at loiter position
            TrajectoryWaypoint(
                position=loiter_pos.copy(),
                yaw=approach_yaw,
                time=approach_time,
                phase="approach",
            ),
            # Start of loiter (same position, marks phase change)
            TrajectoryWaypoint(
                position=loiter_pos.copy(),
                yaw=approach_yaw,
                time=approach_time + 0.01,
                phase="loiter",
            ),
            # End of loiter — start turning to depart
            TrajectoryWaypoint(
                position=loiter_pos.copy(),
                yaw=depart_yaw,
                time=loiter_end_time,
                phase="depart",
            ),
            # Departed
            TrajectoryWaypoint(
                position=depart_pos.copy(),
                yaw=depart_yaw,
                time=total_time,
                phase="depart",
            ),
        ]
        
        self._total_duration = total_time
    
    def _build_arc(self):
        """
        Build ARC trajectory.
        
        Human walks in an arc around the robot, staying at arc_radius distance.
        Good for incidental scenarios where the human curves past the workspace.
        
        Geometry:
            spawn ----> (arc around robot) ----> exit
        """
        cfg = self.config
        
        # Direction from robot to spawn (initial radial direction)
        from_robot = cfg.spawn_pos - cfg.robot_pos
        start_dist = np.linalg.norm(from_robot)
        
        if start_dist < 0.01:
            from_robot = np.array([1.0, 0.0])
            start_dist = 1.0
        
        # Starting angle (from robot's perspective)
        start_angle = np.arctan2(from_robot[1], from_robot[0])
        
        # Arc extent
        arc_extent_rad = np.deg2rad(cfg.arc_extent)
        end_angle = start_angle + arc_extent_rad
        
        # Generate waypoints along the arc
        num_arc_points = max(int(cfg.arc_extent / 15), 4)  # One point per ~15°
        
        # Total arc length
        arc_length = cfg.arc_radius * abs(arc_extent_rad)
        total_time = arc_length / max(cfg.walk_speed, 0.1)
        
        # First waypoint: walk from spawn to arc start
        arc_start = cfg.robot_pos + cfg.arc_radius * np.array([
            np.cos(start_angle), np.sin(start_angle)
        ])
        
        # If spawn is not on the arc, add a lead-in segment
        lead_in_dist = np.linalg.norm(arc_start - cfg.spawn_pos)
        lead_in_time = lead_in_dist / max(cfg.walk_speed, 0.1)
        
        self._waypoints = []
        
        if lead_in_dist > 0.1:
            lead_in_dir = arc_start - cfg.spawn_pos
            lead_in_yaw = np.arctan2(lead_in_dir[1], lead_in_dir[0])
            self._waypoints.append(TrajectoryWaypoint(
                position=cfg.spawn_pos.copy(),
                yaw=lead_in_yaw,
                time=0.0,
                phase="walk",
            ))
        
        # Arc waypoints
        for i in range(num_arc_points + 1):
            frac = i / num_arc_points
            angle = start_angle + frac * arc_extent_rad
            
            pos = cfg.robot_pos + cfg.arc_radius * np.array([
                np.cos(angle), np.sin(angle)
            ])
            
            # Yaw: tangent to the arc (perpendicular to radial direction)
            tangent_angle = angle + np.pi / 2  # tangent to circle
            
            t = lead_in_time + frac * total_time
            
            self._waypoints.append(TrajectoryWaypoint(
                position=pos.copy(),
                yaw=tangent_angle,
                time=t,
                phase="walk",
            ))
        
        self._total_duration = lead_in_time + total_time
    
    def get_clip_time_mapping(self, clip_duration: float, clip_fps: float) -> float:
        """
        Map trajectory time to AMASS clip frame, speed-matching
        to preserve foot plant timing.
        
        The clip is played at a rate that matches the trajectory walking speed
        to the clip's original root speed. This prevents foot sliding.
        
        Args:
            clip_duration: Duration of the AMASS clip in seconds
            clip_fps: Frame rate of the AMASS clip
            
        Returns:
            Speed multiplier to apply to clip playback
        """
        # If trajectory walk speed matches typical AMASS walk speed (~1.2 m/s),
        # play clip at 1x. If faster/slower, scale accordingly.
        # Typical AMASS walking clip root speed is ~1.0-1.4 m/s
        TYPICAL_AMASS_WALK_SPEED = 1.2  # m/s
        
        speed_ratio = self.config.walk_speed / TYPICAL_AMASS_WALK_SPEED
        return speed_ratio
    
    def closest_distance_to_robot(self) -> float:
        """
        Compute the minimum distance the trajectory gets to the robot.
        Useful for validation.
        """
        if not self._waypoints:
            return float('inf')
        
        min_dist = float('inf')
        for wp in self._waypoints:
            dist = np.linalg.norm(wp.position - self.config.robot_pos)
            min_dist = min(min_dist, dist)
        
        return min_dist
