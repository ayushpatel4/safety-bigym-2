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
    STATIONARY = auto()
    COWORKER_PATROL = auto()


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

    # --- COWORKER_PATROL parameters ---
    # Time spent near the robot per visit (reaches are active here).
    patrol_near_loiter: float = 8.0     # seconds
    # Time spent away from the robot before returning (arm stays at rest).
    patrol_away_loiter: float = 3.5     # seconds
    # How far from the robot the AWAY position sits.
    patrol_away_distance: float = 2.5   # meters
    # Number of away excursions to chain into the trajectory.
    patrol_excursions: int = 2
    # Per-visit distance jitter for the NEAR position. Each NEAR
    # (initial walk-in and every return) samples its distance as
    # ``closest_approach + N(0, near_distance_std)``, clipped to keep
    # the human in a reasonable co-worker band. Lets the human stand
    # slightly closer or further across visits within a single episode.
    patrol_near_distance_std: float = 0.12   # meters
    patrol_near_distance_clip: float = 0.25  # meters (max abs jitter)
    # Random seed driving per-excursion angle / distance variation.
    # None falls back to the planner's own internal stream so different
    # planners produce different layouts.
    patrol_seed: Optional[int] = None

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
        elif config.trajectory_type == TrajectoryType.STATIONARY:
            self._build_stationary()
        elif config.trajectory_type == TrajectoryType.COWORKER_PATROL:
            self._build_coworker_patrol()
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
    
    def _build_stationary(self):
        """
        Build STATIONARY trajectory.

        The human is parked at ``closest_approach`` distance from the
        robot, along the spawn->robot ray, facing the robot for the
        whole episode. Phase is reported as "loiter" at all times. Used
        by the COWORKER disruption "spawn in place" variant so the
        HumanController immediately blends to IK targets without an
        approach/depart phase.

        We project to ``closest_approach`` rather than parking at
        ``spawn_pos`` because the default spawn distance (~1.4 m) puts
        the human out of arm range. This matches APPROACH_LOITER_DEPART's
        loiter geometry so the coworker is actually reachable.
        """
        cfg = self.config

        to_robot = cfg.robot_pos - cfg.spawn_pos
        dist_to_robot = float(np.linalg.norm(to_robot))
        if dist_to_robot < 1e-6:
            forward = np.array([np.cos(cfg.approach_yaw), np.sin(cfg.approach_yaw)])
            face_yaw = cfg.approach_yaw
        else:
            forward = to_robot / dist_to_robot
            face_yaw = float(np.arctan2(forward[1], forward[0]))

        approach_dist = max(dist_to_robot - cfg.closest_approach, 0.0)
        loiter_pos = cfg.spawn_pos + forward * approach_dist

        # Loiter duration is honoured so downstream "is this episode still
        # in loiter?" queries work; default to a long horizon.
        loiter_end = max(cfg.loiter_duration, 1.0)

        self._waypoints = [
            TrajectoryWaypoint(
                position=loiter_pos.copy(),
                yaw=face_yaw,
                time=0.0,
                phase="loiter",
            ),
            TrajectoryWaypoint(
                position=loiter_pos.copy(),
                yaw=face_yaw,
                time=loiter_end,
                phase="loiter",
            ),
        ]
        self._total_duration = loiter_end

    def _build_coworker_patrol(self):
        """
        Build COWORKER_PATROL trajectory.

        Like APPROACH_LOITER_DEPART, the human first walks from spawn
        in to a NEAR position at ``closest_approach`` distance. Then,
        instead of staying there for the whole episode, it cycles:

            loiter at NEAR -> walk to AWAY -> loiter at AWAY ->
            walk back to NEAR (resampled angle) -> loiter at NEAR -> ...

        AWAY is at a different angle around the robot from the previous
        NEAR (90°-270° offset), so the human visibly walks off and
        comes back from a different direction. The arm controller
        suppresses reach during AWAY loiter (the shoulder-to-target
        distance check), so the arm hangs at the side rather than
        flailing toward an out-of-range target.

        Phases emitted: "approach" / "loiter" / "depart" — same labels
        the HumanController already handles. Reach is gated on
        geometric reach distance, not phase, so we don't need a custom
        "AWAY loiter" phase string.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.patrol_seed)

        to_robot = cfg.robot_pos - cfg.spawn_pos
        dist_to_robot = float(np.linalg.norm(to_robot))
        if dist_to_robot < 1e-6:
            forward = np.array([np.cos(cfg.approach_yaw), np.sin(cfg.approach_yaw)])
        else:
            forward = to_robot / dist_to_robot

        def face_from(pos: np.ndarray) -> float:
            v = cfg.robot_pos - pos
            n = np.linalg.norm(v)
            if n < 1e-6:
                return cfg.approach_yaw
            return float(np.arctan2(v[1], v[0]))

        std = max(cfg.patrol_near_distance_std, 0.0)
        clip = max(cfg.patrol_near_distance_clip, 0.0)

        def sample_near_distance() -> float:
            """Per-visit NEAR distance: mean=closest_approach,
            stdev=patrol_near_distance_std, clipped to ±clip metres."""
            if std <= 0.0:
                return float(cfg.closest_approach)
            jitter = float(rng.normal(0.0, std))
            jitter = float(np.clip(jitter, -clip, clip))
            return float(cfg.closest_approach + jitter)

        def near_at_angle(angle_rad: float, distance: float) -> np.ndarray:
            return cfg.robot_pos + distance * np.array(
                [np.cos(angle_rad), np.sin(angle_rad)]
            )

        def away_at_angle(angle_rad: float) -> np.ndarray:
            return cfg.robot_pos + cfg.patrol_away_distance * np.array(
                [np.cos(angle_rad), np.sin(angle_rad)]
            )

        # Initial NEAR: along spawn->robot ray, distance sampled around
        # closest_approach. Tracks the latest NEAR distance so the walk
        # length (used for timing the initial approach) stays correct.
        near_distance = sample_near_distance()
        approach_dist = max(dist_to_robot - near_distance, 0.0)
        near_angle = float(
            np.arctan2(cfg.spawn_pos[1] - cfg.robot_pos[1],
                       cfg.spawn_pos[0] - cfg.robot_pos[0])
        )
        near_pos = near_at_angle(near_angle, near_distance)

        # Walk speed must be sane.
        walk_v = max(cfg.walk_speed, 0.1)

        waypoints: List[TrajectoryWaypoint] = []
        t = 0.0

        # --- Initial approach from spawn to first NEAR ---
        waypoints.append(TrajectoryWaypoint(
            position=cfg.spawn_pos.copy(),
            yaw=face_from(cfg.spawn_pos),
            time=t, phase="approach",
        ))
        t += approach_dist / walk_v
        waypoints.append(TrajectoryWaypoint(
            position=near_pos.copy(),
            yaw=face_from(near_pos),
            time=t, phase="approach",
        ))

        # --- N patrol cycles (loiter NEAR -> away -> back to a new NEAR) ---
        L_near = max(cfg.patrol_near_loiter, 0.5)
        L_away = max(cfg.patrol_away_loiter, 0.5)
        n_excursions = max(int(cfg.patrol_excursions), 1)

        current_pos = near_pos
        current_angle = near_angle

        for _ in range(n_excursions):
            # Loiter at NEAR (reach cycle runs).
            waypoints.append(TrajectoryWaypoint(
                position=current_pos.copy(),
                yaw=face_from(current_pos),
                time=t + 0.01, phase="loiter",
            ))
            t += L_near
            waypoints.append(TrajectoryWaypoint(
                position=current_pos.copy(),
                yaw=face_from(current_pos),
                time=t, phase="loiter",
            ))

            # Sample AWAY angle 90°-270° offset from current NEAR angle.
            offset = float(rng.uniform(np.pi / 2, 3 * np.pi / 2))
            sign = float(rng.choice([-1.0, 1.0]))
            away_angle = current_angle + sign * offset
            away_pos = away_at_angle(away_angle)

            # Depart to AWAY (use "depart" so the HumanController's
            # outer blend tapers IK back toward AMASS as we leave the
            # loiter point — same flow as APPROACH_LOITER_DEPART exit).
            waypoints.append(TrajectoryWaypoint(
                position=current_pos.copy(),
                yaw=float(np.arctan2(
                    away_pos[1] - current_pos[1],
                    away_pos[0] - current_pos[0],
                )),
                time=t + 0.01, phase="depart",
            ))
            walk_d = float(np.linalg.norm(away_pos - current_pos))
            t += walk_d / walk_v
            waypoints.append(TrajectoryWaypoint(
                position=away_pos.copy(),
                yaw=face_from(away_pos),
                time=t, phase="depart",
            ))

            # Loiter at AWAY. Phase is "loiter" so the controller will
            # blend toward IK targets, but the arm controller's reach
            # gate will detect that the target is too far and emit a
            # rest-pose qpos — net result: human stands at AWAY with
            # arms at the side until it's time to return.
            waypoints.append(TrajectoryWaypoint(
                position=away_pos.copy(),
                yaw=face_from(away_pos),
                time=t + 0.01, phase="loiter",
            ))
            t += L_away
            waypoints.append(TrajectoryWaypoint(
                position=away_pos.copy(),
                yaw=face_from(away_pos),
                time=t, phase="loiter",
            ))

            # Pick a new NEAR angle for the return so the human comes
            # back from a different direction than where it left, and a
            # new NEAR distance sampled around closest_approach (so the
            # human stands slightly closer or further than last visit).
            angle_jitter = float(rng.uniform(-np.pi / 3, np.pi / 3))
            new_near_angle = current_angle + angle_jitter
            new_near = near_at_angle(new_near_angle, sample_near_distance())

            # Approach back to the new NEAR.
            waypoints.append(TrajectoryWaypoint(
                position=away_pos.copy(),
                yaw=float(np.arctan2(
                    new_near[1] - away_pos[1],
                    new_near[0] - away_pos[0],
                )),
                time=t + 0.01, phase="approach",
            ))
            walk_d = float(np.linalg.norm(new_near - away_pos))
            t += walk_d / walk_v
            waypoints.append(TrajectoryWaypoint(
                position=new_near.copy(),
                yaw=face_from(new_near),
                time=t, phase="approach",
            ))

            current_pos = new_near
            current_angle = new_near_angle

        # Final long loiter at the last NEAR position.
        waypoints.append(TrajectoryWaypoint(
            position=current_pos.copy(),
            yaw=face_from(current_pos),
            time=t + 0.01, phase="loiter",
        ))
        final_t = t + max(cfg.loiter_duration, 5.0)
        waypoints.append(TrajectoryWaypoint(
            position=current_pos.copy(),
            yaw=face_from(current_pos),
            time=final_t, phase="loiter",
        ))

        self._waypoints = waypoints
        self._total_duration = final_t

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
