"""
Visual Demo: Trajectory Planner Paths

Generates a 2D plot showing all three trajectory types:
- PASS_BY (walks past with offset)
- APPROACH_LOITER_DEPART (approach, pause, leave)
- ARC (curves around robot)

Usage:
    ./venv/bin/python scripts/demo_trajectory_paths.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.human.trajectory_planner import (
    TrajectoryPlanner,
    TrajectoryConfig,
    TrajectoryType,
)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def sample_trajectory(planner, n_samples=200):
    """Sample positions along a trajectory."""
    times = np.linspace(0, planner.duration, n_samples)
    positions = []
    phases = []
    for t in times:
        x, y, yaw, phase = planner.get_pose(t)
        positions.append([x, y])
        phases.append(phase)
    return np.array(positions), phases, times


def print_trajectory(name, planner):
    """Print trajectory info to console."""
    positions, phases, times = sample_trajectory(planner, n_samples=20)
    
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Duration: {planner.duration:.2f}s")
    print(f"  Waypoints: {len(planner.waypoints)}")
    print(f"  Min distance to robot: {planner.closest_distance_to_robot():.2f}m")
    print(f"  {'Time':>6s}  {'X':>7s}  {'Y':>7s}  {'Dist':>6s}  Phase")
    print(f"  {'─'*50}")
    
    prev_phase = None
    for i, t in enumerate(times):
        x, y = positions[i]
        phase = phases[i]
        dist = np.sqrt(x**2 + y**2)
        marker = " ◀" if phase != prev_phase else ""
        print(f"  {t:6.2f}s  {x:7.2f}  {y:7.2f}  {dist:6.2f}  {phase}{marker}")
        prev_phase = phase


def main():
    robot_pos = np.array([0.0, 0.0])
    spawn_pos = np.array([3.0, 0.0])
    
    # Create all three trajectory types
    configs = {
        "PASS_BY": TrajectoryConfig(
            trajectory_type=TrajectoryType.PASS_BY,
            robot_pos=robot_pos,
            spawn_pos=spawn_pos,
            approach_yaw=np.pi,
            pass_by_offset=1.0,
            pass_by_side=1,
            walk_speed=1.2,
        ),
        "APPROACH_LOITER_DEPART": TrajectoryConfig(
            trajectory_type=TrajectoryType.APPROACH_LOITER_DEPART,
            robot_pos=robot_pos,
            spawn_pos=spawn_pos,
            approach_yaw=np.pi,
            closest_approach=1.0,
            loiter_duration=2.0,
            departure_angle=150.0,
            walk_speed=1.2,
        ),
        "ARC": TrajectoryConfig(
            trajectory_type=TrajectoryType.ARC,
            robot_pos=robot_pos,
            spawn_pos=spawn_pos,
            arc_radius=1.5,
            arc_extent=120.0,
            walk_speed=1.0,
        ),
    }
    
    planners = {name: TrajectoryPlanner(cfg) for name, cfg in configs.items()}
    
    # Console output
    print("=" * 60)
    print("  TRAJECTORY PLANNER — PATH VISUALIZATION")
    print("=" * 60)
    
    for name, planner in planners.items():
        print_trajectory(name, planner)
    
    # Matplotlib plot
    if not HAS_MPL:
        print("\n⚠️  matplotlib not installed — skipping plot")
        print("   Install with: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Trajectory Planner — Human Root Paths", fontsize=14, fontweight='bold')
    
    phase_colors = {
        "walk": "#3498db",
        "approach": "#2ecc71",
        "loiter": "#e74c3c",
        "depart": "#9b59b6",
    }
    
    for ax, (name, planner) in zip(axes, planners.items()):
        positions, phases, times = sample_trajectory(planner)
        
        # Plot path segments colored by phase
        for i in range(len(positions) - 1):
            color = phase_colors.get(phases[i], "#95a5a6")
            ax.plot(
                [positions[i, 0], positions[i+1, 0]],
                [positions[i, 1], positions[i+1, 1]],
                color=color, linewidth=2
            )
        
        # Start and end markers
        ax.plot(*positions[0], 'o', color='green', markersize=10, label='Start', zorder=5)
        ax.plot(*positions[-1], 's', color='red', markersize=10, label='End', zorder=5)
        
        # Robot position
        circle = plt.Circle(robot_pos, 0.15, color='orange', fill=True, alpha=0.8, label='Robot', zorder=5)
        ax.add_patch(circle)
        
        # Safety radius (for reference)
        safety = plt.Circle(robot_pos, 1.0, color='orange', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(safety)
        
        # Waypoint markers
        for wp in planner.waypoints:
            ax.plot(*wp.position, 'x', color='gray', markersize=6, alpha=0.5)
        
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 5)
        ax.set_ylim(-3, 3)
        ax.legend(fontsize=8, loc='upper right')
    
    # Add phase legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=c, linewidth=2, label=p.capitalize())
        for p, c in phase_colors.items()
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save
    out_path = Path(__file__).parent.parent / "trajectory_paths.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {out_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
