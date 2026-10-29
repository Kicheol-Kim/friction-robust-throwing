import argparse
import ast
import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mujoco

from mujoco_throwing_env import MujocoThrowingEnv


ACTION_COLUMNS = ("Cmd_VelX(m/s)", "Cmd_VelY(m/s)", "Cmd_VelZ(m/s)")
TIME_COLUMN = "Time(s)"
DEFAULT_TARGET = (1.65, 0.0, 0.5)


@dataclass(frozen=True)
class SimulationResult:
    mu: float
    times: np.ndarray
    positions: np.ndarray
    impact: np.ndarray | None


def get_latest_trajectory_file() -> Path:
    trajectory_dir = Path(__file__).resolve().parents[1] / "data" / "rl_trajectories"
    csv_files = glob.glob(str(trajectory_dir / "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No trajectory CSV files found in {trajectory_dir}")
    return Path(max(csv_files, key=os.path.getmtime))


def load_trajectory(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))

    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row and row[0].strip() == TIME_COLUMN
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"Could not find the {TIME_COLUMN!r} CSV header in {path}")

    metadata: dict[str, str] = {}
    for row in rows[:header_index]:
        if not row:
            continue
        text = ",".join(row).strip().lstrip("#").strip()
        if ":" in text:
            key, value = text.split(":", maxsplit=1)
            metadata[key.strip()] = value.strip()

    frame = pd.read_csv(path, skiprows=header_index)
    required_columns = (TIME_COLUMN, *ACTION_COLUMNS)
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"Trajectory CSV is missing columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError(f"Trajectory CSV contains no action samples: {path}")

    frame = frame.loc[:, required_columns].apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Trajectory CSV contains non-finite times or actions: {path}")
    if np.any(np.diff(frame[TIME_COLUMN].to_numpy()) <= 0):
        raise ValueError(f"Trajectory timestamps are not strictly increasing: {path}")

    return frame, metadata


def parse_target(metadata: dict[str, str], override: tuple[float, ...] | None) -> np.ndarray:
    if override is not None:
        target = np.asarray(override, dtype=np.float64)
    elif "Target_Pos_XYZ" in metadata:
        try:
            target = np.asarray(ast.literal_eval(metadata["Target_Pos_XYZ"]), dtype=np.float64)
        except (ValueError, SyntaxError) as error:
            raise ValueError("Invalid Target_Pos_XYZ value in trajectory metadata") from error
    else:
        target = np.asarray(DEFAULT_TARGET, dtype=np.float64)

    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Target position must contain exactly three finite coordinates")
    return target


def has_ground_contact(env: MujocoThrowingEnv, ground_geom_id: int, projectile_geom_id: int) -> bool:
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        if {contact.geom1, contact.geom2} == {ground_geom_id, projectile_geom_id}:
            return True
    return False


def simulate_trajectory(
    env: MujocoThrowingEnv,
    actions: np.ndarray,
    friction: float,
    max_flight_seconds: float,
    ground_geom_id: int,
    projectile_geom_id: int,
    seed: int,
) -> SimulationResult:
    env.reset(seed=seed)
    for geom_id in env.gripper_geom_ids:
        env.model.geom_friction[geom_id, 0] = friction

    positions = [env.data.xpos[env.ball_body_id].copy()]
    times = [float(env.data.time)]
    impact_position: np.ndarray | None = None

    for action in actions:
        env.step(action)
        positions.append(env.data.xpos[env.ball_body_id].copy())
        times.append(float(env.data.time))
        if has_ground_contact(env, ground_geom_id, projectile_geom_id):
            impact_position = positions[-1].copy()
            break

    if impact_position is None:
        max_tail_steps = int(np.ceil(max_flight_seconds / env.dt))
        motor_dof = env.model.jnt_dofadr[env.motor_joint_id]
        for _ in range(max_tail_steps):
            motor_velocity = env.data.qvel[motor_dof]
            env.data.ctrl[env.motor_act_id] = np.clip(
                2.0 * (env.omega_motor - motor_velocity), -12.0, 12.0
            )
            mujoco.mj_step(env.model, env.data)
            positions.append(env.data.xpos[env.ball_body_id].copy())
            times.append(float(env.data.time))
            if has_ground_contact(env, ground_geom_id, projectile_geom_id):
                impact_position = positions[-1].copy()
                break

    return SimulationResult(
        mu=friction,
        times=np.asarray(times),
        positions=np.asarray(positions),
        impact=impact_position,
    )


def build_analysis_plot(
    frame: pd.DataFrame,
    target: np.ndarray,
    simulations: list[SimulationResult],
    trajectory_name: str,
    output_path: Path,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"MuJoCo replay and ground-impact analysis\n{trajectory_name}", fontsize=14)
    command_axes = fig.add_subplot(2, 2, 1)
    flight_axes = fig.add_subplot(2, 2, 2, projection="3d")
    landing_axes = fig.add_subplot(2, 2, 3)
    error_axes = fig.add_subplot(2, 2, 4)

    time_values = frame[TIME_COLUMN].to_numpy()
    for column, label in zip(ACTION_COLUMNS, ("X", "Y", "Z")):
        command_axes.plot(time_values, frame[column], label=f"Command {label}")
    command_axes.set_title("Recorded TCP velocity commands")
    command_axes.set_xlabel("Time (s)")
    command_axes.set_ylabel("Velocity (m/s)")
    command_axes.grid(True, linestyle="--", alpha=0.5)
    command_axes.legend()

    mu_values = np.asarray([result.mu for result in simulations])
    color_map = plt.colormaps["viridis"]
    color_min = float(mu_values.min())
    color_max = float(mu_values.max())
    if color_min == color_max:
        color_max = color_min + 1e-9
    color_norm = plt.Normalize(vmin=color_min, vmax=color_max)
    markers = []

    all_positions = np.concatenate(
        [result.positions for result in simulations], axis=0
    )
    for result in simulations:
        mu = result.mu
        color = color_map(color_norm(mu))
        positions = result.positions
        flight_axes.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            linewidth=1.4,
            alpha=0.8,
        )
        marker, = flight_axes.plot(
            [positions[0, 0]],
            [positions[0, 1]],
            [positions[0, 2]],
            marker="o",
            color=color,
            markersize=5,
            linestyle="None",
        )
        markers.append(marker)

        landing = result.impact
        if landing is not None:
            landing_axes.plot(
                positions[:, 0],
                positions[:, 1],
                color=color,
                linewidth=0.8,
                alpha=0.3,
            )
            landing_axes.scatter(landing[0], landing[1], color=color, edgecolor="black")

    flight_axes.scatter(
        target[0], target[1], 0.0, marker="x", color="red", s=70, label="Target on ground"
    )
    flight_axes.scatter(
        target[0], target[1], target[2], marker="*", color="red", s=90, label="3D target"
    )
    flight_axes.set_title("Simulated projectile trajectories")
    flight_axes.set_xlabel("X (m)")
    flight_axes.set_ylabel("Y (m)")
    flight_axes.set_zlabel("Z (m)")
    x_padding = max(0.05, 0.05 * float(np.ptp(all_positions[:, 0])))
    y_extent = max(0.25, float(np.max(np.abs(all_positions[:, 1]))) * 1.1)
    z_top = max(float(np.max(all_positions[:, 2])), float(target[2])) + 0.05
    flight_axes.set_xlim(
        min(float(np.min(all_positions[:, 0])), float(target[0])) - x_padding,
        max(float(np.max(all_positions[:, 0])), float(target[0])) + x_padding,
    )
    flight_axes.set_ylim(-y_extent, y_extent)
    flight_axes.set_zlim(0.0, z_top)
    flight_axes.legend(loc="upper left", fontsize=8)
    flight_axes.set_box_aspect(
        (
            max(float(np.ptp(all_positions[:, 0])), 0.25),
            max(2.0 * y_extent, 0.25),
            max(z_top, 0.25),
        )
    )

    landing_axes.scatter(
        target[0], target[1], marker="x", color="red", s=90, label="Target on ground"
    )
    landing_axes.set_title("Ground impact locations (top view)")
    landing_axes.set_xlabel("Impact X (m)")
    landing_axes.set_ylabel("Impact Y (m)")
    landing_axes.axis("equal")
    landing_axes.grid(True, linestyle="--", alpha=0.5)
    landing_axes.legend()
    colorbar_axes = fig.add_axes((0.92, 0.57, 0.018, 0.28))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        cax=colorbar_axes,
        label="Gripper friction coefficient",
    )

    landed = [result for result in simulations if result.impact is not None]
    errors = []
    for result in landed:
        if result.impact is not None:
            errors.append(float(np.linalg.norm(result.impact[:2] - target[:2])))
    error_axes.plot(
        [result.mu for result in landed],
        np.asarray(errors) * 1000.0,
        marker="o",
    )
    error_axes.axhline(0.0, color="red", linestyle="--", label="Target")
    error_axes.set_title("Horizontal impact error by friction")
    error_axes.set_xlabel("Gripper friction coefficient")
    error_axes.set_ylabel("XY distance from target projection (mm)")
    error_axes.grid(True, linestyle="--", alpha=0.5)

    fig.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.08,
        top=0.9,
        wspace=0.3,
        hspace=0.3,
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")

    if show:
        max_frames = max(len(result.positions) for result in simulations)

        def animate(frame_index: int) -> list[object]:
            for marker, result in zip(markers, simulations):
                positions = result.positions
                current = positions[min(frame_index, len(positions) - 1)]
                marker.set_data([current[0]], [current[1]])
                marker.set_3d_properties([current[2]])
            return markers

        playback = animation.FuncAnimation(
            fig,
            animate,
            frames=max_frames,
            interval=20,
            blit=False,
            repeat=False,
        )
        setattr(fig, "_trajectory_animation", playback)
        plt.show()
    else:
        plt.close(fig)


def evaluate_and_plot(
    trajectory_path: Path | None = None,
    target_override: tuple[float, float, float] | None = None,
    mu_min: float = 0.1,
    mu_max: float = 0.5,
    mu_count: int = 21,
    max_flight_seconds: float = 5.0,
    output_dir: Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    path = trajectory_path or get_latest_trajectory_file()
    frame, metadata = load_trajectory(path)
    target = parse_target(metadata, target_override)
    if not np.isfinite((mu_min, mu_max, max_flight_seconds)).all():
        raise ValueError("Friction bounds and maximum flight time must be finite")
    if mu_count < 1 or mu_min < 0 or mu_max < mu_min or max_flight_seconds <= 0:
        raise ValueError("Invalid friction sweep or maximum flight time")

    try:
        omega_motor = float(metadata.get("Motor_Omega_Rads", 15.0))
    except ValueError as error:
        raise ValueError("Invalid Motor_Omega_Rads value in trajectory metadata") from error

    env = MujocoThrowingEnv(
        mu_range=(mu_min, mu_max),
        target_pos=tuple(target),
        omega_motor=omega_motor,
    )
    ground_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    projectile_geom_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, "proj_geom"
    )
    if ground_geom_id < 0 or projectile_geom_id < 0:
        raise ValueError("MuJoCo scene must define geoms named 'ground' and 'proj_geom'")

    actions = frame.loc[:, ACTION_COLUMNS].to_numpy(dtype=np.float64)
    simulations = [
        simulate_trajectory(
            env=env,
            actions=actions,
            friction=float(mu),
            max_flight_seconds=max_flight_seconds,
            ground_geom_id=ground_geom_id,
            projectile_geom_id=projectile_geom_id,
            seed=0,
        )
        for mu in np.linspace(mu_min, mu_max, mu_count)
    ]
    env.close()

    rows = []
    for result in simulations:
        impact = result.impact
        landed = impact is not None
        if impact is None:
            impact_time = impact_x = impact_y = impact_z = xy_error = np.nan
        else:
            impact_time = float(result.times[-1])
            impact_x, impact_y, impact_z = map(float, impact)
            xy_error = float(np.linalg.norm(impact[:2] - target[:2]))
        rows.append(
            {
                "friction": result.mu,
                "landed": landed,
                "impact_time_s": impact_time,
                "impact_x_m": impact_x,
                "impact_y_m": impact_y,
                "impact_z_m": impact_z,
                "target_xy_error_m": xy_error,
            }
        )
    results = pd.DataFrame(rows)

    result_dir = output_dir or path.parent / "analysis"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_csv = result_dir / f"{path.stem}_impacts.csv"
    result_plot = result_dir / f"{path.stem}_analysis.png"
    results.to_csv(result_csv, index=False, float_format="%.6f")
    build_analysis_plot(
        frame=frame,
        target=target,
        simulations=simulations,
        trajectory_name=path.name,
        output_path=result_plot,
        show=show,
    )

    landed_results = results.loc[results["landed"]]
    print(f"Trajectory: {path}")
    print(f"Target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) m")
    print(f"Replayed command samples: {len(frame)}")
    print(f"Ground impacts: {len(landed_results)}/{len(results)}")
    if not landed_results.empty:
        errors_mm = landed_results["target_xy_error_m"] * 1000.0
        print(
            "XY impact error: mean={:.1f} mm, std={:.1f} mm, max={:.1f} mm".format(
                errors_mm.mean(), errors_mm.std(ddof=0), errors_mm.max()
            )
        )
        print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    else:
        print("No ground impacts were detected within the configured flight time.")
    print(f"Impact data: {result_csv}")
    print(f"Analysis plot: {result_plot}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a saved 3D TCP trajectory in MuJoCo and analyze projectile ground impacts."
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Trajectory CSV to replay (defaults to the most recently modified CSV).",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the target position stored in the CSV metadata.",
    )
    parser.add_argument("--mu-min", type=float, default=0.1)
    parser.add_argument("--mu-max", type=float, default=0.5)
    parser.add_argument("--mu-count", type=int, default=21)
    parser.add_argument("--max-flight-seconds", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot and impact CSV without opening an animation window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    evaluate_and_plot(
        trajectory_path=arguments.trajectory,
        target_override=tuple(arguments.target) if arguments.target else None,
        mu_min=arguments.mu_min,
        mu_max=arguments.mu_max,
        mu_count=arguments.mu_count,
        max_flight_seconds=arguments.max_flight_seconds,
        output_dir=arguments.output_dir,
        show=not arguments.no_show,
    )
