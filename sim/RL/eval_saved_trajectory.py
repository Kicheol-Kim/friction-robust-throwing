import argparse
import ast
import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button, Slider

from mujoco_throwing_env import MujocoThrowingEnv


ACTION_COLUMNS = ("Cmd_VelX(m/s)", "Cmd_VelY(m/s)", "Cmd_VelZ(m/s)")
POSITION_COLUMNS = ("TCP_PosX(m)", "TCP_PosY(m)", "TCP_PosZ(m)")
VELOCITY_COLUMNS = ("TCP_VelX(m/s)", "TCP_VelY(m/s)", "TCP_VelZ(m/s)")
TIME_COLUMN = "Time(s)"
DEFAULT_TARGET = (1.65, 0.0, 0.5)
AXIS_COLORS = ("tab:blue", "tab:orange", "tab:green")


@dataclass(frozen=True)
class SimulationResult:
    mu: float
    times: np.ndarray
    positions: np.ndarray
    impact: np.ndarray | None
    tcp_positions: np.ndarray
    qpos: np.ndarray


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
    required_columns = (
        TIME_COLUMN,
        *POSITION_COLUMNS,
        *VELOCITY_COLUMNS,
        *ACTION_COLUMNS,
    )
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


def parse_target(
    metadata: dict[str, str], override: tuple[float, ...] | None
) -> np.ndarray:
    if override is not None:
        target = np.asarray(override, dtype=np.float64)
    elif "Target_Pos_XYZ" in metadata:
        try:
            target = np.asarray(
                ast.literal_eval(metadata["Target_Pos_XYZ"]), dtype=np.float64
            )
        except (ValueError, SyntaxError) as error:
            raise ValueError("Invalid Target_Pos_XYZ value in trajectory metadata") from error
    else:
        target = np.asarray(DEFAULT_TARGET, dtype=np.float64)

    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Target position must contain exactly three finite coordinates")
    return target


def set_equal_3d_limits(axes: plt.Axes, points: np.ndarray) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    half_span = max(float(np.max(maximum - minimum)) / 2.0, 0.1) * 1.05
    axes.set_xlim(center[0] - half_span, center[0] + half_span)
    axes.set_ylim(center[1] - half_span, center[1] + half_span)
    axes.set_zlim(center[2] - half_span, center[2] + half_span)
    axes.set_box_aspect((1, 1, 1))


def simulate_trajectory(
    env: MujocoThrowingEnv,
    actions: np.ndarray,
    friction: float,
    max_flight_seconds: float,
    seed: int,
) -> SimulationResult:
    env.reset(seed=seed)
    env.model.geom_friction[env.gripper_geom_ids, 0] = friction

    positions = [env.data.xpos[env.ball_body_id].copy()]
    times = [float(env.data.time)]
    tcp_positions = [env.data.xpos[env.tcp_body_id].copy()]
    qpos_frames = [env.data.qpos.copy()]
    impact_position: np.ndarray | None = None

    for action in actions:
        _, _, terminated, _, _ = env.step(action)
        positions.append(env.data.xpos[env.ball_body_id].copy())
        times.append(float(env.data.time))
        tcp_positions.append(env.data.xpos[env.tcp_body_id].copy())
        qpos_frames.append(env.data.qpos.copy())
        if terminated:
            impact_position = positions[-1].copy()
            break

    if impact_position is None:
        max_tail_steps = int(np.ceil(max_flight_seconds / env.dt))
        for _ in range(max_tail_steps):
            env.data.qpos[env.robot_qpos_ids] = env.joint_targets
            env.data.qvel[env.robot_dof_ids] = 0.0
            mujoco.mj_forward(env.model, env.data)
            motor_velocity = env.data.qvel[env.motor_dof_id]
            env.data.ctrl[env.motor_act_id] = np.clip(
                2.0 * (env.omega_motor - motor_velocity), -12.0, 12.0
            )
            mujoco.mj_step(env.model, env.data)
            env.data.qpos[env.robot_qpos_ids] = env.joint_targets
            env.data.qvel[env.robot_dof_ids] = 0.0
            mujoco.mj_forward(env.model, env.data)
            positions.append(env.data.xpos[env.ball_body_id].copy())
            times.append(float(env.data.time))
            tcp_positions.append(env.data.xpos[env.tcp_body_id].copy())
            qpos_frames.append(env.data.qpos.copy())
            if env._has_ground_contact():
                impact_position = positions[-1].copy()
                break

    return SimulationResult(
        mu=friction,
        times=np.asarray(times),
        positions=np.asarray(positions),
        impact=impact_position,
        tcp_positions=np.asarray(tcp_positions),
        qpos=np.asarray(qpos_frames),
    )


def build_animation_window(
    model: mujoco.MjModel, result: SimulationResult
) -> tuple[plt.Figure, animation.FuncAnimation]:
    render_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    mocap_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_tcp_mocap"
    )
    mocap_id = int(model.body_mocapid[mocap_body_id])
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    scene_points = np.concatenate((result.positions, result.tcp_positions), axis=0)
    minimum = scene_points.min(axis=0)
    maximum = scene_points.max(axis=0)
    camera.lookat[:] = (minimum + maximum) / 2.0
    camera.distance = max(float(np.max(maximum - minimum)) + 1.8, 2.5)
    camera.azimuth = 135
    camera.elevation = -20

    figure, axes = plt.subplots(figsize=(11, 7))
    figure.suptitle("MuJoCo robot and side-arm throwing replay")
    axes.axis("off")
    figure.subplots_adjust(bottom=0.18, top=0.88)

    render_data.qpos[:] = result.qpos[0]
    mujoco.mj_forward(model, render_data)
    renderer.update_scene(render_data, camera=camera)
    image = axes.imshow(renderer.render())
    frame_count = len(result.qpos)
    slider_axes = figure.add_axes((0.10, 0.07, 0.56, 0.04))
    frame_slider = Slider(
        slider_axes,
        "Frame",
        valmin=0,
        valmax=max(frame_count - 1, 1),
        valinit=0,
        valstep=1,
    )
    play_axes = figure.add_axes((0.72, 0.055, 0.11, 0.07))
    replay_axes = figure.add_axes((0.85, 0.055, 0.11, 0.07))
    play_button = Button(play_axes, "Pause")
    replay_button = Button(replay_axes, "Replay")
    state = {"frame": 0, "playing": True, "closed": False}

    def render_frame(frame_index: int, update_slider: bool = True):
        index = min(max(int(frame_index), 0), frame_count - 1)
        render_data.qpos[:] = result.qpos[index]
        render_data.mocap_pos[mocap_id] = result.tcp_positions[index]
        render_data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, render_data)
        renderer.update_scene(render_data, camera=camera)
        image.set_data(renderer.render())
        axes.set_title(
            f"t={result.times[index]:.2f} s | friction μ={result.mu:.2f}"
        )
        state["frame"] = index
        if update_slider:
            frame_slider.eventson = False
            frame_slider.set_val(index)
            frame_slider.eventson = True
        figure.canvas.draw_idle()
        return (image,)

    def on_slider_change(value: float):
        if state["playing"]:
            playback.event_source.stop()
            state["playing"] = False
            play_button.label.set_text("Play")
        render_frame(round(value), update_slider=False)

    playback = animation.FuncAnimation(
        figure,
        render_frame,
        frames=range(frame_count),
        interval=20,
        blit=False,
        repeat=False,
    )
    frame_slider.on_changed(on_slider_change)

    def on_play_pause(_event):
        state["playing"] = not state["playing"]
        if state["playing"]:
            playback.event_source.start()
            play_button.label.set_text("Pause")
        else:
            playback.event_source.stop()
            play_button.label.set_text("Play")

    def on_replay(_event):
        state["playing"] = True
        playback.frame_seq = playback.new_frame_seq()
        frame_slider.set_val(0)
        play_button.label.set_text("Pause")
        playback.event_source.start()

    def on_close(_event):
        if not state["closed"]:
            renderer.close()
            state["closed"] = True

    play_button.on_clicked(on_play_pause)
    replay_button.on_clicked(on_replay)
    figure.canvas.mpl_connect("close_event", on_close)
    render_frame(0)
    figure._mujoco_renderer = renderer
    figure._mujoco_render_data = render_data
    figure._trajectory_animation = playback
    figure._animation_controls = (frame_slider, play_button, replay_button)
    return figure, playback


def build_analysis_plot(
    frame: pd.DataFrame,
    target: np.ndarray,
    simulations: list[SimulationResult],
    trajectory_name: str,
    output_path: Path,
) -> plt.Figure:
    figure = plt.figure(figsize=(20, 7.5))
    figure.suptitle(
        f"Saved TCP trajectory and MuJoCo throw analysis\n{trajectory_name}",
        fontsize=14,
    )
    grid = figure.add_gridspec(1, 3, width_ratios=(1.15, 1.25, 1.15))
    tcp_axes = figure.add_subplot(grid[0, 0])
    position_axes = tcp_axes.twinx()
    flight_axes = figure.add_subplot(grid[0, 1], projection="3d")
    landing_axes = figure.add_subplot(grid[0, 2])

    time_values = frame[TIME_COLUMN].to_numpy()
    handles = []
    for velocity_column, position_column, label, color in zip(
        VELOCITY_COLUMNS, POSITION_COLUMNS, ("X", "Y", "Z"), AXIS_COLORS
    ):
        velocity_line, = tcp_axes.plot(
            time_values,
            frame[velocity_column],
            color=color,
            linestyle="-",
            linewidth=1.6,
            label=f"{label} velocity",
        )
        position_line, = position_axes.plot(
            time_values,
            frame[position_column],
            color=color,
            linestyle="--",
            linewidth=1.6,
            label=f"{label} position",
        )
        handles.extend((velocity_line, position_line))
    tcp_axes.set_title("Recorded TCP trajectory")
    tcp_axes.set_xlabel("Time (s)")
    tcp_axes.set_ylabel("Velocity (m/s)")
    position_axes.set_ylabel("Position from home (m)")
    tcp_axes.grid(True, linestyle="--", alpha=0.5)
    tcp_axes.legend(handles=handles, loc="best", fontsize=8, ncol=2)

    mu_values = np.asarray([result.mu for result in simulations])
    color_map = plt.colormaps["viridis"]
    color_min = float(mu_values.min())
    color_max = float(mu_values.max())
    if color_min == color_max:
        color_max = color_min + 1e-9
    color_norm = plt.Normalize(vmin=color_min, vmax=color_max)
    flight_points = []
    for result in simulations:
        positions = result.positions
        flight_axes.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color_map(color_norm(result.mu)),
            linewidth=1.0,
            alpha=0.55,
        )
        flight_points.append(positions)
    flight_axes.scatter(
        target[0], target[1], 0.0, marker="x", color="red", s=60, label="Target XY"
    )
    flight_axes.scatter(
        target[0],
        target[1],
        target[2],
        marker="*",
        color="red",
        s=85,
        label="Target reference",
    )
    flight_axes.set_title("Projectile flight by friction coefficient")
    flight_axes.set_xlabel("X (m)")
    flight_axes.set_ylabel("Y (m)")
    flight_axes.set_zlabel("Z (m)")
    flight_axes.set_proj_type("ortho")
    flight_axes.view_init(elev=22, azim=-58)
    set_equal_3d_limits(
        flight_axes,
        np.concatenate((*flight_points, target[None, :], np.array([[target[0], target[1], 0.0]]))),
    )
    flight_axes.legend(loc="upper left", fontsize=7)

    landed = [result for result in simulations if result.impact is not None]
    impact_positions = (
        np.asarray([result.impact[:2] for result in landed])
        if landed
        else np.empty((0, 2))
    )
    mean_impact = impact_positions.mean(axis=0) if len(impact_positions) else None
    covariance = (
        np.cov(impact_positions, rowvar=False, ddof=0)
        if len(impact_positions) > 1
        else np.zeros((2, 2))
    )
    landing_axes.scatter(
        target[0],
        target[1],
        marker="x",
        color="red",
        s=90,
        label="Target on ground",
    )
    if landed:
        landing_axes.scatter(
            impact_positions[:, 0],
            impact_positions[:, 1],
            c=[result.mu for result in landed],
            cmap=color_map,
            norm=color_norm,
            edgecolor="black",
            linewidth=0.5,
            s=45,
            label="Projectile impacts",
        )
        landing_axes.scatter(
            mean_impact[0],
            mean_impact[1],
            marker="*",
            color="black",
            edgecolor="white",
            linewidth=0.8,
            s=180,
            label="Mean impact",
            zorder=5,
        )
        if len(landed) > 1 and np.any(covariance):
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            eigenvectors = eigenvectors[:, order]
            angle = np.degrees(
                np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            )
            landing_axes.add_patch(
                Ellipse(
                    mean_impact,
                    width=2.0 * np.sqrt(eigenvalues[0]),
                    height=2.0 * np.sqrt(eigenvalues[1]),
                    angle=angle,
                    fill=False,
                    edgecolor="black",
                    linestyle=":",
                    linewidth=1.3,
                    label="1-sigma covariance",
                )
            )

        errors = mean_impact - target[:2]
        distance_error = float(np.linalg.norm(errors))
        stats = (
            f"Impacts: {len(landed)}/{len(simulations)}\n"
            f"Mean: ({mean_impact[0]:.3f}, {mean_impact[1]:.3f}) m\n"
            f"Variance: X={covariance[0, 0]:.2e}, "
            f"Y={covariance[1, 1]:.2e} m²\n"
            f"Target error: X={errors[0] * 1000:+.1f}, "
            f"Y={errors[1] * 1000:+.1f} mm\n"
            f"Distance error: {distance_error * 1000:.1f} mm"
        )
        landing_axes.text(
            0.03,
            0.97,
            stats,
            transform=landing_axes.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
    else:
        landing_axes.text(
            0.03,
            0.97,
            f"No impacts detected ({len(simulations)} trials)",
            transform=landing_axes.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
    landing_axes.set_xlabel("Impact X (m)")
    landing_axes.set_ylabel("Impact Y (m)")
    landing_axes.set_title("Ground impact distribution (XY)")
    landing_axes.set_aspect("equal", adjustable="datalim")
    landing_axes.grid(True, linestyle="--", alpha=0.5)
    landing_axes.legend(loc="lower right", fontsize=8)

    figure.subplots_adjust(left=0.07, right=0.96, bottom=0.08, top=0.88, wspace=0.38)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return figure


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
    model = env.model
    actions = frame.loc[:, ACTION_COLUMNS].to_numpy(dtype=np.float64)
    simulations = [
        simulate_trajectory(
            env=env,
            actions=actions,
            friction=float(mu),
            max_flight_seconds=max_flight_seconds,
            seed=0,
        )
        for mu in np.linspace(mu_min, mu_max, mu_count)
    ]
    try:
        reference_mu = float(metadata.get("Friction_Coefficient", 0.3))
    except ValueError as error:
        raise ValueError("Invalid Friction_Coefficient value in trajectory metadata") from error
    animated_result = min(simulations, key=lambda result: abs(result.mu - reference_mu))
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

    animation_figure = None
    if show:
        animation_figure, _ = build_animation_window(model, animated_result)
    analysis_figure = build_analysis_plot(
        frame=frame,
        target=target,
        simulations=simulations,
        trajectory_name=path.name,
        output_path=result_plot,
    )
    if show:
        plt.show()
    else:
        plt.close(analysis_figure)
        if animation_figure is not None:
            plt.close(animation_figure)

    landed_results = results.loc[results["landed"]]
    print(f"Trajectory: {path}")
    print(f"Target XY: ({target[0]:.3f}, {target[1]:.3f}) m")
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
        help="Save the plot and impact CSV without opening the GUI windows.",
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
