"""Testing and benchmark evaluation for constrained finite-eMBB completion-time CTD4."""
from __future__ import annotations

import os
import time
import numpy as np
import torch

from resource_algorithms import serve_one_slot, estimate_embb_completion_potential
from uav_env import _sliced_task_reward


def _windows_extended_path(path):
    """Compatibility helper: keep a normal filesystem path.

    Output directories are intentionally compact, so no Windows extended-path
    prefix is needed.  A malformed Windows extended-path prefix caused WinError 123
    in the previous revision.
    """
    return os.fspath(path)

def _safe_makedirs(path):
    os.makedirs(_windows_extended_path(path), exist_ok=True)


def _safe_savefig(plt_module, file_path, dpi):
    plt_module.savefig(_windows_extended_path(file_path), dpi=dpi)


def _safe_savefig_with_outside_legend(plt_module, file_path, dpi):
    """Save a figure containing an outside legend without clipping it."""
    plt_module.savefig(
        _windows_extended_path(file_path),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
    )





def _benchmark_fs_tag(name):
    """Compact filesystem-only tag; display/algorithm names are unchanged."""
    return {
        "SafeNearestUnfinished": "sn",
        "WorkloadDistance": "wd",
        "MakespanAwareOneStepGreedy": "g",
    }.get(str(name), str(name))

def safe_mean(x):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else 0.0


def safe_std(x):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else 0.0


def moving_average(x, beta=0.9):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = beta * out[i - 1] + (1.0 - beta) * arr[i]
    return out


def _load_model(agent, load_model_path):
    if load_model_path is None:
        return
    if os.path.isfile(load_model_path):
        agent.load_actor(load_model_path)
        print(f"已加载Actor：{load_model_path}")
        return
    checkpoint = os.path.join(load_model_path, getattr(agent, "checkpoint_filename", "ctd4_checkpoint.pt"))
    actor = os.path.join(load_model_path, getattr(agent, "actor_filename", "ctd4_actor.pt"))
    if os.path.exists(checkpoint):
        agent.load(load_model_path)
        print(f"已加载完整模型：{load_model_path}")
    elif os.path.exists(actor):
        agent.load_actor(actor)
        print(f"已加载Actor：{actor}")
    else:
        raise FileNotFoundError(f"模型目录中未找到checkpoint或actor：{load_model_path}")


def _apply_trajectory_plot_style():
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    mpl.rcParams["axes.unicode_minus"] = False


def _collect_nfz_polygons(cfg, nfz_polygons_padded=None, nfz_vertex_count=None):
    polygons = []
    if nfz_polygons_padded is not None and nfz_vertex_count is not None:
        padded = np.asarray(nfz_polygons_padded, dtype=np.float32)
        counts = np.asarray(nfz_vertex_count, dtype=np.int32).reshape(-1)
        for j in range(min(padded.shape[0], counts.size)):
            n = int(counts[j])
            if n >= 3:
                polygons.append(np.asarray(padded[j, :n], dtype=np.float32))
    else:
        polygons = [np.asarray(poly, dtype=np.float32) for poly in getattr(cfg, "nfz_polygons", [])]
    return polygons


def _algorithm_display_name(name):
    mapping = {
        "SafeNearestUnfinished": "NU-Target",
        "WorkloadDistance": "WD-Target",
        "MakespanAwareOneStepGreedy": "OS-Greedy",
        "SGMH-CTD4": "SGMH-CTD4",
        "CTD4": "SGMH-CTD4",
    }
    return mapping.get(str(name), str(name))


def _lighten_color(color, amount):
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(color), dtype=np.float64)
    amount = float(np.clip(amount, 0.0, 1.0))
    out = rgb + (1.0 - rgb) * amount
    return tuple(np.clip(out, 0.0, 1.0))


def _base_uav_colors():
    return ["#b22222", "#1f4e79", "#2e8b57"]


def _trajectory_legend_layout(mode="single", algo_n=1):
    if mode == "single":
        return {
            "fontsize": 15.6,
            "loc": "upper left",
            "bbox_to_anchor": (1.01, 1.00),
            "ncol": 1,
            "rect": [0.0, 0.0, 0.73, 1.0],
            "columnspacing": 1.00,
            "handlelength": 2.20,
            "labelspacing": 1.02,
            "borderpad": 0.86,
        }
    if int(algo_n) <= 2:
        return {
            "fontsize": 14.8,
            "loc": "upper left",
            "bbox_to_anchor": (1.01, 1.00),
            "ncol": 1,
            "rect": [0.0, 0.0, 0.71, 1.0],
            "columnspacing": 1.00,
            "handlelength": 2.24,
            "labelspacing": 0.98,
            "borderpad": 0.82,
        }
    return {
        "fontsize": 14.2,
        "loc": "upper left",
        "bbox_to_anchor": (1.01, 1.00),
        "ncol": 1,
        "rect": [0.0, 0.0, 0.69, 1.0],
        "columnspacing": 0.96,
        "handlelength": 2.16,
        "labelspacing": 0.94,
        "borderpad": 0.80,
    }


def plot_uav_trajectory(
    uav_pos,
    user_pos,
    cfg,
    path,
    save_name,
    title,
    user_pos_list=None,
    residual_energy_list=None,
    active_mask_list=None,
    associated_uav_list=None,
    completion_slot=None,
    remaining_data_bits=None,
    annotate_every=20,
    nfz_polygons_padded=None,
    nfz_vertex_count=None,
):
    try:
        import matplotlib.pyplot as plt

        _apply_trajectory_plot_style()
        _safe_makedirs(path)
        uav_pos = np.asarray(uav_pos, dtype=np.float32)
        user_pos = np.asarray(user_pos, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(14, 10))

        polygons = _collect_nfz_polygons(
            cfg,
            nfz_polygons_padded=nfz_polygons_padded,
            nfz_vertex_count=nfz_vertex_count,
        )
        nfz_face = "#d9d9d9"
        nfz_edge = "#4d4d4d"
        for j, poly in enumerate(polygons):
            closed = np.vstack([poly, poly[0]])
            ax.fill(
                closed[:, 0], closed[:, 1],
                facecolor=nfz_face, edgecolor=nfz_edge,
                linewidth=1.8, alpha=0.32,
                label="NFZ" if j == 0 else None,
                zorder=1,
            )
            ax.plot(closed[:, 0], closed[:, 1], color=nfz_edge, linewidth=1.8, zorder=2)

        user_slice = np.asarray(getattr(cfg, "user_slice", np.zeros(user_pos.shape[0])), dtype=np.int32)
        embb_mask = user_slice == int(getattr(cfg, "slice_embb", 0))
        urllc_mask = user_slice == int(getattr(cfg, "slice_urllc", 1))
        mmtc_mask = user_slice == int(getattr(cfg, "slice_mmtc", 2))

        user_styles = [
            (embb_mask, dict(label="eMBB users", marker="o", s=68, facecolors="#d62728", edgecolors="black", linewidths=0.9, alpha=0.95)),
            (urllc_mask, dict(label="URLLC users", marker="^", s=78, facecolors="#1f77b4", edgecolors="black", linewidths=0.9, alpha=0.95)),
            (mmtc_mask, dict(label="mMTC users", marker="D", s=66, facecolors="#2ca02c", edgecolors="black", linewidths=0.9, alpha=0.95)),
        ]
        for mask, style in user_styles:
            if np.any(mask):
                ax.scatter(user_pos[mask, 0], user_pos[mask, 1], zorder=4, **style)

        uav_colors = _base_uav_colors()
        start_marker = "s"
        end_marker = "X"
        step_gap = max(int(annotate_every), 1)
        show_annotations = bool(getattr(cfg, "trajectory_show_step_index", False))

        for m in range(uav_pos.shape[1]):
            traj = uav_pos[:, m, :]
            color = uav_colors[m % len(uav_colors)]
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=color, linewidth=2.8,
                label=f"UAV{m + 1} trajectory",
                zorder=6,
            )
            ax.scatter(
                traj[0, 0], traj[0, 1],
                marker=start_marker, s=140,
                facecolors="white", edgecolors=color, linewidths=2.0,
                label=f"UAV{m + 1} start",
                zorder=8,
            )
            ax.scatter(
                traj[-1, 0], traj[-1, 1],
                marker=end_marker, s=180,
                facecolors=color, edgecolors="black", linewidths=0.9,
                label=f"UAV{m + 1} end",
                zorder=9,
            )
            if show_annotations:
                for t in range(0, traj.shape[0], step_gap):
                    ax.text(
                        traj[t, 0], traj[t, 1], str(t),
                        fontsize=9, color=color,
                        ha="center", va="center",
                        zorder=10,
                    )

        ax.set_xlim(0, cfg.area_size)
        ax.set_ylim(0, cfg.area_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)", fontsize=26)
        ax.set_ylabel("y (m)", fontsize=26)
        ax.tick_params(axis="both", labelsize=21)
        ax.grid(True, alpha=0.28, linewidth=0.8)
        ax.set_axisbelow(True)

        handles, labels = ax.get_legend_handles_labels()
        uniq_handles, uniq_labels = [], []
        for h, l in zip(handles, labels):
            if l and l not in uniq_labels:
                uniq_handles.append(h)
                uniq_labels.append(l)
        legend_cfg = _trajectory_legend_layout(mode="single")
        ax.legend(
            uniq_handles,
            uniq_labels,
            fontsize=legend_cfg["fontsize"],
            loc=legend_cfg["loc"],
            bbox_to_anchor=legend_cfg["bbox_to_anchor"],
            ncol=legend_cfg["ncol"],
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="black",
            columnspacing=legend_cfg["columnspacing"],
            handlelength=legend_cfg["handlelength"],
            borderpad=legend_cfg["borderpad"],
            labelspacing=legend_cfg["labelspacing"],
        )
        fig.tight_layout(rect=legend_cfg["rect"])
        _safe_savefig_with_outside_legend(plt, os.path.join(path, save_name), dpi=320)
        plt.close(fig)
    except Exception as exc:
        print(f"轨迹图保存失败 {save_name}: {exc}")


def _save_trajectory_record(data_dir, episode_index, env, ep_metric):
    """Save one episode's geometry and trajectory for later overlay plots."""
    _safe_makedirs(data_dir)
    np.savez_compressed(
        _windows_extended_path(os.path.join(data_dir, f"episode_{int(episode_index):04d}.npz")),
        uav_pos=np.asarray(env.uav_pos_list, dtype=np.float32),
        user_pos=np.asarray(env.user_pos, dtype=np.float32),
        user_pos_list=np.asarray(env.user_pos_list, dtype=np.float32),
        completion_slot=np.asarray(env.completion_slot, dtype=np.float32),
        remaining_data_bits=np.asarray(env.remaining_data_bits, dtype=np.float32),
        nfz_polygons_padded=np.asarray(
            ep_metric["ep_nfz_polygons_padded"], dtype=np.float32
        ),
        nfz_vertex_count=np.asarray(
            ep_metric["ep_nfz_vertex_count"], dtype=np.int32
        ),
        slot_count=np.asarray([env.slot_count], dtype=np.int32),
    )


def _load_trajectory_record(data_dir, episode_index):
    file_path = os.path.join(data_dir, f"episode_{int(episode_index):04d}.npz")
    fs_path = _windows_extended_path(file_path)
    if not os.path.exists(fs_path):
        return None
    with np.load(fs_path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def plot_multi_algorithm_trajectory(
    trajectory_records,
    cfg,
    path,
    save_name,
    title,
):
    """Overlay several algorithms' multi-UAV trajectories in one figure."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        if not trajectory_records:
            return
        _apply_trajectory_plot_style()
        _safe_makedirs(path)

        first = next(iter(trajectory_records.values()))
        user_pos = np.asarray(first["user_pos"], dtype=np.float32)
        polygons = _collect_nfz_polygons(
            cfg,
            nfz_polygons_padded=first.get("nfz_polygons_padded"),
            nfz_vertex_count=first.get("nfz_vertex_count"),
        )

        fig, ax = plt.subplots(figsize=(14.8, 10.5))

        nfz_face = "#d9d9d9"
        nfz_edge = "#4d4d4d"
        for poly in polygons:
            closed = np.vstack([poly, poly[0]])
            ax.fill(
                closed[:, 0], closed[:, 1],
                facecolor=nfz_face, edgecolor=nfz_edge,
                linewidth=1.8, alpha=0.32, zorder=1,
            )
            ax.plot(closed[:, 0], closed[:, 1], color=nfz_edge, linewidth=1.8, zorder=2)

        user_slice = np.asarray(
            getattr(cfg, "user_slice", np.zeros(user_pos.shape[0])), dtype=np.int32
        )
        embb_mask = user_slice == int(getattr(cfg, "slice_embb", 0))
        urllc_mask = user_slice == int(getattr(cfg, "slice_urllc", 1))
        mmtc_mask = user_slice == int(getattr(cfg, "slice_mmtc", 2))
        if np.any(embb_mask):
            ax.scatter(
                user_pos[embb_mask, 0], user_pos[embb_mask, 1],
                s=68, marker="o", facecolors="#d62728", edgecolors="black",
                linewidths=0.9, alpha=0.95, zorder=4,
            )
        if np.any(urllc_mask):
            ax.scatter(
                user_pos[urllc_mask, 0], user_pos[urllc_mask, 1],
                s=78, marker="^", facecolors="#1f77b4", edgecolors="black",
                linewidths=0.9, alpha=0.95, zorder=4,
            )
        if np.any(mmtc_mask):
            ax.scatter(
                user_pos[mmtc_mask, 0], user_pos[mmtc_mask, 1],
                s=66, marker="D", facecolors="#2ca02c", edgecolors="black",
                linewidths=0.9, alpha=0.95, zorder=4,
            )

        uav_colors = _base_uav_colors()
        algorithm_names = list(trajectory_records.keys())
        lighten_schedule = [0.00, 0.18, 0.34, 0.46]
        line_styles = ["-", "--", "-.", ":"]
        start_marker = "s"
        end_marker = "X"

        for algo_index, (algorithm_name, record) in enumerate(trajectory_records.items()):
            uav_pos = np.asarray(record["uav_pos"], dtype=np.float32)
            line_style = line_styles[min(algo_index, len(line_styles) - 1)]
            shade = lighten_schedule[min(algo_index, len(lighten_schedule) - 1)]
            for m in range(uav_pos.shape[1]):
                traj = uav_pos[:, m, :]
                base_color = uav_colors[m % len(uav_colors)]
                color = _lighten_color(base_color, shade)
                ax.plot(
                    traj[:, 0], traj[:, 1],
                    linestyle=line_style, color=color, linewidth=2.6,
                    solid_capstyle="round", dash_capstyle="round",
                    zorder=6,
                )
                ax.scatter(
                    traj[0, 0], traj[0, 1],
                    marker=start_marker, s=120,
                    facecolors="white", edgecolors=color, linewidths=1.8,
                    zorder=8,
                )
                ax.scatter(
                    traj[-1, 0], traj[-1, 1],
                    marker=end_marker, s=155,
                    facecolors=color, edgecolors="black", linewidths=0.8,
                    zorder=9,
                )

        ax.set_xlim(0, cfg.area_size)
        ax.set_ylim(0, cfg.area_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)", fontsize=26)
        ax.set_ylabel("y (m)", fontsize=26)
        ax.tick_params(axis="both", labelsize=21)
        ax.grid(True, alpha=0.28, linewidth=0.8)
        ax.set_axisbelow(True)

        legend_handles = [
            Patch(facecolor=nfz_face, edgecolor=nfz_edge, linewidth=1.6, alpha=0.32, label="NFZ"),
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#d62728", markeredgecolor="black", markersize=10, label="eMBB users"),
            Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="#1f77b4", markeredgecolor="black", markersize=11, label="URLLC users"),
            Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="#2ca02c", markeredgecolor="black", markersize=10, label="mMTC users"),
        ]
        for m, base_color in enumerate(uav_colors[: int(getattr(cfg, "uav_n", 3))]):
            for algo_index, name in enumerate(algorithm_names):
                line_style = line_styles[min(algo_index, len(line_styles) - 1)]
                shade = lighten_schedule[min(algo_index, len(lighten_schedule) - 1)]
                display_name = _algorithm_display_name(name)
                color = _lighten_color(base_color, shade)
                legend_handles.append(
                    Line2D(
                        [0], [0], color=color, linewidth=2.8,
                        linestyle=line_style, label=f"UAV{m + 1} {display_name}",
                    )
                )
        for m, base_color in enumerate(uav_colors[: int(getattr(cfg, "uav_n", 3))]):
            legend_handles.append(
                Line2D(
                    [0], [0], marker=start_marker, linestyle="None",
                    markerfacecolor="white", markeredgecolor=base_color,
                    markeredgewidth=1.8, markersize=10,
                    label=f"UAV{m + 1} start",
                )
            )
            legend_handles.append(
                Line2D(
                    [0], [0], marker=end_marker, linestyle="None",
                    markerfacecolor=base_color, markeredgecolor="black",
                    markeredgewidth=0.8, markersize=11,
                    label=f"UAV{m + 1} end",
                )
            )

        legend_cfg = _trajectory_legend_layout(mode="multi", algo_n=len(algorithm_names))
        ax.legend(
            handles=legend_handles,
            fontsize=legend_cfg["fontsize"],
            loc=legend_cfg["loc"],
            bbox_to_anchor=legend_cfg["bbox_to_anchor"],
            ncol=legend_cfg["ncol"],
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="black",
            columnspacing=legend_cfg["columnspacing"],
            handlelength=legend_cfg["handlelength"],
            borderpad=legend_cfg["borderpad"],
            labelspacing=legend_cfg["labelspacing"],
        )
        fig.tight_layout(rect=legend_cfg["rect"])
        _safe_savefig_with_outside_legend(plt, os.path.join(path, save_name), dpi=320)
        plt.close(fig)
    except Exception as exc:
        print(f"多算法轨迹对比图保存失败 {save_name}: {exc}")


def _generate_requested_trajectory_comparisons(
    ctd4_cfg,
    main_cfg,
    benchmark_results,
):
    """Generate the four user-requested per-episode trajectory overlays."""
    ctd4_data_dir = os.path.join(ctd4_cfg.result_path, "test_trajectory_data")
    benchmark_root = os.path.join(ctd4_cfg.result_path, "bm")
    output_root = os.path.join(ctd4_cfg.result_path, "cmp")

    proposed_name = str(getattr(ctd4_cfg, "algorithm", "CTD4"))
    comparison_specs = [
        ("g", [proposed_name, "MakespanAwareOneStepGreedy"]),
        ("gw", [proposed_name, "MakespanAwareOneStepGreedy", "WorkloadDistance"]),
    ]

    generated_dirs = []
    for folder_name, algorithms in comparison_specs:
        required_benchmarks = [name for name in algorithms if name != proposed_name]
        missing = [name for name in required_benchmarks if name not in benchmark_results]
        if missing:
            print(
                f"跳过轨迹对比 {folder_name}：未运行 " + ", ".join(missing)
            )
            continue

        output_dir = os.path.join(output_root, folder_name)
        _safe_makedirs(output_dir)
        generated_n = 0
        for ep in range(1, int(ctd4_cfg.test_ep_n) + 1):
            records = {}
            ctd4_record = _load_trajectory_record(ctd4_data_dir, ep)
            if ctd4_record is None:
                continue
            records[proposed_name] = ctd4_record

            complete = True
            for name in required_benchmarks:
                data_dir = os.path.join(benchmark_root, _benchmark_fs_tag(name), "td")
                record = _load_trajectory_record(data_dir, ep)
                if record is None:
                    complete = False
                    break
                records[name] = record
            if not complete:
                continue

            slot_text = ", ".join(
                f"{name}={int(np.asarray(record['slot_count']).reshape(-1)[0])}"
                for name, record in records.items()
            )
            plot_multi_algorithm_trajectory(
                trajectory_records=records,
                cfg=main_cfg,
                path=output_dir,
                save_name=f"e{ep:04d}.png",
                title=f"EP {ep} trajectory comparison | slots: {slot_text}",
            )
            generated_n += 1

        generated_dirs.append((folder_name, generated_n))
        print(f"轨迹对比图已保存：{output_dir}（{generated_n}个回合）")

    if generated_dirs:
        _safe_makedirs(output_root)
        with open(
            os.path.join(output_root, "comparison_directories.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for folder_name, generated_n in generated_dirs:
                f.write(f"{folder_name}: {generated_n} episodes\n")

def _heading_action(cfg, headings):
    headings = np.asarray(headings, dtype=np.float32).reshape(-1)
    action = np.zeros(cfg.action_dim, dtype=np.float32)
    for m, theta in enumerate(headings):
        base = m * cfg.local_action_dim
        action[base:base + 2] = cfg.heading_to_local_traj_action(float(theta))
    return action


def _decision_user_pos(cfg, env):
    return env.get_service_user_pos(cfg)


def _assign_unique_targets(cfg, env, score_matrix, maximize=False):
    """Greedy one-to-one UAV/user matching without fixed UAV-order bias."""
    unfinished = np.where(
        env.remaining_data_bits > float(cfg.task_data_eps_bits)
    )[0].astype(np.int32)
    if unfinished.size == 0:
        return -np.ones(cfg.uav_n, dtype=np.int32)

    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    targets = -np.ones(cfg.uav_n, dtype=np.int32)
    free_uavs = list(range(int(cfg.uav_n)))
    free_users = [int(k) for k in unfinished.tolist()]

    while free_uavs and free_users:
        best_pair = None
        best_value = -np.inf if maximize else np.inf
        for m in free_uavs:
            for k in free_users:
                value = float(score_matrix[m, k])
                better = value > best_value if maximize else value < best_value
                if better:
                    best_value = value
                    best_pair = (m, k)
        if best_pair is None:
            break
        m, k = best_pair
        targets[m] = k
        free_uavs.remove(m)
        free_users.remove(k)

    # If unfinished users are fewer than UAVs, let any unassigned UAV choose its
    # own best unfinished target; duplicate targets are then unavoidable.
    for m in free_uavs:
        values = score_matrix[m, unfinished]
        j = int(np.argmax(values) if maximize else np.argmin(values))
        targets[m] = int(unfinished[j])
    return targets


def _targets_to_action(cfg, env, targets):
    user_pos = _decision_user_pos(cfg, env)
    headings = np.zeros(cfg.uav_n, dtype=np.float32)
    center = np.array([0.5 * cfg.area_size, 0.5 * cfg.area_size], dtype=np.float32)
    for m in range(cfg.uav_n):
        if 0 <= int(targets[m]) < cfg.user_n:
            vec = user_pos[int(targets[m])] - env.uav_pos[m]
        else:
            vec = center - env.uav_pos[m]
        headings[m] = float(np.arctan2(vec[1], vec[0]))
    return _heading_action(cfg, headings)


def _safe_nearest_unfinished(cfg, env):
    """Distance-only heuristic: pair UAVs with nearest unfinished eMBB users."""
    user_pos = _decision_user_pos(cfg, env)
    dist = np.linalg.norm(
        env.uav_pos[:, None, :] - user_pos[None, :, :], axis=2
    )
    targets = _assign_unique_targets(cfg, env, dist, maximize=False)
    return _targets_to_action(cfg, env, targets)


def _workload_distance(cfg, env):
    """Workload-distance heuristic balancing remaining service and travel time.

    Score(m,k) = estimated remaining transmission slots of user k
                 / (1 + estimated flight slots from UAV m to user k).
    """
    user_pos = _decision_user_pos(cfg, env)
    dist = np.linalg.norm(
        env.uav_pos[:, None, :] - user_pos[None, :, :], axis=2
    )
    ref_rate = np.maximum(np.asarray(env.reference_user_rate, dtype=np.float64), 1.0)
    tx_slots = np.asarray(env.remaining_data_bits, dtype=np.float64) / np.maximum(
        ref_rate * float(cfg.slot_time), 1.0
    )
    fly_slots = dist / max(float(cfg.move_dist_each_step), 1e-9)
    score = tx_slots[None, :] / (1.0 + fly_slots)
    targets = _assign_unique_targets(cfg, env, score, maximize=True)
    return _targets_to_action(cfg, env, targets)


def _evaluate_action_global_reward(cfg, env, raw_action):
    """Return the exact proposed-algorithm global reward for one candidate action.

    The candidate next state is simulated with the same shield, mobility,
    association, interference, resource allocation, URLLC puncturing and mMTC
    model as ``Env.step``.  The scalar score itself is computed by the actual
    environment reward helper ``_sliced_task_reward``; no separate Greedy
    objective is maintained here.
    """
    action, _ = env._shield_action_by_nearest_safe_heading(cfg, raw_action)
    heading, _ = cfg.parse_joint_action(action, active_mask=env.active_mask)
    next_uav, _, _, _ = cfg.move_uav_one_slot(
        env.uav_pos, heading, active_mask=env.active_mask
    )

    # Exact service-user state used by Env.step at this slot.
    service_user_pos, service_user_speed, service_user_heading = (
        env.preview_user_state_for_current_step(cfg)
    )
    embb_idx = np.asarray(cfg.embb_user_idx, dtype=np.int32)
    unfinished_before = np.zeros(cfg.user_n, dtype=bool)
    unfinished_before[embb_idx] = (
        env.remaining_data_bits[embb_idx] > float(cfg.task_data_eps_bits)
    )
    result = serve_one_slot(
        cfg=cfg,
        uav_pos=next_uav,
        user_pos=service_user_pos,
        residual_energy=env.residual_energy,
        active_mask=env.active_mask,
        ep_id=env.ep_id,
        step_idx=env.step_id,
        avg_user_rate=env.avg_user_rate,
        unfinished_mask=unfinished_before,
        remaining_data_bits=env.remaining_data_bits,
    )

    before = np.asarray(env.remaining_data_bits, dtype=np.float32)
    after = before.copy()
    served = np.asarray(result["served_bits_user"], dtype=np.float32)
    after[embb_idx] = np.maximum(before[embb_idx] - served[embb_idx], 0.0)
    non_embb = np.setdiff1d(np.arange(cfg.user_n), embb_idx)
    after[non_embb] = 0.0
    terminated = bool(
        np.all(after[embb_idx] <= float(cfg.task_data_eps_bits))
    )

    potential_before_info = getattr(env, "current_completion_potential_info", None)
    if potential_before_info is None:
        potential_before_info = estimate_embb_completion_potential(
            cfg=cfg,
            uav_pos=env.uav_pos,
            user_pos=env._decision_user_pos(cfg),
            remaining_data_bits=before,
            initial_data_bits=env.initial_data_bits,
            active_mask=env.active_mask,
        )
    phi_before = float(potential_before_info["potential"])

    if terminated:
        phi_after = 0.0
    else:
        if bool(getattr(cfg, "use_perfect_next_user_position_observation", True)):
            potential_user_pos, _, _ = cfg.move_users_one_slot(
                user_pos=service_user_pos,
                user_speed=service_user_speed,
                user_heading=service_user_heading,
                ep_id=env.ep_id,
                step_idx=env.step_id + 1,
            )
        else:
            potential_user_pos = service_user_pos
        potential_after_info = estimate_embb_completion_potential(
            cfg=cfg,
            uav_pos=next_uav,
            user_pos=potential_user_pos,
            remaining_data_bits=after,
            initial_data_bits=env.initial_data_bits,
            active_mask=env.active_mask,
        )
        phi_after = float(potential_after_info["potential"])

    traffic = result.get("traffic_metrics", {})
    arrivals = int(traffic.get("urllc_arrival_count", 0))
    successes = int(traffic.get("urllc_success_count", 0))
    mmtc_ratio = float(traffic.get("mmtc_success_ratio", 1.0))

    # _sliced_task_reward reads the URLLC history *after* the current slot has
    # been appended in Env.step.  Temporarily append only for scoring and undo.
    env.urllc_slot_arrival_history.append(arrivals)
    env.urllc_slot_success_history.append(successes)
    try:
        reward, _, _ = _sliced_task_reward(
            env,
            cfg,
            before,
            after,
            current_mmtc_ratio=mmtc_ratio,
            phi_before=phi_before,
            phi_after=phi_after,
            terminated=terminated,
            truncated=False,
        )
    finally:
        env.urllc_slot_arrival_history.pop()
        env.urllc_slot_success_history.pop()
    return float(reward)


def _makespan_aware_one_step_greedy(cfg, env):
    """Coordinate-wise one-step Greedy maximizing the same global reward."""
    base_action = _workload_distance(cfg, env)
    base_heading, _ = cfg.parse_joint_action(base_action, active_mask=env.active_mask)
    headings = base_heading.copy()
    candidate_n = 8
    candidates = np.linspace(-np.pi, np.pi, candidate_n, endpoint=False)
    for m in range(cfg.uav_n):
        best_theta = float(headings[m])
        best_score = None
        for theta in candidates:
            trial = headings.copy()
            trial[m] = float(theta)
            action = _heading_action(cfg, trial)
            score = _evaluate_action_global_reward(cfg, env, action)
            if best_score is None or score > best_score:
                best_score = score
                best_theta = float(theta)
        headings[m] = best_theta
    return _heading_action(cfg, headings)


def _benchmark_action(name, cfg, env, rng):
    if name == "SafeNearestUnfinished":
        return _safe_nearest_unfinished(cfg, env)
    if name == "WorkloadDistance":
        return _workload_distance(cfg, env)
    if name == "MakespanAwareOneStepGreedy":
        return _makespan_aware_one_step_greedy(cfg, env)
    raise ValueError(f"未知benchmark：{name}")


_SCALAR_KEYS = [
    "ep_slot_count",
    "ep_all_users_completed",
    "ep_time_limit_hit",
    "ep_completed_user_n",
    "ep_unfinished_user_n",
    "ep_user_completion_ratio",
    "ep_data_completion_ratio",
    "ep_total_initial_data_bits",
    "ep_total_remaining_data_bits",
    "ep_total_served_bits",
    "ep_mean_completion_slot",
    "ep_p95_completion_slot",
    "ep_max_completion_slot",
    "ep_avg_sum_rate",
    "ep_avg_mean_rate",
    "ep_avg_progress",
    "ep_avg_fairness",
    "ep_shield_ratio",
    "ep_raw_boundary_attempt_ratio",
    "ep_raw_nfz_attempt_ratio",
    "ep_urllc_reliability",
    "ep_urllc_constraint_gap",
    "ep_urllc_fail_link_ratio",
    "ep_urllc_fail_cap_ratio",
    "ep_urllc_fail_budget_ratio",
    "ep_urllc_fail_available_ratio",
    "ep_urllc_fail_resource_ratio",
    "ep_urllc_avg_required_rb",
    "ep_urllc_p95_required_rb",
    "ep_urllc_cap_hit_tti_ratio",
    "ep_urllc_full_rb_hit_tti_ratio",
    "ep_urllc_avg_idle_rb_used",
    "ep_urllc_avg_total_rb_used",
    "ep_urllc_max_punctured_rb_one_uav_tti",
    "ep_urllc_max_total_rb_one_uav_tti",
    "ep_mmtc_connection_success_ratio",
    "ep_mmtc_constraint_gap",
    "ep_qos_feasible",
    "ep_avg_puncture_ratio",
    "ep_avg_qos_penalty",
]


def _new_metric_lists():
    return {key: [] for key in ["ep_reward"] + _SCALAR_KEYS}


def _append_metric(lists, reward, ep_metric):
    lists["ep_reward"].append(float(reward))
    for key in _SCALAR_KEYS:
        lists[key].append(ep_metric[key])


def _finalize_metric(lists):
    out = {}
    for key, values in lists.items():
        if key in ("ep_all_users_completed", "ep_time_limit_hit", "ep_qos_feasible"):
            out[key] = np.asarray(values, dtype=bool)
        else:
            out[key] = np.asarray(values, dtype=np.float32)
    return out


def _save_result(metric_dict, result_path, prefix):
    _safe_makedirs(result_path)
    np.savez(os.path.join(result_path, f"{prefix}_results.npz"), **metric_dict)
    try:
        import matplotlib.pyplot as plt
        plots = [
            ("ep_reward", "Episode reward", "r.png"),
            ("ep_slot_count", "Completion / cutoff slots", "slots.png"),
            ("ep_data_completion_ratio", "Data completion ratio", "data.png"),
            ("ep_user_completion_ratio", "User completion ratio", "users.png"),
            ("ep_avg_sum_rate", "Average eMBB sum rate (bps)", "rate.png"),
            ("ep_urllc_reliability", "Episode URLLC reliability", "urllc.png"),
            ("ep_mmtc_connection_success_ratio", "Episode-average mMTC connectivity", "mmtc.png"),
            ("ep_qos_feasible", "Episode QoS feasible", "qos.png"),
        ]
        for key, ylabel, plot_file in plots:
            y = np.asarray(metric_dict[key], dtype=np.float32)
            plt.figure()
            plt.plot(y, label="Raw")
            plt.plot(moving_average(y), label="Moving average")
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            _safe_savefig(plt, os.path.join(result_path, plot_file), dpi=300)
            plt.close()
    except Exception as exc:
        print(f"{prefix}曲线保存失败：", exc)


def _qos_satisfied_episode_stats(metric, key, target):
    values = np.asarray(metric[key], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    total = int(values.size)
    satisfied = int(np.sum(values >= float(target) - 1e-12)) if total > 0 else 0
    ratio = float(satisfied / total) if total > 0 else 0.0
    return satisfied, total, ratio


def _print_added_test_statistics(name, metric, main_cfg):
    urllc_ok, urllc_total, urllc_ratio = _qos_satisfied_episode_stats(
        metric,
        "ep_urllc_reliability",
        main_cfg.urllc_reliability_target,
    )
    mmtc_ok, mmtc_total, mmtc_ratio = _qos_satisfied_episode_stats(
        metric,
        "ep_mmtc_connection_success_ratio",
        main_cfg.mmtc_connectivity_target,
    )
    decision_us = safe_mean(metric.get("avg_decision_time_us", []))
    print(
        f"    DecisionTime/slot={decision_us:.3f} us | "
        f"URLLC QoS满足={urllc_ok}/{urllc_total}回合 ({urllc_ratio * 100.0:.2f}%) | "
        f"mMTC QoS满足={mmtc_ok}/{mmtc_total}回合 ({mmtc_ratio * 100.0:.2f}%)"
    )


def _print_summary(name, metric, main_cfg=None):
    print(
        f"{name}: slots={safe_mean(metric['ep_slot_count']):.2f}±{safe_std(metric['ep_slot_count']):.2f} | "
        f"all-complete={safe_mean(metric['ep_all_users_completed']):.3f} | "
        f"user-done={safe_mean(metric['ep_user_completion_ratio']):.3f} | "
        f"data-done={safe_mean(metric['ep_data_completion_ratio']):.3f} | "
        f"sum-rate={safe_mean(metric['ep_avg_sum_rate'])/1e6:.2f}Mbps | "
        f"URLLC={safe_mean(metric['ep_urllc_reliability']):.4f} | "
        f"mMTC={safe_mean(metric['ep_mmtc_connection_success_ratio']):.3f} | "
        f"QoSOK={safe_mean(metric['ep_qos_feasible']):.3f}"
    )
    print(
        f"    URLLCDiag: FailFBL/Resource="
        f"{safe_mean(metric['ep_urllc_fail_link_ratio']):.4f}/"
        f"{safe_mean(metric['ep_urllc_fail_resource_ratio']):.4f} | "
        f"ReqRB(avg/p95)={safe_mean(metric['ep_urllc_avg_required_rb']):.2f}/"
        f"{safe_mean(metric['ep_urllc_p95_required_rb']):.2f} | "
        f"FullRBHitTTI={safe_mean(metric['ep_urllc_full_rb_hit_tti_ratio']):.4f} | "
        f"IdleRB(avg/slot)={safe_mean(metric['ep_urllc_avg_idle_rb_used']):.2f}"
    )
    if main_cfg is not None:
        _print_added_test_statistics(name, metric, main_cfg)


def _save_ctd4_greedy_slot_comparison(
    ctd4_metric,
    greedy_metric,
    result_path,
    greedy_name="MakespanAwareOneStepGreedy",
    file_tag="greedy",
    proposed_name="CTD4",
):
    ctd4_slots = np.asarray(ctd4_metric["ep_slot_count"], dtype=np.float32).reshape(-1)
    greedy_slots = np.asarray(greedy_metric["ep_slot_count"], dtype=np.float32).reshape(-1)

    if ctd4_slots.size != greedy_slots.size:
        raise ValueError(
            f"{proposed_name}与{greedy_name}测试回合数不一致：{ctd4_slots.size} vs {greedy_slots.size}"
        )

    slot_diff = ctd4_slots - greedy_slots
    ctd4_better = slot_diff < 0
    ctd4_worse = slot_diff > 0
    equal = slot_diff == 0

    print(f"\n================ {proposed_name} vs {greedy_name}逐回合slot对比 ================")
    print(f"差值定义：{proposed_name} slot - {greedy_name} slot；正数表示所提算法使用更多slot。")
    for ep, (ctd4_slot, greedy_slot, diff) in enumerate(
        zip(ctd4_slots, greedy_slots, slot_diff), start=1
    ):
        if diff < 0:
            comparison = f"{proposed_name}更少"
        elif diff > 0:
            comparison = f"{proposed_name}更多"
        else:
            comparison = "相同"
        print(
            f"EP {ep:>3d}: {proposed_name}={int(round(float(ctd4_slot))):>3d} slot | "
            f"{greedy_name}={int(round(float(greedy_slot))):>3d} slot | "
            f"{proposed_name}-{greedy_name}={int(round(float(diff))):+d} slot | {comparison}"
        )

    print("--------------------------------------------------------------")
    print(
        f"平均slot：{proposed_name}={safe_mean(ctd4_slots):.2f}，"
        f"{greedy_name}={safe_mean(greedy_slots):.2f}，"
        f"平均差值（{proposed_name}-{greedy_name}）={safe_mean(slot_diff):+.2f} slot"
    )
    print(
        f"累计差值（{proposed_name}-{greedy_name}）={float(np.sum(slot_diff)):+.0f} slot | "
        f"{proposed_name}更少={int(np.sum(ctd4_better))}回合 | "
        f"{proposed_name}更多={int(np.sum(ctd4_worse))}回合 | "
        f"相同={int(np.sum(equal))}回合"
    )
    print("==============================================================")

    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment

    _safe_makedirs(result_path)
    excel_path = os.path.join(result_path, f"{str(proposed_name).lower()}_vs_{file_tag}_slot_comparison.xlsx")
    workbook = Workbook()

    detail_sheet = workbook.active
    detail_sheet.title = "逐回合对比"
    detail_headers = [
        "测试回合",
        "所提算法slot数",
        f"{greedy_name} slot数",
        f"所提算法-{greedy_name}(slot)",
        "对比结果",
    ]
    detail_sheet.append(detail_headers)
    for cell in detail_sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    for ep, (ctd4_slot, greedy_slot, diff) in enumerate(
        zip(ctd4_slots, greedy_slots, slot_diff), start=1
    ):
        if diff < 0:
            comparison = "所提算法slot更少"
        elif diff > 0:
            comparison = "所提算法slot更多"
        else:
            comparison = "slot数相同"
        detail_sheet.append(
            [
                ep,
                int(round(float(ctd4_slot))),
                int(round(float(greedy_slot))),
                int(round(float(diff))),
                comparison,
            ]
        )
    detail_sheet.freeze_panes = "A2"
    detail_sheet.auto_filter.ref = detail_sheet.dimensions
    detail_sheet.column_dimensions["A"].width = 12
    detail_sheet.column_dimensions["B"].width = 15
    detail_sheet.column_dimensions["C"].width = 15
    detail_sheet.column_dimensions["D"].width = 23
    detail_sheet.column_dimensions["E"].width = 25

    summary_sheet = workbook.create_sheet("汇总")
    summary_sheet.append(["指标", "数值"])
    for cell in summary_sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
    summary_rows = [
        ("测试回合数", int(ctd4_slots.size)),
        ("所提算法平均slot数", safe_mean(ctd4_slots)),
        (f"{greedy_name}平均slot数", safe_mean(greedy_slots)),
        (f"平均差值（所提算法-{greedy_name}）", safe_mean(slot_diff)),
        (f"累计差值（所提算法-{greedy_name}）", float(np.sum(slot_diff))),
        ("所提算法slot更少的回合数", int(np.sum(ctd4_better))),
        ("所提算法slot更多的回合数", int(np.sum(ctd4_worse))),
        ("slot数相同的回合数", int(np.sum(equal))),
    ]
    for row in summary_rows:
        summary_sheet.append(list(row))
    summary_sheet.column_dimensions["A"].width = 36
    summary_sheet.column_dimensions["B"].width = 20

    workbook.save(excel_path)
    print(f"{proposed_name}与{greedy_name}逐回合slot对比Excel已保存：{excel_path}")


def ctd4_test_func(ctd4_cfg, env, agent, main_cfg, load_model_path=None):
    _load_model(agent, load_model_path)
    metric_lists = _new_metric_lists()
    decision_times = []
    traj_dir = os.path.join(ctd4_cfg.result_path, "test_trajectories")
    trajectory_data_dir = os.path.join(
        ctd4_cfg.result_path, "test_trajectory_data"
    )

    proposed_name = str(getattr(ctd4_cfg, "algorithm", "CTD4"))
    print(f"\n开始测试 {proposed_name}：eMBB有限文件 + URLLC穿孔 + mMTC连接约束")
    with torch.no_grad():
        for ep in range(ctd4_cfg.test_ep_n):
            real_ep = ctd4_cfg.train_ep_n + ep
            env.reset(main_cfg, ep_id=real_ep)
            obs = env.get_obs(main_cfg)
            done = False
            ep_reward = 0.0
            while not done:
                t0 = time.perf_counter()
                action = agent.choose_action(obs)
                decision_times.append((time.perf_counter() - t0) * 1e6)
                _, reward, done = env.step(main_cfg, real_ep, action)
                ep_reward += float(reward)
                if not done:
                    obs = env.get_obs(main_cfg)
            ep_metric = env.get_ep_metric(main_cfg)
            _append_metric(metric_lists, ep_reward, ep_metric)
            _save_trajectory_record(
                trajectory_data_dir, ep + 1, env, ep_metric
            )
            if bool(getattr(ctd4_cfg, "test_plot_all_trajectories", True)):
                plot_uav_trajectory(
                    uav_pos=np.asarray(env.uav_pos_list),
                    user_pos=np.asarray(env.user_pos),
                    cfg=main_cfg,
                    path=traj_dir,
                    save_name=f"episode_{ep + 1:04d}.png",
                    title=f"{proposed_name} EP {ep + 1} | slots={env.slot_count} | {env.prev_done_reason}",
                    user_pos_list=np.asarray(env.user_pos_list),
                    completion_slot=env.completion_slot,
                    remaining_data_bits=env.remaining_data_bits,
                    annotate_every=ctd4_cfg.trajectory_annotate_every,
                    nfz_polygons_padded=ep_metric["ep_nfz_polygons_padded"],
                    nfz_vertex_count=ep_metric["ep_nfz_vertex_count"],
                )
            if (ep + 1) % 10 == 0 or ep == 0:
                print(
                    f"Test {ep + 1:>3d}/{ctd4_cfg.test_ep_n} | slots={env.slot_count} | "
                    f"eMBBdone={ep_metric['ep_completed_user_n']}/{main_cfg.embb_user_n} | "
                    f"data={ep_metric['ep_data_completion_ratio']:.3f} | "
                    f"URLLC={ep_metric['ep_urllc_reliability']:.4f} | "
                    f"mMTC={ep_metric['ep_mmtc_connection_success_ratio']:.3f} | "
                    f"UResFail={ep_metric['ep_urllc_fail_resource_ratio']:.4f} | "
                    f"FullRBHit={ep_metric['ep_urllc_full_rb_hit_tti_ratio']:.4f}"
                )

    metric = _finalize_metric(metric_lists)
    metric["avg_decision_time_us"] = np.asarray(
        [safe_mean(decision_times)], dtype=np.float32
    )
    _save_result(metric, ctd4_cfg.result_path, "test")
    _print_summary(proposed_name, metric, main_cfg)
    return safe_mean(metric["ep_reward"]), metric, safe_mean(decision_times)


def evaluate_benchmarks(
    ctd4_cfg,
    env,
    main_cfg,
    benchmark_names=None,
    ctd4_result=None,
):
    if benchmark_names is None:
        names = list(ctd4_cfg.benchmark_names)
    else:
        names = list(benchmark_names)
    if len(names) == 0:
        print("\n未启用任何benchmark，跳过benchmark测试。")
        return {}
    results = {}
    root = os.path.join(ctd4_cfg.result_path, "bm")
    _safe_makedirs(root)

    for name in names:
        lists = _new_metric_lists()
        decision_times = []
        fs_tag = _benchmark_fs_tag(name)
        traj_dir = os.path.join(root, fs_tag, "tr")
        trajectory_data_dir = os.path.join(root, fs_tag, "td")
        print(f"\nBenchmark: {name}")
        for ep in range(ctd4_cfg.test_ep_n):
            real_ep = ctd4_cfg.train_ep_n + ep
            env.reset(main_cfg, ep_id=real_ep)
            done = False
            ep_reward = 0.0
            rng = np.random.default_rng(main_cfg.seed + real_ep * 1009)
            while not done:
                t0 = time.perf_counter()
                action = _benchmark_action(name, main_cfg, env, rng)
                decision_times.append((time.perf_counter() - t0) * 1e6)
                _, reward, done = env.step(main_cfg, real_ep, action)
                ep_reward += float(reward)
            ep_metric = env.get_ep_metric(main_cfg)
            _append_metric(lists, ep_reward, ep_metric)
            if ctd4_result is not None and name in (
                "MakespanAwareOneStepGreedy",
                "WorkloadDistance",
            ):
                _save_trajectory_record(
                    trajectory_data_dir, ep + 1, env, ep_metric
                )
            if bool(getattr(ctd4_cfg, "benchmark_plot_all_trajectories", True)):
                plot_uav_trajectory(
                    uav_pos=np.asarray(env.uav_pos_list),
                    user_pos=np.asarray(env.user_pos),
                    cfg=main_cfg,
                    path=traj_dir,
                    save_name=f"episode_{ep + 1:04d}.png",
                    title=f"{name} EP {ep + 1} | slots={env.slot_count}",
                    user_pos_list=np.asarray(env.user_pos_list),
                    completion_slot=env.completion_slot,
                    remaining_data_bits=env.remaining_data_bits,
                    annotate_every=ctd4_cfg.trajectory_annotate_every,
                    nfz_polygons_padded=ep_metric["ep_nfz_polygons_padded"],
                    nfz_vertex_count=ep_metric["ep_nfz_vertex_count"],
                )
        metric = _finalize_metric(lists)
        metric["avg_decision_time_us"] = np.asarray([safe_mean(decision_times)], dtype=np.float32)
        results[name] = metric
        out_dir = os.path.join(root, fs_tag)
        _save_result(metric, out_dir, name)
        _print_summary(name, metric, main_cfg)

    if ctd4_result is not None:
        ctd4_metric = ctd4_result.get("metric_dict", {})
        if "ep_slot_count" in ctd4_metric:
            if "MakespanAwareOneStepGreedy" in results:
                _save_ctd4_greedy_slot_comparison(
                    ctd4_metric=ctd4_metric,
                    greedy_metric=results["MakespanAwareOneStepGreedy"],
                    result_path=root,
                    greedy_name="MakespanAwareOneStepGreedy",
                    file_tag="greedy",
                    proposed_name=str(getattr(ctd4_cfg, "algorithm", "CTD4")),
                )

    if ctd4_result is not None:
        _generate_requested_trajectory_comparisons(
            ctd4_cfg=ctd4_cfg,
            main_cfg=main_cfg,
            benchmark_results=results,
        )

    # Compact comparison file and figure.
    summary_names = []
    summary_slots = []
    summary_success = []
    if ctd4_result is not None:
        cmetric = ctd4_result.get("metric_dict", {})
        if "ep_slot_count" in cmetric:
            summary_names.append(str(getattr(ctd4_cfg, "algorithm", "CTD4")))
            summary_slots.append(safe_mean(cmetric["ep_slot_count"]))
            summary_success.append(safe_mean(cmetric["ep_all_users_completed"]))
    for name, metric in results.items():
        summary_names.append(name)
        summary_slots.append(safe_mean(metric["ep_slot_count"]))
        summary_success.append(safe_mean(metric["ep_all_users_completed"]))
    np.savez(
        os.path.join(root, "benchmark_summary.npz"),
        names=np.asarray(summary_names, dtype=str),
        average_slots=np.asarray(summary_slots, dtype=np.float32),
        all_complete_ratio=np.asarray(summary_success, dtype=np.float32),
    )
    try:
        import matplotlib.pyplot as plt
        x = np.arange(len(summary_names))
        plt.figure(figsize=(10, 5))
        plt.bar(x, summary_slots)
        plt.xticks(x, summary_names, rotation=25, ha="right")
        plt.ylabel("Average completion/cutoff slots")
        plt.tight_layout()
        _safe_savefig(plt, os.path.join(root, "completion_slot_comparison.png"), dpi=300)
        plt.close()
    except Exception as exc:
        print("benchmark对比图保存失败：", exc)

    print("\n================ 最终：各算法决策时长与QoS达标回合统计 ================")
    if ctd4_result is not None:
        cmetric = ctd4_result.get("metric_dict", {})
        if "ep_slot_count" in cmetric:
            _print_added_test_statistics(
                str(getattr(ctd4_cfg, "algorithm", "CTD4")),
                cmetric,
                main_cfg,
            )
    for name, metric in results.items():
        _print_added_test_statistics(name, metric, main_cfg)
    print("====================================================================")

    manifest_path = os.path.join(root, "dirs.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        if ctd4_result is not None:
            f.write(f"{getattr(ctd4_cfg, 'algorithm', 'CTD4')} trajectories: ../../test_trajectories\n")
        for name in names:
            f.write(f"{name}: {_benchmark_fs_tag(name)}/tr\n")
        if ctd4_result is not None:
            f.write(
                f"Every test episode is plotted for {getattr(ctd4_cfg, 'algorithm', 'CTD4')} and every benchmark when "
                "the corresponding *_plot_all_trajectories flags are True.\n"
            )
            f.write(
                "Requested overlay comparisons: ../../cmp/\n"
            )
        else:
            f.write(
                "The proposed DRL algorithm is disabled; only benchmark trajectories are generated when "
                "benchmark_plot_all_trajectories is True.\n"
            )
    return results
