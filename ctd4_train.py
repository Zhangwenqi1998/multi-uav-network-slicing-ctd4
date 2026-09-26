"""Training entry for finite-eMBB completion-time CTD4 with URLLC/mMTC constraints."""
from __future__ import annotations

import os
import random
import copy
from typing import Dict, List, Tuple

import numpy as np
import torch

from uav_env import GaussianNoise
from ctd4_test import plot_uav_trajectory


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(x, beta=0.9):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = beta * out[i - 1] + (1.0 - beta) * x[i]
    return out


def _mean_last(values, window=100, default=0.0):
    if len(values) == 0:
        return float(default)
    arr = np.asarray(values[-int(window):], dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float(default)


def _mean_last_axis0(values, window=100, size=3):
    if len(values) == 0:
        return np.zeros(size, dtype=np.float32)
    return np.nanmean(np.asarray(values[-int(window):], dtype=np.float32), axis=0)


def _save_curves(rewards, metric_dict, result_path):
    os.makedirs(result_path, exist_ok=True)
    np.savez(
        os.path.join(result_path, "train_results.npz"),
        rewards=np.asarray(rewards, dtype=np.float32),
        ma_rewards=moving_average(rewards),
        **metric_dict,
    )
    try:
        import matplotlib.pyplot as plt

        plots = [
            (rewards, "Episode reward", "train_reward.png"),
            (metric_dict["ep_slot_count"], "Completion / cutoff slots", "train_slot_count.png"),
            (metric_dict["ep_data_completion_ratio"], "Data completion ratio", "train_data_completion_ratio.png"),
            (metric_dict["ep_user_completion_ratio"], "User completion ratio", "train_user_completion_ratio.png"),
            (metric_dict["ep_avg_sum_rate"], "Average sum rate (bps)", "train_avg_sum_rate.png"),
            (metric_dict["ep_raw_boundary_attempt_ratio"], "Raw boundary attempt ratio", "train_raw_boundary_ratio.png"),
            (metric_dict["ep_raw_nfz_attempt_ratio"], "Raw NFZ attempt ratio", "train_raw_nfz_ratio.png"),
            (metric_dict["ep_urllc_reliability"], "Episode URLLC reliability", "train_urllc_reliability.png"),
            (metric_dict["ep_mmtc_connection_success_ratio"], "Episode-average mMTC connectivity", "train_mmtc_connectivity.png"),
            (metric_dict["ep_qos_feasible"], "Episode QoS feasible", "train_qos_feasible.png"),
            (metric_dict["ep_qos_lambda_urllc"], "URLLC Lagrange multiplier", "train_qos_lambda_urllc.png"),
            (metric_dict["ep_qos_lambda_mmtc"], "mMTC Lagrange multiplier", "train_qos_lambda_mmtc.png"),
            (metric_dict["ep_avg_potential_user_bottleneck"], "Potential user bottleneck", "train_potential_user_bottleneck.png"),
            (metric_dict["ep_avg_potential_uav_bottleneck"], "Potential UAV bottleneck", "train_potential_uav_bottleneck.png"),
            (metric_dict["actor_lr"], "Actor learning rate", "train_actor_lr.png"),
            (metric_dict["critic_lr"], "Critic learning rate", "train_critic_lr.png"),
        ]
        for y, ylabel, name in plots:
            y = np.asarray(y, dtype=np.float32)
            if y.size == 0:
                continue
            plt.figure()
            plt.plot(y, label="Raw")
            plt.plot(moving_average(y), label="Moving average")
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, name), dpi=300)
            plt.close()
    except Exception as exc:
        print("训练曲线保存失败：", exc)


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _validation_representative_indices(
    slot_count: np.ndarray,
    success: np.ndarray,
    max_step: int,
    representative_n: int,
) -> List[int]:
    """Return best/median/worst deterministic validation episodes."""
    slot_count = np.asarray(slot_count, dtype=np.float64).reshape(-1)
    success = np.asarray(success, dtype=bool).reshape(-1)
    effective = slot_count + (~success).astype(np.float64) * float(max_step)
    order = np.argsort(effective)
    if order.size == 0:
        return []
    n = max(1, int(representative_n))
    if n >= order.size:
        return order.tolist()
    positions = np.linspace(0, order.size - 1, n)
    selected = []
    for pos in positions:
        idx = int(order[int(round(float(pos)))])
        if idx not in selected:
            selected.append(idx)
    return selected


def _save_validation_history(history: Dict[str, list], result_path: str):
    validation_root = os.path.join(result_path, "fixed_validation")
    os.makedirs(validation_root, exist_ok=True)
    arrays = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in history.items()
    }
    np.savez(os.path.join(validation_root, "validation_history.npz"), **arrays)
    try:
        import matplotlib.pyplot as plt

        episode = arrays.get("train_episode", np.zeros(0, dtype=np.float32))
        plots = [
            ("mean_slots", "Fixed-validation completion slots", "validation_slots.png"),
            ("success_ratio", "Fixed-validation all-complete ratio", "validation_success.png"),
            ("score", "Validation selection score", "validation_score.png"),
        ]
        for key, ylabel, filename in plots:
            y = arrays.get(key, np.zeros(0, dtype=np.float32))
            if y.size == 0:
                continue
            plt.figure()
            plt.plot(episode, y, marker="o")
            plt.xlabel("Training episode")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(validation_root, filename), dpi=300)
            plt.close()
    except Exception as exc:
        print("固定验证曲线保存失败：", exc)


def _run_fixed_validation(
    ctd4_cfg,
    env,
    agent,
    main_cfg,
    train_episode: int,
    validation_history: Dict[str, list],
):
    """Evaluate the deterministic actor on the same independent scenarios.

    Validation does not add transitions, update the networks or consume the
    training RNG stream.  The fixed episode IDs guarantee directly comparable
    checkpoints throughout training.
    """
    rng_state = _capture_rng_state()
    rewards: List[float] = []
    slots: List[float] = []
    success: List[float] = []
    data_done: List[float] = []
    user_done: List[float] = []
    urllc_rel: List[float] = []
    mmtc_rel: List[float] = []
    qos_feasible: List[float] = []
    snapshots: List[dict] = []

    validation_root = os.path.join(
        ctd4_cfg.result_path,
        "fixed_validation",
        f"train_ep_{int(train_episode):05d}",
    )
    os.makedirs(validation_root, exist_ok=True)

    try:
        with torch.no_grad():
            for val_idx in range(int(ctd4_cfg.validation_ep_n)):
                real_ep = int(ctd4_cfg.validation_ep_id_start) + val_idx
                env.reset(main_cfg, ep_id=real_ep)
                obs = env.get_obs(main_cfg)
                done = False
                ep_reward = 0.0
                while not done:
                    action = agent.choose_action(obs)
                    _, reward, done = env.step(main_cfg, real_ep, action)
                    ep_reward += float(reward)
                    if not done:
                        obs = env.get_obs(main_cfg)

                metric = env.get_ep_metric(main_cfg)
                rewards.append(ep_reward)
                slots.append(float(metric["ep_slot_count"]))
                success.append(float(metric["ep_all_users_completed"]))
                data_done.append(float(metric["ep_data_completion_ratio"]))
                user_done.append(float(metric["ep_user_completion_ratio"]))
                urllc_rel.append(float(metric["ep_urllc_reliability"]))
                mmtc_rel.append(float(metric["ep_mmtc_connection_success_ratio"]))
                qos_feasible.append(float(metric["ep_qos_feasible"]))
                snapshots.append({
                    "val_idx": val_idx,
                    "real_ep": real_ep,
                    "slot_count": int(metric["ep_slot_count"]),
                    "done_reason": str(metric["ep_done_reason"]),
                    "uav_pos": np.asarray(env.uav_pos_list, dtype=np.float32).copy(),
                    "user_pos": np.asarray(env.user_pos, dtype=np.float32).copy(),
                    "user_pos_list": np.asarray(env.user_pos_list, dtype=np.float32).copy(),
                    "completion_slot": np.asarray(env.completion_slot, dtype=np.int32).copy(),
                    "remaining_data_bits": np.asarray(env.remaining_data_bits, dtype=np.float32).copy(),
                    "nfz_polygons_padded": np.asarray(metric["ep_nfz_polygons_padded"], dtype=np.float32).copy(),
                    "nfz_vertex_count": np.asarray(metric["ep_nfz_vertex_count"], dtype=np.int32).copy(),
                })
    finally:
        _restore_rng_state(rng_state)

    slots_arr = np.asarray(slots, dtype=np.float32)
    success_arr = np.asarray(success, dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    data_arr = np.asarray(data_done, dtype=np.float32)
    user_arr = np.asarray(user_done, dtype=np.float32)
    urllc_arr = np.asarray(urllc_rel, dtype=np.float32)
    mmtc_arr = np.asarray(mmtc_rel, dtype=np.float32)
    qos_arr = np.asarray(qos_feasible, dtype=np.float32)

    mean_slots = float(np.mean(slots_arr))
    std_slots = float(np.std(slots_arr))
    success_ratio = float(np.mean(success_arr))
    mean_reward = float(np.mean(rewards_arr))
    mean_data = float(np.mean(data_arr))
    mean_user = float(np.mean(user_arr))
    mean_urllc = float(np.mean(urllc_arr))
    mean_mmtc = float(np.mean(mmtc_arr))
    qos_feasible_ratio = float(np.mean(qos_arr))
    # Formal QoS constraints are episode-level.  Average the per-scenario gaps
    # instead of taking the gap of an average, so strong scenarios cannot hide
    # violations in other fixed-validation scenarios.
    mean_u_gap = float(np.mean(np.maximum(
        float(main_cfg.urllc_reliability_target) - urllc_arr, 0.0
    )))
    mean_m_gap = float(np.mean(np.maximum(
        float(main_cfg.mmtc_connectivity_target) - mmtc_arr, 0.0
    )))
    # Constrained lexicographic model selection.  The tuple is compared in
    # order, so no arbitrary scalar weight can trade QoS against completion time:
    #   1) maximize all-complete ratio;
    #   2) among incomplete models, maximize data completion;
    #   3) maximize episode-level QoS-feasible ratio;
    #   4) minimize average QoS residual;
    #   5) minimize mean completion slots.
    selection_key = (
        float(1.0 - success_ratio),
        float(1.0 - mean_data),
        float(1.0 - qos_feasible_ratio),
        float(mean_u_gap + mean_m_gap),
        float(mean_slots),
    )
    # Keep a scalar diagnostic for old plotting/result readers.  It is not used
    # to select the best checkpoint.
    score = float(
        mean_slots
        + float(main_cfg.max_step) * (1.0 - success_ratio)
        + float(main_cfg.max_step) * (1.0 - mean_data)
        + float(main_cfg.max_step) * (mean_u_gap + mean_m_gap)
    )

    np.savez(
        os.path.join(validation_root, "validation_results.npz"),
        real_episode_ids=np.arange(
            int(ctd4_cfg.validation_ep_id_start),
            int(ctd4_cfg.validation_ep_id_start) + int(ctd4_cfg.validation_ep_n),
            dtype=np.int64,
        ),
        rewards=rewards_arr,
        slot_count=slots_arr,
        all_completed=success_arr.astype(bool),
        data_completion_ratio=data_arr,
        user_completion_ratio=user_arr,
        urllc_reliability=urllc_arr,
        mmtc_connection_success_ratio=mmtc_arr,
        qos_feasible=qos_arr.astype(bool),
        mean_slots=np.asarray([mean_slots], dtype=np.float32),
        std_slots=np.asarray([std_slots], dtype=np.float32),
        success_ratio=np.asarray([success_ratio], dtype=np.float32),
        mean_urllc_reliability=np.asarray([mean_urllc], dtype=np.float32),
        mean_mmtc_connectivity=np.asarray([mean_mmtc], dtype=np.float32),
        mean_urllc_gap=np.asarray([mean_u_gap], dtype=np.float32),
        mean_mmtc_gap=np.asarray([mean_m_gap], dtype=np.float32),
        qos_feasible_ratio=np.asarray([qos_feasible_ratio], dtype=np.float32),
        score=np.asarray([score], dtype=np.float32),
        selection_key=np.asarray(selection_key, dtype=np.float64),
    )

    if bool(getattr(ctd4_cfg, "validation_plot_all_episodes", False)):
        plot_indices = list(range(len(snapshots)))
    else:
        plot_indices = _validation_representative_indices(
            slots_arr,
            success_arr > 0.5,
            max_step=main_cfg.max_step,
            representative_n=ctd4_cfg.validation_representative_n,
        )
    traj_dir = os.path.join(validation_root, "trajectories")
    rank_names = ["best", "median", "worst"]
    for rank, idx in enumerate(plot_indices):
        snap = snapshots[int(idx)]
        label = (
            f"all_{int(idx) + 1:03d}"
            if bool(getattr(ctd4_cfg, "validation_plot_all_episodes", False))
            else rank_names[min(rank, len(rank_names) - 1)]
        )
        plot_uav_trajectory(
            uav_pos=snap["uav_pos"],
            user_pos=snap["user_pos"],
            cfg=main_cfg,
            path=traj_dir,
            save_name=(
                f"{label}_fixed_ep_{int(snap['val_idx']) + 1:03d}_"
                f"slots_{int(snap['slot_count']):03d}.png"
            ),
            title=(
                f"Validation @ Train EP {train_episode} | {label} | "
                f"fixed scenario {int(snap['val_idx']) + 1} | "
                f"slots={int(snap['slot_count'])}"
            ),
            user_pos_list=snap["user_pos_list"],
            completion_slot=snap["completion_slot"],
            remaining_data_bits=snap["remaining_data_bits"],
            annotate_every=ctd4_cfg.trajectory_annotate_every,
            nfz_polygons_padded=snap["nfz_polygons_padded"],
            nfz_vertex_count=snap["nfz_vertex_count"],
        )

    validation_history["train_episode"].append(float(train_episode))
    validation_history["mean_slots"].append(mean_slots)
    validation_history["std_slots"].append(std_slots)
    validation_history["success_ratio"].append(success_ratio)
    validation_history["mean_reward"].append(mean_reward)
    validation_history["data_completion_ratio"].append(mean_data)
    validation_history["user_completion_ratio"].append(mean_user)
    validation_history["urllc_reliability"].append(mean_urllc)
    validation_history["mmtc_connectivity"].append(mean_mmtc)
    validation_history["qos_feasible_ratio"].append(qos_feasible_ratio)
    validation_history["score"].append(score)
    _save_validation_history(validation_history, ctd4_cfg.result_path)

    print(
        f"    FixedVal({ctd4_cfg.validation_ep_n}EP, no-noise) | "
        f"Slots={mean_slots:.2f}±{std_slots:.2f} | Complete={success_ratio:.3f} | "
        f"DataDone={mean_data:.3f} | URLLC={mean_urllc:.4f} | "
        f"mMTC={mean_mmtc:.3f} | QoSOK={qos_feasible_ratio:.3f} | "
        f"SelectKey={tuple(round(x, 6) for x in selection_key)}"
    )
    return {
        "mean_slots": mean_slots,
        "std_slots": std_slots,
        "success_ratio": success_ratio,
        "mean_reward": mean_reward,
        "data_completion_ratio": mean_data,
        "user_completion_ratio": mean_user,
        "urllc_reliability": mean_urllc,
        "mmtc_connectivity": mean_mmtc,
        "mean_urllc_gap": mean_u_gap,
        "mean_mmtc_gap": mean_m_gap,
        "qos_feasible_ratio": qos_feasible_ratio,
        "score": score,
        "selection_key": selection_key,
    }


def pretrain_actor_with_teacher(ctd4_cfg, env, agent, main_cfg):
    """Old slice-aware teacher is intentionally disabled for this task."""
    return None


def _update_qos_lagrange(main_cfg, ep_metric):
    """One projected dual-ascent update from episode-level QoS residuals."""
    if not bool(getattr(main_cfg, "use_adaptive_qos_lagrange", False)):
        return (
            float(getattr(main_cfg, "qos_lagrange_urllc", 0.0)),
            float(getattr(main_cfg, "qos_lagrange_mmtc", 0.0)),
        )

    u_rel = float(ep_metric.get("ep_urllc_reliability", 1.0))
    m_rel = float(ep_metric.get("ep_mmtc_connection_success_ratio", 1.0))
    u_residual = float(main_cfg.urllc_reliability_target) - u_rel
    m_residual = float(main_cfg.mmtc_connectivity_target) - m_rel

    lambda_u = float(getattr(main_cfg, "qos_lagrange_urllc", 0.0))
    lambda_m = float(getattr(main_cfg, "qos_lagrange_mmtc", 0.0))
    lambda_u += float(main_cfg.qos_lagrange_urllc_lr) * u_residual
    lambda_m += float(main_cfg.qos_lagrange_mmtc_lr) * m_residual
    lambda_u = float(np.clip(
        lambda_u,
        float(main_cfg.qos_lagrange_urllc_min),
        float(main_cfg.qos_lagrange_urllc_max),
    ))
    lambda_m = float(np.clip(
        lambda_m,
        float(main_cfg.qos_lagrange_mmtc_min),
        float(main_cfg.qos_lagrange_mmtc_max),
    ))
    main_cfg.qos_lagrange_urllc = lambda_u
    main_cfg.qos_lagrange_mmtc = lambda_m
    return lambda_u, lambda_m


def ctd4_train_func(ctd4_cfg, env, agent, main_cfg):
    set_global_seed(main_cfg.seed)
    os.makedirs(ctd4_cfg.result_path, exist_ok=True)
    os.makedirs(ctd4_cfg.model_path, exist_ok=True)

    rewards = []
    metrics = {
        "ep_slot_count": [],
        "ep_all_users_completed": [],
        "ep_time_limit_hit": [],
        "ep_completed_user_n": [],
        "ep_unfinished_user_n": [],
        "ep_user_completion_ratio": [],
        "ep_data_completion_ratio": [],
        "ep_total_initial_data_bits": [],
        "ep_total_remaining_data_bits": [],
        "ep_total_served_bits": [],
        "ep_mean_completion_slot": [],
        "ep_p95_completion_slot": [],
        "ep_max_completion_slot": [],
        "ep_avg_sum_rate": [],
        "ep_avg_mean_rate": [],
        "ep_avg_progress": [],
        "ep_shield_ratio": [],
        "ep_raw_boundary_attempt_ratio": [],
        "ep_raw_nfz_attempt_ratio": [],
        "ep_urllc_reliability": [],
        "ep_urllc_constraint_gap": [],
        "ep_urllc_fail_link_ratio": [],
        "ep_urllc_fail_cap_ratio": [],
        "ep_urllc_fail_budget_ratio": [],
        "ep_urllc_fail_available_ratio": [],
        "ep_urllc_fail_resource_ratio": [],
        "ep_urllc_avg_required_rb": [],
        "ep_urllc_p95_required_rb": [],
        "ep_urllc_cap_hit_tti_ratio": [],
        "ep_urllc_full_rb_hit_tti_ratio": [],
        "ep_urllc_avg_idle_rb_used": [],
        "ep_urllc_avg_total_rb_used": [],
        "ep_urllc_max_punctured_rb_one_uav_tti": [],
        "ep_urllc_max_total_rb_one_uav_tti": [],
        "ep_mmtc_connection_success_ratio": [],
        "ep_mmtc_constraint_gap": [],
        "ep_qos_feasible": [],
        "ep_avg_puncture_ratio": [],
        "ep_avg_qos_penalty": [],
        "ep_qos_lambda_urllc": [],
        "ep_qos_lambda_mmtc": [],
        "ep_final_completion_potential": [],
        "ep_avg_potential_user_bottleneck": [],
        "ep_avg_potential_uav_bottleneck": [],
        "ep_avg_potential_mean_remaining": [],
        "ep_avg_agent_reward": [],
        "ep_agent_illegal_ratio": [],
        "ep_avg_agent_local_progress_contribution": [],
        "ep_avg_agent_credit_reward": [],
        "critic_loss": [],
        "actor_loss": [],
        "actor_q": [],
        "reward_mean": [],
        "actor_lr": [],
        "critic_lr": [],
    }
    validation_history = {
        "train_episode": [],
        "mean_slots": [],
        "std_slots": [],
        "success_ratio": [],
        "mean_reward": [],
        "data_completion_ratio": [],
        "user_completion_ratio": [],
        "urllc_reliability": [],
        "mmtc_connectivity": [],
        "qos_feasible_ratio": [],
        "score": [],
    }

    noise = GaussianNoise(
        action_dim=ctd4_cfg.action_dim,
        action_low=ctd4_cfg.action_low,
        action_high=ctd4_cfg.action_high,
        max_sigma=ctd4_cfg.exploration_noise_init,
        min_sigma=ctd4_cfg.exploration_noise_min,
        decay_period=ctd4_cfg.exploration_noise_decay_episodes,
    )

    total_steps = 0
    best_validation_score = float("inf")
    best_validation_key = None
    best_validation_episode = -1
    best_actor_state = None
    best_qos_lambdas = (
        float(getattr(main_cfg, "qos_lagrange_urllc", 0.0)),
        float(getattr(main_cfg, "qos_lagrange_mmtc", 0.0)),
    )
    best_dir = os.path.join(ctd4_cfg.model_path, "best")

    print(f"\n开始训练 Shared-Actor MA-{ctd4_cfg.algorithm}：eMBB有限文件 + URLLC穿孔 + mMTC连接约束")
    print(
        f"EP={ctd4_cfg.train_ep_n} | time_limit={ctd4_cfg.max_step} | "
        f"state={ctd4_cfg.state_dim} | obs={ctd4_cfg.local_obs_dim} | action={ctd4_cfg.action_dim}"
    )
    print(
        f"随机探索={ctd4_cfg.random_steps} steps | update_after={ctd4_cfg.update_after} | "
        f"exploration noise decay={ctd4_cfg.exploration_noise_decay_episodes} episodes | "
        f"target-policy noise decay={ctd4_cfg.policy_noise_decay_steps} updates"
    )
    if bool(getattr(ctd4_cfg, "use_lr_decay", False)):
        print(
            "学习率分段线性衰减（按episode）："
            f"hold/mid/end={ctd4_cfg.lr_decay_hold_episodes}/"
            f"{ctd4_cfg.lr_decay_mid_episodes}/{ctd4_cfg.lr_decay_end_episodes} EP | "
            f"Actor={ctd4_cfg.actor_lr:.1e}->{ctd4_cfg.actor_lr_mid:.1e}->"
            f"{ctd4_cfg.actor_lr_min:.1e} | "
            f"Critic={ctd4_cfg.critic_lr:.1e}->{ctd4_cfg.critic_lr_mid:.1e}->"
            f"{ctd4_cfg.critic_lr_min:.1e} | Replay={ctd4_cfg.memory_capacity}"
        )
    if ctd4_cfg.use_fixed_validation:
        print(
            f"固定无噪声验证：每{ctd4_cfg.validation_interval}EP，"
            f"固定{ctd4_cfg.validation_ep_n}个独立场景；按完成/QoS/Slots字典序保存best模型。"
        )
    print(
        f"真实终止=全部eMBB文件完成；max_step={main_cfg.max_step}仅为truncation并继续bootstrap，"
        "不附加cutoff reward。"
    )
    print(
        "Reward potential=用户瓶颈预计完成时间 + UAV负载瓶颈 + 平均剩余比例；"
        "QoS使用episode级自适应Lagrange乘子。"
    )
    print("-" * 96)

    for ep_id in range(ctd4_cfg.train_ep_n):
        episode_index = ep_id + 1
        agent.set_training_episode(episode_index)
        state = env.reset(main_cfg, ep_id=ep_id)
        obs = env.get_obs(main_cfg)
        episode_done = False
        ep_reward = 0.0
        last_loss = None

        while not episode_done:
            if total_steps < int(ctd4_cfg.random_steps):
                theta = np.random.uniform(-np.pi, np.pi, size=ctd4_cfg.agent_n)
                action = np.stack([np.cos(theta), np.sin(theta)], axis=1).reshape(-1).astype(np.float32)
            else:
                action = agent.choose_action(obs)
                if ctd4_cfg.use_exploration_noise:
                    # Common episode-wise exploration schedule: identical sigma at
                    # the same training episode for CTD4/DDPG/TD3, independent of slot count.
                    action = noise.get_action(action, episode_index)
            action = np.asarray(action, dtype=np.float32).reshape(-1)

            next_state, reward, episode_done = env.step(
                main_cfg=main_cfg, ep_id=ep_id, action=action
            )
            terminated = bool(getattr(env, "prev_terminated", False))
            truncated = bool(getattr(env, "prev_truncated", False))

            if terminated:
                next_obs = np.zeros(
                    (ctd4_cfg.agent_n, ctd4_cfg.local_obs_dim), dtype=np.float32
                )
            else:
                # Includes time-limit truncation: preserve the actual final state
                # and store done=0 so Bellman targets bootstrap correctly.
                next_obs = env.get_obs(main_cfg)

            raw_action = np.asarray(
                getattr(env, "prev_raw_action", action), dtype=np.float32
            ).reshape(-1)
            reward_vec = np.asarray(
                getattr(
                    env,
                    "prev_agent_reward",
                    np.repeat(float(reward), ctd4_cfg.agent_n),
                ),
                dtype=np.float32,
            ).reshape(-1)
            if ctd4_cfg.use_reward_clip:
                reward_vec = np.clip(
                    reward_vec,
                    ctd4_cfg.reward_clip_low,
                    ctd4_cfg.reward_clip_high,
                )

            agent.store_transition(
                state=state,
                obs=obs,
                action=raw_action,
                reward=reward_vec,
                next_state=next_state,
                next_obs=next_obs,
                done=terminated,
            )

            if total_steps >= int(ctd4_cfg.update_after):
                if total_steps % int(ctd4_cfg.update_every) == 0:
                    for _ in range(int(ctd4_cfg.update_times)):
                        info = agent.train()
                        if info is not None:
                            last_loss = info

            state = next_state
            obs = next_obs
            ep_reward += float(reward)
            total_steps += 1

            if truncated and not episode_done:
                raise RuntimeError("truncation必须结束当前episode。")

        ep_metric = env.get_ep_metric(main_cfg)
        rewards.append(float(ep_reward))
        for key in metrics:
            if key in ("critic_loss", "actor_loss", "actor_q", "reward_mean", "actor_lr", "critic_lr"):
                continue
            metrics[key].append(ep_metric[key])

        # Projected dual ascent uses the completed episode's aggregate QoS and
        # updates the prices applied from the next episode onward.
        next_lambda_u, next_lambda_m = _update_qos_lagrange(main_cfg, ep_metric)

        if last_loss is not None:
            metrics["critic_loss"].append(float(last_loss.get("critic_loss", np.nan)))
            actor_loss = last_loss.get("actor_loss", np.nan)
            actor_q = last_loss.get("actor_q", np.nan)
            metrics["actor_loss"].append(np.nan if actor_loss is None else float(actor_loss))
            metrics["actor_q"].append(np.nan if actor_q is None else float(actor_q))
            metrics["reward_mean"].append(float(last_loss.get("reward_mean", np.nan)))
            metrics["actor_lr"].append(float(last_loss.get("actor_lr", np.nan)))
            metrics["critic_lr"].append(float(last_loss.get("critic_lr", np.nan)))
        else:
            metrics["critic_loss"].append(np.nan)
            metrics["actor_loss"].append(np.nan)
            metrics["actor_q"].append(np.nan)
            metrics["reward_mean"].append(np.nan)
            metrics["actor_lr"].append(np.nan)
            metrics["critic_lr"].append(np.nan)

        if (ep_id + 1) % int(ctd4_cfg.print_interval) == 0 or ep_id == 0:
            window = min(100, len(rewards))
            success_rate = _mean_last(metrics["ep_all_users_completed"], window)
            slot_avg = _mean_last(metrics["ep_slot_count"], window)
            data_ratio = _mean_last(metrics["ep_data_completion_ratio"], window)
            user_ratio = _mean_last(metrics["ep_user_completion_ratio"], window)
            rb = _mean_last(metrics["ep_raw_boundary_attempt_ratio"], window)
            rn = _mean_last(metrics["ep_raw_nfz_attempt_ratio"], window)
            u_rel = _mean_last(metrics["ep_urllc_reliability"], window)
            m_rel = _mean_last(metrics["ep_mmtc_connection_success_ratio"], window)
            qos_ok = _mean_last(metrics["ep_qos_feasible"], window)
            punc = _mean_last(metrics["ep_avg_puncture_ratio"], window)
            u_fail_link = _mean_last(metrics["ep_urllc_fail_link_ratio"], window)
            u_fail_resource = _mean_last(metrics["ep_urllc_fail_resource_ratio"], window)
            u_req_avg = _mean_last(metrics["ep_urllc_avg_required_rb"], window)
            u_req_p95 = _mean_last(metrics["ep_urllc_p95_required_rb"], window)
            u_full_rb_hit = _mean_last(metrics["ep_urllc_full_rb_hit_tti_ratio"], window)
            u_idle = _mean_last(metrics["ep_urllc_avg_idle_rb_used"], window)
            agent_r = _mean_last_axis0(
                metrics["ep_avg_agent_reward"], window, ctd4_cfg.agent_n
            )
            illegal = _mean_last_axis0(
                metrics["ep_agent_illegal_ratio"], window, ctd4_cfg.agent_n
            )
            if total_steps < int(ctd4_cfg.random_steps):
                exploration_text = "random"
            else:
                sigma = noise.get_sigma(episode_index)
                exploration_text = f"sigma={sigma:.4f}rad"
            print(
                f"EP {ep_id + 1:>5d}/{ctd4_cfg.train_ep_n} | "
                f"R={_mean_last(rewards, window):.3f} | Slots={slot_avg:.2f} | "
                f"Complete={success_rate:.3f} | UserDone={user_ratio:.3f} | "
                f"DataDone={data_ratio:.3f} | URLLC={u_rel:.4f} | mMTC={m_rel:.3f} | "
                f"QoSOK={qos_ok:.3f} | Punc={punc:.4f} | RawB={rb:.3f} | RawNFZ={rn:.3f} | "
                f"Explore={exploration_text}"
            )
            actor_lr_now = float(getattr(agent, "current_actor_lr", ctd4_cfg.actor_lr))
            critic_lr_now = float(getattr(agent, "current_critic_lr", ctd4_cfg.critic_lr))
            print(
                f"    AgentReward={np.array2string(agent_r, precision=3, separator=',')} | "
                f"IllegalRatio={np.array2string(illegal, precision=3, separator=',')} | "
                f"LR(A/C)={actor_lr_now:.2e}/{critic_lr_now:.2e} | "
                f"Lambda(U/M)={next_lambda_u:.3f}/{next_lambda_m:.3f} | "
                f"Replay={len(agent.memory)}/{ctd4_cfg.memory_capacity}"
            )
            print(
                f"    URLLCDiag: FailFBL/Resource={u_fail_link:.4f}/{u_fail_resource:.4f} | "
                f"ReqRB(avg/p95)={u_req_avg:.2f}/{u_req_p95:.2f} | "
                f"FullRBHitTTI={u_full_rb_hit:.4f} | IdleRB(avg/slot)={u_idle:.2f}"
            )

        if ctd4_cfg.save and (ep_id + 1) % int(ctd4_cfg.save_freq) == 0:
            save_dir = os.path.join(ctd4_cfg.model_path, f"ep_{ep_id + 1}")
            agent.save(save_dir)
            try:
                traj_dir = os.path.join(ctd4_cfg.result_path, "train_trajectories")
                plot_uav_trajectory(
                    uav_pos=np.asarray(env.uav_pos_list),
                    user_pos=np.asarray(env.user_pos),
                    cfg=main_cfg,
                    path=traj_dir,
                    save_name=f"episode_{ep_id + 1:05d}.png",
                    title=f"Train EP {ep_id + 1} | slots={env.slot_count}",
                    user_pos_list=np.asarray(env.user_pos_list),
                    completion_slot=np.asarray(env.completion_slot),
                    remaining_data_bits=np.asarray(env.remaining_data_bits),
                    annotate_every=ctd4_cfg.trajectory_annotate_every,
                    nfz_polygons_padded=ep_metric["ep_nfz_polygons_padded"],
                    nfz_vertex_count=ep_metric["ep_nfz_vertex_count"],
                )
            except Exception as exc:
                print("训练轨迹图保存失败：", exc)

        do_validation = bool(ctd4_cfg.use_fixed_validation) and (
            (ep_id + 1) % int(ctd4_cfg.validation_interval) == 0
            or (ep_id + 1) == int(ctd4_cfg.train_ep_n)
        )
        if do_validation:
            summary = _run_fixed_validation(
                ctd4_cfg=ctd4_cfg,
                env=env,
                agent=agent,
                main_cfg=main_cfg,
                train_episode=ep_id + 1,
                validation_history=validation_history,
            )
            candidate_key = tuple(float(x) for x in summary["selection_key"])
            if best_validation_key is None or candidate_key < best_validation_key:
                best_validation_key = candidate_key
                best_validation_score = float(summary["score"])
                best_validation_episode = int(ep_id + 1)
                best_actor_state = copy.deepcopy(agent.actor.state_dict())
                best_qos_lambdas = (
                    float(getattr(main_cfg, "qos_lagrange_urllc", 0.0)),
                    float(getattr(main_cfg, "qos_lagrange_mmtc", 0.0)),
                )
                if ctd4_cfg.save and ctd4_cfg.validation_save_best:
                    agent.save(best_dir)
                    agent.save_actor(best_dir)
                    with open(
                        os.path.join(best_dir, "best_validation.txt"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(f"train_episode: {best_validation_episode}\n")
                        f.write(f"selection_key: {best_validation_key}\n")
                        f.write(f"diagnostic_score: {best_validation_score:.8f}\n")
                        f.write(f"mean_slots: {summary['mean_slots']:.8f}\n")
                        f.write(f"std_slots: {summary['std_slots']:.8f}\n")
                        f.write(f"success_ratio: {summary['success_ratio']:.8f}\n")
                        f.write(f"data_completion_ratio: {summary['data_completion_ratio']:.8f}\n")
                        f.write(f"urllc_reliability: {summary['urllc_reliability']:.8f}\n")
                        f.write(f"mmtc_connectivity: {summary['mmtc_connectivity']:.8f}\n")
                        f.write(f"qos_feasible_ratio: {summary['qos_feasible_ratio']:.8f}\n")
                        f.write(f"qos_lambda_urllc: {best_qos_lambdas[0]:.8f}\n")
                        f.write(f"qos_lambda_mmtc: {best_qos_lambdas[1]:.8f}\n")
                    print(
                        f"    新best模型：Train EP {best_validation_episode} | "
                        f"selection_key={best_validation_key}"
                    )

    # Always preserve the actual last iterate.  By default the lexicographic
    # validation-best model is then restored and becomes the model used by the
    # immediate final test and the `models/final` directory.
    if ctd4_cfg.save:
        last_dir = os.path.join(ctd4_cfg.model_path, "final_last")
        agent.save(last_dir)
        agent.save_actor(last_dir)

    selected_from = "final_training_episode"
    selected_episode = int(ctd4_cfg.train_ep_n)
    if (
        bool(getattr(ctd4_cfg, "validation_use_best_for_final_test", True))
        and best_validation_episode > 0
        and best_actor_state is not None
    ):
        best_checkpoint = os.path.join(best_dir, ctd4_cfg.model_checkpoint_name)
        if ctd4_cfg.save and os.path.exists(best_checkpoint):
            agent.load(best_dir)
        else:
            agent.actor.load_state_dict(best_actor_state)
            agent.target_actor.load_state_dict(agent.actor.state_dict())
        selected_from = "lexicographic_validation_best"
        selected_episode = int(best_validation_episode)
        if best_qos_lambdas is not None:
            main_cfg.qos_lagrange_urllc = float(best_qos_lambdas[0])
            main_cfg.qos_lagrange_mmtc = float(best_qos_lambdas[1])

    if ctd4_cfg.save:
        final_dir = os.path.join(ctd4_cfg.model_path, "final")
        agent.save(final_dir)
        agent.save_actor(final_dir)
        with open(
            os.path.join(final_dir, "selected_model.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"selected_from: {selected_from}\n")
            f.write(f"train_episode: {selected_episode}\n")
            if best_validation_key is not None:
                f.write(f"selection_key: {best_validation_key}\n")
                f.write(f"diagnostic_score: {best_validation_score:.8f}\n")
            f.write(
                f"qos_lambda_urllc: {float(getattr(main_cfg, 'qos_lagrange_urllc', 0.0)):.8f}\n"
            )
            f.write(
                f"qos_lambda_mmtc: {float(getattr(main_cfg, 'qos_lagrange_mmtc', 0.0)):.8f}\n"
            )

    metric_dict = {}
    for key, values in metrics.items():
        if key in ("ep_all_users_completed", "ep_time_limit_hit", "ep_qos_feasible"):
            metric_dict[key] = np.asarray(values, dtype=bool)
        else:
            metric_dict[key] = np.asarray(values, dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    if ctd4_cfg.save:
        _save_curves(rewards_arr, metric_dict, ctd4_cfg.result_path)

    print("\n训练结束。")
    print(
        f"最后100EP：平均slots={_mean_last(metrics['ep_slot_count'], 100):.2f}, "
        f"全部完成率={_mean_last(metrics['ep_all_users_completed'], 100):.3f}, "
        f"数据完成率={_mean_last(metrics['ep_data_completion_ratio'], 100):.3f}, "
        f"URLLC={_mean_last(metrics['ep_urllc_reliability'], 100):.4f}, "
        f"mMTC={_mean_last(metrics['ep_mmtc_connection_success_ratio'], 100):.3f}, "
        f"QoS可行率={_mean_last(metrics['ep_qos_feasible'], 100):.3f}"
    )
    if best_validation_episode > 0:
        print(
            f"固定验证best：Train EP {best_validation_episode} | "
            f"selection_key={best_validation_key}。"
        )
    print(
        f"最终测试模型：{selected_from}，Train EP {selected_episode}。"
    )
    return rewards_arr, moving_average(rewards_arr), metric_dict
