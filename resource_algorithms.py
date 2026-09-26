"""
ZHANG Wenqi

User association and radio service utilities.  The active runtime definition
at the end of this file performs load-aware eMBB association/allocation, pure
URLLC puncturing, and RB-free mMTC access-connectivity evaluation. Earlier
helpers are kept only for backward compatibility with old result readers.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 方向角 -> 二维连续动作
# ============================================================
def _set_heading_action(action, cfg, m, theta):
    base = m * cfg.local_action_dim
    action[base:base + 2] = cfg.heading_to_local_traj_action(float(theta))



# ============================================================
# 用户关联：最大接收信号强度
# ============================================================
def max_signal_user_association(
    cfg,
    rx_signal_matrix,
    main_lobe_mask,
    active_mask=None
):
    """
    用户关联到“接收信号强度最高”的 active UAV。

    Parameters
    ----------
    cfg:
        MainConfig
    rx_signal_matrix:
        [M, K]，P_m * G_mk * h_mk
    main_lobe_mask:
        [M, K] bool，用户是否处于UAV主瓣覆盖范围内。
        当 cfg.require_main_lobe_for_service=False 时，它只用于速率计算，不限制关联。
    active_mask:
        [M] bool，UAV是否仍有能量、可服务

    Returns
    -------
    associated_uav:
        [K]，associated_uav[k] = m；只有当所有UAV都inactive时才可能为-1。
    association_matrix:
        [M, K]，one-hot关联矩阵
    """

    rx_signal_matrix = np.asarray(rx_signal_matrix, dtype=np.float32)
    main_lobe_mask = np.asarray(main_lobe_mask, dtype=bool)

    m_n, user_n = rx_signal_matrix.shape

    if active_mask is None:
        active_mask = np.ones(m_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    if active_mask.size != m_n:
        raise ValueError(
            f"active_mask维度错误：期望 {m_n}，实际 {active_mask.size}"
        )

    # 可关联条件：UAV active；如果开启主瓣约束，则还要求用户处于主瓣覆盖内。
    if getattr(cfg, "require_main_lobe_for_service", True):
        valid_mask = main_lobe_mask.copy()
    else:
        valid_mask = np.ones_like(main_lobe_mask, dtype=bool)

    valid_mask = valid_mask & active_mask[:, None]

    masked_signal = rx_signal_matrix.copy()
    masked_signal[~valid_mask] = -np.inf

    associated_uav = -np.ones(user_n, dtype=np.int32)
    association_matrix = np.zeros((m_n, user_n), dtype=np.float32)

    for k in range(user_n):
        if not np.any(valid_mask[:, k]):
            continue

        best_m = int(np.argmax(masked_signal[:, k]))

        if not np.isfinite(masked_signal[best_m, k]):
            continue

        associated_uav[k] = best_m
        association_matrix[best_m, k] = 1.0

    return associated_uav.astype(np.int32), association_matrix.astype(np.float32)


# ============================================================
# 用户关联：最近UAV
# 备用benchmark，不作为默认关联方式
# ============================================================
def nearest_uav_user_association(
    cfg,
    uav_pos,
    user_pos,
    main_lobe_mask,
    active_mask=None
):
    """
    用户关联到最近的可用UAV。

    注意：
    这不是当前默认方案。默认方案是最大接收信号强度关联。
    """

    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    main_lobe_mask = np.asarray(main_lobe_mask, dtype=bool)

    dist_3d = cfg.calc_uav_user_distance_3d(
        uav_pos=uav_pos,
        user_pos=user_pos
    )

    m_n, user_n = dist_3d.shape

    if active_mask is None:
        active_mask = np.ones(m_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    if getattr(cfg, "require_main_lobe_for_service", True):
        valid_mask = main_lobe_mask.copy()
    else:
        valid_mask = np.ones_like(main_lobe_mask, dtype=bool)

    valid_mask = valid_mask & active_mask[:, None]

    masked_dist = dist_3d.copy()
    masked_dist[~valid_mask] = np.inf

    associated_uav = -np.ones(user_n, dtype=np.int32)
    association_matrix = np.zeros((m_n, user_n), dtype=np.float32)

    for k in range(user_n):
        if not np.any(valid_mask[:, k]):
            continue

        best_m = int(np.argmin(masked_dist[:, k]))

        if not np.isfinite(masked_dist[best_m, k]):
            continue

        associated_uav[k] = best_m
        association_matrix[best_m, k] = 1.0

    return associated_uav.astype(np.int32), association_matrix.astype(np.float32)


# ============================================================
# 平均带宽和功率分配
# ============================================================
def equal_bandwidth_power_allocation(cfg, association_matrix, active_mask=None):
    """
    每架UAV对其关联用户平均分配带宽和发射功率。

    Parameters
    ----------
    association_matrix:
        [M, K]，one-hot关联矩阵
    active_mask:
        [M] bool

    Returns
    -------
    bandwidth_alloc:
        [M, K]，Hz
    power_alloc:
        [M, K]，W
    user_count_each_uav:
        [M]，每架UAV关联用户数
    """

    association_matrix = np.asarray(association_matrix, dtype=np.float32)

    m_n, user_n = association_matrix.shape

    if active_mask is None:
        active_mask = np.ones(m_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    bandwidth_alloc = np.zeros((m_n, user_n), dtype=np.float32)
    power_alloc = np.zeros((m_n, user_n), dtype=np.float32)

    user_count_each_uav = np.sum(association_matrix > 0.5, axis=1).astype(np.int32)

    for m in range(m_n):
        if not active_mask[m]:
            continue

        user_idx = np.where(association_matrix[m] > 0.5)[0]

        if user_idx.size <= 0:
            continue

        bandwidth_each_user = cfg.bandwidth_hz / float(user_idx.size)
        power_each_user = cfg.tx_power_w_each_uav / float(user_idx.size)

        bandwidth_alloc[m, user_idx] = bandwidth_each_user
        power_alloc[m, user_idx] = power_each_user

    return (
        bandwidth_alloc.astype(np.float32),
        power_alloc.astype(np.float32),
        user_count_each_uav.astype(np.int32)
    )


# ============================================================
# 切片感知速率驱动资源分配
# ============================================================
def slice_aware_rate_resource_allocation(
    cfg,
    association_matrix,
    channel_gain,
    antenna_gain,
    active_mask=None,
    avg_user_rate=None,
    last_user_rate=None,
    slice_budget_ratio=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
):
    """
    eMBB / URLLC / mMTC two-level resource allocation.

    Level 1: 每架UAV使用DRL输出的切片预算比例。
    Level 2: 切片内按业务优先级分配带宽和功率。
    """

    association_matrix = np.asarray(association_matrix, dtype=np.float32)
    channel_gain = np.asarray(channel_gain, dtype=np.float32)
    antenna_gain = np.asarray(antenna_gain, dtype=np.float32)

    m_n, user_n = association_matrix.shape

    if active_mask is None:
        active_mask = np.ones(m_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    if avg_user_rate is None:
        avg_user_rate = np.zeros(user_n, dtype=np.float32)
    else:
        avg_user_rate = np.asarray(avg_user_rate, dtype=np.float32).reshape(-1)

    if last_user_rate is None:
        last_user_rate = np.zeros(user_n, dtype=np.float32)
    else:
        last_user_rate = np.asarray(last_user_rate, dtype=np.float32).reshape(-1)

    if urllc_queue_bits is None:
        urllc_queue_bits = np.zeros(user_n, dtype=np.float32)
    else:
        urllc_queue_bits = np.asarray(urllc_queue_bits, dtype=np.float32).reshape(-1)

    if urllc_deadline is None:
        urllc_deadline = np.zeros(user_n, dtype=np.float32)
    else:
        urllc_deadline = np.asarray(urllc_deadline, dtype=np.float32).reshape(-1)

    if mmtc_active is None:
        mmtc_active = np.zeros(user_n, dtype=np.float32)
    else:
        mmtc_active = np.asarray(mmtc_active, dtype=np.float32).reshape(-1)

    if mmtc_pending_bits is None:
        mmtc_pending_bits = np.zeros(user_n, dtype=np.float32)
    else:
        mmtc_pending_bits = np.asarray(mmtc_pending_bits, dtype=np.float32).reshape(-1)

    use_external_slice_budget = bool(getattr(cfg, "use_drl_slice_budget", False)) and slice_budget_ratio is not None
    if use_external_slice_budget:
        slice_budget_ratio = np.asarray(slice_budget_ratio, dtype=np.float32)
        if slice_budget_ratio.shape != (m_n, int(cfg.slice_n)):
            raise ValueError(f"slice_budget_ratio维度错误：期望 {(m_n, int(cfg.slice_n))}，实际 {slice_budget_ratio.shape}")
        slice_budget_ratio = np.maximum(slice_budget_ratio, 0.0)
        row_sum = np.sum(slice_budget_ratio, axis=1, keepdims=True)
        uniform_budget = np.ones((m_n, int(cfg.slice_n)), dtype=np.float32) / float(cfg.slice_n)
        slice_budget_ratio = np.where(
            row_sum > float(getattr(cfg, "slice_budget_eps", 1e-8)),
            slice_budget_ratio / np.maximum(row_sum, float(getattr(cfg, "slice_budget_eps", 1e-8))),
            uniform_budget,
        ).astype(np.float32)

    user_slice = np.asarray(cfg.user_slice, dtype=np.int32).reshape(-1)

    bandwidth_alloc = np.zeros((m_n, user_n), dtype=np.float32)
    power_alloc = np.zeros((m_n, user_n), dtype=np.float32)
    user_count_each_uav = np.sum(association_matrix > 0.5, axis=1).astype(np.int32)
    slice_budget_bandwidth = np.zeros((m_n, cfg.slice_n), dtype=np.float32)
    slice_budget_power = np.zeros((m_n, cfg.slice_n), dtype=np.float32)
    priority_matrix = np.zeros((m_n, user_n), dtype=np.float32)

    noise_full_band = cfg.bandwidth_hz * cfg.noise_psd_w_per_hz
    rx_power = cfg.tx_power_w_each_uav * antenna_gain * channel_gain
    spectral_eff_proxy = np.log2(1.0 + rx_power / max(noise_full_band, 1e-30)).astype(np.float32)
    spectral_eff_proxy = np.maximum(spectral_eff_proxy, float(cfg.alloc_spectral_eff_eps))

    for m in range(m_n):
        if not active_mask[m]:
            continue

        assoc_user_idx = np.where(association_matrix[m] > 0.5)[0]
        if assoc_user_idx.size <= 0:
            continue

        slice_score = np.zeros(cfg.slice_n, dtype=np.float32)
        user_priority = np.zeros(user_n, dtype=np.float32)

        for k in assoc_user_idx:
            sl = int(user_slice[k])
            base_weight = float(cfg.slice_resource_weight[sl])
            channel_term = float(spectral_eff_proxy[m, k])

            if sl == cfg.slice_embb:
                # eMBB：偏向频谱效率高且历史平均速率不太高的用户。
                pf_term = 1.0 / max(float(avg_user_rate[k]) / max(cfg.rate_norm, cfg.rate_eps), 0.2)
                service_term = min(4.0, pf_term)

            elif sl == cfg.slice_urllc:
                queue_norm = float(urllc_queue_bits[k]) / max(float(cfg.urllc_packet_bits), 1e-9)
                if urllc_queue_bits[k] > 0.0 and urllc_deadline[k] > 0.0:
                    deadline_urgency = 1.0 / max(float(urllc_deadline[k]), 1.0)
                elif urllc_queue_bits[k] > 0.0:
                    deadline_urgency = 1.0
                else:
                    deadline_urgency = 0.0
                service_term = 1.0 + cfg.alloc_urllc_queue_boost * queue_norm + cfg.alloc_urllc_deadline_boost * deadline_urgency

            elif sl == cfg.slice_mmtc:
                if mmtc_active[k] > 0.5 and mmtc_pending_bits[k] > 0.0:
                    service_term = 1.0 + cfg.alloc_mmtc_active_boost
                else:
                    # 非激活mMTC设备不应消耗太多用户级资源。
                    service_term = 0.05

            else:
                service_term = 1.0

            priority = max(base_weight * channel_term * service_term, float(cfg.alloc_min_priority))
            user_priority[k] = priority
            slice_score[sl] += priority

        total_slice_score = float(np.sum(slice_score))
        if use_external_slice_budget:
            slice_budget_m = slice_budget_ratio[m].copy().astype(np.float32)
            if bool(getattr(cfg, "redistribute_empty_slice_budget", False)):
                present_slice_mask = np.zeros(cfg.slice_n, dtype=bool)
                for sl in range(cfg.slice_n):
                    present_slice_mask[sl] = np.any(user_slice[assoc_user_idx] == sl)
                slice_budget_m[~present_slice_mask] = 0.0
                present_sum = float(np.sum(slice_budget_m))
                if present_sum <= float(getattr(cfg, "slice_budget_eps", 1e-8)):
                    present_count = int(np.sum(present_slice_mask))
                    if present_count > 0:
                        slice_budget_m[present_slice_mask] = 1.0 / float(present_count)
                else:
                    slice_budget_m = slice_budget_m / present_sum
        else:
            if total_slice_score <= cfg.alloc_priority_eps:
                slice_budget_m = np.ones(cfg.slice_n, dtype=np.float32) / float(cfg.slice_n)
            else:
                slice_budget_m = slice_score / max(total_slice_score, cfg.alloc_priority_eps)

        for sl in range(cfg.slice_n):
            slice_user_idx = assoc_user_idx[user_slice[assoc_user_idx] == sl]
            bw_s = cfg.bandwidth_hz * float(slice_budget_m[sl])
            p_s = cfg.tx_power_w_each_uav * float(slice_budget_m[sl])
            slice_budget_bandwidth[m, sl] = bw_s
            slice_budget_power[m, sl] = p_s

            if slice_user_idx.size <= 0:
                continue

            priority_s = user_priority[slice_user_idx]
            priority_sum_s = float(np.sum(priority_s))
            if priority_sum_s <= cfg.alloc_priority_eps:
                ratio = np.ones(slice_user_idx.size, dtype=np.float32) / float(slice_user_idx.size)
            else:
                ratio = priority_s / priority_sum_s

            bandwidth_alloc[m, slice_user_idx] = bw_s * ratio
            power_alloc[m, slice_user_idx] = p_s * ratio
            priority_matrix[m, slice_user_idx] = priority_s

    return (
        bandwidth_alloc.astype(np.float32),
        power_alloc.astype(np.float32),
        user_count_each_uav.astype(np.int32),
        slice_budget_bandwidth.astype(np.float32),
        slice_budget_power.astype(np.float32),
        priority_matrix.astype(np.float32),
    )


# ============================================================
# 平均UAV间干扰功率谱密度
# ============================================================
def calc_average_inter_uav_interference_psd(
    cfg,
    channel_gain,
    antenna_gain,
    active_mask=None
):
    """
    参考文章中的平均UAV间干扰模型：

        I_psd[m, k] = P_m * G_mk * h_mk / B

    表示UAV m 对用户 k 造成的平均干扰功率谱密度。

    Parameters
    ----------
    channel_gain:
        [M, K]，A2G信道功率增益 h_mk
    antenna_gain:
        [M, K]，天线增益 G_mk
    active_mask:
        [M] bool

    Returns
    -------
    interference_psd:
        [M, K]，W/Hz
    """

    channel_gain = np.asarray(channel_gain, dtype=np.float32)
    antenna_gain = np.asarray(antenna_gain, dtype=np.float32)

    m_n, user_n = channel_gain.shape

    if active_mask is None:
        active_mask = np.ones(m_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    interference_psd = (
        cfg.tx_power_w_each_uav
        * antenna_gain
        * channel_gain
        / max(cfg.bandwidth_hz, 1e-30)
    ).astype(np.float32)

    interference_psd[~active_mask, :] = 0.0

    return interference_psd.astype(np.float32)


# ============================================================
# 根据关联和资源分配计算用户速率
# ============================================================
def calc_user_rate_with_average_interference(
    cfg,
    channel_gain,
    antenna_gain,
    association_matrix,
    bandwidth_alloc,
    power_alloc,
    active_mask=None
):
    """
    计算每个用户的下行速率。

    若用户k由UAV m服务：

        SINR =
            p_mk * G_mk * h_mk
            /
            (beta_mk * N0 + beta_mk * sum_{m' != m} I_psd[m', k])

        R_k = beta_mk * log2(1 + SINR)

    Parameters
    ----------
    channel_gain:
        [M, K]
    antenna_gain:
        [M, K]
    association_matrix:
        [M, K]
    bandwidth_alloc:
        [M, K]
    power_alloc:
        [M, K]
    active_mask:
        [M] bool

    Returns
    -------
    user_rate:
        [K]，bps
    sinr_matrix:
        [M, K]，仅关联链路对应位置有有效SINR
    desired_power_matrix:
        [M, K]
    interference_power_matrix:
        [M, K]
    noise_power_matrix:
        [M, K]
    """

    channel_gain = np.asarray(channel_gain, dtype=np.float32)
    antenna_gain = np.asarray(antenna_gain, dtype=np.float32)
    association_matrix = np.asarray(association_matrix, dtype=np.float32)
    bandwidth_alloc = np.asarray(bandwidth_alloc, dtype=np.float32)
    power_alloc = np.asarray(power_alloc, dtype=np.float32)

    m_n, user_n = channel_gain.shape

    if active_mask is None:
        active_mask = np.ones(m_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    interference_psd = calc_average_inter_uav_interference_psd(
        cfg=cfg,
        channel_gain=channel_gain,
        antenna_gain=antenna_gain,
        active_mask=active_mask
    )

    desired_power_matrix = (
        power_alloc
        * antenna_gain
        * channel_gain
        * association_matrix
    ).astype(np.float32)

    sinr_matrix = np.zeros((m_n, user_n), dtype=np.float32)
    interference_power_matrix = np.zeros((m_n, user_n), dtype=np.float32)
    noise_power_matrix = np.zeros((m_n, user_n), dtype=np.float32)

    user_rate = np.zeros(user_n, dtype=np.float32)

    for k in range(user_n):
        serving_uav = np.where(association_matrix[:, k] > 0.5)[0]

        if serving_uav.size <= 0:
            continue

        m = int(serving_uav[0])

        if not active_mask[m]:
            continue

        beta_mk = float(bandwidth_alloc[m, k])
        p_mk = float(power_alloc[m, k])

        if beta_mk <= 0.0 or p_mk <= 0.0:
            continue

        desired_power = float(
            p_mk * antenna_gain[m, k] * channel_gain[m, k]
        )

        if desired_power <= 0.0:
            continue

        if cfg.full_frequency_reuse and cfg.use_average_inter_uav_interference:
            interference_psd_sum = 0.0

            for mp in range(m_n):
                if mp == m:
                    continue

                if not active_mask[mp]:
                    continue

                interference_psd_sum += float(interference_psd[mp, k])

            interference_power = beta_mk * interference_psd_sum
        else:
            interference_power = 0.0

        noise_power = beta_mk * cfg.noise_psd_w_per_hz

        denominator = max(
            noise_power + interference_power,
            1e-30
        )

        sinr = desired_power / denominator

        rate = beta_mk * np.log2(1.0 + sinr)

        sinr_matrix[m, k] = float(sinr)
        user_rate[k] = float(rate)
        interference_power_matrix[m, k] = float(interference_power)
        noise_power_matrix[m, k] = float(noise_power)

    return {
        "user_rate": user_rate.astype(np.float32),
        "sinr": sinr_matrix.astype(np.float32),
        "desired_power": desired_power_matrix.astype(np.float32),
        "interference_power": interference_power_matrix.astype(np.float32),
        "noise_power": noise_power_matrix.astype(np.float32),
        "interference_psd": interference_psd.astype(np.float32),
    }


# ============================================================
# 指标统计
# ============================================================
def calc_slot_metrics(
    cfg,
    user_rate,
    association_matrix,
    associated_uav,
    avg_user_rate=None
):
    user_rate = np.asarray(user_rate, dtype=np.float32).reshape(-1)
    association_matrix = np.asarray(association_matrix, dtype=np.float32)
    associated_uav = np.asarray(associated_uav, dtype=np.int32).reshape(-1)

    if avg_user_rate is None:
        avg_user_rate = np.zeros_like(user_rate, dtype=np.float32)
    else:
        avg_user_rate = np.asarray(avg_user_rate, dtype=np.float32).reshape(-1)

    sum_rate = float(np.sum(user_rate))
    mean_rate = float(np.mean(user_rate))
    served_bits = float(sum_rate * cfg.slot_time)

    # 当前slot速率公平性
    instantaneous_fairness = float(cfg.calc_jain_fairness(user_rate))

    # 历史平均速率 + 当前速率，用来近似反映累计吞吐量公平性
    if np.sum(avg_user_rate) <= cfg.rate_eps:
        fairness_value = user_rate
    else:
        fairness_value = avg_user_rate + user_rate

    fairness = float(cfg.calc_jain_fairness(fairness_value))

    associated_user = associated_uav >= 0
    associated_user_n = int(np.sum(associated_user))
    associated_ratio = float(associated_user_n / max(cfg.user_n, 1))

    active_link_n = int(np.sum(association_matrix > 0.5))

    if user_rate.size > 0:
        min_rate = float(np.min(user_rate))
        max_rate = float(np.max(user_rate))
    else:
        min_rate = 0.0
        max_rate = 0.0

    return {
        "sum_rate": float(sum_rate),
        "mean_rate": float(mean_rate),
        "served_bits": float(served_bits),

        "fairness": float(fairness),
        "instantaneous_fairness": float(instantaneous_fairness),

        # 当前问题不再把coverage/unserved作为核心指标；
        # 这里仅保留“成功关联比例”，避免把关联误写成覆盖性能。
        "associated_user_n": int(associated_user_n),
        "associated_ratio": float(associated_ratio),

        "active_link_n": int(active_link_n),

        "min_rate": float(min_rate),
        "max_rate": float(max_rate),
    }


# ============================================================
# 一个slot完整服务流程
# ============================================================
def serve_one_slot(
    cfg,
    uav_pos,
    user_pos,
    residual_energy=None,
    active_mask=None,
    ep_id=0,
    step_idx=0,
    avg_user_rate=None,
    association_mode="max_signal",
    slice_budget_ratio=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
):
    """
    当前slot完整通信流程：

    1. 计算A2G信道增益。
    2. 计算方向性天线增益和主瓣覆盖矩阵。
    3. 用户关联到 active UAV 中接收信号最强的UAV；主瓣/副瓣差异只影响速率，不再作为默认硬关联约束。
    4. 每架UAV执行切片感知资源分配，或按配置退化为平均分配。
    5. 考虑平均UAV间干扰，计算用户速率。
    6. 输出指标。

    Parameters
    ----------
    cfg:
        MainConfig
    uav_pos:
        [M, 2]
    user_pos:
        [K, 2]
    residual_energy:
        [M]
    active_mask:
        [M] bool
    ep_id:
        episode编号
    step_idx:
        当前slot编号
    avg_user_rate:
        [K]，历史平均速率
    association_mode:
        "max_signal" 或 "nearest"
    slice_budget_ratio:
        [M, S]，DRL输出的UAV级切片预算比例；每行和为1

    Returns
    -------
    dict
    """

    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)

    if not bool(getattr(cfg, "use_energy_management", False)):
        active_mask = np.ones(cfg.uav_n, dtype=bool)
    elif active_mask is None:
        if residual_energy is None:
            active_mask = np.ones(cfg.uav_n, dtype=bool)
        else:
            residual_energy = np.asarray(residual_energy, dtype=np.float32)
            active_mask = residual_energy > cfg.uav_energy_threshold
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    if avg_user_rate is None:
        avg_user_rate = np.zeros(cfg.user_n, dtype=np.float32)
    else:
        avg_user_rate = np.asarray(avg_user_rate, dtype=np.float32).reshape(-1)

    # ========================================================
    # 1) 信道增益
    # ========================================================
    channel_gain = cfg.calc_channel_gain(
        uav_pos=uav_pos,
        user_pos=user_pos,
        ep_id=ep_id,
        step_idx=step_idx
    )

    # ========================================================
    # 2) 天线增益与主瓣覆盖
    # ========================================================
    antenna_gain, main_lobe_mask = cfg.calc_antenna_gain_matrix(
        uav_pos=uav_pos,
        user_pos=user_pos
    )

    # 接收信号强度：P_m * G_mk * h_mk
    rx_signal_matrix = (
        cfg.tx_power_w_each_uav
        * antenna_gain
        * channel_gain
    ).astype(np.float32)

    # ========================================================
    # 3) 用户关联
    # ========================================================
    if association_mode == "max_signal":
        associated_uav, association_matrix = max_signal_user_association(
            cfg=cfg,
            rx_signal_matrix=rx_signal_matrix,
            main_lobe_mask=main_lobe_mask,
            active_mask=active_mask
        )

    elif association_mode == "nearest":
        associated_uav, association_matrix = nearest_uav_user_association(
            cfg=cfg,
            uav_pos=uav_pos,
            user_pos=user_pos,
            main_lobe_mask=main_lobe_mask,
            active_mask=active_mask
        )

    else:
        raise ValueError(f"未知association_mode: {association_mode}")

    # ========================================================
    # 4) 带宽 / 功率分配
    # ========================================================
    if getattr(cfg, "resource_allocation_mode", "slice_aware") == "slice_aware":
        (
            bandwidth_alloc,
            power_alloc,
            user_count_each_uav,
            slice_budget_bandwidth,
            slice_budget_power,
            priority_matrix,
        ) = slice_aware_rate_resource_allocation(
            cfg=cfg,
            association_matrix=association_matrix,
            channel_gain=channel_gain,
            antenna_gain=antenna_gain,
            active_mask=active_mask,
            avg_user_rate=avg_user_rate,
            slice_budget_ratio=slice_budget_ratio,
            urllc_queue_bits=urllc_queue_bits,
            urllc_deadline=urllc_deadline,
            mmtc_active=mmtc_active,
            mmtc_pending_bits=mmtc_pending_bits,
        )

    elif getattr(cfg, "resource_allocation_mode", "slice_aware") == "equal":
        bandwidth_alloc, power_alloc, user_count_each_uav = (
            equal_bandwidth_power_allocation(
                cfg=cfg,
                association_matrix=association_matrix,
                active_mask=active_mask
            )
        )
        slice_budget_bandwidth = np.zeros((cfg.uav_n, cfg.slice_n), dtype=np.float32)
        slice_budget_power = np.zeros((cfg.uav_n, cfg.slice_n), dtype=np.float32)
        priority_matrix = association_matrix.copy().astype(np.float32)

    else:
        raise ValueError(
            f"未知resource_allocation_mode: {cfg.resource_allocation_mode}"
        )

    # ========================================================
    # 5) 考虑UAV间平均干扰，计算速率
    # ========================================================
    rate_result = calc_user_rate_with_average_interference(
        cfg=cfg,
        channel_gain=channel_gain,
        antenna_gain=antenna_gain,
        association_matrix=association_matrix,
        bandwidth_alloc=bandwidth_alloc,
        power_alloc=power_alloc,
        active_mask=active_mask
    )

    user_rate = rate_result["user_rate"]

    # ========================================================
    # 6) 指标统计
    # ========================================================
    metrics = calc_slot_metrics(
        cfg=cfg,
        user_rate=user_rate,
        association_matrix=association_matrix,
        associated_uav=associated_uav,
        avg_user_rate=avg_user_rate
    )

    return {
        "channel_gain": channel_gain.astype(np.float32),
        "antenna_gain": antenna_gain.astype(np.float32),
        "main_lobe_mask": main_lobe_mask.astype(bool),

        "rx_signal_matrix": rx_signal_matrix.astype(np.float32),

        "associated_uav": associated_uav.astype(np.int32),
        "association": association_matrix.astype(np.float32),

        "bandwidth_alloc": bandwidth_alloc.astype(np.float32),
        "power_alloc": power_alloc.astype(np.float32),
        "user_count_each_uav": user_count_each_uav.astype(np.int32),
        "slice_budget_bandwidth": slice_budget_bandwidth.astype(np.float32),
        "slice_budget_power": slice_budget_power.astype(np.float32),
        "priority_matrix": priority_matrix.astype(np.float32),

        "user_rate": user_rate.astype(np.float32),
        "sinr": rate_result["sinr"].astype(np.float32),
        "desired_power": rate_result["desired_power"].astype(np.float32),
        "interference_power": rate_result["interference_power"].astype(np.float32),
        "noise_power": rate_result["noise_power"].astype(np.float32),
        "interference_psd": rate_result["interference_psd"].astype(np.float32),

        "active_mask": active_mask.astype(bool),
        "metrics": metrics,
    }


# ============================================================
# 更新历史平均速率
# ============================================================
def update_average_user_rate(old_avg_rate, new_rate, slot_count):
    """
    递推更新用户历史平均速率。

    old_avg_rate:
        [K]
    new_rate:
        [K]
    slot_count:
        当前已经累计过的slot数量。
        如果slot_count=0，则new_avg = new_rate。
    """

    old_avg_rate = np.asarray(old_avg_rate, dtype=np.float32)
    new_rate = np.asarray(new_rate, dtype=np.float32)

    slot_count = int(slot_count)

    if slot_count <= 0:
        return new_rate.astype(np.float32)

    new_avg = (
        old_avg_rate * float(slot_count)
        + new_rate
    ) / float(slot_count + 1)

    return new_avg.astype(np.float32)


# ============================================================
# 简单benchmark：随机方向动作
# ============================================================
def random_direction_action(cfg, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    action = np.zeros(cfg.action_dim, dtype=np.float32)
    random_heading = rng.uniform(-np.pi, np.pi, size=cfg.uav_n).astype(np.float32)

    for m in range(cfg.uav_n):
        _set_heading_action(action, cfg, m, random_heading[m])

    return np.clip(action, cfg.action_low, cfg.action_high).astype(np.float32)


# ============================================================
# 简单benchmark：朝区域中心飞
# ============================================================
def center_direction_action(cfg, uav_pos, active_mask=None):
    """
    输出连续动作，使UAV朝区域中心飞。

    注意：
    action的前两维为二维方向向量 [cos(theta), sin(theta)]。
    """

    uav_pos = np.asarray(uav_pos, dtype=np.float32)

    if active_mask is None:
        active_mask = np.ones(cfg.uav_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    center = np.array(
        [cfg.area_size / 2.0, cfg.area_size / 2.0],
        dtype=np.float32
    )

    action = np.zeros(cfg.action_dim, dtype=np.float32)

    for m in range(cfg.uav_n):
        if not active_mask[m]:
            continue

        diff = center - uav_pos[m]
        theta = np.arctan2(diff[1], diff[0])
        _set_heading_action(action, cfg, m, theta)

    action = np.clip(
        action,
        cfg.action_low,
        cfg.action_high
    )

    return action.astype(np.float32)


# ============================================================
# 简单benchmark：朝各自最近用户飞
# ============================================================
def nearest_user_direction_action(cfg, uav_pos, user_pos, active_mask=None):
    """
    每架UAV朝距离自己最近的用户飞。

    该函数只生成方向动作，不做通信资源分配。
    """

    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)

    if active_mask is None:
        active_mask = np.ones(cfg.uav_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    action = np.zeros(cfg.action_dim, dtype=np.float32)

    for m in range(cfg.uav_n):
        if not active_mask[m]:
            continue

        diff = user_pos - uav_pos[m][None, :]
        dist = np.linalg.norm(diff, axis=1)

        nearest_k = int(np.argmin(dist))
        target = user_pos[nearest_k]

        direction = target - uav_pos[m]
        theta = np.arctan2(direction[1], direction[0])

        _set_heading_action(action, cfg, m, theta)

    action = np.clip(
        action,
        cfg.action_low,
        cfg.action_high
    )

    return action.astype(np.float32)


# ============================================================
# 简单benchmark：绕区域中心巡航
# ============================================================
def circular_direction_action(cfg, uav_pos, active_mask=None, clockwise=False):
    """
    每架UAV沿区域中心附近的切向方向飞行。

    该策略不使用用户位置，只提供一个稳定的区域巡航基线。
    """

    uav_pos = np.asarray(uav_pos, dtype=np.float32)

    if active_mask is None:
        active_mask = np.ones(cfg.uav_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    center = np.array(
        [cfg.area_size / 2.0, cfg.area_size / 2.0],
        dtype=np.float32
    )

    action = np.zeros(cfg.action_dim, dtype=np.float32)

    for m in range(cfg.uav_n):
        base = m * cfg.local_action_dim
        if not active_mask[m]:
            action[base] = 0.0
            continue

        radial = uav_pos[m] - center

        if np.linalg.norm(radial) <= 1e-6:
            theta = 2.0 * np.pi * float(m) / max(float(cfg.uav_n), 1.0)
        else:
            radial_angle = np.arctan2(radial[1], radial[0])
            if clockwise:
                theta = radial_angle - np.pi / 2.0
            else:
                theta = radial_angle + np.pi / 2.0

        theta = float(cfg.wrap_angle(theta))
        _set_heading_action(action, cfg, m, theta)

    action = np.clip(
        action,
        cfg.action_low,
        cfg.action_high
    )

    return action.astype(np.float32)


# ============================================================
# 简单benchmark：朝历史平均速率最低的用户飞
# ============================================================
def lowest_rate_user_direction_action(
    cfg,
    uav_pos,
    user_pos,
    avg_user_rate,
    active_mask=None
):
    """
    每架active UAV朝一个历史平均速率较低的用户飞。

    为了避免所有UAV都追同一个用户，这里按平均速率从低到高排序，
    然后依次分配给active UAV。
    """

    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    avg_user_rate = np.asarray(avg_user_rate, dtype=np.float32).reshape(-1)

    if active_mask is None:
        active_mask = np.ones(cfg.uav_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    action = np.zeros(cfg.action_dim, dtype=np.float32)
    sorted_user_idx = np.argsort(avg_user_rate)
    active_uav_idx = np.where(active_mask)[0]

    if sorted_user_idx.size <= 0:
        return action.astype(np.float32)

    for order, m in enumerate(active_uav_idx):
        user_rank = order % sorted_user_idx.size
        target_k = int(sorted_user_idx[user_rank])
        target = user_pos[target_k]

        base = m * cfg.local_action_dim
        direction = target - uav_pos[m]
        theta = np.arctan2(direction[1], direction[0])

        _set_heading_action(action, cfg, m, theta)

    action = np.clip(
        action,
        cfg.action_low,
        cfg.action_high
    )

    return action.astype(np.float32)

# ============================================================
# RB-level eMBB/mMTC allocation + TTI-level URLLC puncturing
# ============================================================
def _softmax_budget_to_rb(cfg, slice_budget_ratio, active_mask):
    m_n = int(cfg.uav_n)
    rb_n = int(getattr(cfg, "rb_n", 50))
    slice_budget_ratio = np.asarray(slice_budget_ratio, dtype=np.float32) if slice_budget_ratio is not None else None
    if slice_budget_ratio is None or slice_budget_ratio.shape != (m_n, int(cfg.slice_n)):
        slice_budget_ratio = np.zeros((m_n, int(cfg.slice_n)), dtype=np.float32)
        slice_budget_ratio[:, int(cfg.slice_embb)] = 0.75
        slice_budget_ratio[:, int(cfg.slice_mmtc)] = 0.25
    slice_budget_ratio = np.maximum(slice_budget_ratio, 0.0)
    rb_budget = np.zeros((m_n, int(cfg.slice_n)), dtype=np.int32)
    for m in range(m_n):
        if not active_mask[m]:
            continue
        e_ratio = float(slice_budget_ratio[m, int(cfg.slice_embb)])
        m_ratio = float(slice_budget_ratio[m, int(cfg.slice_mmtc)])
        denom = max(e_ratio + m_ratio, float(getattr(cfg, "slice_budget_eps", 1e-8)))
        mmtc_rb = int(np.round(rb_n * m_ratio / denom))
        mmtc_rb = int(np.clip(mmtc_rb, 0, rb_n))
        embb_rb = int(rb_n - mmtc_rb)
        rb_budget[m, int(cfg.slice_embb)] = embb_rb
        rb_budget[m, int(cfg.slice_mmtc)] = mmtc_rb
        rb_budget[m, int(cfg.slice_urllc)] = 0
    return rb_budget.astype(np.int32)


def _calc_rb_sinr_and_capacity(cfg, channel_gain, antenna_gain, active_mask):
    channel_gain = np.asarray(channel_gain, dtype=np.float32)
    antenna_gain = np.asarray(antenna_gain, dtype=np.float32)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    m_n, user_n = channel_gain.shape
    rb_bw = float(getattr(cfg, "rb_bandwidth_hz", 180e3))
    rb_power = float(getattr(cfg, "rb_power_w_each_uav", cfg.tx_power_w_each_uav / max(getattr(cfg, "rb_n", 1), 1)))
    tti_time = float(getattr(cfg, "tti_time", 1e-3))

    sinr_matrix = np.zeros((m_n, user_n), dtype=np.float32)
    rb_rate_matrix = np.zeros((m_n, user_n), dtype=np.float32)  # bps per RB
    rb_bits_matrix = np.zeros((m_n, user_n), dtype=np.float32)  # bits per RB per TTI
    desired_power_matrix = np.zeros((m_n, user_n), dtype=np.float32)
    interference_power_matrix = np.zeros((m_n, user_n), dtype=np.float32)
    noise_power_matrix = np.zeros((m_n, user_n), dtype=np.float32)

    rx_rb_power = rb_power * antenna_gain * channel_gain
    noise_power = rb_bw * float(cfg.noise_psd_w_per_hz)
    for m in range(m_n):
        if not active_mask[m]:
            continue
        for k in range(user_n):
            desired = float(rx_rb_power[m, k])
            interf = 0.0
            if bool(getattr(cfg, "full_frequency_reuse", True)):
                for mp in range(m_n):
                    if mp == m or not active_mask[mp]:
                        continue
                    interf += float(rx_rb_power[mp, k])
            denom = max(noise_power + interf, 1e-30)
            sinr = desired / denom
            eta = np.log2(1.0 + sinr)
            rb_rate = rb_bw * eta
            sinr_matrix[m, k] = float(sinr)
            rb_rate_matrix[m, k] = float(rb_rate)
            rb_bits_matrix[m, k] = float(rb_rate * tti_time)
            desired_power_matrix[m, k] = float(desired)
            interference_power_matrix[m, k] = float(interf)
            noise_power_matrix[m, k] = float(noise_power)
    return {
        "sinr": sinr_matrix.astype(np.float32),
        "rb_rate": rb_rate_matrix.astype(np.float32),
        "rb_bits_tti": rb_bits_matrix.astype(np.float32),
        "desired_power": desired_power_matrix.astype(np.float32),
        "interference_power": interference_power_matrix.astype(np.float32),
        "noise_power": noise_power_matrix.astype(np.float32),
    }


def _allocate_mmtc_rb(cfg, m, user_idx, rb_budget_m, rb_bits_tti, sinr_row, mmtc_active, mmtc_pending_bits):
    rb_alloc = {}
    if rb_budget_m <= 0 or len(user_idx) == 0:
        return rb_alloc
    tti_n = int(getattr(cfg, "tti_n_per_slot", max(1, round(cfg.slot_time / getattr(cfg, "tti_time", 1e-3)))))
    candidates = []
    for k in user_idx:
        k = int(k)
        if mmtc_active[k] <= 0.5:
            continue
        if sinr_row[k] + 1e-12 < float(cfg.mmtc_sinr_threshold):
            continue
        bits_per_rb_slot = float(rb_bits_tti[k]) * float(tti_n)
        if bits_per_rb_slot <= 1e-12:
            continue
        req_bits = max(float(mmtc_pending_bits[k]), float(cfg.mmtc_packet_bits))
        req_rb = int(np.ceil(req_bits / bits_per_rb_slot))
        req_rb = max(req_rb, 1)
        candidates.append((req_rb, -float(sinr_row[k]), k))
    # maximize number of successful active devices: satisfy low-RB users first
    candidates.sort()
    remaining = int(rb_budget_m)
    for req_rb, _, k in candidates:
        if req_rb <= remaining:
            rb_alloc[int(k)] = int(req_rb)
            remaining -= int(req_rb)
        if remaining <= 0:
            break
    return rb_alloc


def _allocate_embb_rb(cfg, m, user_idx, rb_budget_m, rb_rate_row, avg_user_rate=None):
    rb_alloc = {}
    if rb_budget_m <= 0 or len(user_idx) == 0:
        return rb_alloc
    user_idx = [int(k) for k in user_idx]
    rb_rate = np.asarray(rb_rate_row, dtype=np.float32)
    avg_user_rate = np.zeros_like(rb_rate, dtype=np.float32) if avg_user_rate is None else np.asarray(avg_user_rate, dtype=np.float32)

    # Drop users with zero link capacity.
    user_idx = [k for k in user_idx if rb_rate[k] > 1e-9]
    if len(user_idx) == 0:
        return rb_alloc

    # Give each eMBB user one RB if possible, then greedily allocate by marginal log utility.
    remaining = int(rb_budget_m)
    if remaining >= len(user_idx):
        for k in user_idx:
            rb_alloc[k] = 1
        remaining -= len(user_idx)
    else:
        # If RBs are fewer than users, give them to strongest RB-rate users.
        order = sorted(user_idx, key=lambda kk: float(rb_rate[kk]), reverse=True)
        for k in order[:remaining]:
            rb_alloc[k] = rb_alloc.get(k, 0) + 1
        return rb_alloc

    policy = str(getattr(cfg, "embb_rb_allocation_policy", "log_utility"))
    while remaining > 0:
        best_k = None
        best_gain = -np.inf
        for k in user_idx:
            if policy == "sum_rate":
                gain = float(rb_rate[k])
            else:
                # Proportional/log utility marginal benefit.
                current = float(avg_user_rate[k]) + float(rb_alloc.get(k, 0)) * float(rb_rate[k])
                gain = np.log1p(current + float(rb_rate[k])) - np.log1p(current)
            if gain > best_gain:
                best_gain = gain
                best_k = k
        rb_alloc[best_k] = rb_alloc.get(best_k, 0) + 1
        remaining -= 1
    return rb_alloc


def _generate_urllc_packets_for_slot(cfg, ep_id, step_idx):
    user_slice = np.asarray(cfg.user_slice, dtype=np.int32)
    urllc_idx = np.where(user_slice == int(cfg.slice_urllc))[0]
    tti_n = int(getattr(cfg, "tti_n_per_slot", 1000))
    lam = float(getattr(cfg, "urllc_arrival_rate", 10.0)) * float(cfg.slot_time)
    seed = int(cfg.seed) + int(ep_id) * 1000003 + int(step_idx) * 19391 + 44017
    rng = np.random.default_rng(seed)
    packets = []
    for k in urllc_idx:
        n = int(rng.poisson(lam=max(lam, 0.0)))
        if n <= 0:
            continue
        tti_arr = rng.integers(low=0, high=max(tti_n, 1), size=n)
        for a in tti_arr:
            packets.append({
                "user": int(k),
                "arrival_tti": int(a),
                "deadline_tti": int(getattr(cfg, "urllc_deadline_tti", 1)),
                "bits": float(getattr(cfg, "urllc_packet_bits", 512.0)),
            })
    packets.sort(key=lambda p: (int(p["arrival_tti"]), int(p["deadline_tti"]), int(p["user"])))
    return packets


def _apply_urllc_puncturing(cfg, packets, associated_uav, sinr_matrix, rb_bits_tti, rb_rate_matrix, embb_rb_base_user, active_mask):
    m_n, user_n = embb_rb_base_user.shape
    tti_n = int(getattr(cfg, "tti_n_per_slot", 1000))
    punctured_rb_user = np.zeros((m_n, user_n), dtype=np.int32)
    urllc_rb_user = np.zeros((m_n, user_n), dtype=np.int32)
    urllc_bits_user = np.zeros(user_n, dtype=np.float32)
    urllc_success_each = np.zeros(m_n, dtype=np.float32)
    urllc_violation_each = np.zeros(m_n, dtype=np.float32)
    punctured_rb_each = np.zeros(m_n, dtype=np.float32)

    success = 0
    violation = 0
    # group packets by TTI. First version has deadline_tti=1, but code is compatible with same-TTI processing.
    by_tti = {}
    for pkt in packets:
        by_tti.setdefault(int(pkt["arrival_tti"]), []).append(pkt)

    for tau in sorted(by_tti.keys()):
        embb_available = embb_rb_base_user.copy().astype(np.int32)
        pkts = by_tti[tau]
        # Determine serving UAV and RB demand; serve smaller RB demand first to maximize URLLC successes.
        enriched = []
        for pkt in pkts:
            k = int(pkt["user"])
            # choose active UAV with best SINR for URLLC reliability.
            valid_m = np.where(active_mask)[0]
            if valid_m.size <= 0:
                enriched.append((np.inf, k, -1, pkt))
                continue
            m = int(valid_m[np.argmax(sinr_matrix[valid_m, k])])
            c = float(rb_bits_tti[m, k])
            if c <= 1e-12:
                req = np.inf
            else:
                req = int(np.ceil(float(pkt["bits"]) / c))
                req = max(req, 1)
            enriched.append((req, k, m, pkt))
        enriched.sort(key=lambda x: (x[0], x[1]))

        for req, k, m, pkt in enriched:
            if m < 0 or (not np.isfinite(req)):
                violation += 1
                continue
            req = int(req)
            embb_idx = np.where(embb_available[m] > 0)[0]
            if embb_idx.size <= 0 or int(np.sum(embb_available[m, embb_idx])) < req:
                violation += 1
                urllc_violation_each[m] += 1.0
                continue
            # Puncture RBs with minimum eMBB instantaneous rate loss.
            for _ in range(req):
                embb_idx = np.where(embb_available[m] > 0)[0]
                if embb_idx.size <= 0:
                    break
                loss = rb_rate_matrix[m, embb_idx]
                j = int(embb_idx[int(np.argmin(loss))])
                embb_available[m, j] -= 1
                punctured_rb_user[m, j] += 1
                punctured_rb_each[m] += 1.0
            success += 1
            urllc_success_each[m] += 1.0
            urllc_rb_user[m, k] += req
            urllc_bits_user[k] += float(pkt["bits"])
    arrivals = int(len(packets))
    return {
        "arrivals": arrivals,
        "success": int(success),
        "violation": int(violation),
        "punctured_rb_user": punctured_rb_user.astype(np.int32),
        "urllc_rb_user": urllc_rb_user.astype(np.int32),
        "urllc_bits_user": urllc_bits_user.astype(np.float32),
        "urllc_success_each_uav": urllc_success_each.astype(np.float32),
        "urllc_violation_each_uav": urllc_violation_each.astype(np.float32),
        "punctured_rb_each_uav": punctured_rb_each.astype(np.float32),
    }


# Override original serve_one_slot with RB-level implementation.
def serve_one_slot(
    cfg,
    uav_pos,
    user_pos,
    residual_energy=None,
    active_mask=None,
    ep_id=0,
    step_idx=0,
    avg_user_rate=None,
    association_mode="max_signal",
    slice_budget_ratio=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
):
    """
    RB级通信流程（trajectory-only版本）：
    1) 计算信道、天线增益和关联；
    2) Actor不再决策eMBB/mMTC切片预算；每RB等功率；
    3) 每架UAV先用最小所需RB满足active且SINR达标的mMTC设备；
    4) 未被mMTC占用的RB全部给eMBB；
    5) URLLC在1 ms TTI级Poisson到达，只从eMBB RB中puncture；puncture只持续当前TTI；
    6) 输出puncturing后的eMBB有效速率、URLLC包成功率、mMTC成功率与mMTC瓶颈诊断。
    """
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    if not bool(getattr(cfg, "use_energy_management", False)):
        active_mask = np.ones(cfg.uav_n, dtype=bool)
    elif active_mask is None:
        if residual_energy is None:
            active_mask = np.ones(cfg.uav_n, dtype=bool)
        else:
            active_mask = np.asarray(residual_energy, dtype=np.float32) > float(cfg.uav_energy_threshold)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if avg_user_rate is None:
        avg_user_rate = np.zeros(cfg.user_n, dtype=np.float32)
    else:
        avg_user_rate = np.asarray(avg_user_rate, dtype=np.float32).reshape(-1)
    if mmtc_active is None:
        mmtc_active = np.zeros(cfg.user_n, dtype=np.float32)
    else:
        mmtc_active = np.asarray(mmtc_active, dtype=np.float32).reshape(-1)
    if mmtc_pending_bits is None:
        mmtc_pending_bits = np.zeros(cfg.user_n, dtype=np.float32)
    else:
        mmtc_pending_bits = np.asarray(mmtc_pending_bits, dtype=np.float32).reshape(-1)

    # 1) Channel and antenna
    channel_gain = cfg.calc_channel_gain(uav_pos=uav_pos, user_pos=user_pos, ep_id=ep_id, step_idx=step_idx)
    antenna_gain, main_lobe_mask = cfg.calc_antenna_gain_matrix(uav_pos=uav_pos, user_pos=user_pos)
    rx_signal_matrix = (cfg.tx_power_w_each_uav * antenna_gain * channel_gain).astype(np.float32)

    # 2) Association
    if association_mode == "max_signal":
        associated_uav, association_matrix = max_signal_user_association(cfg, rx_signal_matrix, main_lobe_mask, active_mask)
    elif association_mode == "nearest":
        associated_uav, association_matrix = nearest_uav_user_association(cfg, uav_pos, user_pos, main_lobe_mask, active_mask)
    else:
        raise ValueError(f"未知association_mode: {association_mode}")

    # 3) Per-RB SINR/capacity under equal per-RB power.
    rb_info = _calc_rb_sinr_and_capacity(cfg, channel_gain, antenna_gain, active_mask)
    sinr_matrix = rb_info["sinr"]
    rb_rate_matrix = rb_info["rb_rate"]      # bps per RB
    rb_bits_tti = rb_info["rb_bits_tti"]     # bits per RB per TTI

    m_n, user_n = association_matrix.shape
    user_slice = np.asarray(cfg.user_slice, dtype=np.int32)
    rb_n = int(getattr(cfg, "rb_n", 50))

    # 轨迹-only版本：不再使用Actor输出的slice_budget_ratio。
    # rb_budget记录每个UAV本slot实际被mMTC/eMBB占用的RB数量，方便日志和诊断。
    rb_budget = np.zeros((m_n, int(cfg.slice_n)), dtype=np.int32)
    rb_alloc_base = np.zeros((m_n, user_n), dtype=np.int32)
    rb_alloc_mmtc = np.zeros((m_n, user_n), dtype=np.int32)
    rb_alloc_embb_base = np.zeros((m_n, user_n), dtype=np.int32)
    priority_matrix = np.zeros((m_n, user_n), dtype=np.float32)

    for m in range(m_n):
        if not active_mask[m]:
            continue
        assoc_idx = np.where(association_matrix[m] > 0.5)[0]
        embb_idx = assoc_idx[user_slice[assoc_idx] == int(cfg.slice_embb)]
        mmtc_idx = assoc_idx[user_slice[assoc_idx] == int(cfg.slice_mmtc)]

        # mMTC: 不再由Actor给预算，而是从本UAV全部RB中按需优先占用。
        # active且SINR达标的mMTC用户按照所需RB从小到大服务，以最大化成功设备数。
        mmtc_alloc = _allocate_mmtc_rb(
            cfg,
            m,
            mmtc_idx,
            rb_n,
            rb_bits_tti[m],
            sinr_matrix[m],
            mmtc_active,
            mmtc_pending_bits,
        )
        used_mmtc = int(np.clip(sum(mmtc_alloc.values()), 0, rb_n))
        embb_budget = int(max(0, rb_n - used_mmtc))

        rb_budget[m, int(cfg.slice_mmtc)] = used_mmtc
        rb_budget[m, int(cfg.slice_embb)] = embb_budget
        rb_budget[m, int(cfg.slice_urllc)] = 0

        for k, n in mmtc_alloc.items():
            rb_alloc_mmtc[m, k] = int(n)
            rb_alloc_base[m, k] = int(n)
            priority_matrix[m, k] = 1.0 / float(max(n, 1))

        # eMBB: 使用mMTC按需占用后的剩余RB。
        embb_alloc = _allocate_embb_rb(cfg, m, embb_idx, embb_budget, rb_rate_matrix[m], avg_user_rate)
        for k, n in embb_alloc.items():
            rb_alloc_embb_base[m, k] = int(n)
            rb_alloc_base[m, k] = int(n)
            priority_matrix[m, k] = float(rb_rate_matrix[m, k])

    # 4) URLLC arrival and eMBB-only puncturing at TTI level.
    packets = _generate_urllc_packets_for_slot(cfg, ep_id=ep_id, step_idx=step_idx)
    punct = _apply_urllc_puncturing(
        cfg=cfg,
        packets=packets,
        associated_uav=associated_uav,
        sinr_matrix=sinr_matrix,
        rb_bits_tti=rb_bits_tti,
        rb_rate_matrix=rb_rate_matrix,
        embb_rb_base_user=rb_alloc_embb_base,
        active_mask=active_mask,
    )
    punctured_rb_user = punct["punctured_rb_user"]
    urllc_bits_user = punct["urllc_bits_user"]

    # 5) Convert RB allocations into user rates over the 1s control slot.
    tti_n = int(getattr(cfg, "tti_n_per_slot", 1000))
    slot_time = float(cfg.slot_time)
    embb_base_user_rate = np.zeros(user_n, dtype=np.float32)
    embb_eff_user_rate = np.zeros(user_n, dtype=np.float32)
    mmtc_user_rate = np.zeros(user_n, dtype=np.float32)
    urllc_user_rate = np.zeros(user_n, dtype=np.float32)
    user_rate = np.zeros(user_n, dtype=np.float32)

    for m in range(m_n):
        for k in range(user_n):
            rb_rate = float(rb_rate_matrix[m, k])
            if rb_alloc_embb_base[m, k] > 0:
                base_rate = float(rb_alloc_embb_base[m, k]) * rb_rate
                loss_rate = float(punctured_rb_user[m, k]) * rb_rate / max(float(tti_n), 1.0)
                eff_rate = max(base_rate - loss_rate, 0.0)
                embb_base_user_rate[k] += base_rate
                embb_eff_user_rate[k] += eff_rate
                user_rate[k] += eff_rate
            if rb_alloc_mmtc[m, k] > 0:
                rate = float(rb_alloc_mmtc[m, k]) * rb_rate
                mmtc_user_rate[k] += rate
                user_rate[k] += rate
    if slot_time > 0.0:
        urllc_user_rate = urllc_bits_user / slot_time
        user_rate += urllc_user_rate.astype(np.float32)

    # 6) mMTC success statistics and bottleneck diagnostics.
    #
    # 诊断目标：判断mMTC卡在某个成功率附近时，到底是“RB不够”，
    # 还是“即使给足RB也因为SINR/覆盖不足而无法成功”。
    mmtc_success = 0
    mmtc_success_each = np.zeros(m_n, dtype=np.float32)
    mmtc_active_each = np.zeros(m_n, dtype=np.float32)
    mmtc_sinr_ok_each = np.zeros(m_n, dtype=np.float32)
    mmtc_bits_ok_each = np.zeros(m_n, dtype=np.float32)

    mmtc_idx_all = np.where(user_slice == int(cfg.slice_mmtc))[0]
    active_count = 0
    mmtc_sinr_ok_count = 0
    mmtc_bits_ok_count = 0
    mmtc_full_rb_upper_count = 0
    mmtc_fail_sinr_count = 0
    mmtc_fail_rb_given_sinr_ok_count = 0
    mmtc_fail_both_count = 0
    mmtc_required_rb_sum = 0.0
    mmtc_allocated_rb_sum = 0.0
    mmtc_required_rb_valid_count = 0

    tti_n_float = float(max(int(getattr(cfg, "tti_n_per_slot", 1000)), 1))
    rb_n_total = int(getattr(cfg, "rb_n", 50))

    for k in mmtc_idx_all:
        k = int(k)
        if mmtc_active[k] <= 0.5:
            continue

        active_count += 1
        m = int(associated_uav[k])
        if 0 <= m < m_n:
            mmtc_active_each[m] += 1.0

        req_bits = max(float(mmtc_pending_bits[k]), float(cfg.mmtc_packet_bits))
        allocated_rb = int(rb_alloc_mmtc[m, k]) if 0 <= m < m_n else 0
        mmtc_allocated_rb_sum += float(allocated_rb)

        if 0 <= m < m_n:
            rb_bits_slot = float(rb_bits_tti[m, k]) * tti_n_float
            if rb_bits_slot > 1e-12:
                req_rb = int(np.ceil(req_bits / rb_bits_slot))
                req_rb = max(req_rb, 1)
                mmtc_required_rb_sum += float(req_rb)
                mmtc_required_rb_valid_count += 1
            else:
                req_rb = np.inf
        else:
            rb_bits_slot = 0.0
            req_rb = np.inf

        bits_ok = float(mmtc_user_rate[k]) * slot_time + 1e-9 >= req_bits
        sinr_ok = (0 <= m < m_n) and float(sinr_matrix[m, k]) >= float(cfg.mmtc_sinr_threshold)

        # SINR upper bound：如果给足RB，仅受SINR门限约束时能成功的比例。
        if sinr_ok:
            mmtc_sinr_ok_count += 1
            if 0 <= m < m_n:
                mmtc_sinr_ok_each[m] += 1.0

        # 当前RB预算下，是否传完bits；这里不考虑SINR门限。
        if bits_ok:
            mmtc_bits_ok_count += 1
            if 0 <= m < m_n:
                mmtc_bits_ok_each[m] += 1.0

        # Full-RB upper bound：假设该UAV全部RB都可给这个mMTC用户，是否既满足SINR又能传完。
        # 若该值仍约等于当前成功率，说明继续增加mMTC预算意义不大。
        if sinr_ok and np.isfinite(req_rb) and req_rb <= rb_n_total:
            mmtc_full_rb_upper_count += 1

        if bits_ok and sinr_ok:
            mmtc_success += 1
            if 0 <= m < m_n:
                mmtc_success_each[m] += 1.0
        else:
            if (not sinr_ok) and (not bits_ok):
                mmtc_fail_both_count += 1
            elif not sinr_ok:
                mmtc_fail_sinr_count += 1
            elif sinr_ok and (not bits_ok):
                # 这部分才是真正“RB不够”导致的失败。
                mmtc_fail_rb_given_sinr_ok_count += 1

    mmtc_success_ratio = float(mmtc_success / max(active_count, 1)) if active_count > 0 else 1.0
    mmtc_active_ratio = float(active_count / max(len(mmtc_idx_all), 1))
    mmtc_sinr_ok_ratio = float(mmtc_sinr_ok_count / max(active_count, 1)) if active_count > 0 else 1.0
    mmtc_bits_ok_ratio = float(mmtc_bits_ok_count / max(active_count, 1)) if active_count > 0 else 1.0
    mmtc_full_rb_upper_ratio = float(mmtc_full_rb_upper_count / max(active_count, 1)) if active_count > 0 else 1.0
    mmtc_fail_sinr_ratio = float(mmtc_fail_sinr_count / max(active_count, 1)) if active_count > 0 else 0.0
    mmtc_fail_rb_ratio = float(mmtc_fail_rb_given_sinr_ok_count / max(active_count, 1)) if active_count > 0 else 0.0
    mmtc_fail_both_ratio = float(mmtc_fail_both_count / max(active_count, 1)) if active_count > 0 else 0.0
    mmtc_avg_required_rb = float(mmtc_required_rb_sum / max(mmtc_required_rb_valid_count, 1)) if mmtc_required_rb_valid_count > 0 else 0.0
    mmtc_avg_allocated_rb = float(mmtc_allocated_rb_sum / max(active_count, 1)) if active_count > 0 else 0.0
    mmtc_total_budget_rb = float(np.sum(rb_budget[:, int(cfg.slice_mmtc)]))
    mmtc_total_available_rb = float(np.sum(active_mask.astype(np.float32)) * float(rb_n_total))

    arrivals = int(punct["arrivals"])
    success = int(punct["success"])
    violation = int(punct["violation"])
    urllc_success_ratio = float(success / max(arrivals, 1)) if arrivals > 0 else 1.0
    urllc_violation_ratio = float(violation / max(arrivals, 1)) if arrivals > 0 else 0.0
    total_embb_rb_time = float(np.sum(rb_alloc_embb_base) * max(tti_n, 1))
    total_punctured_rb = float(np.sum(punctured_rb_user))
    puncture_ratio = total_punctured_rb / max(total_embb_rb_time, 1.0)

    # 7) Overall metrics.
    metrics = calc_slot_metrics(cfg, user_rate, association_matrix, associated_uav, avg_user_rate)
    metrics["embb_base_sum_rate"] = float(np.sum(embb_base_user_rate))
    metrics["embb_effective_sum_rate"] = float(np.sum(embb_eff_user_rate))
    metrics["embb_loss_rate"] = float(max(np.sum(embb_base_user_rate) - np.sum(embb_eff_user_rate), 0.0))
    metrics["puncture_ratio"] = float(np.clip(puncture_ratio, 0.0, 1.0))
    metrics["urllc_arrival_count"] = int(arrivals)
    metrics["urllc_success_count"] = int(success)
    metrics["urllc_violation_count"] = int(violation)
    metrics["mmtc_active_count"] = int(active_count)
    metrics["mmtc_success_count"] = int(mmtc_success)
    metrics["mmtc_sinr_ok_count"] = int(mmtc_sinr_ok_count)
    metrics["mmtc_bits_ok_count"] = int(mmtc_bits_ok_count)
    metrics["mmtc_full_rb_upper_count"] = int(mmtc_full_rb_upper_count)
    metrics["mmtc_fail_sinr_count"] = int(mmtc_fail_sinr_count)
    metrics["mmtc_fail_rb_count"] = int(mmtc_fail_rb_given_sinr_ok_count)
    metrics["mmtc_fail_both_count"] = int(mmtc_fail_both_count)
    metrics["mmtc_sinr_ok_ratio"] = float(np.clip(mmtc_sinr_ok_ratio, 0.0, 1.0))
    metrics["mmtc_bits_ok_ratio"] = float(np.clip(mmtc_bits_ok_ratio, 0.0, 1.0))
    metrics["mmtc_full_rb_upper_ratio"] = float(np.clip(mmtc_full_rb_upper_ratio, 0.0, 1.0))
    metrics["mmtc_fail_sinr_ratio"] = float(np.clip(mmtc_fail_sinr_ratio, 0.0, 1.0))
    metrics["mmtc_fail_rb_ratio"] = float(np.clip(mmtc_fail_rb_ratio, 0.0, 1.0))
    metrics["mmtc_fail_both_ratio"] = float(np.clip(mmtc_fail_both_ratio, 0.0, 1.0))
    metrics["mmtc_avg_required_rb"] = float(mmtc_avg_required_rb)
    metrics["mmtc_avg_allocated_rb"] = float(mmtc_avg_allocated_rb)
    metrics["mmtc_total_budget_rb"] = float(mmtc_total_budget_rb)
    metrics["mmtc_total_available_rb"] = float(mmtc_total_available_rb)

    slice_budget_bandwidth = np.zeros((m_n, int(cfg.slice_n)), dtype=np.float32)
    slice_budget_power = np.zeros((m_n, int(cfg.slice_n)), dtype=np.float32)
    rb_bw = float(getattr(cfg, "rb_bandwidth_hz", 180e3))
    rb_power = float(getattr(cfg, "rb_power_w_each_uav", cfg.tx_power_w_each_uav / max(getattr(cfg, "rb_n", 1), 1)))
    for m in range(m_n):
        for sl in range(int(cfg.slice_n)):
            slice_budget_bandwidth[m, sl] = float(rb_budget[m, sl]) * rb_bw
            slice_budget_power[m, sl] = float(rb_budget[m, sl]) * rb_power

    bandwidth_alloc = rb_alloc_base.astype(np.float32) * rb_bw
    power_alloc = rb_alloc_base.astype(np.float32) * rb_power
    user_count_each_uav = np.sum(association_matrix > 0.5, axis=1).astype(np.int32)

    traffic_metrics = {
        "urllc_success_ratio": float(np.clip(urllc_success_ratio, 0.0, 1.0)),
        "urllc_violation_ratio": float(np.clip(urllc_violation_ratio, 0.0, 1.0)),
        "urllc_success_count": int(success),
        "urllc_violation_count": int(violation),
        "urllc_arrival_count": int(arrivals),
        "urllc_success_each_uav": punct["urllc_success_each_uav"].astype(np.float32),
        "urllc_violation_each_uav": punct["urllc_violation_each_uav"].astype(np.float32),
        "mmtc_success_ratio": float(np.clip(mmtc_success_ratio, 0.0, 1.0)),
        "mmtc_active_ratio": float(np.clip(mmtc_active_ratio, 0.0, 1.0)),
        "mmtc_coverage_ratio": float(np.clip(mmtc_success_ratio, 0.0, 1.0)),
        "mmtc_success_count": int(mmtc_success),
        "mmtc_active_count": int(active_count),
        "mmtc_sinr_ok_count": int(mmtc_sinr_ok_count),
        "mmtc_bits_ok_count": int(mmtc_bits_ok_count),
        "mmtc_full_rb_upper_count": int(mmtc_full_rb_upper_count),
        "mmtc_fail_sinr_count": int(mmtc_fail_sinr_count),
        "mmtc_fail_rb_count": int(mmtc_fail_rb_given_sinr_ok_count),
        "mmtc_fail_both_count": int(mmtc_fail_both_count),
        "mmtc_sinr_ok_ratio": float(np.clip(mmtc_sinr_ok_ratio, 0.0, 1.0)),
        "mmtc_bits_ok_ratio": float(np.clip(mmtc_bits_ok_ratio, 0.0, 1.0)),
        "mmtc_full_rb_upper_ratio": float(np.clip(mmtc_full_rb_upper_ratio, 0.0, 1.0)),
        "mmtc_fail_sinr_ratio": float(np.clip(mmtc_fail_sinr_ratio, 0.0, 1.0)),
        "mmtc_fail_rb_ratio": float(np.clip(mmtc_fail_rb_ratio, 0.0, 1.0)),
        "mmtc_fail_both_ratio": float(np.clip(mmtc_fail_both_ratio, 0.0, 1.0)),
        "mmtc_avg_required_rb": float(mmtc_avg_required_rb),
        "mmtc_avg_allocated_rb": float(mmtc_avg_allocated_rb),
        "mmtc_total_budget_rb": float(mmtc_total_budget_rb),
        "mmtc_total_available_rb": float(mmtc_total_available_rb),
        "mmtc_success_each_uav": mmtc_success_each.astype(np.float32),
        "mmtc_active_each_uav": mmtc_active_each.astype(np.float32),
        "mmtc_sinr_ok_each_uav": mmtc_sinr_ok_each.astype(np.float32),
        "mmtc_bits_ok_each_uav": mmtc_bits_ok_each.astype(np.float32),
        "punctured_rb_each_uav": punct["punctured_rb_each_uav"].astype(np.float32),
        "puncture_ratio": float(np.clip(puncture_ratio, 0.0, 1.0)),
    }

    return {
        "channel_gain": channel_gain.astype(np.float32),
        "antenna_gain": antenna_gain.astype(np.float32),
        "main_lobe_mask": main_lobe_mask.astype(bool),
        "rx_signal_matrix": rx_signal_matrix.astype(np.float32),
        "associated_uav": associated_uav.astype(np.int32),
        "association": association_matrix.astype(np.float32),
        "bandwidth_alloc": bandwidth_alloc.astype(np.float32),
        "power_alloc": power_alloc.astype(np.float32),
        "rb_alloc_base": rb_alloc_base.astype(np.int32),
        "rb_alloc_embb_base": rb_alloc_embb_base.astype(np.int32),
        "rb_alloc_mmtc": rb_alloc_mmtc.astype(np.int32),
        "rb_punctured": punctured_rb_user.astype(np.int32),
        "rb_budget": rb_budget.astype(np.int32),
        "user_count_each_uav": user_count_each_uav.astype(np.int32),
        "slice_budget_bandwidth": slice_budget_bandwidth.astype(np.float32),
        "slice_budget_power": slice_budget_power.astype(np.float32),
        "priority_matrix": priority_matrix.astype(np.float32),
        "user_rate": user_rate.astype(np.float32),
        "embb_base_user_rate": embb_base_user_rate.astype(np.float32),
        "embb_effective_user_rate": embb_eff_user_rate.astype(np.float32),
        "embb_loss_user_rate": np.maximum(embb_base_user_rate - embb_eff_user_rate, 0.0).astype(np.float32),
        "mmtc_user_rate": mmtc_user_rate.astype(np.float32),
        "urllc_user_rate": urllc_user_rate.astype(np.float32),
        "sinr": sinr_matrix.astype(np.float32),
        "rb_rate": rb_rate_matrix.astype(np.float32),
        "rb_bits_tti": rb_bits_tti.astype(np.float32),
        "desired_power": rb_info["desired_power"].astype(np.float32),
        "interference_power": rb_info["interference_power"].astype(np.float32),
        "noise_power": rb_info["noise_power"].astype(np.float32),
        "interference_psd": calc_average_inter_uav_interference_psd(cfg, channel_gain, antenna_gain, active_mask),
        "active_mask": active_mask.astype(bool),
        "metrics": metrics,
        "traffic_metrics": traffic_metrics,
    }

# ============================================================================
# Finite-data equal-RB service model (runtime override)
# ============================================================================
def _finite_equal_rb_allocation(cfg, association_matrix, rb_rate_matrix, active_mask):
    """Allocate integer RBs as evenly as possible among associated users."""
    association_matrix = np.asarray(association_matrix, dtype=np.float32)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    m_n, user_n = association_matrix.shape
    rb_n = int(getattr(cfg, "rb_n", 50))
    rb_alloc = np.zeros((m_n, user_n), dtype=np.int32)

    for m in range(m_n):
        if not active_mask[m]:
            continue
        idx = np.where(association_matrix[m] > 0.5)[0]
        if idx.size == 0:
            continue
        base = rb_n // int(idx.size)
        rem = rb_n % int(idx.size)
        rb_alloc[m, idx] = base
        if rem > 0:
            # The remainder is given to the strongest per-RB links.  The
            # difference is at most one RB, so the allocation remains equal.
            order = idx[np.argsort(-rb_rate_matrix[m, idx], kind="stable")]
            rb_alloc[m, order[:rem]] += 1
    return rb_alloc.astype(np.int32)


def _finite_association(cfg, rx_signal_matrix, main_lobe_mask, active_mask, unfinished_mask):
    associated_uav, association = max_signal_user_association(
        cfg=cfg,
        rx_signal_matrix=rx_signal_matrix,
        main_lobe_mask=main_lobe_mask,
        active_mask=active_mask,
    )
    unfinished_mask = np.asarray(unfinished_mask, dtype=bool).reshape(-1)
    completed = ~unfinished_mask
    if np.any(completed):
        associated_uav[completed] = -1
        association[:, completed] = 0.0
    return associated_uav.astype(np.int32), association.astype(np.float32)


def _finite_rate_result(cfg, channel_gain, antenna_gain, association, active_mask):
    # A UAV with no unfinished associated user does not transmit data RBs and
    # therefore should not create full-band interference in this slot.
    tx_mask = np.asarray(active_mask, dtype=bool).reshape(-1) & (
        np.sum(np.asarray(association) > 0.5, axis=1) > 0
    )
    rb_info = _calc_rb_sinr_and_capacity(
        cfg=cfg,
        channel_gain=channel_gain,
        antenna_gain=antenna_gain,
        active_mask=tx_mask,
    )
    rb_alloc = _finite_equal_rb_allocation(
        cfg=cfg,
        association_matrix=association,
        rb_rate_matrix=rb_info["rb_rate"],
        active_mask=active_mask,
    )
    user_rate = np.sum(rb_alloc.astype(np.float32) * rb_info["rb_rate"], axis=0)
    rb_bw = float(getattr(cfg, "rb_bandwidth_hz", 180e3))
    rb_power = float(
        getattr(
            cfg,
            "rb_power_w_each_uav",
            cfg.tx_power_w_each_uav / max(int(getattr(cfg, "rb_n", 1)), 1),
        )
    )
    return {
        **rb_info,
        "rb_alloc": rb_alloc.astype(np.int32),
        "user_rate": user_rate.astype(np.float32),
        "bandwidth_alloc": (rb_alloc.astype(np.float32) * rb_bw).astype(np.float32),
        "power_alloc": (rb_alloc.astype(np.float32) * rb_power).astype(np.float32),
        "tx_mask": tx_mask.astype(bool),
    }


def estimate_reference_user_rate(cfg, uav_pos, user_pos, active_mask=None):
    """
    Legacy diagnostic helper retained for backward compatibility.

    The current task samples file sizes directly and does not call this helper.
    It computes an expected large-scale-rate snapshot with the environment's
    maximum-signal association and equal-RB/full-reuse model.
    """
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    if active_mask is None:
        active_mask = np.ones(cfg.uav_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    channel_gain = cfg.calc_large_scale_channel_gain(uav_pos, user_pos)
    antenna_gain, main_lobe_mask = cfg.calc_antenna_gain_matrix(uav_pos, user_pos)
    rx_signal = cfg.tx_power_w_each_uav * antenna_gain * channel_gain
    unfinished = np.ones(cfg.user_n, dtype=bool)
    _, association = _finite_association(
        cfg, rx_signal, main_lobe_mask, active_mask, unfinished
    )
    rate_result = _finite_rate_result(
        cfg, channel_gain, antenna_gain, association, active_mask
    )
    return rate_result["user_rate"].astype(np.float32)


def serve_one_slot(
    cfg,
    uav_pos,
    user_pos,
    residual_energy=None,
    active_mask=None,
    ep_id=0,
    step_idx=0,
    avg_user_rate=None,
    association_mode="max_signal",
    slice_budget_ratio=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
    unfinished_mask=None,
    remaining_data_bits=None,
):
    """Serve one 1-s slot for homogeneous finite-data users."""
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    if active_mask is None or not bool(getattr(cfg, "use_energy_management", False)):
        active_mask = np.ones(cfg.uav_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if unfinished_mask is None:
        unfinished_mask = np.ones(cfg.user_n, dtype=bool)
    else:
        unfinished_mask = np.asarray(unfinished_mask, dtype=bool).reshape(-1)
    if remaining_data_bits is None:
        remaining_data_bits = np.full(cfg.user_n, np.inf, dtype=np.float32)
    else:
        remaining_data_bits = np.maximum(
            np.asarray(remaining_data_bits, dtype=np.float32).reshape(-1), 0.0
        )

    channel_gain = cfg.calc_channel_gain(
        uav_pos=uav_pos,
        user_pos=user_pos,
        ep_id=ep_id,
        step_idx=step_idx,
    )
    antenna_gain, main_lobe_mask = cfg.calc_antenna_gain_matrix(uav_pos, user_pos)
    rx_signal_matrix = (
        cfg.tx_power_w_each_uav * antenna_gain * channel_gain
    ).astype(np.float32)

    if association_mode != "max_signal":
        raise ValueError("finite-data版本只保留最大接收信号关联。")
    associated_uav, association = _finite_association(
        cfg,
        rx_signal_matrix,
        main_lobe_mask,
        active_mask,
        unfinished_mask,
    )
    rate_result = _finite_rate_result(
        cfg, channel_gain, antenna_gain, association, active_mask
    )
    user_rate = rate_result["user_rate"].astype(np.float32)
    user_rate[~unfinished_mask] = 0.0

    capacity_bits = user_rate * float(cfg.slot_time)
    served_bits_user = np.minimum(capacity_bits, remaining_data_bits).astype(np.float32)
    served_bits_user[~unfinished_mask] = 0.0

    unfinished_n = int(np.sum(unfinished_mask))
    active_rates = user_rate[unfinished_mask]
    sum_rate = float(np.sum(user_rate))
    mean_rate = float(np.mean(active_rates)) if active_rates.size > 0 else 0.0
    served_bits = float(np.sum(served_bits_user))
    user_count_each_uav = np.sum(association > 0.5, axis=1).astype(np.int32)

    rb_alloc = rate_result["rb_alloc"]
    rb_budget = np.zeros((cfg.uav_n, int(getattr(cfg, "slice_n", 3))), dtype=np.int32)
    rb_budget[:, 0] = np.sum(rb_alloc, axis=1).astype(np.int32)
    rb_bw = float(getattr(cfg, "rb_bandwidth_hz", 180e3))
    rb_power = float(getattr(cfg, "rb_power_w_each_uav", cfg.tx_power_w_each_uav / max(cfg.rb_n, 1)))
    slice_budget_bandwidth = rb_budget.astype(np.float32) * rb_bw
    slice_budget_power = rb_budget.astype(np.float32) * rb_power

    fairness_values = served_bits_user[unfinished_mask]
    fairness = float(cfg.calc_jain_fairness(fairness_values)) if fairness_values.size > 0 else 1.0
    metrics = {
        "sum_rate": sum_rate,
        "mean_rate": mean_rate,
        "served_bits": served_bits,
        "fairness": fairness,
        "instantaneous_fairness": fairness,
        "associated_user_n": int(np.sum(associated_uav >= 0)),
        "associated_ratio": float(np.sum(associated_uav >= 0) / max(unfinished_n, 1)) if unfinished_n > 0 else 1.0,
        "active_link_n": int(np.sum(association > 0.5)),
        "min_rate": float(np.min(active_rates)) if active_rates.size > 0 else 0.0,
        "max_rate": float(np.max(active_rates)) if active_rates.size > 0 else 0.0,
        "unfinished_user_n": unfinished_n,
    }

    neutral_traffic = {
        "urllc_success_ratio": 1.0,
        "urllc_violation_ratio": 0.0,
        "urllc_success_count": 0,
        "urllc_violation_count": 0,
        "urllc_arrival_count": 0,
        "urllc_success_each_uav": np.zeros(cfg.uav_n, dtype=np.float32),
        "urllc_violation_each_uav": np.zeros(cfg.uav_n, dtype=np.float32),
        "mmtc_success_ratio": 1.0,
        "mmtc_active_ratio": 0.0,
        "mmtc_coverage_ratio": 1.0,
        "mmtc_success_count": 0,
        "mmtc_active_count": 0,
        "mmtc_success_each_uav": np.zeros(cfg.uav_n, dtype=np.float32),
        "mmtc_active_each_uav": np.zeros(cfg.uav_n, dtype=np.float32),
        "puncture_ratio": 0.0,
    }

    return {
        "channel_gain": channel_gain.astype(np.float32),
        "antenna_gain": antenna_gain.astype(np.float32),
        "main_lobe_mask": main_lobe_mask.astype(bool),
        "rx_signal_matrix": rx_signal_matrix.astype(np.float32),
        "associated_uav": associated_uav.astype(np.int32),
        "association": association.astype(np.float32),
        "bandwidth_alloc": rate_result["bandwidth_alloc"].astype(np.float32),
        "power_alloc": rate_result["power_alloc"].astype(np.float32),
        "rb_alloc_base": rb_alloc.astype(np.int32),
        "rb_alloc_embb_base": rb_alloc.astype(np.int32),
        "rb_alloc_mmtc": np.zeros_like(rb_alloc, dtype=np.int32),
        "rb_punctured": np.zeros_like(rb_alloc, dtype=np.int32),
        "rb_budget": rb_budget.astype(np.int32),
        "user_count_each_uav": user_count_each_uav,
        "slice_budget_bandwidth": slice_budget_bandwidth.astype(np.float32),
        "slice_budget_power": slice_budget_power.astype(np.float32),
        "priority_matrix": association.astype(np.float32),
        "user_rate": user_rate.astype(np.float32),
        "served_bits_user": served_bits_user.astype(np.float32),
        "capacity_bits_user": capacity_bits.astype(np.float32),
        "embb_base_user_rate": user_rate.astype(np.float32),
        "embb_effective_user_rate": user_rate.astype(np.float32),
        "embb_loss_user_rate": np.zeros(cfg.user_n, dtype=np.float32),
        "mmtc_user_rate": np.zeros(cfg.user_n, dtype=np.float32),
        "urllc_user_rate": np.zeros(cfg.user_n, dtype=np.float32),
        "sinr": rate_result["sinr"].astype(np.float32),
        "rb_rate": rate_result["rb_rate"].astype(np.float32),
        "rb_bits_tti": rate_result["rb_bits_tti"].astype(np.float32),
        "desired_power": rate_result["desired_power"].astype(np.float32),
        "interference_power": rate_result["interference_power"].astype(np.float32),
        "noise_power": rate_result["noise_power"].astype(np.float32),
        "interference_psd": calc_average_inter_uav_interference_psd(
            cfg, channel_gain, antenna_gain, rate_result["tx_mask"]
        ),
        "active_mask": active_mask.astype(bool),
        "metrics": metrics,
        "traffic_metrics": neutral_traffic,
    }

# ============================================================================
# Runtime override: eMBB finite files, pure URLLC puncturing, mMTC connectivity
# ============================================================================
def _allocate_embb_completion_time_rb(
    cfg,
    association,
    rb_rate_matrix,
    remaining_data_bits,
    active_mask,
):
    """Low-complexity RB allocation aligned with minimum eMBB completion time.

    Every associated unfinished eMBB user first receives one RB when possible.
    Remaining RBs are greedily assigned to the user with the largest predicted
    remaining completion time, while never exceeding the RB count needed to
    finish that user within the current slot.
    """
    association = np.asarray(association, dtype=np.float32)
    rb_rate_matrix = np.asarray(rb_rate_matrix, dtype=np.float32)
    remaining_data_bits = np.asarray(remaining_data_bits, dtype=np.float32).reshape(-1)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    m_n, user_n = association.shape
    rb_n = int(cfg.rb_n)
    slot_time = float(cfg.slot_time)
    rb_alloc = np.zeros((m_n, user_n), dtype=np.int32)

    for m in range(m_n):
        if not active_mask[m]:
            continue
        idx = np.where(association[m] > 0.5)[0]
        if idx.size == 0:
            continue
        valid = [
            int(k) for k in idx
            if rb_rate_matrix[m, k] > 1e-9
            and remaining_data_bits[k] > float(cfg.task_data_eps_bits)
        ]
        if not valid:
            continue

        need = {
            k: max(
                1,
                int(np.ceil(
                    float(remaining_data_bits[k])
                    / max(float(rb_rate_matrix[m, k]) * slot_time, 1e-12)
                )),
            )
            for k in valid
        }
        remaining = rb_n

        # Ensure broad service when RBs permit; if not, prioritize the largest
        # normalized workload D/C.
        order = sorted(
            valid,
            key=lambda k: float(remaining_data_bits[k])
            / max(float(rb_rate_matrix[m, k]) * slot_time, 1e-12),
            reverse=True,
        )
        for k in order[: min(len(order), remaining)]:
            rb_alloc[m, k] = 1
            remaining -= 1

        while remaining > 0:
            candidates = [k for k in valid if rb_alloc[m, k] < need[k]]
            if not candidates:
                break
            # Current predicted completion time under the already assigned RBs.
            best_k = max(
                candidates,
                key=lambda k: float(remaining_data_bits[k])
                / max(
                    float(rb_alloc[m, k])
                    * float(rb_rate_matrix[m, k])
                    * slot_time,
                    1e-12,
                ),
            )
            rb_alloc[m, best_k] += 1
            remaining -= 1

    return rb_alloc.astype(np.int32)


def _apply_limited_urllc_puncturing(
    cfg,
    packets,
    sinr_matrix,
    rb_bits_tti,
    rb_rate_matrix,
    embb_rb_base_user,
    active_mask,
):
    """Serve FBL URLLC with idle-RB-first access and eMBB puncturing.

    No RB is statically reserved for URLLC and there is no additional
    puncturing-ratio cap.  In each UAV/TTI, the physical pool of ``rb_n`` RBs
    is the only resource upper bound.  A URLLC packet first uses RBs left idle
    by the completion-time-aware eMBB allocation; only the remaining demand
    punctures already assigned eMBB RBs.  When multiple UAVs can serve a
    packet, the scheduler first minimizes the number of eMBB RBs that must be
    punctured, then the total required RB count, and finally prefers the higher
    SINR link.

    The URLLC RB requirement is obtained from the finite-blocklength normal
    approximation via ``cfg.calc_urllc_required_rb``.  Packets arriving in the
    same TTI are processed in increasing best-link RB demand so that the finite
    physical pool serves as many one-TTI-deadline packets as possible.

    Failure causes used by the active model are:
      * link/FBL: no active UAV can carry the packet even with all ``rb_n`` RBs;
      * resource: at least one link is physically feasible in isolation, but
        the remaining RBs in that TTI are insufficient because earlier URLLC
        packets have already consumed them.

    Legacy cap/budget/available diagnostic fields are retained as aliases for
    backward compatibility.  ``fail_cap`` and ``fail_available`` are always
    zero; ``fail_budget`` aliases the physical-resource failure count.
    """
    m_n, user_n = embb_rb_base_user.shape
    rb_n = int(getattr(cfg, "urllc_physical_rb_limit_per_tti", cfg.rb_n))
    if rb_n != int(cfg.rb_n):
        raise ValueError(
            "URLLC physical RB limit must equal the per-UAV RB pool: "
            f"limit={rb_n}, rb_n={cfg.rb_n}"
        )

    punctured_rb_user = np.zeros((m_n, user_n), dtype=np.int32)
    urllc_rb_user = np.zeros((m_n, user_n), dtype=np.int32)
    urllc_bits_user = np.zeros(user_n, dtype=np.float32)
    success_each = np.zeros(m_n, dtype=np.float32)
    violation_each = np.zeros(m_n, dtype=np.float32)
    punctured_each = np.zeros(m_n, dtype=np.float32)
    idle_used_each = np.zeros(m_n, dtype=np.float32)
    total_urllc_rb_each = np.zeros(m_n, dtype=np.float32)

    fail_link = 0
    fail_resource = 0
    full_rb_hit_tti = 0
    tti_with_arrivals = 0
    required_rb_samples = []
    max_punctured_in_one_uav_tti = 0
    max_total_urllc_rb_one_uav_tti = 0

    by_tti = {}
    for pkt in packets:
        by_tti.setdefault(int(pkt["arrival_tti"]), []).append(pkt)

    # ------------------------------------------------------------------
    # Per-slot FBL required-RB cache.
    #
    # During one control slot, ``sinr_matrix[m, k]`` is fixed.  All packets of
    # the current model also have the same 512-bit payload, so recomputing the
    # same (UAV, URLLC-user) FBL RB demand for every packet is redundant.  Build
    # it once here and let every TTI/packet below reuse the result.  The cache is
    # keyed by packet_bits as well, so behavior remains correct if a future
    # experiment mixes URLLC packet sizes.
    # ------------------------------------------------------------------
    active_uavs = np.where(active_mask)[0].astype(np.int32)
    packet_users = sorted({int(pkt["user"]) for pkt in packets})
    packet_bit_values = sorted({float(pkt["bits"]) for pkt in packets})
    required_rb_cache = {}
    for bits in packet_bit_values:
        req_matrix = np.full((m_n, user_n), np.inf, dtype=np.float64)
        for m in active_uavs:
            for k in packet_users:
                req_matrix[int(m), int(k)] = cfg.calc_urllc_required_rb(
                    float(sinr_matrix[int(m), int(k)]),
                    packet_bits=float(bits),
                    max_rb=rb_n,
                )
        required_rb_cache[float(bits)] = req_matrix

    success = 0
    violation = 0
    for tau in sorted(by_tti):
        tti_with_arrivals += 1

        # eMBB allocation is repeated over the TTIs of this 1-s control slot.
        # Some RBs can be left idle when finite eMBB files need fewer than rb_n.
        available_embb = np.asarray(embb_rb_base_user, dtype=np.int32).copy()
        allocated_embb = np.sum(available_embb, axis=1).astype(np.int32)
        if np.any(allocated_embb > rb_n):
            raise RuntimeError("eMBB RB allocation exceeds the physical RB pool.")
        idle_left = np.where(
            active_mask,
            np.maximum(rb_n - allocated_embb, 0),
            0,
        ).astype(np.int32)

        punctured_before = punctured_each.copy()
        total_before = total_urllc_rb_each.copy()

        enriched = []
        group = by_tti[tau]
        for pkt in group:
            k = int(pkt["user"])
            bits = float(pkt["bits"])
            req_matrix = required_rb_cache[bits]
            req_by_m = []
            for m in active_uavs:
                req = float(req_matrix[int(m), k])
                req_by_m.append((req, -float(sinr_matrix[int(m), k]), int(m)))
            req_by_m.sort()
            best_req = req_by_m[0][0] if req_by_m else np.inf
            if np.isfinite(best_req):
                required_rb_samples.append(float(best_req))
            enriched.append((best_req, k, pkt, req_by_m))

        # Shortest feasible packets first improves one-TTI packet reliability
        # under simultaneous URLLC arrivals without changing their deadlines.
        enriched.sort(key=lambda x: (x[0], x[1]))

        for _, k, pkt, req_by_m in enriched:
            feasible = []
            for req, neg_sinr, m in req_by_m:
                if not np.isfinite(req):
                    continue
                req = int(req)
                physical_left = int(idle_left[m] + np.sum(available_embb[m]))
                if req > physical_left:
                    continue
                puncture_need = max(req - int(idle_left[m]), 0)
                feasible.append((puncture_need, req, neg_sinr, int(m)))

            if not feasible:
                violation += 1
                finite = [
                    (int(req), int(m))
                    for req, _neg_sinr, m in req_by_m
                    if np.isfinite(req)
                ]
                if not finite:
                    fail_link += 1
                else:
                    fail_resource += 1

                if req_by_m:
                    # Attribute the violation to the strongest active link for
                    # per-UAV diagnostics; this does not change scheduling.
                    m_best = int(min(req_by_m, key=lambda x: x[1])[2])
                    violation_each[m_best] += 1.0
                continue

            # Idle-RB-first across candidate UAVs: minimize actual eMBB loss.
            puncture_need, req, _neg_sinr, m = min(
                feasible, key=lambda x: (x[0], x[1], x[2], x[3])
            )

            idle_take = min(req, int(idle_left[m]))
            idle_left[m] -= idle_take
            idle_used_each[m] += float(idle_take)
            remaining_to_puncture = int(req - idle_take)

            # If idle RBs are insufficient, puncture the lowest-rate eMBB RBs
            # first to minimize instantaneous eMBB throughput loss.
            for _ in range(remaining_to_puncture):
                candidates = np.where(available_embb[m] > 0)[0]
                if candidates.size == 0:
                    raise RuntimeError(
                        "URLLC physical-resource feasibility check became inconsistent."
                    )
                j = int(candidates[np.argmin(rb_rate_matrix[m, candidates])])
                available_embb[m, j] -= 1
                punctured_rb_user[m, j] += 1
                punctured_each[m] += 1.0

            success += 1
            success_each[m] += 1.0
            total_urllc_rb_each[m] += float(req)
            urllc_rb_user[m, k] += int(req)
            urllc_bits_user[k] += float(pkt["bits"])

        end_resource = idle_left + np.sum(available_embb, axis=1).astype(np.int32)
        if np.any(active_mask & (end_resource <= 0)):
            full_rb_hit_tti += 1

        tti_punctured = punctured_each - punctured_before
        tti_total = total_urllc_rb_each - total_before
        if tti_punctured.size:
            max_punctured_in_one_uav_tti = max(
                max_punctured_in_one_uav_tti, int(np.max(tti_punctured))
            )
        if tti_total.size:
            max_total_urllc_rb_one_uav_tti = max(
                max_total_urllc_rb_one_uav_tti, int(np.max(tti_total))
            )

    req_arr = np.asarray(required_rb_samples, dtype=np.float32)
    full_rb_hit_ratio = float(full_rb_hit_tti / max(tti_with_arrivals, 1))
    return {
        "arrivals": int(len(packets)),
        "success": int(success),
        "violation": int(violation),
        "punctured_rb_user": punctured_rb_user.astype(np.int32),
        "urllc_rb_user": urllc_rb_user.astype(np.int32),
        "urllc_bits_user": urllc_bits_user.astype(np.float32),
        "urllc_success_each_uav": success_each.astype(np.float32),
        "urllc_violation_each_uav": violation_each.astype(np.float32),
        "punctured_rb_each_uav": punctured_each.astype(np.float32),
        "urllc_idle_rb_used_each_uav": idle_used_each.astype(np.float32),
        "urllc_total_rb_used_each_uav": total_urllc_rb_each.astype(np.float32),
        "urllc_fail_link_count": int(fail_link),
        "urllc_fail_resource_count": int(fail_resource),
        "urllc_required_rb_samples": req_arr,
        "urllc_avg_required_rb": float(np.mean(req_arr)) if req_arr.size else 0.0,
        "urllc_p95_required_rb": float(np.percentile(req_arr, 95)) if req_arr.size else 0.0,
        "urllc_tti_with_arrivals": int(tti_with_arrivals),
        "urllc_full_rb_hit_tti_count": int(full_rb_hit_tti),
        "urllc_full_rb_hit_tti_ratio": full_rb_hit_ratio,
        "urllc_max_punctured_rb_one_uav_tti": int(max_punctured_in_one_uav_tti),
        "urllc_max_total_rb_one_uav_tti": int(max_total_urllc_rb_one_uav_tti),
        # Legacy diagnostic aliases retained so old result readers do not break.
        "urllc_fail_cap_count": 0,
        "urllc_fail_budget_count": int(fail_resource),
        "urllc_fail_available_count": 0,
        "urllc_cap_hit_tti_count": int(full_rb_hit_tti),
        "urllc_cap_hit_tti_ratio": full_rb_hit_ratio,
    }

def _calc_mmtc_connectivity(cfg, uav_pos, user_pos, active_mask):
    """Evaluate all mMTC users on a fixed common access/reference channel."""
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    gain = cfg.calc_large_scale_channel_gain(uav_pos, user_pos)
    antenna, _ = cfg.calc_antenna_gain_matrix(uav_pos, user_pos)
    p = float(getattr(cfg, "mmtc_access_power_w_each_uav", cfg.rb_power_w_each_uav))
    bw = float(getattr(cfg, "mmtc_access_bandwidth_hz", cfg.rb_bandwidth_hz))
    rx = p * gain * antenna
    rx[~active_mask, :] = 0.0
    noise = bw * float(cfg.noise_psd_w_per_hz)
    total = np.sum(rx, axis=0, keepdims=True)
    sinr_matrix = rx / np.maximum(noise + total - rx, 1e-30)

    associated = -np.ones(cfg.user_n, dtype=np.int32)
    best_sinr = np.zeros(cfg.user_n, dtype=np.float32)
    connected = np.zeros(cfg.user_n, dtype=np.float32)
    mmtc_idx = np.asarray(cfg.mmtc_user_idx, dtype=np.int32)
    active_uavs = np.where(active_mask)[0]
    if active_uavs.size > 0 and mmtc_idx.size > 0:
        for k in mmtc_idx:
            # Association follows maximum access received signal; SINR is then
            # evaluated against the other UAVs sharing the access channel.
            m = int(active_uavs[np.argmax(rx[active_uavs, k])])
            associated[k] = m
            best_sinr[k] = float(sinr_matrix[m, k])
            connected[k] = float(best_sinr[k] >= float(cfg.mmtc_sinr_threshold))

    success_each = np.zeros(cfg.uav_n, dtype=np.float32)
    total_each = np.zeros(cfg.uav_n, dtype=np.float32)
    for k in mmtc_idx:
        m = int(associated[k])
        if m >= 0:
            total_each[m] += 1.0
            success_each[m] += connected[k]
    total_count = int(mmtc_idx.size)
    success_count = int(np.sum(connected[mmtc_idx])) if total_count > 0 else 0
    ratio = float(success_count / max(total_count, 1)) if total_count > 0 else 1.0
    return {
        "associated_uav": associated,
        "sinr_matrix": sinr_matrix.astype(np.float32),
        "best_sinr": best_sinr.astype(np.float32),
        "connected": connected.astype(np.float32),
        "success_count": success_count,
        "total_count": total_count,
        "success_ratio": ratio,
        "success_each_uav": success_each,
        "total_each_uav": total_each,
    }



def _associate_embb_load_aware(cfg, rb_rate_matrix, remaining_data_bits, active_mask, unfinished_embb):
    """Greedy list-scheduling association using remaining RB-slot workload.

    Users with larger minimum service workload are placed first.  Each user is
    assigned to the UAV that minimizes the projected aggregate workload
    sum_j D_j/(C_mj*slot), which balances completion time while penalizing weak
    links.
    """
    rb_rate_matrix = np.asarray(rb_rate_matrix, dtype=np.float32)
    remaining_data_bits = np.asarray(remaining_data_bits, dtype=np.float32).reshape(-1)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    unfinished_embb = np.asarray(unfinished_embb, dtype=bool).reshape(-1)
    m_n, user_n = rb_rate_matrix.shape
    associated = -np.ones(user_n, dtype=np.int32)
    association = np.zeros((m_n, user_n), dtype=np.float32)
    active = np.where(active_mask)[0]
    users = np.where(unfinished_embb)[0]
    if active.size == 0 or users.size == 0:
        return associated, association

    slot_time = float(cfg.slot_time)
    workload = np.full((m_n, user_n), np.inf, dtype=np.float64)
    for m in active:
        rate = np.maximum(rb_rate_matrix[m], 1e-12)
        workload[m] = remaining_data_bits / (rate * slot_time)
    min_work = np.min(workload[active][:, users], axis=0)
    order = users[np.argsort(-min_work, kind="stable")]
    load = np.zeros(m_n, dtype=np.float64)
    for k in order:
        feasible = [int(m) for m in active if np.isfinite(workload[m, k])]
        if not feasible:
            continue
        m = min(feasible, key=lambda mm: load[mm] + workload[mm, k])
        associated[k] = m
        association[m, k] = 1.0
        load[m] += workload[m, k]
    return associated.astype(np.int32), association.astype(np.float32)

def serve_one_slot(
    cfg,
    uav_pos,
    user_pos,
    residual_energy=None,
    active_mask=None,
    ep_id=0,
    step_idx=0,
    avg_user_rate=None,
    association_mode="max_signal",
    slice_budget_ratio=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
    unfinished_mask=None,
    remaining_data_bits=None,
):
    """One slot of finite eMBB service, URLLC puncturing and mMTC access."""
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    if active_mask is None or not bool(getattr(cfg, "use_energy_management", False)):
        active_mask = np.ones(cfg.uav_n, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if remaining_data_bits is None:
        remaining_data_bits = np.zeros(cfg.user_n, dtype=np.float32)
    else:
        remaining_data_bits = np.maximum(
            np.asarray(remaining_data_bits, dtype=np.float32).reshape(-1), 0.0
        )
    if unfinished_mask is None:
        unfinished_mask = remaining_data_bits > float(cfg.task_data_eps_bits)
    else:
        unfinished_mask = np.asarray(unfinished_mask, dtype=bool).reshape(-1)
    embb_mask = np.zeros(cfg.user_n, dtype=bool)
    embb_mask[np.asarray(cfg.embb_user_idx, dtype=np.int32)] = True
    unfinished_embb = unfinished_mask & embb_mask

    # Instantaneous channel for eMBB and URLLC packet service.
    channel_gain = cfg.calc_channel_gain(
        uav_pos=uav_pos, user_pos=user_pos, ep_id=ep_id, step_idx=step_idx
    )
    antenna_gain, main_lobe_mask = cfg.calc_antenna_gain_matrix(uav_pos, user_pos)
    rx_signal_matrix = (
        cfg.tx_power_w_each_uav * antenna_gain * channel_gain
    ).astype(np.float32)
    rb_info = _calc_rb_sinr_and_capacity(
        cfg, channel_gain, antenna_gain, active_mask
    )

    # eMBB-only association and workload-aware RB allocation.
    associated_uav, association = _associate_embb_load_aware(
        cfg, rb_info["rb_rate"], remaining_data_bits, active_mask, unfinished_embb
    )
    rb_alloc_base = _allocate_embb_completion_time_rb(
        cfg,
        association,
        rb_info["rb_rate"],
        remaining_data_bits,
        active_mask,
    )
    embb_base_user_rate = np.sum(
        rb_alloc_base.astype(np.float32) * rb_info["rb_rate"], axis=0
    ).astype(np.float32)
    embb_base_user_rate[~embb_mask] = 0.0

    # URLLC has no reservation: use idle RBs first, then puncture eMBB RB-time.
    packets = _generate_urllc_packets_for_slot(cfg, ep_id, step_idx)
    punct = _apply_limited_urllc_puncturing(
        cfg,
        packets,
        rb_info["sinr"],
        rb_info["rb_bits_tti"],
        rb_info["rb_rate"],
        rb_alloc_base,
        active_mask,
    )
    punctured_rb_user = punct["punctured_rb_user"]
    punctured_bits_user = np.sum(
        punctured_rb_user.astype(np.float32) * rb_info["rb_bits_tti"], axis=0
    )
    base_bits_user = embb_base_user_rate * float(cfg.slot_time)
    effective_bits_user = np.maximum(base_bits_user - punctured_bits_user, 0.0)
    embb_effective_user_rate = (
        effective_bits_user / max(float(cfg.slot_time), 1e-12)
    ).astype(np.float32)
    embb_effective_user_rate[~embb_mask] = 0.0
    served_bits_user = np.minimum(
        effective_bits_user, remaining_data_bits
    ).astype(np.float32)
    served_bits_user[~unfinished_embb] = 0.0

    # mMTC uses a fixed common access channel and consumes no data RB.
    mconn = _calc_mmtc_connectivity(cfg, uav_pos, user_pos, active_mask)

    arrivals = int(punct["arrivals"])
    urllc_success = int(punct["success"])
    urllc_violation = int(punct["violation"])
    urllc_ratio = float(urllc_success / arrivals) if arrivals > 0 else 1.0
    puncture_den = float(np.sum(rb_alloc_base) * max(int(cfg.tti_n_per_slot), 1))
    puncture_ratio = float(np.sum(punctured_rb_user) / max(puncture_den, 1.0))

    active_rates = embb_effective_user_rate[unfinished_embb]
    sum_rate = float(np.sum(embb_effective_user_rate))
    mean_rate = float(np.mean(active_rates)) if active_rates.size else 0.0
    fairness_values = served_bits_user[unfinished_embb]
    fairness = float(cfg.calc_jain_fairness(fairness_values)) if fairness_values.size else 1.0
    user_count_each_uav = np.sum(association > 0.5, axis=1).astype(np.int32)

    metrics = {
        "sum_rate": sum_rate,
        "mean_rate": mean_rate,
        "served_bits": float(np.sum(served_bits_user)),
        "fairness": fairness,
        "instantaneous_fairness": fairness,
        "associated_user_n": int(np.sum(associated_uav >= 0)),
        "associated_ratio": float(np.sum(associated_uav >= 0) / max(np.sum(unfinished_embb), 1)),
        "active_link_n": int(np.sum(association > 0.5)),
        "min_rate": float(np.min(active_rates)) if active_rates.size else 0.0,
        "max_rate": float(np.max(active_rates)) if active_rates.size else 0.0,
        "unfinished_user_n": int(np.sum(unfinished_embb)),
        "embb_base_sum_rate": float(np.sum(embb_base_user_rate)),
        "embb_effective_sum_rate": sum_rate,
        "embb_loss_rate": float(max(np.sum(embb_base_user_rate) - sum_rate, 0.0)),
        "puncture_ratio": puncture_ratio,
        "urllc_arrival_count": arrivals,
        "urllc_success_count": urllc_success,
        "urllc_violation_count": urllc_violation,
        "urllc_fail_link_count": int(punct["urllc_fail_link_count"]),
        "urllc_fail_cap_count": int(punct["urllc_fail_cap_count"]),
        "urllc_fail_budget_count": int(punct["urllc_fail_budget_count"]),
        "urllc_fail_available_count": int(punct["urllc_fail_available_count"]),
        "urllc_fail_resource_count": int(punct["urllc_fail_resource_count"]),
        "urllc_avg_required_rb": float(punct["urllc_avg_required_rb"]),
        "urllc_p95_required_rb": float(punct["urllc_p95_required_rb"]),
        "urllc_cap_hit_tti_ratio": float(punct["urllc_cap_hit_tti_ratio"]),
        "urllc_full_rb_hit_tti_ratio": float(punct["urllc_full_rb_hit_tti_ratio"]),
        "urllc_idle_rb_used": float(np.sum(punct["urllc_idle_rb_used_each_uav"])),
        "urllc_total_rb_used": float(np.sum(punct["urllc_total_rb_used_each_uav"])),
        "mmtc_success_count": int(mconn["success_count"]),
        "mmtc_active_count": int(mconn["total_count"]),
        "mmtc_sinr_ok_count": int(mconn["success_count"]),
        "mmtc_sinr_ok_ratio": float(mconn["success_ratio"]),
    }

    rb_budget = np.zeros((cfg.uav_n, cfg.slice_n), dtype=np.int32)
    rb_budget[active_mask, int(cfg.slice_embb)] = int(cfg.rb_n)
    rb_bw = float(cfg.rb_bandwidth_hz)
    rb_power = float(cfg.rb_power_w_each_uav)
    slice_budget_bandwidth = rb_budget.astype(np.float32) * rb_bw
    slice_budget_power = rb_budget.astype(np.float32) * rb_power
    bandwidth_alloc = rb_alloc_base.astype(np.float32) * rb_bw
    power_alloc = rb_alloc_base.astype(np.float32) * rb_power
    urllc_user_rate = punct["urllc_bits_user"] / max(float(cfg.slot_time), 1e-12)

    traffic_metrics = {
        "urllc_success_ratio": float(np.clip(urllc_ratio, 0.0, 1.0)),
        "urllc_violation_ratio": float(np.clip(1.0 - urllc_ratio, 0.0, 1.0)) if arrivals > 0 else 0.0,
        "urllc_success_count": urllc_success,
        "urllc_violation_count": urllc_violation,
        "urllc_arrival_count": arrivals,
        "urllc_success_each_uav": punct["urllc_success_each_uav"].astype(np.float32),
        "urllc_violation_each_uav": punct["urllc_violation_each_uav"].astype(np.float32),
        "urllc_fail_link_count": int(punct["urllc_fail_link_count"]),
        "urllc_fail_cap_count": int(punct["urllc_fail_cap_count"]),
        "urllc_fail_budget_count": int(punct["urllc_fail_budget_count"]),
        "urllc_fail_available_count": int(punct["urllc_fail_available_count"]),
        "urllc_fail_resource_count": int(punct["urllc_fail_resource_count"]),
        "urllc_required_rb_samples": punct["urllc_required_rb_samples"].astype(np.float32),
        "urllc_avg_required_rb": float(punct["urllc_avg_required_rb"]),
        "urllc_p95_required_rb": float(punct["urllc_p95_required_rb"]),
        "urllc_tti_with_arrivals": int(punct["urllc_tti_with_arrivals"]),
        "urllc_cap_hit_tti_count": int(punct["urllc_cap_hit_tti_count"]),
        "urllc_cap_hit_tti_ratio": float(punct["urllc_cap_hit_tti_ratio"]),
        "urllc_full_rb_hit_tti_count": int(punct["urllc_full_rb_hit_tti_count"]),
        "urllc_full_rb_hit_tti_ratio": float(punct["urllc_full_rb_hit_tti_ratio"]),
        "urllc_idle_rb_used_each_uav": punct["urllc_idle_rb_used_each_uav"].astype(np.float32),
        "urllc_total_rb_used_each_uav": punct["urllc_total_rb_used_each_uav"].astype(np.float32),
        "urllc_max_punctured_rb_one_uav_tti": int(
            punct["urllc_max_punctured_rb_one_uav_tti"]
        ),
        "urllc_max_total_rb_one_uav_tti": int(
            punct["urllc_max_total_rb_one_uav_tti"]
        ),
        "mmtc_success_ratio": float(np.clip(mconn["success_ratio"], 0.0, 1.0)),
        "mmtc_active_ratio": 1.0,
        "mmtc_coverage_ratio": float(np.clip(mconn["success_ratio"], 0.0, 1.0)),
        "mmtc_success_count": int(mconn["success_count"]),
        "mmtc_active_count": int(mconn["total_count"]),
        "mmtc_sinr_ok_count": int(mconn["success_count"]),
        "mmtc_sinr_ok_ratio": float(np.clip(mconn["success_ratio"], 0.0, 1.0)),
        "mmtc_success_each_uav": mconn["success_each_uav"].astype(np.float32),
        "mmtc_active_each_uav": mconn["total_each_uav"].astype(np.float32),
        "mmtc_best_sinr": mconn["best_sinr"].astype(np.float32),
        "mmtc_connected_user": mconn["connected"].astype(np.float32),
        "mmtc_associated_uav": mconn["associated_uav"].astype(np.int32),
        "punctured_rb_each_uav": punct["punctured_rb_each_uav"].astype(np.float32),
        "puncture_ratio": float(np.clip(puncture_ratio, 0.0, 1.0)),
    }

    return {
        "channel_gain": channel_gain.astype(np.float32),
        "antenna_gain": antenna_gain.astype(np.float32),
        "main_lobe_mask": main_lobe_mask.astype(bool),
        "rx_signal_matrix": rx_signal_matrix.astype(np.float32),
        "associated_uav": associated_uav.astype(np.int32),
        "association": association.astype(np.float32),
        "bandwidth_alloc": bandwidth_alloc.astype(np.float32),
        "power_alloc": power_alloc.astype(np.float32),
        "rb_alloc_base": rb_alloc_base.astype(np.int32),
        "rb_alloc_embb_base": rb_alloc_base.astype(np.int32),
        "rb_alloc_mmtc": np.zeros_like(rb_alloc_base, dtype=np.int32),
        "rb_punctured": punctured_rb_user.astype(np.int32),
        "rb_budget": rb_budget.astype(np.int32),
        "user_count_each_uav": user_count_each_uav,
        "slice_budget_bandwidth": slice_budget_bandwidth.astype(np.float32),
        "slice_budget_power": slice_budget_power.astype(np.float32),
        "priority_matrix": association.astype(np.float32),
        "user_rate": embb_effective_user_rate.astype(np.float32),
        "served_bits_user": served_bits_user.astype(np.float32),
        "capacity_bits_user": effective_bits_user.astype(np.float32),
        "embb_base_user_rate": embb_base_user_rate.astype(np.float32),
        "embb_effective_user_rate": embb_effective_user_rate.astype(np.float32),
        "embb_loss_user_rate": np.maximum(
            embb_base_user_rate - embb_effective_user_rate, 0.0
        ).astype(np.float32),
        "mmtc_user_rate": np.zeros(cfg.user_n, dtype=np.float32),
        "urllc_user_rate": urllc_user_rate.astype(np.float32),
        "sinr": rb_info["sinr"].astype(np.float32),
        "rb_rate": rb_info["rb_rate"].astype(np.float32),
        "rb_bits_tti": rb_info["rb_bits_tti"].astype(np.float32),
        "desired_power": rb_info["desired_power"].astype(np.float32),
        "interference_power": rb_info["interference_power"].astype(np.float32),
        "noise_power": rb_info["noise_power"].astype(np.float32),
        "interference_psd": calc_average_inter_uav_interference_psd(
            cfg, channel_gain, antenna_gain, active_mask
        ),
        "active_mask": active_mask.astype(bool),
        "metrics": metrics,
        "traffic_metrics": traffic_metrics,
    }

# ============================================================================
# Reward-shaping helper: deterministic completion-time potential
# ============================================================================
def estimate_embb_completion_potential(
    cfg,
    uav_pos,
    user_pos,
    remaining_data_bits,
    initial_data_bits,
    active_mask=None,
):
    """Estimate a state-only makespan potential from large-scale channels.

    This helper does not alter the runtime association/resource-allocation
    algorithm.  It only supplies a low-variance shaping signal.  Small-scale
    fading and random URLLC arrivals are intentionally excluded so the same
    physical state maps to a stable potential value.

    Returns
    -------
    dict with ``potential`` and diagnostic components.  ``potential`` is zero
    when all eMBB files are complete and otherwise non-positive.
    """
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    remain_all = np.maximum(
        np.asarray(remaining_data_bits, dtype=np.float32).reshape(-1), 0.0
    )
    init_all = np.maximum(
        np.asarray(initial_data_bits, dtype=np.float32).reshape(-1), 1.0
    )
    if active_mask is None:
        active_mask = np.ones(int(cfg.uav_n), dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    embb_idx = np.asarray(getattr(cfg, "embb_user_idx", np.arange(cfg.user_n)), dtype=np.int32)
    if embb_idx.size == 0:
        return {
            "potential": 0.0,
            "user_bottleneck": 0.0,
            "uav_bottleneck": 0.0,
            "mean_remaining_ratio": 0.0,
            "estimated_user_slots": np.zeros(cfg.user_n, dtype=np.float32),
            "estimated_uav_load_slots": np.zeros(cfg.uav_n, dtype=np.float32),
            "associated_uav": -np.ones(cfg.user_n, dtype=np.int32),
        }

    unfinished = np.zeros(cfg.user_n, dtype=bool)
    unfinished[embb_idx] = remain_all[embb_idx] > float(cfg.task_data_eps_bits)
    if not np.any(unfinished[embb_idx]):
        return {
            "potential": 0.0,
            "user_bottleneck": 0.0,
            "uav_bottleneck": 0.0,
            "mean_remaining_ratio": 0.0,
            "estimated_user_slots": np.zeros(cfg.user_n, dtype=np.float32),
            "estimated_uav_load_slots": np.zeros(cfg.uav_n, dtype=np.float32),
            "associated_uav": -np.ones(cfg.user_n, dtype=np.int32),
        }

    # Deterministic expected channel snapshot using the same full-reuse RB model
    # and the same workload-aware association/allocation rules as runtime.
    channel_gain = cfg.calc_large_scale_channel_gain(uav_pos, user_pos)
    antenna_gain, _ = cfg.calc_antenna_gain_matrix(uav_pos, user_pos)
    rb_info = _calc_rb_sinr_and_capacity(
        cfg, channel_gain, antenna_gain, active_mask
    )
    associated_uav, association = _associate_embb_load_aware(
        cfg,
        rb_info["rb_rate"],
        remain_all,
        active_mask,
        unfinished,
    )
    rb_alloc = _allocate_embb_completion_time_rb(
        cfg,
        association,
        rb_info["rb_rate"],
        remain_all,
        active_mask,
    )
    estimated_rate = np.sum(
        rb_alloc.astype(np.float32) * rb_info["rb_rate"], axis=0
    ).astype(np.float32)

    max_step = max(float(getattr(cfg, "max_step", 1)), 1.0)
    component_clip = max(
        float(getattr(cfg, "reward_potential_component_clip", 2.0)), 1.0
    )
    eps = 1e-9

    user_slots = np.zeros(cfg.user_n, dtype=np.float32)
    valid_rate = estimated_rate > eps
    valid = unfinished & valid_rate
    user_slots[valid] = remain_all[valid] / np.maximum(
        estimated_rate[valid] * float(cfg.slot_time), eps
    )
    # An unfinished user with zero estimated rate is treated as a clipped severe
    # bottleneck instead of producing an infinite potential.
    user_slots[unfinished & (~valid_rate)] = component_clip * max_step
    user_norm = np.clip(user_slots[embb_idx] / max_step, 0.0, component_clip)

    # Per-UAV workload in equivalent full-RB slots: sum(D/C_rb)/RB_count.
    uav_load_slots = np.zeros(cfg.uav_n, dtype=np.float32)
    for m in range(cfg.uav_n):
        idx_m = np.where((associated_uav == m) & unfinished)[0]
        if idx_m.size == 0 or not active_mask[m]:
            continue
        rb_rate = np.maximum(rb_info["rb_rate"][m, idx_m], eps)
        rb_slot_work = remain_all[idx_m] / np.maximum(
            rb_rate * float(cfg.slot_time), eps
        )
        uav_load_slots[m] = float(np.sum(rb_slot_work) / max(int(cfg.rb_n), 1))
    uav_norm = np.clip(uav_load_slots / max_step, 0.0, component_clip)

    temperature = max(
        float(getattr(cfg, "reward_potential_softmax_temperature", 10.0)), 1e-6
    )

    def _softmax_weighted_max(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0 or not np.any(x > 0.0):
            return 0.0
        z = temperature * x
        z -= np.max(z)
        w = np.exp(np.clip(z, -60.0, 0.0))
        w /= max(float(np.sum(w)), 1e-12)
        return float(np.sum(w * x))

    user_bottleneck = _softmax_weighted_max(user_norm)
    uav_bottleneck = _softmax_weighted_max(uav_norm)
    mean_remaining_ratio = float(np.mean(np.clip(
        remain_all[embb_idx] / init_all[embb_idx], 0.0, 1.0
    )))

    w_user = float(getattr(cfg, "reward_potential_user_bottleneck_weight", 0.5))
    w_uav = float(getattr(cfg, "reward_potential_uav_bottleneck_weight", 0.3))
    w_mean = float(getattr(cfg, "reward_potential_mean_remaining_weight", 0.2))
    weight_sum = max(w_user + w_uav + w_mean, 1e-12)
    potential_cost = (
        w_user * user_bottleneck
        + w_uav * uav_bottleneck
        + w_mean * mean_remaining_ratio
    ) / weight_sum

    return {
        "potential": float(-potential_cost),
        "user_bottleneck": float(user_bottleneck),
        "uav_bottleneck": float(uav_bottleneck),
        "mean_remaining_ratio": float(mean_remaining_ratio),
        "estimated_user_slots": user_slots.astype(np.float32),
        "estimated_uav_load_slots": uav_load_slots.astype(np.float32),
        "associated_uav": associated_uav.astype(np.int32),
    }
