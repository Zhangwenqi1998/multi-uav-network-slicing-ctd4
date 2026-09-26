"""
ZHANG Wenqi

Finite eMBB completion-time environment with URLLC and mMTC QoS constraints.  Geometry, mobility, NFZ and action-shield
helpers from the previous environment are retained; the active reset/step/state
and metric methods are installed in the finite-data override section below.
"""

from __future__ import annotations

import numpy as np

from resource_algorithms import (
    serve_one_slot,
    update_average_user_rate,
    estimate_embb_completion_potential,
)


class Env:
    def __init__(self):
        self.state = None
        self.ep_id = None
        self.step_id = None
        self.slot_count = None

        self.uav_init_pos = None
        self.uav_pos = None
        self.user_pos = None
        self.user_speed = None
        self.user_heading = None
        self.residual_energy = None
        self.active_mask = None

        self.nfz_polygons = []
        self.nfz_vertex_count = None
        self.prev_nfz_penalty = 0.0
        self.prev_nfz_violation = None

        # Action shield diagnostics
        self.prev_raw_action = None
        self.prev_executed_action = None
        self.prev_shield_flag = 0.0
        self.prev_shield_penalty = 0.0
        self.prev_raw_boundary_penalty = 0.0
        self.prev_raw_nfz_penalty = 0.0
        self.prev_raw_boundary_violation = None
        self.prev_raw_nfz_violation = None

        self.avg_user_rate = None
        self.last_user_rate = None
        self.association = None
        self.associated_uav = None
        self.prev_heading = None

        # URLLC/mMTC traffic state
        self.urllc_packets = None
        self.urllc_queue_bits = None
        self.urllc_deadline = None
        self.urllc_arrival_count_current = 0
        self.urllc_total_arrivals = 0
        self.urllc_total_success = 0
        self.urllc_total_violations = 0

        self.mmtc_active = None
        self.mmtc_pending_bits = None
        self.mmtc_active_count_current = 0
        self.mmtc_total_active = 0
        self.mmtc_total_success = 0

        self.prev_reward = 0.0
        self.prev_agent_reward = None
        self.prev_agent_sum_rate = None
        self.prev_agent_embb_utility = None
        self.prev_agent_user_count = None
        self.prev_agent_illegal_flag = None
        self.prev_sum_rate = 0.0
        self.prev_mean_rate = 0.0
        self.prev_served_bits = 0.0
        self.prev_energy = 0.0
        self.prev_boundary_penalty = 0.0
        self.prev_active_uav_n = 0
        self.prev_done_reason = ""
        # 保持step()三返回值接口不变，同时显式区分真实终止与时间截断。
        self.prev_terminated = False
        self.prev_truncated = False
        self.prev_slice_metrics = {}
        self.prev_puncture_ratio = 0.0
        self.prev_embb_base_sum_rate = 0.0
        self.prev_embb_effective_sum_rate = 0.0
        self.prev_embb_loss_rate = 0.0

        self._clear_episode_lists()

    def _clear_episode_lists(self):
        self.reward_list = []                  # 全局reward，仅用于统计/benchmark
        self.agent_reward_list = []            # [T, M]，每个UAV独立训练reward
        self.agent_sum_rate_list = []          # [T, M]，每个UAV服务总速率
        self.agent_embb_utility_list = []      # [T, M]，每个UAV服务eMBB效用
        self.agent_user_count_list = []        # [T, M]，每个UAV关联用户数
        self.agent_boundary_penalty_list = []
        self.agent_nfz_penalty_list = []
        self.agent_shield_penalty_list = []
        self.agent_raw_boundary_penalty_list = []
        self.agent_raw_nfz_penalty_list = []
        self.agent_illegal_flag_list = []      # [T, M]，raw action是否越界/进NFZ
        self.agent_urllc_success_count_list = []
        self.agent_urllc_violation_count_list = []
        self.agent_mmtc_success_count_list = []
        self.agent_mmtc_active_count_list = []
        self.sum_rate_list = []
        self.mean_rate_list = []
        self.served_bits_list = []
        self.energy_list = []
        self.boundary_penalty_list = []
        self.nfz_penalty_list = []
        self.nfz_violation_list = []
        self.active_uav_n_list = []

        self.embb_utility_list = []
        self.embb_mean_rate_list = []
        self.embb_sum_rate_list = []
        self.urllc_success_ratio_list = []
        self.urllc_violation_ratio_list = []
        self.urllc_constraint_gap_list = []
        self.mmtc_success_ratio_list = []
        self.mmtc_active_ratio_list = []
        self.mmtc_constraint_gap_list = []
        self.urllc_arrival_count_list = []
        self.urllc_success_count_list = []
        self.urllc_violation_count_list = []
        self.mmtc_active_count_list = []
        self.mmtc_success_count_list = []
        self.mmtc_sinr_ok_count_list = []
        self.mmtc_bits_ok_count_list = []
        self.mmtc_full_rb_upper_count_list = []
        self.mmtc_fail_sinr_count_list = []
        self.mmtc_fail_rb_count_list = []
        self.mmtc_fail_both_count_list = []
        self.mmtc_sinr_ok_ratio_list = []
        self.mmtc_bits_ok_ratio_list = []
        self.mmtc_full_rb_upper_ratio_list = []
        self.mmtc_fail_sinr_ratio_list = []
        self.mmtc_fail_rb_ratio_list = []
        self.mmtc_fail_both_ratio_list = []
        self.mmtc_avg_required_rb_list = []
        self.mmtc_avg_allocated_rb_list = []
        self.mmtc_total_budget_rb_list = []
        self.puncture_ratio_list = []
        self.punctured_rb_list = []
        self.embb_base_sum_rate_list = []
        self.embb_effective_sum_rate_list = []
        self.embb_loss_rate_list = []

        # aliases for old plotting code
        self.fairness_list = []
        # 当前问题中用户总是关联到active UAV中的最强接收信号UAV，
        # covered/unserved不再作为有效业务指标，仅保留必要兼容字段。
        self.slice_rate_max_utility_list = self.embb_utility_list
        self.slice_guarantee_satisfaction_list = self.urllc_success_ratio_list
        self.slice_fair_utility_list = self.mmtc_success_ratio_list
        self.rate_outage_ratio_list = self.urllc_violation_ratio_list
        self.rate_deficit_mean_list = []

        self.uav_pos_list = []
        self.user_pos_list = []
        self.residual_energy_list = []
        self.active_mask_list = []
        self.association_list = []
        self.associated_uav_list = []
        self.action_list = []              # 实际执行动作，仅用于环境轨迹与诊断
        self.raw_action_list = []          # Actor/benchmark原始输出，仅用于诊断
        self.executed_action_list = []     # 与action_list一致，显式保存便于对比
        self.shield_flag_list = []
        self.shield_penalty_list = []
        self.raw_boundary_penalty_list = []
        self.raw_nfz_penalty_list = []
        self.raw_boundary_violation_list = []
        self.raw_nfz_violation_list = []
        self.heading_list = []
        self.slice_budget_ratio_list = []
        self.slice_budget_bandwidth_list = []
        self.slice_budget_power_list = []
        self.rb_budget_list = []
        self.rb_punctured_list = []
        self.user_rate_list = []
        self.urllc_queue_bits_list = []
        self.urllc_deadline_list = []
        self.mmtc_active_list = []
        self.mmtc_pending_bits_list = []

    def reset(self, main_cfg, ep_id=0):
        self.ep_id = int(ep_id)
        self.step_id = 0
        self.slot_count = 0

        main_cfg.set_episode_nfz(ep_id=self.ep_id)
        self.nfz_polygons = [poly.copy() for poly in main_cfg.nfz_polygons]
        _, self.nfz_vertex_count = main_cfg.get_padded_nfz_polygons()

        self.uav_init_pos = main_cfg.sample_uav_init_pos(ep_id=self.ep_id)
        self.uav_pos = self.uav_init_pos.copy()
        self.user_pos = main_cfg.sample_user_pos(ep_id=self.ep_id)
        self.user_speed, self.user_heading = main_cfg.sample_user_motion_state(ep_id=self.ep_id)

        # finite-data版本：不做能量管理，所有UAV全程保持active。
        # residual_energy仅保留为零值兼容字段，不进入state/obs，也不参与终止。
        self.residual_energy = np.zeros(main_cfg.uav_n, dtype=np.float32)
        self.active_mask = np.ones(main_cfg.uav_n, dtype=bool)

        self.avg_user_rate = np.zeros(main_cfg.user_n, dtype=np.float32)
        self.last_user_rate = np.zeros(main_cfg.user_n, dtype=np.float32)
        self.association = np.zeros((main_cfg.uav_n, main_cfg.user_n), dtype=np.float32)
        self.associated_uav = -np.ones(main_cfg.user_n, dtype=np.int32)
        self.prev_heading = np.zeros(main_cfg.uav_n, dtype=np.float32)

        self.urllc_packets = [[] for _ in range(main_cfg.user_n)]
        self.urllc_queue_bits = np.zeros(main_cfg.user_n, dtype=np.float32)
        self.urllc_deadline = np.zeros(main_cfg.user_n, dtype=np.float32)
        self.urllc_arrival_count_current = 0
        self.urllc_total_arrivals = 0
        self.urllc_total_success = 0
        self.urllc_total_violations = 0

        self.mmtc_active = np.zeros(main_cfg.user_n, dtype=np.float32)
        self.mmtc_pending_bits = np.zeros(main_cfg.user_n, dtype=np.float32)
        self.mmtc_active_count_current = 0
        self.mmtc_total_active = 0
        self.mmtc_total_success = 0

        self.prev_reward = 0.0
        self.prev_agent_reward = np.zeros(main_cfg.uav_n, dtype=np.float32)
        self.prev_agent_sum_rate = np.zeros(main_cfg.uav_n, dtype=np.float32)
        self.prev_agent_embb_utility = np.zeros(main_cfg.uav_n, dtype=np.float32)
        self.prev_agent_user_count = np.zeros(main_cfg.uav_n, dtype=np.float32)
        self.prev_agent_illegal_flag = np.zeros(main_cfg.uav_n, dtype=np.float32)
        self.prev_sum_rate = 0.0
        self.prev_mean_rate = 0.0
        self.prev_served_bits = 0.0
        self.prev_energy = 0.0
        self.prev_boundary_penalty = 0.0
        self.prev_nfz_penalty = 0.0
        self.prev_nfz_violation = np.zeros(main_cfg.uav_n, dtype=np.float32)

        self.prev_raw_action = np.zeros(main_cfg.action_dim, dtype=np.float32)
        self.prev_executed_action = np.zeros(main_cfg.action_dim, dtype=np.float32)
        self.prev_shield_flag = 0.0
        self.prev_shield_penalty = 0.0
        self.prev_raw_boundary_penalty = 0.0
        self.prev_raw_nfz_penalty = 0.0
        self.prev_raw_boundary_violation = np.zeros(main_cfg.uav_n, dtype=np.float32)
        self.prev_raw_nfz_violation = np.zeros(main_cfg.uav_n, dtype=np.float32)

        self.prev_active_uav_n = int(np.sum(self.active_mask))
        self.prev_done_reason = ""
        self.prev_terminated = False
        self.prev_truncated = False
        self.prev_slice_metrics = {}
        self.prev_puncture_ratio = 0.0
        self.prev_embb_base_sum_rate = 0.0
        self.prev_embb_effective_sum_rate = 0.0
        self.prev_embb_loss_rate = 0.0

        self.ep_reward = 0.0
        self.ep_agent_reward = np.zeros(main_cfg.uav_n, dtype=np.float32)
        self.ep_total_rate = 0.0
        self.ep_total_served_bits = 0.0
        self.ep_total_energy = 0.0

        self._clear_episode_lists()
        self._generate_slot_traffic(main_cfg, self.step_id)
        self._record_initial_state()

        self.state = self._build_state(main_cfg)
        return self.state.copy()

    def _record_initial_state(self):
        self.uav_pos_list = [self.uav_pos.copy()]
        self.user_pos_list = [self.user_pos.copy()]
        self.residual_energy_list = [self.residual_energy.copy()]
        self.active_mask_list = [self.active_mask.copy()]
        self.association_list = [self.association.copy()]
        self.associated_uav_list = [self.associated_uav.copy()]
        self.user_rate_list = [self.last_user_rate.copy()]
        self.urllc_queue_bits_list = [self.urllc_queue_bits.copy()]
        self.urllc_deadline_list = [self.urllc_deadline.copy()]
        self.mmtc_active_list = [self.mmtc_active.copy()]
        self.mmtc_pending_bits_list = [self.mmtc_pending_bits.copy()]

    def _rng_for_slot(self, main_cfg, slot_idx, salt=0):
        seed = int(main_cfg.seed) + int(self.ep_id) * 1000003 + int(slot_idx) * 9176 + int(salt)
        return np.random.default_rng(seed)

    def _generate_slot_traffic(self, main_cfg, slot_idx):
        """
        生成当前控制slot开始前可观测的mMTC激活状态。

        新版URLLC不在slot开始前进入队列，而是在serve_one_slot内部按
        TTI级Poisson过程突发到达并puncture eMBB RB。因此这里仅重置
        URLLC队列诊断量并生成mMTC active/pending bits。
        """
        self.urllc_packets = [[] for _ in range(main_cfg.user_n)]
        self.urllc_arrival_count_current = 0
        self.urllc_queue_bits[:] = 0.0
        self.urllc_deadline[:] = 0.0

        rng = self._rng_for_slot(main_cfg, slot_idx, salt=7727)
        self.mmtc_active[:] = 0.0
        self.mmtc_pending_bits[:] = 0.0
        self.mmtc_active_count_current = 0
        mmtc_idx = np.where(main_cfg.user_slice == main_cfg.slice_mmtc)[0]
        if mmtc_idx.size > 0:
            active = rng.random(mmtc_idx.size) < float(main_cfg.mmtc_active_prob)
            for k, flag in zip(mmtc_idx, active):
                if flag:
                    self.mmtc_active[int(k)] = 1.0
                    self.mmtc_pending_bits[int(k)] = float(main_cfg.mmtc_packet_bits)
                    self.mmtc_active_count_current += 1

    def _sync_urllc_state(self, main_cfg):
        self.urllc_queue_bits[:] = 0.0
        self.urllc_deadline[:] = 0.0
        for k, packets in enumerate(self.urllc_packets):
            if len(packets) <= 0:
                continue
            self.urllc_queue_bits[k] = float(sum(max(pkt[0], 0.0) for pkt in packets))
            self.urllc_deadline[k] = float(min(max(int(pkt[1]), 0) for pkt in packets))

    def _serve_urlcc_and_mmtc(self, main_cfg, user_rate, associated_uav, sinr_matrix):
        service_bits = np.asarray(user_rate, dtype=np.float32).reshape(-1) * float(main_cfg.slot_time)
        associated_uav = np.asarray(associated_uav, dtype=np.int32).reshape(-1)
        sinr_matrix = np.asarray(sinr_matrix, dtype=np.float32)

        m_n = int(main_cfg.uav_n)

        urllc_success = 0
        urllc_violation = 0
        urllc_success_each = np.zeros(m_n, dtype=np.float32)
        urllc_violation_each = np.zeros(m_n, dtype=np.float32)

        urllc_idx = np.where(main_cfg.user_slice == main_cfg.slice_urllc)[0]
        for k in urllc_idx:
            k = int(k)
            m = int(associated_uav[k])
            remain_service = float(service_bits[k])
            new_packets = []
            for bits, deadline in self.urllc_packets[k]:
                bits = float(bits)
                deadline = int(deadline)
                if remain_service > 0.0:
                    served = min(bits, remain_service)
                    bits -= served
                    remain_service -= served
                if bits <= 1e-9:
                    urllc_success += 1
                    if 0 <= m < m_n:
                        urllc_success_each[m] += 1.0
                    self.urllc_total_success += 1
                    continue
                deadline -= 1
                if deadline <= 0:
                    urllc_violation += 1
                    if 0 <= m < m_n:
                        urllc_violation_each[m] += 1.0
                    self.urllc_total_violations += 1
                else:
                    new_packets.append([bits, deadline])
            self.urllc_packets[k] = new_packets

        self._sync_urllc_state(main_cfg)

        if self.urllc_arrival_count_current > 0:
            urllc_violation_ratio = urllc_violation / float(max(self.urllc_arrival_count_current, 1))
        else:
            urllc_violation_ratio = 0.0
        urllc_success_ratio = 1.0 - min(1.0, float(urllc_violation_ratio))

        mmtc_success = 0
        mmtc_success_each = np.zeros(m_n, dtype=np.float32)
        mmtc_active_each = np.zeros(m_n, dtype=np.float32)

        mmtc_idx = np.where(main_cfg.user_slice == main_cfg.slice_mmtc)[0]
        for k in mmtc_idx:
            k = int(k)
            if self.mmtc_active[k] <= 0.5:
                continue
            m = int(associated_uav[k])
            if 0 <= m < m_n:
                mmtc_active_each[m] += 1.0
            sinr = float(sinr_matrix[m, k]) if 0 <= m < main_cfg.uav_n else 0.0
            bits_ok = float(service_bits[k]) >= float(main_cfg.mmtc_packet_bits)
            sinr_ok = sinr >= float(main_cfg.mmtc_sinr_threshold)
            if bits_ok and sinr_ok:
                mmtc_success += 1
                if 0 <= m < m_n:
                    mmtc_success_each[m] += 1.0
                self.mmtc_total_success += 1

        if self.mmtc_active_count_current > 0:
            mmtc_success_ratio = mmtc_success / float(self.mmtc_active_count_current)
        else:
            mmtc_success_ratio = 1.0
        mmtc_active_ratio = float(self.mmtc_active_count_current / max(len(mmtc_idx), 1))
        mmtc_coverage_ratio = float(mmtc_success_ratio if self.mmtc_active_count_current > 0 else 1.0)

        return {
            "urllc_success_ratio": float(np.clip(urllc_success_ratio, 0.0, 1.0)),
            "urllc_violation_ratio": float(np.clip(urllc_violation_ratio, 0.0, 1.0)),
            "urllc_success_count": int(urllc_success),
            "urllc_violation_count": int(urllc_violation),
            "urllc_arrival_count": int(self.urllc_arrival_count_current),
            "urllc_success_each_uav": urllc_success_each.astype(np.float32),
            "urllc_violation_each_uav": urllc_violation_each.astype(np.float32),
            "mmtc_success_ratio": float(np.clip(mmtc_success_ratio, 0.0, 1.0)),
            "mmtc_active_ratio": float(np.clip(mmtc_active_ratio, 0.0, 1.0)),
            "mmtc_coverage_ratio": float(np.clip(mmtc_coverage_ratio, 0.0, 1.0)),
            "mmtc_success_count": int(mmtc_success),
            "mmtc_active_count": int(self.mmtc_active_count_current),
            "mmtc_success_each_uav": mmtc_success_each.astype(np.float32),
            "mmtc_active_each_uav": mmtc_active_each.astype(np.float32),
        }

    def _calc_action_safety_violation(self, main_cfg, action):
        """
        计算原始动作若直接执行时的边界/NFZ违规量与安全距离软惩罚。

        注意：
        1) 这里只评估raw动作，不修改环境状态；
        2) 安全距离按每架UAV独立计算；
        3) 合法动作也可能因为距离边界或NFZ不足50m而受到软惩罚；
        4) 真正非法动作仍由per-agent hard negative reward覆盖。
        """
        action = self._parse_action(main_cfg, action)
        heading, _ = main_cfg.parse_joint_action(
            action_cont=action,
            active_mask=self.active_mask,
        )

        heading_for_move = heading.copy()
        heading_for_move[~self.active_mask] = self.prev_heading[~self.active_mask]

        _, raw_next_pos, boundary_violation, nfz_violation = main_cfg.move_uav_one_slot(
            uav_pos=self.uav_pos,
            heading=heading_for_move,
            active_mask=self.active_mask,
        )

        raw_next_pos = np.asarray(raw_next_pos, dtype=np.float32)
        boundary_violation = np.asarray(boundary_violation, dtype=np.float32).reshape(-1)
        nfz_violation = np.asarray(nfz_violation, dtype=np.float32).reshape(-1)
        active = np.asarray(self.active_mask, dtype=bool).reshape(-1)
        violation_sum = boundary_violation + nfz_violation
        max_active_violation = float(np.max(violation_sum[active])) if np.any(active) else 0.0

        boundary_penalty = self._calc_boundary_penalty(main_cfg, boundary_violation)
        nfz_penalty = self._calc_nfz_penalty(main_cfg, nfz_violation)

        # --------------------------------------------------------
        # raw下一位置到矩形区域边界的有符号距离
        # 合法时为正；越界时为负，绝对值为越界距离。
        # --------------------------------------------------------
        area_size = float(main_cfg.area_size)
        x = raw_next_pos[:, 0]
        y = raw_next_pos[:, 1]
        boundary_clearance = np.minimum.reduce(
            [x, y, area_size - x, area_size - y]
        ).astype(np.float32)
        boundary_illegal = boundary_violation > float(
            getattr(main_cfg, "shield_violation_eps", 1e-6)
        )
        boundary_clearance[boundary_illegal] = -boundary_violation[boundary_illegal]

        # --------------------------------------------------------
        # raw下一位置到最近NFZ边界的有符号距离
        # 对线段穿越/进入NFZ的动作，nfz_violation已经为正，因此记为负距离；
        # 对合法动作，使用raw下一位置到最近NFZ多边形边界的距离。
        # --------------------------------------------------------
        nfz_clearance = np.full(main_cfg.uav_n, np.inf, dtype=np.float32)
        if bool(getattr(main_cfg, "use_nfz", False)) and len(main_cfg.nfz_polygons) > 0:
            eps = float(getattr(main_cfg, "shield_violation_eps", 1e-6))
            for m in range(main_cfg.uav_n):
                if nfz_violation[m] > eps:
                    nfz_clearance[m] = -float(nfz_violation[m])
                    continue

                dist_m, _, inside_m = main_cfg.point_to_any_nfz_distance_and_closest(
                    raw_next_pos[m]
                )
                if inside_m:
                    nfz_clearance[m] = -eps
                else:
                    nfz_clearance[m] = float(dist_m)

        min_safety_clearance = np.minimum(
            boundary_clearance,
            nfz_clearance,
        ).astype(np.float32)

        # --------------------------------------------------------
        # 50m缓冲区内的连续二次软惩罚
        # d >= margin: 0
        # 0 <= d < margin: max_penalty * (1-d/margin)^power
        # d < 0: 先记为max_penalty，但随后非法动作会被hard negative覆盖。
        # --------------------------------------------------------
        margin = max(
            float(getattr(main_cfg, "safe_distance_penalty_margin", 50.0)),
            1e-9,
        )
        max_penalty = max(
            float(getattr(main_cfg, "safe_distance_penalty_max", 15.0)),
            0.0,
        )
        power = max(
            float(getattr(main_cfg, "safe_distance_penalty_power", 2.0)),
            1.0,
        )

        safe_clearance_for_penalty = np.maximum(min_safety_clearance, 0.0)
        proximity_ratio = np.clip(
            1.0 - safe_clearance_for_penalty / margin,
            0.0,
            1.0,
        )
        safety_distance_penalty_each = (
            max_penalty * np.power(proximity_ratio, power)
        ).astype(np.float32)
        safety_distance_penalty_each[~active] = 0.0

        return {
            "boundary_violation": boundary_violation.astype(np.float32),
            "nfz_violation": nfz_violation.astype(np.float32),
            "boundary_penalty": float(boundary_penalty),
            "nfz_penalty": float(nfz_penalty),
            "raw_next_pos": raw_next_pos.astype(np.float32),
            "raw_boundary_clearance": boundary_clearance.astype(np.float32),
            "raw_nfz_clearance": nfz_clearance.astype(np.float32),
            "raw_min_safety_clearance": min_safety_clearance.astype(np.float32),
            "safety_distance_penalty_each": safety_distance_penalty_each.astype(np.float32),
            "max_active_violation": float(max_active_violation),
            "total_penalty": float(boundary_penalty + nfz_penalty),
        }

    # ============================================================
    # Action shield：逐UAV将非法动作替换为最近合法方向
    # ============================================================
    def _shield_action_by_nearest_safe_heading(self, main_cfg, raw_action):
        """
        对每架UAV独立处理非法动作：
        1) 只修正原始动作非法的UAV；其他UAV动作保持完全不变。
        2) 对非法UAV，在方向空间中寻找与原始方向角距离最近的合法方向。
        3) 这里只做边界/NFZ几何检查，不再调用通信资源分配或一步reward估计。
        """
        raw_action = self._parse_action(main_cfg, raw_action)
        raw_safety = self._calc_action_safety_violation(main_cfg, raw_action)

        eps = float(getattr(main_cfg, "shield_violation_eps", 1e-6))
        use_shield = bool(getattr(main_cfg, "use_action_shield", False))

        raw_boundary_violation = np.asarray(
            raw_safety["boundary_violation"],
            dtype=np.float32,
        ).reshape(-1)
        raw_nfz_violation = np.asarray(
            raw_safety["nfz_violation"],
            dtype=np.float32,
        ).reshape(-1)
        active = np.asarray(self.active_mask, dtype=bool).reshape(-1)

        illegal_flag_each = (
            active
            & ((raw_boundary_violation + raw_nfz_violation) > eps)
        )

        info = {
            "shield_flag": float(np.any(illegal_flag_each)),
            "shield_penalty": float(raw_safety["boundary_penalty"] + raw_safety["nfz_penalty"]),
            "raw_boundary_penalty": float(raw_safety["boundary_penalty"]),
            "raw_nfz_penalty": float(raw_safety["nfz_penalty"]),
            "raw_boundary_violation": raw_boundary_violation.copy(),
            "raw_nfz_violation": raw_nfz_violation.copy(),
            "raw_boundary_clearance": np.asarray(
                raw_safety.get("raw_boundary_clearance", np.full(main_cfg.uav_n, np.inf)),
                dtype=np.float32,
            ).copy(),
            "raw_nfz_clearance": np.asarray(
                raw_safety.get("raw_nfz_clearance", np.full(main_cfg.uav_n, np.inf)),
                dtype=np.float32,
            ).copy(),
            "raw_min_safety_clearance": np.asarray(
                raw_safety.get("raw_min_safety_clearance", np.full(main_cfg.uav_n, np.inf)),
                dtype=np.float32,
            ).copy(),
            "safety_distance_penalty_each": np.asarray(
                raw_safety.get("safety_distance_penalty_each", np.zeros(main_cfg.uav_n)),
                dtype=np.float32,
            ).copy(),
            "illegal_flag_each": illegal_flag_each.astype(np.float32),
        }

        if (not use_shield) or (not np.any(illegal_flag_each)):
            if not use_shield:
                info["shield_flag"] = 0.0
                info["shield_penalty"] = 0.0
            return raw_action.copy(), info

        raw_heading, _ = main_cfg.parse_joint_action(
            action_cont=raw_action,
            active_mask=self.active_mask,
        )

        # 从原始联合动作开始，只改动非法UAV对应的方向维度。
        executed_action_matrix = raw_action.reshape(
            main_cfg.uav_n,
            main_cfg.local_action_dim,
        ).copy()

        candidate_n = int(getattr(main_cfg, "shield_candidate_n", 72))
        candidate_n = max(candidate_n, 8)
        refine_iter = int(getattr(main_cfg, "shield_refine_iter", 12))
        refine_iter = max(refine_iter, 0)

        def circular_gap(theta_a, theta_b):
            return abs(float(main_cfg.wrap_angle(float(theta_a) - float(theta_b))))

        def build_trial_action(m, theta):
            trial_matrix = executed_action_matrix.copy()
            trial_matrix[m, :2] = main_cfg.heading_to_local_traj_action(float(theta))
            return trial_matrix.reshape(-1).astype(np.float32)

        def own_violation(m, theta):
            trial_action = build_trial_action(m, theta)
            safety = self._calc_action_safety_violation(main_cfg, trial_action)
            value = (
                float(safety["boundary_violation"][m])
                + float(safety["nfz_violation"][m])
            )
            return float(value)

        for m in range(main_cfg.uav_n):
            # 合法UAV以及inactive UAV均不修改。
            if not illegal_flag_each[m]:
                continue

            theta_raw = float(raw_heading[m])

            # 候选点围绕原始方向均匀展开，并按环形角距离从近到远排序。
            candidate_angles = [
                float(main_cfg.wrap_angle(theta_raw + 2.0 * np.pi * j / candidate_n))
                for j in range(candidate_n)
            ]
            candidate_angles.append(float(main_cfg.wrap_angle(self.prev_heading[m])))

            unique_candidates = []
            for theta in candidate_angles:
                if not any(circular_gap(theta, old) <= 1e-10 for old in unique_candidates):
                    unique_candidates.append(theta)
            unique_candidates.sort(key=lambda theta: circular_gap(theta, theta_raw))

            nearest_safe_theta = None
            fallback_theta = theta_raw
            fallback_key = (np.inf, np.inf)

            for theta in unique_candidates:
                violation = own_violation(m, theta)
                gap = circular_gap(theta, theta_raw)

                key = (violation, gap)
                if key < fallback_key:
                    fallback_key = key
                    fallback_theta = float(theta)

                if violation <= eps:
                    nearest_safe_theta = float(theta)
                    break

            # 极少数情况下粗网格没有找到安全方向，再使用1度细网格兜底。
            if nearest_safe_theta is None:
                fine_n = max(360, candidate_n)
                fine_candidates = [
                    float(main_cfg.wrap_angle(theta_raw + 2.0 * np.pi * j / fine_n))
                    for j in range(fine_n)
                ]
                fine_candidates.sort(key=lambda theta: circular_gap(theta, theta_raw))

                for theta in fine_candidates:
                    violation = own_violation(m, theta)
                    gap = circular_gap(theta, theta_raw)

                    key = (violation, gap)
                    if key < fallback_key:
                        fallback_key = key
                        fallback_theta = float(theta)

                    if violation <= eps:
                        nearest_safe_theta = float(theta)
                        break

            if nearest_safe_theta is None:
                # 理论上当前位置合法时通常总能找到安全方向；数值极端情况下
                # 退化为违规量最小的方向，随后仍由原环境几何投影保证状态安全。
                corrected_theta = float(fallback_theta)
            else:
                corrected_theta = float(nearest_safe_theta)

                # 在“原始非法方向 -> 最近安全候选方向”之间二分，进一步逼近
                # 合法边界，从而得到更接近原始动作的连续安全方向。
                signed_gap = float(main_cfg.wrap_angle(nearest_safe_theta - theta_raw))
                if abs(signed_gap) > 1e-12 and refine_iter > 0:
                    low = 0.0      # 原始方向：非法
                    high = 1.0     # 已找到方向：合法

                    for _ in range(refine_iter):
                        mid = 0.5 * (low + high)
                        theta_mid = float(main_cfg.wrap_angle(theta_raw + mid * signed_gap))
                        if own_violation(m, theta_mid) <= eps:
                            high = mid
                        else:
                            low = mid

                    corrected_theta = float(
                        main_cfg.wrap_angle(theta_raw + high * signed_gap)
                    )

            # 只覆盖当前非法UAV的方向；其余UAV动作矩阵元素保持原值。
            executed_action_matrix[m, :2] = main_cfg.heading_to_local_traj_action(corrected_theta)

        executed_action = executed_action_matrix.reshape(-1).astype(np.float32)
        return executed_action, info

    def _calc_per_uav_performance(self, main_cfg, user_rate, associated_uav):
        """
        统计每个UAV本slot的局部服务贡献。
        这些量只用于per-agent reward和日志，不改变资源分配结果。
        """
        user_rate = np.asarray(user_rate, dtype=np.float32).reshape(-1)
        associated_uav = np.asarray(associated_uav, dtype=np.int32).reshape(-1)

        m_n = int(main_cfg.uav_n)
        sum_rate_each = np.zeros(m_n, dtype=np.float32)
        embb_sum_rate_each = np.zeros(m_n, dtype=np.float32)
        user_count_each = np.zeros(m_n, dtype=np.float32)
        embb_user_count_each = np.zeros(m_n, dtype=np.float32)

        for k in range(main_cfg.user_n):
            m = int(associated_uav[k])
            if not (0 <= m < m_n):
                continue
            rate_k = float(user_rate[k])
            sum_rate_each[m] += rate_k
            user_count_each[m] += 1.0
            if int(main_cfg.user_slice[k]) == int(main_cfg.slice_embb):
                embb_sum_rate_each[m] += rate_k
                embb_user_count_each[m] += 1.0

        embb_utility_each = embb_sum_rate_each / max(float(main_cfg.reward_rate_scale), float(main_cfg.rate_eps))

        return {
            "sum_rate_each_uav": sum_rate_each.astype(np.float32),
            "served_bits_each_uav": (sum_rate_each * float(main_cfg.slot_time)).astype(np.float32),
            "embb_sum_rate_each_uav": embb_sum_rate_each.astype(np.float32),
            "embb_utility_each_uav": embb_utility_each.astype(np.float32),
            "user_count_each_uav": user_count_each.astype(np.float32),
            "embb_user_count_each_uav": embb_user_count_each.astype(np.float32),
        }

    def _calc_agent_reward(
        self,
        main_cfg,
        slice_metrics,
        traffic_metrics,
        per_uav_perf,
        boundary_violation,
        nfz_violation,
        shield_info,
        active_mask_before,
    ):
        """
        每个agent独立reward。

        为降低URLLC/mMTC小样本计数造成的局部reward跳变：
        1) 局部项只保留本UAV可归因的eMBB有效吞吐量；
        2) URLLC/mMTC约束只使用全网成功率gap；
        3) 默认30%局部eMBB + 70%全局通信任务均分；
        4) 安全惩罚仍只作用于责任UAV；Kalman critic融合不变。
        """
        m_n = int(main_cfg.uav_n)
        active = np.asarray(active_mask_before, dtype=bool).reshape(-1)

        embb = float(slice_metrics["embb_utility"])
        urllc_gap = float(slice_metrics["urllc_constraint_gap"])
        mmtc_gap = float(slice_metrics["mmtc_constraint_gap"])

        urllc_w = float(main_cfg.reward_urllc_constraint_weight)
        mmtc_w = float(main_cfg.reward_mmtc_constraint_weight)
        embb_w = float(main_cfg.reward_embb_weight)

        global_task_reward = (
            embb_w * embb
            - urllc_w * urllc_gap
            - mmtc_w * mmtc_gap
        )
        global_share = global_task_reward / float(max(m_n, 1))

        local_embb = np.asarray(
            per_uav_perf["embb_utility_each_uav"],
            dtype=np.float32,
        ).reshape(-1)
        local_task_reward = (embb_w * local_embb).astype(np.float32)

        # 以下局部QoS量仅用于日志诊断，不进入训练reward。
        traffic_metrics = traffic_metrics or {}
        urllc_success_each = np.asarray(
            traffic_metrics.get("urllc_success_each_uav", np.zeros(m_n)),
            dtype=np.float32,
        ).reshape(-1)
        urllc_violation_each = np.asarray(
            traffic_metrics.get("urllc_violation_each_uav", np.zeros(m_n)),
            dtype=np.float32,
        ).reshape(-1)
        urllc_total_each = urllc_success_each + urllc_violation_each
        local_urllc_success_ratio = np.ones(m_n, dtype=np.float32)
        has_urllc = urllc_total_each > 0.0
        local_urllc_success_ratio[has_urllc] = (
            urllc_success_each[has_urllc]
            / np.maximum(urllc_total_each[has_urllc], 1.0)
        )
        local_urllc_gap = np.maximum(
            float(main_cfg.urllc_success_threshold) - local_urllc_success_ratio,
            0.0,
        ).astype(np.float32)

        mmtc_success_each = np.asarray(
            traffic_metrics.get("mmtc_success_each_uav", np.zeros(m_n)),
            dtype=np.float32,
        ).reshape(-1)
        mmtc_active_each = np.asarray(
            traffic_metrics.get("mmtc_active_each_uav", np.zeros(m_n)),
            dtype=np.float32,
        ).reshape(-1)
        local_mmtc_success_ratio = np.ones(m_n, dtype=np.float32)
        has_mmtc = mmtc_active_each > 0.0
        local_mmtc_success_ratio[has_mmtc] = (
            mmtc_success_each[has_mmtc]
            / np.maximum(mmtc_active_each[has_mmtc], 1.0)
        )
        local_mmtc_gap = np.maximum(
            float(main_cfg.mmtc_success_threshold) - local_mmtc_success_ratio,
            0.0,
        ).astype(np.float32)

        boundary_violation = np.asarray(boundary_violation, dtype=np.float32).reshape(-1)
        nfz_violation = np.asarray(nfz_violation, dtype=np.float32).reshape(-1)
        raw_boundary_violation = np.asarray(
            shield_info.get("raw_boundary_violation", np.zeros(m_n)),
            dtype=np.float32,
        ).reshape(-1)
        raw_nfz_violation = np.asarray(
            shield_info.get("raw_nfz_violation", np.zeros(m_n)),
            dtype=np.float32,
        ).reshape(-1)
        safety_distance_penalty_each = np.asarray(
            shield_info.get("safety_distance_penalty_each", np.zeros(m_n)),
            dtype=np.float32,
        ).reshape(-1)

        denom = max(float(main_cfg.move_dist_each_step), 1e-9)
        boundary_penalty_each = boundary_violation / denom
        nfz_penalty_each = nfz_violation / denom
        raw_boundary_penalty_each = raw_boundary_violation / denom
        raw_nfz_penalty_each = raw_nfz_violation / denom

        shield_flag = float(shield_info.get("shield_flag", 0.0))
        shield_penalty_each = shield_flag * (
            raw_boundary_penalty_each + raw_nfz_penalty_each
        )

        illegal_eps = float(getattr(main_cfg, "shield_violation_eps", 1e-6))
        illegal_flag_each = (
            (raw_boundary_violation + raw_nfz_violation) > illegal_eps
        ).astype(np.float32)

        local_ratio = float(getattr(main_cfg, "agent_reward_local_ratio", 0.30))
        global_ratio = float(getattr(main_cfg, "agent_reward_global_ratio", 0.70))
        ratio_sum = local_ratio + global_ratio
        if ratio_sum <= 1e-12:
            local_ratio, global_ratio = 0.30, 0.70
        else:
            local_ratio /= ratio_sum
            global_ratio /= ratio_sum

        reward_each = (
            local_ratio * local_task_reward
            + global_ratio * global_share
            - float(main_cfg.reward_boundary_weight) * boundary_penalty_each
            - float(main_cfg.reward_nfz_weight) * nfz_penalty_each
            - safety_distance_penalty_each
        ).astype(np.float32)

        # 保留当前版本的raw非法hard-negative机制，不改变安全学习问题定义。
        illegal_fixed_penalty = float(
            getattr(main_cfg, "reward_illegal_action_fixed_penalty", 50.0)
        )
        raw_violation_penalty_each = (
            float(main_cfg.reward_boundary_weight) * raw_boundary_penalty_each
            + float(main_cfg.reward_nfz_weight) * raw_nfz_penalty_each
        ).astype(np.float32)
        illegal_mask = illegal_flag_each > 0.5
        reward_each[illegal_mask] = -(
            illegal_fixed_penalty + raw_violation_penalty_each[illegal_mask]
        ).astype(np.float32)

        if active.size == reward_each.size:
            reward_each[~active] = 0.0

        return {
            "agent_reward": reward_each.astype(np.float32),
            "agent_global_share": np.full(m_n, global_share, dtype=np.float32),
            "agent_local_task_reward": local_task_reward.astype(np.float32),
            "agent_local_embb": local_embb.astype(np.float32),
            "agent_local_urllc_success_ratio": local_urllc_success_ratio.astype(np.float32),
            "agent_local_urllc_gap": local_urllc_gap.astype(np.float32),
            "agent_local_mmtc_success_ratio": local_mmtc_success_ratio.astype(np.float32),
            "agent_local_mmtc_gap": local_mmtc_gap.astype(np.float32),
            "agent_boundary_penalty": boundary_penalty_each.astype(np.float32),
            "agent_nfz_penalty": nfz_penalty_each.astype(np.float32),
            "agent_shield_penalty": shield_penalty_each.astype(np.float32),
            "agent_raw_boundary_penalty": raw_boundary_penalty_each.astype(np.float32),
            "agent_raw_nfz_penalty": raw_nfz_penalty_each.astype(np.float32),
            "agent_safety_distance_penalty": safety_distance_penalty_each.astype(np.float32),
            "agent_illegal_flag": illegal_flag_each.astype(np.float32),
        }

    def preview_user_state_for_current_step(self, main_cfg):
        """
        返回本控制step结束、实际计算通信reward时使用的用户状态。

        本版本采用完美一步用户位置已知假设：Actor在选择a_t前可观测
        w_{t+1}。move_users_one_slot()按(ep_id, step_id)确定性复现随机量，
        因而这里预览的用户状态与随后step()真正采用的状态严格一致。
        该函数无副作用，不修改环境。
        """
        if self.ep_id is None:
            raise RuntimeError("请先调用 env.reset()。")
        next_user_pos, next_user_speed, next_user_heading = main_cfg.move_users_one_slot(
            user_pos=self.user_pos,
            user_speed=self.user_speed,
            user_heading=self.user_heading,
            ep_id=self.ep_id,
            step_idx=self.step_id,
        )
        return (
            np.asarray(next_user_pos, dtype=np.float32).copy(),
            np.asarray(next_user_speed, dtype=np.float32).copy(),
            np.asarray(next_user_heading, dtype=np.float32).copy(),
        )

    def get_service_user_pos(self, main_cfg):
        """本step通信/reward阶段使用的已知用户位置w_{t+1}。"""
        return self.preview_user_state_for_current_step(main_cfg)[0]

    def step(self, main_cfg, ep_id, action):
        if self.ep_id is None:
            raise RuntimeError("请先调用 env.reset()。")
        if int(ep_id) != self.ep_id:
            raise ValueError(f"ep_id不一致：环境ep_id={self.ep_id}, 输入ep_id={ep_id}")

        self.prev_done_reason = ""
        self.prev_terminated = False
        self.prev_truncated = False

        # 防御性检查：旧版固定时域代码的防御性检查（后续runtime override会替换）。
        if self.step_id >= main_cfg.max_step:
            self.prev_done_reason = "fixed_horizon"
            self.prev_terminated = True
            terminal_state = self._build_terminal_state(main_cfg)
            self.state = terminal_state.copy()
            return terminal_state, 0.0, True

        raw_action = self._parse_action(main_cfg, action)
        active_mask_before = self.active_mask.copy()
        action, shield_info = self._shield_action_by_nearest_safe_heading(
            main_cfg=main_cfg,
            raw_action=raw_action,
        )

        heading, slice_budget_ratio = main_cfg.parse_joint_action(action_cont=action, active_mask=self.active_mask)
        heading_for_move = heading.copy()
        heading_for_move[~self.active_mask] = self.prev_heading[~self.active_mask]

        next_uav_pos, _, boundary_violation, nfz_violation = main_cfg.move_uav_one_slot(
            uav_pos=self.uav_pos,
            heading=heading_for_move,
            active_mask=self.active_mask,
        )
        self.uav_pos = next_uav_pos.copy()
        boundary_penalty = self._calc_boundary_penalty(main_cfg, boundary_violation)
        nfz_penalty = self._calc_nfz_penalty(main_cfg, nfz_violation)
        shield_penalty = float(shield_info.get("shield_penalty", 0.0))

        # 使用决策前已经提供给state/obs的同一组w_{t+1}，保证观测与
        # 本step通信reward所依据的用户位置严格一致。
        self.user_pos, self.user_speed, self.user_heading = (
            self.preview_user_state_for_current_step(main_cfg)
        )

        resource_result = serve_one_slot(
            cfg=main_cfg,
            uav_pos=self.uav_pos,
            user_pos=self.user_pos,
            residual_energy=self.residual_energy,
            active_mask=self.active_mask,
            ep_id=self.ep_id,
            step_idx=self.step_id,
            avg_user_rate=self.avg_user_rate,
            slice_budget_ratio=slice_budget_ratio,
            urllc_queue_bits=self.urllc_queue_bits,
            urllc_deadline=self.urllc_deadline,
            mmtc_active=self.mmtc_active,
            mmtc_pending_bits=self.mmtc_pending_bits,
        )

        user_rate = resource_result["user_rate"].astype(np.float32)
        self.association = resource_result["association"].astype(np.float32)
        self.associated_uav = resource_result["associated_uav"].astype(np.int32)
        self.last_user_rate = user_rate.copy()
        metrics = resource_result["metrics"]

        traffic_metrics = resource_result.get("traffic_metrics", {})
        self.urllc_arrival_count_current = int(traffic_metrics.get("urllc_arrival_count", 0))
        self.urllc_total_arrivals += int(traffic_metrics.get("urllc_arrival_count", 0))
        self.urllc_total_success += int(traffic_metrics.get("urllc_success_count", 0))
        self.urllc_total_violations += int(traffic_metrics.get("urllc_violation_count", 0))
        self.mmtc_total_active += int(traffic_metrics.get("mmtc_active_count", 0))
        self.mmtc_total_success += int(traffic_metrics.get("mmtc_success_count", 0))
        self.urllc_queue_bits[:] = 0.0
        self.urllc_deadline[:] = 0.0

        self.avg_user_rate = update_average_user_rate(
            old_avg_rate=self.avg_user_rate,
            new_rate=user_rate,
            slot_count=self.slot_count,
        )

        slice_metrics = main_cfg.calc_slice_metrics(
            user_rate=user_rate,
            avg_user_rate=self.avg_user_rate,
            urllc_success_ratio=traffic_metrics["urllc_success_ratio"],
            urllc_violation_ratio=traffic_metrics["urllc_violation_ratio"],
            mmtc_success_ratio=traffic_metrics.get("mmtc_success_ratio", 1.0),
            mmtc_active_ratio=traffic_metrics.get("mmtc_active_ratio", 0.0),
            mmtc_coverage_ratio=traffic_metrics.get("mmtc_coverage_ratio", 0.0),
            puncture_ratio=traffic_metrics.get("puncture_ratio", 0.0),
            embb_base_sum_rate=metrics.get("embb_base_sum_rate", None),
            embb_loss_rate=metrics.get("embb_loss_rate", 0.0),
            urllc_arrival_count=traffic_metrics.get("urllc_arrival_count", 0),
            urllc_success_count=traffic_metrics.get("urllc_success_count", 0),
            urllc_violation_count=traffic_metrics.get("urllc_violation_count", 0),
        )

        per_uav_perf = self._calc_per_uav_performance(
            main_cfg=main_cfg,
            user_rate=user_rate,
            associated_uav=self.associated_uav,
        )
        agent_reward_info = self._calc_agent_reward(
            main_cfg=main_cfg,
            slice_metrics=slice_metrics,
            traffic_metrics=traffic_metrics,
            per_uav_perf=per_uav_perf,
            boundary_violation=boundary_violation,
            nfz_violation=nfz_violation,
            shield_info=shield_info,
            active_mask_before=active_mask_before,
        )
        agent_reward = agent_reward_info["agent_reward"].astype(np.float32)

        # 不计算或扣除能量；所有UAV在episode内始终active。
        slot_energy = 0.0
        self.active_mask[:] = True

        reward = self._calc_reward(
            main_cfg=main_cfg,
            slice_metrics=slice_metrics,
            boundary_penalty=boundary_penalty,
            nfz_penalty=nfz_penalty,
            shield_penalty=shield_penalty,
        )

        self.prev_heading = heading_for_move.copy()
        self.prev_sum_rate = float(metrics["sum_rate"])
        self.prev_mean_rate = float(metrics["mean_rate"])
        self.prev_served_bits = float(metrics["served_bits"])
        self.prev_energy = float(slot_energy)
        self.prev_boundary_penalty = float(boundary_penalty)
        self.prev_nfz_penalty = float(nfz_penalty)
        self.prev_nfz_violation = np.asarray(nfz_violation, dtype=np.float32).copy()
        self.prev_raw_action = raw_action.copy()
        self.prev_executed_action = action.copy()
        self.prev_shield_flag = float(shield_info.get("shield_flag", 0.0))
        self.prev_shield_penalty = float(shield_penalty)
        self.prev_raw_boundary_penalty = float(shield_info.get("raw_boundary_penalty", 0.0))
        self.prev_raw_nfz_penalty = float(shield_info.get("raw_nfz_penalty", 0.0))
        self.prev_raw_boundary_violation = np.asarray(
            shield_info.get("raw_boundary_violation", np.zeros(main_cfg.uav_n)),
            dtype=np.float32,
        ).copy()
        self.prev_raw_nfz_violation = np.asarray(
            shield_info.get("raw_nfz_violation", np.zeros(main_cfg.uav_n)),
            dtype=np.float32,
        ).copy()
        self.prev_active_uav_n = int(np.sum(self.active_mask))
        self.prev_reward = float(reward)
        self.prev_agent_reward = agent_reward.copy()
        self.prev_agent_sum_rate = per_uav_perf["sum_rate_each_uav"].copy()
        self.prev_agent_embb_utility = per_uav_perf["embb_utility_each_uav"].copy()
        self.prev_agent_user_count = per_uav_perf["user_count_each_uav"].copy()
        self.prev_agent_illegal_flag = agent_reward_info["agent_illegal_flag"].copy()
        self.prev_slice_metrics = dict(slice_metrics)
        self.prev_puncture_ratio = float(slice_metrics.get("puncture_ratio", 0.0))
        self.prev_embb_base_sum_rate = float(slice_metrics.get("embb_base_sum_rate", 0.0))
        self.prev_embb_effective_sum_rate = float(slice_metrics.get("embb_effective_sum_rate", slice_metrics.get("embb_sum_rate", 0.0)))
        self.prev_embb_loss_rate = float(slice_metrics.get("embb_loss_rate", 0.0))

        self.ep_reward += float(reward)
        self.ep_agent_reward += agent_reward.astype(np.float32)
        self.ep_total_rate += float(metrics["sum_rate"])
        self.ep_total_served_bits += float(metrics["served_bits"])
        self.ep_total_energy += float(slot_energy)

        self.reward_list.append(float(reward))
        self.agent_reward_list.append(agent_reward.copy())
        self.agent_sum_rate_list.append(per_uav_perf["sum_rate_each_uav"].copy())
        self.agent_embb_utility_list.append(per_uav_perf["embb_utility_each_uav"].copy())
        self.agent_user_count_list.append(per_uav_perf["user_count_each_uav"].copy())
        self.agent_boundary_penalty_list.append(agent_reward_info["agent_boundary_penalty"].copy())
        self.agent_nfz_penalty_list.append(agent_reward_info["agent_nfz_penalty"].copy())
        self.agent_shield_penalty_list.append(agent_reward_info["agent_shield_penalty"].copy())
        self.agent_raw_boundary_penalty_list.append(agent_reward_info["agent_raw_boundary_penalty"].copy())
        self.agent_raw_nfz_penalty_list.append(agent_reward_info["agent_raw_nfz_penalty"].copy())
        self.agent_illegal_flag_list.append(agent_reward_info["agent_illegal_flag"].copy())
        self.agent_urllc_success_count_list.append(traffic_metrics["urllc_success_each_uav"].copy())
        self.agent_urllc_violation_count_list.append(traffic_metrics["urllc_violation_each_uav"].copy())
        self.agent_mmtc_success_count_list.append(traffic_metrics["mmtc_success_each_uav"].copy())
        self.agent_mmtc_active_count_list.append(traffic_metrics["mmtc_active_each_uav"].copy())
        self.sum_rate_list.append(float(metrics["sum_rate"]))
        self.mean_rate_list.append(float(metrics["mean_rate"]))
        self.served_bits_list.append(float(metrics["served_bits"]))
        self.energy_list.append(float(slot_energy))
        self.boundary_penalty_list.append(float(boundary_penalty))
        self.nfz_penalty_list.append(float(nfz_penalty))
        self.nfz_violation_list.append(np.asarray(nfz_violation, dtype=np.float32).copy())
        self.shield_flag_list.append(float(self.prev_shield_flag))
        self.shield_penalty_list.append(float(self.prev_shield_penalty))
        self.raw_boundary_penalty_list.append(float(self.prev_raw_boundary_penalty))
        self.raw_nfz_penalty_list.append(float(self.prev_raw_nfz_penalty))
        self.raw_boundary_violation_list.append(self.prev_raw_boundary_violation.copy())
        self.raw_nfz_violation_list.append(self.prev_raw_nfz_violation.copy())
        self.active_uav_n_list.append(int(np.sum(self.active_mask)))

        self.embb_utility_list.append(float(slice_metrics["embb_utility"]))
        self.embb_mean_rate_list.append(float(slice_metrics["embb_mean_rate"]))
        self.embb_sum_rate_list.append(float(slice_metrics["embb_sum_rate"]))
        self.urllc_success_ratio_list.append(float(slice_metrics["urllc_success_ratio"]))
        self.urllc_violation_ratio_list.append(float(slice_metrics["urllc_violation_ratio"]))
        self.urllc_constraint_gap_list.append(float(slice_metrics["urllc_constraint_gap"]))
        self.mmtc_success_ratio_list.append(float(slice_metrics["mmtc_success_ratio"]))
        self.mmtc_active_ratio_list.append(float(slice_metrics["mmtc_active_ratio"]))
        self.mmtc_constraint_gap_list.append(float(slice_metrics["mmtc_constraint_gap"]))
        self.urllc_arrival_count_list.append(float(traffic_metrics.get("urllc_arrival_count", 0)))
        self.urllc_success_count_list.append(float(traffic_metrics.get("urllc_success_count", 0)))
        self.urllc_violation_count_list.append(float(traffic_metrics.get("urllc_violation_count", 0)))
        self.mmtc_active_count_list.append(float(traffic_metrics.get("mmtc_active_count", 0)))
        self.mmtc_success_count_list.append(float(traffic_metrics.get("mmtc_success_count", 0)))
        self.mmtc_sinr_ok_count_list.append(float(traffic_metrics.get("mmtc_sinr_ok_count", 0)))
        self.mmtc_bits_ok_count_list.append(float(traffic_metrics.get("mmtc_bits_ok_count", 0)))
        self.mmtc_full_rb_upper_count_list.append(float(traffic_metrics.get("mmtc_full_rb_upper_count", 0)))
        self.mmtc_fail_sinr_count_list.append(float(traffic_metrics.get("mmtc_fail_sinr_count", 0)))
        self.mmtc_fail_rb_count_list.append(float(traffic_metrics.get("mmtc_fail_rb_count", 0)))
        self.mmtc_fail_both_count_list.append(float(traffic_metrics.get("mmtc_fail_both_count", 0)))
        self.mmtc_sinr_ok_ratio_list.append(float(traffic_metrics.get("mmtc_sinr_ok_ratio", 1.0)))
        self.mmtc_bits_ok_ratio_list.append(float(traffic_metrics.get("mmtc_bits_ok_ratio", 1.0)))
        self.mmtc_full_rb_upper_ratio_list.append(float(traffic_metrics.get("mmtc_full_rb_upper_ratio", 1.0)))
        self.mmtc_fail_sinr_ratio_list.append(float(traffic_metrics.get("mmtc_fail_sinr_ratio", 0.0)))
        self.mmtc_fail_rb_ratio_list.append(float(traffic_metrics.get("mmtc_fail_rb_ratio", 0.0)))
        self.mmtc_fail_both_ratio_list.append(float(traffic_metrics.get("mmtc_fail_both_ratio", 0.0)))
        self.mmtc_avg_required_rb_list.append(float(traffic_metrics.get("mmtc_avg_required_rb", 0.0)))
        self.mmtc_avg_allocated_rb_list.append(float(traffic_metrics.get("mmtc_avg_allocated_rb", 0.0)))
        self.mmtc_total_budget_rb_list.append(float(traffic_metrics.get("mmtc_total_budget_rb", 0.0)))
        self.puncture_ratio_list.append(float(slice_metrics.get("puncture_ratio", 0.0)))
        self.punctured_rb_list.append(float(np.sum(resource_result.get("rb_punctured", np.zeros((main_cfg.uav_n, main_cfg.user_n), dtype=np.float32)))))
        self.embb_base_sum_rate_list.append(float(slice_metrics.get("embb_base_sum_rate", 0.0)))
        self.embb_effective_sum_rate_list.append(float(slice_metrics.get("embb_effective_sum_rate", slice_metrics.get("embb_sum_rate", 0.0))))
        self.embb_loss_rate_list.append(float(slice_metrics.get("embb_loss_rate", 0.0)))
        self.rate_deficit_mean_list.append(float(0.5 * (slice_metrics["urllc_constraint_gap"] + slice_metrics["mmtc_constraint_gap"])))

        self.fairness_list.append(float(metrics.get("fairness", 0.0)))

        self.action_list.append(action.copy())
        self.raw_action_list.append(raw_action.copy())
        self.executed_action_list.append(action.copy())
        self.heading_list.append(heading_for_move.copy())
        self.slice_budget_ratio_list.append(slice_budget_ratio.copy())
        self.slice_budget_bandwidth_list.append(resource_result.get("slice_budget_bandwidth", np.zeros((main_cfg.uav_n, main_cfg.slice_n), dtype=np.float32)).copy())
        self.slice_budget_power_list.append(resource_result.get("slice_budget_power", np.zeros((main_cfg.uav_n, main_cfg.slice_n), dtype=np.float32)).copy())
        self.rb_budget_list.append(resource_result.get("rb_budget", np.zeros((main_cfg.uav_n, main_cfg.slice_n), dtype=np.int32)).copy())
        self.rb_punctured_list.append(resource_result.get("rb_punctured", np.zeros((main_cfg.uav_n, main_cfg.user_n), dtype=np.int32)).copy())
        self.uav_pos_list.append(self.uav_pos.copy())
        self.user_pos_list.append(self.user_pos.copy())
        self.residual_energy_list.append(self.residual_energy.copy())
        self.active_mask_list.append(self.active_mask.copy())
        self.association_list.append(self.association.copy())
        self.associated_uav_list.append(self.associated_uav.copy())
        self.user_rate_list.append(self.last_user_rate.copy())

        self.step_id += 1
        self.slot_count += 1

        # 旧版固定时域终止逻辑（后续finite-data runtime override会替换）。
        terminated = bool(self.step_id >= main_cfg.max_step)
        truncated = False
        done = terminated

        self.prev_terminated = terminated
        self.prev_truncated = False
        self.prev_done_reason = "fixed_horizon" if terminated else ""

        if terminated:
            next_state = self._build_terminal_state(main_cfg)
        else:
            self._generate_slot_traffic(main_cfg, self.step_id)
            self.urllc_queue_bits_list.append(self.urllc_queue_bits.copy())
            self.urllc_deadline_list.append(self.urllc_deadline.copy())
            self.mmtc_active_list.append(self.mmtc_active.copy())
            self.mmtc_pending_bits_list.append(self.mmtc_pending_bits.copy())
            next_state = self._build_state(main_cfg)

        self.state = next_state.copy()
        return next_state.copy(), float(reward), bool(done)

    def _calc_reward(self, main_cfg, slice_metrics, boundary_penalty=0.0, nfz_penalty=0.0, shield_penalty=0.0):
        """
        全局通信任务reward。

        GlobalR只衡量网络通信目标：puncturing后的eMBB有效总速率，
        以及URLLC/mMTC低于目标成功率的约束gap。
        boundary、NFZ、shield和puncture ratio仍保留为独立诊断指标，
        但不重复混入GlobalR；逐UAV安全责任由_calc_agent_reward处理。
        """
        embb = float(slice_metrics["embb_utility"])
        urllc_gap = float(slice_metrics["urllc_constraint_gap"])
        mmtc_gap = float(slice_metrics["mmtc_constraint_gap"])

        reward = float(main_cfg.reward_embb_weight) * embb
        reward -= float(main_cfg.reward_urllc_constraint_weight) * urllc_gap
        reward -= float(main_cfg.reward_mmtc_constraint_weight) * mmtc_gap
        return float(reward)

    def _calc_boundary_penalty(self, main_cfg, boundary_violation):
        boundary_violation = np.asarray(boundary_violation, dtype=np.float32)
        denom = max(main_cfg.move_dist_each_step * main_cfg.uav_n, 1e-9)
        return float(np.sum(boundary_violation) / denom)

    def _calc_nfz_penalty(self, main_cfg, nfz_violation):
        nfz_violation = np.asarray(nfz_violation, dtype=np.float32)
        denom = max(main_cfg.move_dist_each_step * main_cfg.uav_n, 1e-9)
        return float(np.sum(nfz_violation) / denom)

    def get_ep_metric(self, main_cfg):
        actual_slot_n = max(int(self.slot_count), 1)
        actual_time = actual_slot_n * main_cfg.slot_time
        ep_avg_sum_rate = self.ep_total_served_bits / max(actual_time, 1e-9)
        ep_avg_mean_rate = ep_avg_sum_rate / float(main_cfg.user_n)
        # 能量管理关闭；兼容字段固定为0，避免产生无意义的除零巨大值。
        ep_energy_efficiency = 0.0

        ep_avg_boundary_penalty = float(np.mean(self.boundary_penalty_list)) if self.boundary_penalty_list else 0.0
        ep_avg_nfz_penalty = float(np.mean(self.nfz_penalty_list)) if self.nfz_penalty_list else 0.0
        ep_avg_shield_penalty = float(np.mean(self.shield_penalty_list)) if self.shield_penalty_list else 0.0
        ep_shield_count = int(np.sum(np.asarray(self.shield_flag_list, dtype=np.float32))) if self.shield_flag_list else 0
        ep_shield_ratio = float(ep_shield_count / max(len(self.shield_flag_list), 1))
        ep_avg_raw_boundary_penalty = float(np.mean(self.raw_boundary_penalty_list)) if self.raw_boundary_penalty_list else 0.0
        ep_avg_raw_nfz_penalty = float(np.mean(self.raw_nfz_penalty_list)) if self.raw_nfz_penalty_list else 0.0
        ep_nfz_polygons_padded, ep_nfz_vertex_count = main_cfg.get_padded_nfz_polygons()
        ep_nfz_count = int(len(main_cfg.nfz_polygons))
        ep_nfz_area_ratio = float(main_cfg.calc_nfz_area_ratio())

        ep_avg_embb_utility = float(np.mean(self.embb_utility_list)) if self.embb_utility_list else 0.0
        ep_avg_embb_mean_rate = float(np.mean(self.embb_mean_rate_list)) if self.embb_mean_rate_list else 0.0
        ep_avg_embb_sum_rate = float(np.mean(self.embb_sum_rate_list)) if self.embb_sum_rate_list else 0.0
        ep_avg_urllc_success_ratio = float(np.mean(self.urllc_success_ratio_list)) if self.urllc_success_ratio_list else 1.0
        ep_avg_urllc_violation_ratio = float(np.mean(self.urllc_violation_ratio_list)) if self.urllc_violation_ratio_list else 0.0
        ep_avg_urllc_constraint_gap = float(np.mean(self.urllc_constraint_gap_list)) if self.urllc_constraint_gap_list else 0.0
        ep_avg_mmtc_success_ratio = float(np.mean(self.mmtc_success_ratio_list)) if self.mmtc_success_ratio_list else 1.0
        ep_avg_mmtc_active_ratio = float(np.mean(self.mmtc_active_ratio_list)) if self.mmtc_active_ratio_list else 0.0
        ep_avg_mmtc_constraint_gap = float(np.mean(self.mmtc_constraint_gap_list)) if self.mmtc_constraint_gap_list else 0.0
        ep_avg_mmtc_sinr_ok_ratio = float(np.mean(self.mmtc_sinr_ok_ratio_list)) if self.mmtc_sinr_ok_ratio_list else 1.0
        ep_avg_mmtc_bits_ok_ratio = float(np.mean(self.mmtc_bits_ok_ratio_list)) if self.mmtc_bits_ok_ratio_list else 1.0
        ep_avg_mmtc_full_rb_upper_ratio = float(np.mean(self.mmtc_full_rb_upper_ratio_list)) if self.mmtc_full_rb_upper_ratio_list else 1.0
        ep_avg_mmtc_fail_sinr_ratio = float(np.mean(self.mmtc_fail_sinr_ratio_list)) if self.mmtc_fail_sinr_ratio_list else 0.0
        ep_avg_mmtc_fail_rb_ratio = float(np.mean(self.mmtc_fail_rb_ratio_list)) if self.mmtc_fail_rb_ratio_list else 0.0
        ep_avg_mmtc_fail_both_ratio = float(np.mean(self.mmtc_fail_both_ratio_list)) if self.mmtc_fail_both_ratio_list else 0.0
        ep_avg_mmtc_required_rb = float(np.mean(self.mmtc_avg_required_rb_list)) if self.mmtc_avg_required_rb_list else 0.0
        ep_avg_mmtc_allocated_rb = float(np.mean(self.mmtc_avg_allocated_rb_list)) if self.mmtc_avg_allocated_rb_list else 0.0
        ep_avg_mmtc_budget_rb = float(np.mean(self.mmtc_total_budget_rb_list)) if self.mmtc_total_budget_rb_list else 0.0
        ep_mmtc_sinr_ok_count = int(np.sum(np.asarray(self.mmtc_sinr_ok_count_list, dtype=np.float32))) if self.mmtc_sinr_ok_count_list else 0
        ep_mmtc_bits_ok_count = int(np.sum(np.asarray(self.mmtc_bits_ok_count_list, dtype=np.float32))) if self.mmtc_bits_ok_count_list else 0
        ep_mmtc_full_rb_upper_count = int(np.sum(np.asarray(self.mmtc_full_rb_upper_count_list, dtype=np.float32))) if self.mmtc_full_rb_upper_count_list else 0
        ep_mmtc_fail_sinr_count = int(np.sum(np.asarray(self.mmtc_fail_sinr_count_list, dtype=np.float32))) if self.mmtc_fail_sinr_count_list else 0
        ep_mmtc_fail_rb_count = int(np.sum(np.asarray(self.mmtc_fail_rb_count_list, dtype=np.float32))) if self.mmtc_fail_rb_count_list else 0
        ep_mmtc_fail_both_count = int(np.sum(np.asarray(self.mmtc_fail_both_count_list, dtype=np.float32))) if self.mmtc_fail_both_count_list else 0
        ep_avg_puncture_ratio = float(np.mean(self.puncture_ratio_list)) if self.puncture_ratio_list else 0.0
        ep_avg_punctured_rb = float(np.mean(self.punctured_rb_list)) if self.punctured_rb_list else 0.0
        ep_avg_embb_base_sum_rate = float(np.mean(self.embb_base_sum_rate_list)) if self.embb_base_sum_rate_list else 0.0
        ep_avg_embb_effective_sum_rate = float(np.mean(self.embb_effective_sum_rate_list)) if self.embb_effective_sum_rate_list else ep_avg_embb_sum_rate
        ep_avg_embb_loss_rate = float(np.mean(self.embb_loss_rate_list)) if self.embb_loss_rate_list else 0.0

        if len(self.slice_budget_ratio_list) > 0:
            ep_avg_slice_budget_ratio = np.mean(np.asarray(self.slice_budget_ratio_list, dtype=np.float32), axis=0).astype(np.float32)
            ep_final_slice_budget_ratio = np.asarray(self.slice_budget_ratio_list[-1], dtype=np.float32)
        else:
            ep_avg_slice_budget_ratio = np.ones((main_cfg.uav_n, main_cfg.slice_n), dtype=np.float32) / float(main_cfg.slice_n)
            ep_final_slice_budget_ratio = ep_avg_slice_budget_ratio.copy()

        total_urllc_success_ratio = 1.0 - self.urllc_total_violations / float(max(self.urllc_total_arrivals, 1))
        total_mmtc_success_ratio = self.mmtc_total_success / float(max(self.mmtc_total_active, 1)) if self.mmtc_total_active > 0 else 1.0

        m_n = int(main_cfg.uav_n)
        if len(self.agent_reward_list) > 0:
            step_agent_reward = np.asarray(self.agent_reward_list, dtype=np.float32)
            step_agent_sum_rate = np.asarray(self.agent_sum_rate_list, dtype=np.float32)
            step_agent_embb_utility = np.asarray(self.agent_embb_utility_list, dtype=np.float32)
            step_agent_user_count = np.asarray(self.agent_user_count_list, dtype=np.float32)
            step_agent_boundary_penalty = np.asarray(self.agent_boundary_penalty_list, dtype=np.float32)
            step_agent_nfz_penalty = np.asarray(self.agent_nfz_penalty_list, dtype=np.float32)
            step_agent_shield_penalty = np.asarray(self.agent_shield_penalty_list, dtype=np.float32)
            step_agent_raw_boundary_penalty = np.asarray(self.agent_raw_boundary_penalty_list, dtype=np.float32)
            step_agent_raw_nfz_penalty = np.asarray(self.agent_raw_nfz_penalty_list, dtype=np.float32)
            step_agent_illegal_flag = np.asarray(self.agent_illegal_flag_list, dtype=np.float32)
            step_agent_urllc_success_count = np.asarray(self.agent_urllc_success_count_list, dtype=np.float32)
            step_agent_urllc_violation_count = np.asarray(self.agent_urllc_violation_count_list, dtype=np.float32)
            step_agent_mmtc_success_count = np.asarray(self.agent_mmtc_success_count_list, dtype=np.float32)
            step_agent_mmtc_active_count = np.asarray(self.agent_mmtc_active_count_list, dtype=np.float32)
        else:
            step_agent_reward = np.zeros((0, m_n), dtype=np.float32)
            step_agent_sum_rate = np.zeros((0, m_n), dtype=np.float32)
            step_agent_embb_utility = np.zeros((0, m_n), dtype=np.float32)
            step_agent_user_count = np.zeros((0, m_n), dtype=np.float32)
            step_agent_boundary_penalty = np.zeros((0, m_n), dtype=np.float32)
            step_agent_nfz_penalty = np.zeros((0, m_n), dtype=np.float32)
            step_agent_shield_penalty = np.zeros((0, m_n), dtype=np.float32)
            step_agent_raw_boundary_penalty = np.zeros((0, m_n), dtype=np.float32)
            step_agent_raw_nfz_penalty = np.zeros((0, m_n), dtype=np.float32)
            step_agent_illegal_flag = np.zeros((0, m_n), dtype=np.float32)
            step_agent_urllc_success_count = np.zeros((0, m_n), dtype=np.float32)
            step_agent_urllc_violation_count = np.zeros((0, m_n), dtype=np.float32)
            step_agent_mmtc_success_count = np.zeros((0, m_n), dtype=np.float32)
            step_agent_mmtc_active_count = np.zeros((0, m_n), dtype=np.float32)

        ep_agent_reward = np.sum(step_agent_reward, axis=0) if step_agent_reward.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_avg_agent_reward = np.mean(step_agent_reward, axis=0) if step_agent_reward.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_avg_agent_sum_rate = np.mean(step_agent_sum_rate, axis=0) if step_agent_sum_rate.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_avg_agent_embb_utility = np.mean(step_agent_embb_utility, axis=0) if step_agent_embb_utility.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_avg_agent_user_count = np.mean(step_agent_user_count, axis=0) if step_agent_user_count.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_avg_agent_boundary_penalty = np.mean(step_agent_boundary_penalty, axis=0) if step_agent_boundary_penalty.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_avg_agent_nfz_penalty = np.mean(step_agent_nfz_penalty, axis=0) if step_agent_nfz_penalty.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_avg_agent_shield_penalty = np.mean(step_agent_shield_penalty, axis=0) if step_agent_shield_penalty.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_avg_agent_raw_boundary_penalty = np.mean(step_agent_raw_boundary_penalty, axis=0) if step_agent_raw_boundary_penalty.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_avg_agent_raw_nfz_penalty = np.mean(step_agent_raw_nfz_penalty, axis=0) if step_agent_raw_nfz_penalty.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_agent_illegal_count = np.sum(step_agent_illegal_flag, axis=0) if step_agent_illegal_flag.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_agent_illegal_ratio = np.mean(step_agent_illegal_flag, axis=0) if step_agent_illegal_flag.size > 0 else np.zeros(m_n, dtype=np.float32)

        urllc_success_sum_each = np.sum(step_agent_urllc_success_count, axis=0) if step_agent_urllc_success_count.size > 0 else np.zeros(m_n, dtype=np.float32)
        urllc_violation_sum_each = np.sum(step_agent_urllc_violation_count, axis=0) if step_agent_urllc_violation_count.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_agent_urllc_success_ratio = urllc_success_sum_each / np.maximum(urllc_success_sum_each + urllc_violation_sum_each, 1.0)
        mmtc_success_sum_each = np.sum(step_agent_mmtc_success_count, axis=0) if step_agent_mmtc_success_count.size > 0 else np.zeros(m_n, dtype=np.float32)
        mmtc_active_sum_each = np.sum(step_agent_mmtc_active_count, axis=0) if step_agent_mmtc_active_count.size > 0 else np.zeros(m_n, dtype=np.float32)
        ep_agent_mmtc_success_ratio = np.where(mmtc_active_sum_each > 0.0, mmtc_success_sum_each / np.maximum(mmtc_active_sum_each, 1.0), 1.0)

        return {
            "ep_reward": float(self.ep_reward),
            "ep_agent_reward": ep_agent_reward.astype(np.float32),
            "ep_avg_agent_reward": ep_avg_agent_reward.astype(np.float32),
            "ep_slot_count": int(self.slot_count),
            "ep_done_reason": str(self.prev_done_reason),
            "ep_total_bits": float(self.ep_total_served_bits),
            "ep_total_served_bits": float(self.ep_total_served_bits),
            "ep_total_energy": float(self.ep_total_energy),
            "ep_energy_efficiency": float(ep_energy_efficiency),
            "ep_avg_sum_rate": float(ep_avg_sum_rate),
            "ep_avg_mean_rate": float(ep_avg_mean_rate),
            "ep_avg_agent_sum_rate": ep_avg_agent_sum_rate.astype(np.float32),
            "ep_avg_agent_embb_utility": ep_avg_agent_embb_utility.astype(np.float32),
            "ep_avg_agent_user_count": ep_avg_agent_user_count.astype(np.float32),
            "ep_avg_agent_boundary_penalty": ep_avg_agent_boundary_penalty.astype(np.float32),
            "ep_avg_agent_nfz_penalty": ep_avg_agent_nfz_penalty.astype(np.float32),
            "ep_avg_agent_shield_penalty": ep_avg_agent_shield_penalty.astype(np.float32),
            "ep_avg_agent_raw_boundary_penalty": ep_avg_agent_raw_boundary_penalty.astype(np.float32),
            "ep_avg_agent_raw_nfz_penalty": ep_avg_agent_raw_nfz_penalty.astype(np.float32),
            "ep_agent_illegal_count": ep_agent_illegal_count.astype(np.float32),
            "ep_agent_illegal_ratio": ep_agent_illegal_ratio.astype(np.float32),
            "ep_agent_urllc_success_ratio": ep_agent_urllc_success_ratio.astype(np.float32),
            "ep_agent_mmtc_success_ratio": ep_agent_mmtc_success_ratio.astype(np.float32),

            "ep_avg_embb_utility": float(ep_avg_embb_utility),
            "ep_avg_embb_mean_rate": float(ep_avg_embb_mean_rate),
            "ep_avg_embb_sum_rate": float(ep_avg_embb_sum_rate),
            "ep_avg_urllc_success_ratio": float(ep_avg_urllc_success_ratio),
            "ep_avg_urllc_violation_ratio": float(ep_avg_urllc_violation_ratio),
            "ep_avg_urllc_constraint_gap": float(ep_avg_urllc_constraint_gap),
            "ep_avg_mmtc_success_ratio": float(ep_avg_mmtc_success_ratio),
            "ep_avg_mmtc_active_ratio": float(ep_avg_mmtc_active_ratio),
            "ep_avg_mmtc_constraint_gap": float(ep_avg_mmtc_constraint_gap),
            "ep_avg_mmtc_sinr_ok_ratio": float(ep_avg_mmtc_sinr_ok_ratio),
            "ep_avg_mmtc_bits_ok_ratio": float(ep_avg_mmtc_bits_ok_ratio),
            "ep_avg_mmtc_full_rb_upper_ratio": float(ep_avg_mmtc_full_rb_upper_ratio),
            "ep_avg_mmtc_fail_sinr_ratio": float(ep_avg_mmtc_fail_sinr_ratio),
            "ep_avg_mmtc_fail_rb_ratio": float(ep_avg_mmtc_fail_rb_ratio),
            "ep_avg_mmtc_fail_both_ratio": float(ep_avg_mmtc_fail_both_ratio),
            "ep_avg_mmtc_required_rb": float(ep_avg_mmtc_required_rb),
            "ep_avg_mmtc_allocated_rb": float(ep_avg_mmtc_allocated_rb),
            "ep_avg_mmtc_budget_rb": float(ep_avg_mmtc_budget_rb),
            "ep_mmtc_sinr_ok_count": int(ep_mmtc_sinr_ok_count),
            "ep_mmtc_bits_ok_count": int(ep_mmtc_bits_ok_count),
            "ep_mmtc_full_rb_upper_count": int(ep_mmtc_full_rb_upper_count),
            "ep_mmtc_fail_sinr_count": int(ep_mmtc_fail_sinr_count),
            "ep_mmtc_fail_rb_count": int(ep_mmtc_fail_rb_count),
            "ep_mmtc_fail_both_count": int(ep_mmtc_fail_both_count),
            "ep_avg_puncture_ratio": float(ep_avg_puncture_ratio),
            "ep_avg_punctured_rb": float(ep_avg_punctured_rb),
            "ep_avg_embb_base_sum_rate": float(ep_avg_embb_base_sum_rate),
            "ep_avg_embb_effective_sum_rate": float(ep_avg_embb_effective_sum_rate),
            "ep_avg_embb_loss_rate": float(ep_avg_embb_loss_rate),
            "ep_total_urllc_success_ratio": float(np.clip(total_urllc_success_ratio, 0.0, 1.0)),
            "ep_total_mmtc_success_ratio": float(np.clip(total_mmtc_success_ratio, 0.0, 1.0)),
            "ep_urllc_arrivals": int(self.urllc_total_arrivals),
            "ep_urllc_violations": int(self.urllc_total_violations),
            "ep_mmtc_active_count": int(self.mmtc_total_active),
            "ep_mmtc_success_count": int(self.mmtc_total_success),

            "ep_avg_boundary_penalty": float(ep_avg_boundary_penalty),
            "ep_avg_nfz_penalty": float(ep_avg_nfz_penalty),
            "ep_avg_shield_penalty": float(ep_avg_shield_penalty),
            "ep_shield_count": int(ep_shield_count),
            "ep_shield_ratio": float(ep_shield_ratio),
            "ep_avg_raw_boundary_penalty": float(ep_avg_raw_boundary_penalty),
            "ep_avg_raw_nfz_penalty": float(ep_avg_raw_nfz_penalty),
            "ep_nfz_count": int(ep_nfz_count),
            "ep_nfz_area_ratio": float(ep_nfz_area_ratio),
            "ep_nfz_polygons_padded": ep_nfz_polygons_padded.copy(),
            "ep_nfz_vertex_count": ep_nfz_vertex_count.copy(),
            "ep_avg_slice_budget_ratio": ep_avg_slice_budget_ratio.copy(),
            "ep_final_slice_budget_ratio": ep_final_slice_budget_ratio.copy(),
            "ep_final_active_uav_n": int(np.sum(self.active_mask)),

            # old aliases kept only for compatibility.
            "ep_avg_fairness": float(np.mean(self.fairness_list)) if self.fairness_list else 0.0,
            "ep_final_fairness": float(main_cfg.calc_jain_fairness(self.avg_user_rate)),
            "ep_avg_slice_rate_max_utility": float(ep_avg_embb_utility),
            "ep_avg_slice_rate_max_sum_rate": float(ep_avg_embb_sum_rate),
            "ep_avg_slice_guarantee_satisfaction": float(ep_avg_urllc_success_ratio),
            "ep_avg_slice_fair_utility": float(ep_avg_mmtc_success_ratio),
            "ep_avg_rate_outage_ratio": float(ep_avg_urllc_violation_ratio),
            "ep_avg_rate_deficit_mean": float(0.5 * (ep_avg_urllc_constraint_gap + ep_avg_mmtc_constraint_gap)),
            "ep_final_slice_rate_max_avg_rate": float(ep_avg_embb_mean_rate),
            "ep_final_slice_guarantee_avg_satisfaction": float(ep_avg_urllc_success_ratio),
            "ep_final_slice_fair_jain": float(ep_avg_mmtc_success_ratio),
            "ep_final_slice_fair_min_avg_rate": 0.0,

            "ep_uav_init_pos": self.uav_init_pos.copy(),
            "ep_final_uav_pos": self.uav_pos.copy(),
            "ep_user_pos": self.user_pos.copy(),
            "ep_residual_energy": self.residual_energy.copy(),
            "ep_active_mask": self.active_mask.copy(),
            "ep_avg_user_rate": self.avg_user_rate.copy(),
            "ep_last_user_rate": self.last_user_rate.copy(),
            "ep_unserved_time": np.zeros(main_cfg.user_n, dtype=np.float32),
            "ep_user_slice": main_cfg.user_slice.copy(),
            "ep_user_rate_req": main_cfg.user_rate_req.copy(),
            "ep_association": self.association.copy(),
            "ep_associated_uav": self.associated_uav.copy(),

            "ep_uav_pos_list": np.asarray(self.uav_pos_list, dtype=np.float32),
            "ep_user_pos_list": np.asarray(self.user_pos_list, dtype=np.float32),
            "ep_residual_energy_list": np.asarray(self.residual_energy_list, dtype=np.float32),
            "ep_active_mask_list": np.asarray(self.active_mask_list, dtype=bool),
            "ep_association_list": np.asarray(self.association_list, dtype=np.float32),
            "ep_associated_uav_list": np.asarray(self.associated_uav_list, dtype=np.int32),
            "ep_action_list": np.asarray(self.action_list, dtype=np.float32),
            "ep_raw_action_list": np.asarray(self.raw_action_list, dtype=np.float32),
            "ep_executed_action_list": np.asarray(self.executed_action_list, dtype=np.float32),
            "ep_heading_list": np.asarray(self.heading_list, dtype=np.float32),
            "ep_slice_budget_ratio_list": np.asarray(self.slice_budget_ratio_list, dtype=np.float32),
            "ep_slice_budget_bandwidth_list": np.asarray(self.slice_budget_bandwidth_list, dtype=np.float32),
            "ep_slice_budget_power_list": np.asarray(self.slice_budget_power_list, dtype=np.float32),
            "ep_rb_budget_list": np.asarray(self.rb_budget_list, dtype=np.int32),
            "ep_rb_punctured_list": np.asarray(self.rb_punctured_list, dtype=np.int32),
            "ep_user_rate_list": np.asarray(self.user_rate_list, dtype=np.float32),
            "ep_urllc_queue_bits_list": np.asarray(self.urllc_queue_bits_list, dtype=np.float32),
            "ep_urllc_deadline_list": np.asarray(self.urllc_deadline_list, dtype=np.float32),
            "ep_mmtc_active_list": np.asarray(self.mmtc_active_list, dtype=np.float32),
            "ep_mmtc_pending_bits_list": np.asarray(self.mmtc_pending_bits_list, dtype=np.float32),

            "step_reward": np.asarray(self.reward_list, dtype=np.float32),
            "step_agent_reward": step_agent_reward.astype(np.float32),
            "step_agent_sum_rate": step_agent_sum_rate.astype(np.float32),
            "step_agent_embb_utility": step_agent_embb_utility.astype(np.float32),
            "step_agent_user_count": step_agent_user_count.astype(np.float32),
            "step_agent_boundary_penalty": step_agent_boundary_penalty.astype(np.float32),
            "step_agent_nfz_penalty": step_agent_nfz_penalty.astype(np.float32),
            "step_agent_shield_penalty": step_agent_shield_penalty.astype(np.float32),
            "step_agent_raw_boundary_penalty": step_agent_raw_boundary_penalty.astype(np.float32),
            "step_agent_raw_nfz_penalty": step_agent_raw_nfz_penalty.astype(np.float32),
            "step_agent_illegal_flag": step_agent_illegal_flag.astype(np.float32),
            "step_agent_urllc_success_count": step_agent_urllc_success_count.astype(np.float32),
            "step_agent_urllc_violation_count": step_agent_urllc_violation_count.astype(np.float32),
            "step_agent_mmtc_success_count": step_agent_mmtc_success_count.astype(np.float32),
            "step_agent_mmtc_active_count": step_agent_mmtc_active_count.astype(np.float32),
            "step_sum_rate": np.asarray(self.sum_rate_list, dtype=np.float32),
            "step_mean_rate": np.asarray(self.mean_rate_list, dtype=np.float32),
            "step_served_bits": np.asarray(self.served_bits_list, dtype=np.float32),
            "step_energy": np.asarray(self.energy_list, dtype=np.float32),
            "step_boundary_penalty": np.asarray(self.boundary_penalty_list, dtype=np.float32),
            "step_nfz_penalty": np.asarray(self.nfz_penalty_list, dtype=np.float32),
            "step_nfz_violation": np.asarray(self.nfz_violation_list, dtype=np.float32),
            "step_shield_flag": np.asarray(self.shield_flag_list, dtype=np.float32),
            "step_shield_penalty": np.asarray(self.shield_penalty_list, dtype=np.float32),
            "step_raw_boundary_penalty": np.asarray(self.raw_boundary_penalty_list, dtype=np.float32),
            "step_raw_nfz_penalty": np.asarray(self.raw_nfz_penalty_list, dtype=np.float32),
            "step_raw_boundary_violation": np.asarray(self.raw_boundary_violation_list, dtype=np.float32),
            "step_raw_nfz_violation": np.asarray(self.raw_nfz_violation_list, dtype=np.float32),
            "step_active_uav_n": np.asarray(self.active_uav_n_list, dtype=np.int32),
            "step_embb_utility": np.asarray(self.embb_utility_list, dtype=np.float32),
            "step_embb_sum_rate": np.asarray(self.embb_sum_rate_list, dtype=np.float32),
            "step_urllc_success_ratio": np.asarray(self.urllc_success_ratio_list, dtype=np.float32),
            "step_urllc_violation_ratio": np.asarray(self.urllc_violation_ratio_list, dtype=np.float32),
            "step_mmtc_success_ratio": np.asarray(self.mmtc_success_ratio_list, dtype=np.float32),
            "step_mmtc_active_ratio": np.asarray(self.mmtc_active_ratio_list, dtype=np.float32),
            "step_urllc_arrival_count": np.asarray(self.urllc_arrival_count_list, dtype=np.float32),
            "step_urllc_success_count": np.asarray(self.urllc_success_count_list, dtype=np.float32),
            "step_urllc_violation_count": np.asarray(self.urllc_violation_count_list, dtype=np.float32),
            "step_mmtc_active_count": np.asarray(self.mmtc_active_count_list, dtype=np.float32),
            "step_mmtc_success_count": np.asarray(self.mmtc_success_count_list, dtype=np.float32),
            "step_puncture_ratio": np.asarray(self.puncture_ratio_list, dtype=np.float32),
            "step_punctured_rb": np.asarray(self.punctured_rb_list, dtype=np.float32),
            "step_embb_base_sum_rate": np.asarray(self.embb_base_sum_rate_list, dtype=np.float32),
            "step_embb_effective_sum_rate": np.asarray(self.embb_effective_sum_rate_list, dtype=np.float32),
            "step_embb_loss_rate": np.asarray(self.embb_loss_rate_list, dtype=np.float32),
            "step_fairness": np.asarray(self.fairness_list, dtype=np.float32),
            "step_slice_rate_max_utility": np.asarray(self.embb_utility_list, dtype=np.float32),
            "step_slice_rate_max_sum_rate": np.asarray(self.embb_sum_rate_list, dtype=np.float32),
            "step_slice_guarantee_satisfaction": np.asarray(self.urllc_success_ratio_list, dtype=np.float32),
            "step_slice_fair_utility": np.asarray(self.mmtc_success_ratio_list, dtype=np.float32),
            "step_rate_outage_ratio": np.asarray(self.urllc_violation_ratio_list, dtype=np.float32),
            "step_rate_deficit_mean": np.asarray(self.rate_deficit_mean_list, dtype=np.float32),
        }

    def _decision_user_pos(self, main_cfg):
        """返回当前策略决策可观测的用户位置。"""
        if bool(getattr(main_cfg, "use_perfect_next_user_position_observation", True)):
            return self.get_service_user_pos(main_cfg)
        return np.asarray(self.user_pos, dtype=np.float32).copy()

    def _build_state(self, main_cfg):
        # 默认直接包含本step通信阶段实际使用的w_{t+1}；可通过配置切回w_t做消融。
        decision_user_pos = self._decision_user_pos(main_cfg)
        return main_cfg.build_state(
            step_idx=self.step_id,
            uav_pos=self.uav_pos,
            user_pos=decision_user_pos,
            residual_energy=self.residual_energy,
            avg_user_rate=self.avg_user_rate,
            urllc_queue_bits=self.urllc_queue_bits,
            urllc_deadline=self.urllc_deadline,
            mmtc_active=self.mmtc_active,
            mmtc_pending_bits=self.mmtc_pending_bits,
            prev_heading=self.prev_heading,
            associated_uav=self.associated_uav,
            ep_id=self.ep_id,
        )

    def _parse_action(self, main_cfg, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != main_cfg.action_dim:
            raise ValueError(f"动作维度错误：期望 {main_cfg.action_dim}，实际 {action.size}")
        return np.clip(action, main_cfg.action_low, main_cfg.action_high).astype(np.float32)

    def get_obs(self, main_cfg):
        if self.ep_id is None:
            raise RuntimeError("请先调用 env.reset()。")
        if self.state is None or np.allclose(self.state, 0.0):
            return self._build_terminal_obs(main_cfg)
        decision_user_pos = self._decision_user_pos(main_cfg)
        return main_cfg.build_local_obs_all(
            uav_pos=self.uav_pos,
            user_pos=decision_user_pos,
            residual_energy=self.residual_energy,
            avg_user_rate=self.avg_user_rate,
            urllc_queue_bits=self.urllc_queue_bits,
            urllc_deadline=self.urllc_deadline,
            mmtc_active=self.mmtc_active,
            mmtc_pending_bits=self.mmtc_pending_bits,
            prev_heading=self.prev_heading,
            associated_uav=self.associated_uav,
            ep_id=self.ep_id,
            step_idx=self.step_id,
        )

    def _build_terminal_state(self, main_cfg):
        return np.zeros(main_cfg.state_dim, dtype=np.float32)

    def _build_terminal_obs(self, main_cfg):
        return np.zeros((main_cfg.agent_n, main_cfg.local_obs_dim), dtype=np.float32)


class GaussianNoise:
    """对每个二维方向动作在角度域添加高斯噪声。"""

    def __init__(self, action_dim, action_low=-1.0, action_high=1.0, max_sigma=0.35, min_sigma=0.035, decay_period=80000):
        self.action_dim = int(action_dim)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.max_sigma = float(max_sigma)
        self.min_sigma = float(min_sigma)
        self.decay_period = max(int(decay_period), 1)

    def get_sigma(self, step):
        step = int(step)
        frac = min(float(step) / float(self.decay_period), 1.0)
        return float(self.max_sigma - frac * (self.max_sigma - self.min_sigma))

    def get_action(self, action, step):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"动作维度错误：期望 {self.action_dim}，实际 {action.size}")
        if self.action_dim % 2 != 0:
            raise ValueError("角度域探索要求action_dim为2的整数倍。")

        action_mat = action.reshape(-1, 2)
        theta = np.arctan2(action_mat[:, 1], action_mat[:, 0])
        sigma = self.get_sigma(step)
        theta = theta + np.random.normal(
            loc=0.0,
            scale=sigma,
            size=theta.shape,
        ).astype(np.float32)

        noisy = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        return noisy.reshape(-1).astype(np.float32)


# ============================================================================
# Finite-data completion-time environment overrides
# ============================================================================

_ORIGINAL_ENV_CLEAR_LISTS = Env._clear_episode_lists
_ORIGINAL_ENV_RECORD_INITIAL = Env._record_initial_state


def _finite_clear_episode_lists(self):
    _ORIGINAL_ENV_CLEAR_LISTS(self)
    self.remaining_data_bits_list = []
    self.remaining_data_ratio_list = []
    self.unfinished_user_n_list = []
    self.completed_user_n_list = []
    self.newly_completed_n_list = []
    self.task_progress_list = []
    self.task_reward_list = []
    self.served_bits_user_list = []
    self.completion_slot_snapshot_list = []


Env._clear_episode_lists = _finite_clear_episode_lists


def _finite_reset(self, main_cfg, ep_id=0):
    self.ep_id = int(ep_id)
    self.step_id = 0
    self.slot_count = 0

    main_cfg.set_episode_nfz(ep_id=self.ep_id)
    self.nfz_polygons = [poly.copy() for poly in main_cfg.nfz_polygons]
    _, self.nfz_vertex_count = main_cfg.get_padded_nfz_polygons()

    self.uav_init_pos = main_cfg.sample_uav_init_pos(ep_id=self.ep_id)
    self.uav_pos = self.uav_init_pos.copy()
    self.user_pos = main_cfg.sample_user_pos(ep_id=self.ep_id)
    self.user_speed, self.user_heading = main_cfg.sample_user_motion_state(ep_id=self.ep_id)

    self.residual_energy = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.active_mask = np.ones(main_cfg.uav_n, dtype=bool)
    self.avg_user_rate = np.zeros(main_cfg.user_n, dtype=np.float32)
    self.last_user_rate = np.zeros(main_cfg.user_n, dtype=np.float32)
    self.association = np.zeros((main_cfg.uav_n, main_cfg.user_n), dtype=np.float32)
    self.associated_uav = -np.ones(main_cfg.user_n, dtype=np.int32)
    self.prev_heading = np.zeros(main_cfg.uav_n, dtype=np.float32)

    # Legacy traffic fields remain neutral for compatibility.
    self.urllc_packets = [[] for _ in range(main_cfg.user_n)]
    self.urllc_queue_bits = np.zeros(main_cfg.user_n, dtype=np.float32)
    self.urllc_deadline = np.zeros(main_cfg.user_n, dtype=np.float32)
    self.urllc_arrival_count_current = 0
    self.urllc_total_arrivals = 0
    self.urllc_total_success = 0
    self.urllc_total_violations = 0
    self.mmtc_active = np.zeros(main_cfg.user_n, dtype=np.float32)
    self.mmtc_pending_bits = np.zeros(main_cfg.user_n, dtype=np.float32)
    self.mmtc_active_count_current = 0
    self.mmtc_total_active = 0
    self.mmtc_total_success = 0

    # File sizes are sampled directly, independent of initial geometry/channel.
    self.initial_data_bits = main_cfg.sample_initial_data_bits(ep_id=self.ep_id)
    self.reference_user_rate = np.full(
        main_cfg.user_n,
        float(main_cfg.nominal_workload_rate_bps),
        dtype=np.float32,
    )
    self.target_service_slots = main_cfg.sample_target_service_slots(
        ep_id=self.ep_id,
        initial_data_bits=self.initial_data_bits,
    )
    self.remaining_data_bits = self.initial_data_bits.copy()
    self.completed_mask = np.zeros(main_cfg.user_n, dtype=bool)
    self.completion_slot = -np.ones(main_cfg.user_n, dtype=np.int32)

    self.prev_reward = 0.0
    self.prev_agent_reward = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.prev_agent_sum_rate = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.prev_agent_embb_utility = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.prev_agent_user_count = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.prev_agent_illegal_flag = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.prev_sum_rate = 0.0
    self.prev_mean_rate = 0.0
    self.prev_served_bits = 0.0
    self.prev_energy = 0.0
    self.prev_boundary_penalty = 0.0
    self.prev_nfz_penalty = 0.0
    self.prev_nfz_violation = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.prev_raw_action = np.zeros(main_cfg.action_dim, dtype=np.float32)
    self.prev_executed_action = np.zeros(main_cfg.action_dim, dtype=np.float32)
    self.prev_shield_flag = 0.0
    self.prev_shield_penalty = 0.0
    self.prev_raw_boundary_penalty = 0.0
    self.prev_raw_nfz_penalty = 0.0
    self.prev_raw_boundary_violation = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.prev_raw_nfz_violation = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.prev_active_uav_n = main_cfg.uav_n
    self.prev_done_reason = ""
    self.prev_terminated = False
    self.prev_truncated = False
    self.prev_slice_metrics = {}
    self.prev_puncture_ratio = 0.0
    self.prev_embb_base_sum_rate = 0.0
    self.prev_embb_effective_sum_rate = 0.0
    self.prev_embb_loss_rate = 0.0

    self.ep_reward = 0.0
    self.ep_agent_reward = np.zeros(main_cfg.uav_n, dtype=np.float32)
    self.ep_total_rate = 0.0
    self.ep_total_served_bits = 0.0
    self.ep_total_energy = 0.0

    self._clear_episode_lists()
    _ORIGINAL_ENV_RECORD_INITIAL(self)
    self.remaining_data_bits_list = [self.remaining_data_bits.copy()]
    self.remaining_data_ratio_list = [
        (self.remaining_data_bits / np.maximum(self.initial_data_bits, 1.0)).copy()
    ]
    self.unfinished_user_n_list = [int(main_cfg.user_n)]
    self.completed_user_n_list = [0]
    self.completion_slot_snapshot_list = [self.completion_slot.copy()]

    self.state = self._build_state(main_cfg)
    return self.state.copy()


Env.reset = _finite_reset


def _finite_build_state(self, main_cfg):
    decision_user_pos = self._decision_user_pos(main_cfg)
    return main_cfg.build_state(
        step_idx=self.step_id,
        uav_pos=self.uav_pos,
        user_pos=decision_user_pos,
        initial_data_bits=self.initial_data_bits,
        remaining_data_bits=self.remaining_data_bits,
        reference_rate=self.reference_user_rate,
        ep_id=self.ep_id,
    )


Env._build_state = _finite_build_state


def _finite_get_obs(self, main_cfg):
    if self.prev_terminated:
        return self._build_terminal_obs(main_cfg)
    decision_user_pos = self._decision_user_pos(main_cfg)
    return main_cfg.build_local_obs_all(
        uav_pos=self.uav_pos,
        user_pos=decision_user_pos,
        step_idx=self.step_id,
        initial_data_bits=self.initial_data_bits,
        remaining_data_bits=self.remaining_data_bits,
        reference_rate=self.reference_user_rate,
        ep_id=self.ep_id,
    )


Env.get_obs = _finite_get_obs


def _finite_task_reward(self, main_cfg, before_remaining, after_remaining, truncated=False):
    """Minimum-time reward with policy-invariant potential-based shaping.

    Base task reward is -1 for every executed slot.  The shaping term is
        gamma * Phi(s_next) - Phi(s)
    with Phi(s) = -mean_k(remaining_k / initial_k).  For a true terminal state
    Phi(s_next)=0.  This is the standard potential-based form for the same gamma
    used by the critic, so it adds dense learning signal without changing the
    optimal policy of the original makespan objective.
    """
    init = np.maximum(self.initial_data_bits, 1.0)
    before_ratio = float(np.mean(np.maximum(before_remaining, 0.0) / init))
    after_ratio = float(np.mean(np.maximum(after_remaining, 0.0) / init))
    progress = max(before_ratio - after_ratio, 0.0)

    phi_before = -before_ratio
    true_terminal = bool(
        np.all(np.asarray(after_remaining) <= float(main_cfg.task_data_eps_bits))
    )
    phi_after = 0.0 if true_terminal else -after_ratio
    shaping = (
        float(main_cfg.reward_potential_gamma) * phi_after - phi_before
    )

    reward = -float(main_cfg.reward_step_cost)
    reward += float(main_cfg.reward_progress_weight) * shaping
    if truncated:
        unfinished_ratio = float(
            np.mean(after_remaining > float(main_cfg.task_data_eps_bits))
        )
        reward -= float(main_cfg.reward_time_limit_penalty) * (
            0.5 * after_ratio + 0.5 * unfinished_ratio
        )
    return float(reward), float(progress)


def _finite_agent_reward(
    self,
    main_cfg,
    task_reward,
    boundary_violation,
    nfz_violation,
    shield_info,
    local_progress_contribution=None,
):
    """Build per-UAV rewards with centered contribution credit.

    All UAVs retain the same global task reward.  A centered local contribution
    term improves credit assignment without changing the team-average reward:
        beta * (p_m - mean_m p_m).
    Illegal raw actions still receive an explicit per-agent hard penalty.
    """
    m_n = int(main_cfg.uav_n)
    denom = max(float(main_cfg.move_dist_each_step), 1e-9)
    boundary_violation = np.asarray(boundary_violation, dtype=np.float32).reshape(-1)
    nfz_violation = np.asarray(nfz_violation, dtype=np.float32).reshape(-1)
    raw_b = np.asarray(
        shield_info.get("raw_boundary_violation", np.zeros(m_n)), dtype=np.float32
    ).reshape(-1)
    raw_n = np.asarray(
        shield_info.get("raw_nfz_violation", np.zeros(m_n)), dtype=np.float32
    ).reshape(-1)
    near_penalty = np.asarray(
        shield_info.get("safety_distance_penalty_each", np.zeros(m_n)),
        dtype=np.float32,
    ).reshape(-1)

    if local_progress_contribution is None:
        local_progress = np.zeros(m_n, dtype=np.float32)
    else:
        local_progress = np.asarray(
            local_progress_contribution, dtype=np.float32
        ).reshape(-1)
        if local_progress.size != m_n:
            raise ValueError(
                f"local_progress_contribution维度错误：期望{m_n}，实际{local_progress.size}"
            )
    centered_progress = local_progress - float(np.mean(local_progress))
    credit_weight = float(getattr(main_cfg, "reward_agent_credit_weight", 0.0))
    credit_reward = credit_weight * centered_progress

    b_pen = boundary_violation / denom
    n_pen = nfz_violation / denom
    raw_b_pen = raw_b / denom
    raw_n_pen = raw_n / denom
    illegal_eps = float(getattr(main_cfg, "shield_violation_eps", 1e-6))
    illegal = ((raw_b + raw_n) > illegal_eps).astype(np.float32)

    reward_each = np.full(m_n, float(task_reward), dtype=np.float32)
    reward_each += credit_reward.astype(np.float32)
    reward_each -= float(main_cfg.reward_boundary_weight) * b_pen
    reward_each -= float(main_cfg.reward_nfz_weight) * n_pen
    reward_each -= near_penalty

    fixed = float(getattr(main_cfg, "reward_illegal_action_fixed_penalty", 15.0))
    raw_pen = (
        float(main_cfg.reward_boundary_weight) * raw_b_pen
        + float(main_cfg.reward_nfz_weight) * raw_n_pen
    )
    mask = illegal > 0.5
    reward_each[mask] = -(fixed + raw_pen[mask])
    return {
        "agent_reward": reward_each.astype(np.float32),
        "agent_illegal_flag": illegal.astype(np.float32),
        "agent_boundary_penalty": b_pen.astype(np.float32),
        "agent_nfz_penalty": n_pen.astype(np.float32),
        "agent_shield_penalty": (illegal * (raw_b_pen + raw_n_pen)).astype(np.float32),
        "agent_raw_boundary_penalty": raw_b_pen.astype(np.float32),
        "agent_raw_nfz_penalty": raw_n_pen.astype(np.float32),
        "agent_local_progress_contribution": local_progress.astype(np.float32),
        "agent_centered_progress": centered_progress.astype(np.float32),
        "agent_credit_reward": credit_reward.astype(np.float32),
    }


def _finite_step(self, main_cfg, ep_id, action):
    if self.ep_id is None:
        raise RuntimeError("请先调用 env.reset()。")
    if int(ep_id) != int(self.ep_id):
        raise ValueError(f"ep_id不一致：环境={self.ep_id}, 输入={ep_id}")

    if np.all(self.completed_mask):
        self.prev_terminated = True
        self.prev_truncated = False
        self.prev_done_reason = "all_users_completed"
        terminal = self._build_terminal_state(main_cfg)
        self.state = terminal.copy()
        return terminal, 0.0, True

    if self.step_id >= int(main_cfg.max_step):
        self.prev_terminated = False
        self.prev_truncated = True
        self.prev_done_reason = "time_limit"
        current = self._build_state(main_cfg)
        self.state = current.copy()
        return current, 0.0, True

    self.prev_done_reason = ""
    self.prev_terminated = False
    self.prev_truncated = False

    raw_action = self._parse_action(main_cfg, action)
    executed_action, shield_info = self._shield_action_by_nearest_safe_heading(
        main_cfg=main_cfg,
        raw_action=raw_action,
    )
    heading, slice_budget_ratio = main_cfg.parse_joint_action(
        action_cont=executed_action, active_mask=self.active_mask
    )

    next_uav_pos, _, boundary_violation, nfz_violation = main_cfg.move_uav_one_slot(
        uav_pos=self.uav_pos,
        heading=heading,
        active_mask=self.active_mask,
    )
    self.uav_pos = next_uav_pos.copy()
    self.user_pos, self.user_speed, self.user_heading = (
        self.preview_user_state_for_current_step(main_cfg)
    )

    before_remaining = self.remaining_data_bits.copy()
    unfinished_before = before_remaining > float(main_cfg.task_data_eps_bits)
    resource_result = serve_one_slot(
        cfg=main_cfg,
        uav_pos=self.uav_pos,
        user_pos=self.user_pos,
        residual_energy=self.residual_energy,
        active_mask=self.active_mask,
        ep_id=self.ep_id,
        step_idx=self.step_id,
        avg_user_rate=self.avg_user_rate,
        unfinished_mask=unfinished_before,
        remaining_data_bits=before_remaining,
    )

    user_rate = resource_result["user_rate"].astype(np.float32)
    served_bits_user = resource_result["served_bits_user"].astype(np.float32)
    self.remaining_data_bits = np.maximum(
        before_remaining - served_bits_user, 0.0
    ).astype(np.float32)
    completed_after = self.remaining_data_bits <= float(main_cfg.task_data_eps_bits)
    newly_completed = completed_after & (~self.completed_mask)
    self.completion_slot[newly_completed] = int(self.step_id + 1)
    self.completed_mask = completed_after.copy()

    service_association = resource_result["association"].astype(np.float32)
    service_associated_uav = resource_result["associated_uav"].astype(np.int32)
    # Association after the slot represents the next decision state: users that
    # finished in this slot are immediately removed.  Service attribution below
    # still uses service_associated_uav from the just-completed slot.
    self.association = service_association.copy()
    self.associated_uav = service_associated_uav.copy()
    self.association[:, self.completed_mask] = 0.0
    self.associated_uav[self.completed_mask] = -1
    self.last_user_rate = user_rate.copy()
    self.avg_user_rate = update_average_user_rate(
        self.avg_user_rate, user_rate, self.slot_count
    )

    # Advance the episode clock before deciding termination/truncation.
    self.step_id += 1
    self.slot_count += 1
    terminated = bool(np.all(self.completed_mask))
    truncated = bool((self.step_id >= int(main_cfg.max_step)) and not terminated)
    episode_done = terminated or truncated

    task_reward, progress = _finite_task_reward(
        self,
        main_cfg,
        before_remaining,
        self.remaining_data_bits,
        truncated=truncated,
    )
    reward = float(task_reward)
    agent_reward_info = _finite_agent_reward(
        self,
        main_cfg,
        task_reward=task_reward,
        boundary_violation=boundary_violation,
        nfz_violation=nfz_violation,
        shield_info=shield_info,
    )
    agent_reward = agent_reward_info["agent_reward"]

    per_uav_perf = self._calc_per_uav_performance(
        main_cfg=main_cfg,
        user_rate=user_rate,
        associated_uav=service_associated_uav,
    )
    metrics = resource_result["metrics"]

    self.prev_heading = heading.copy()
    self.prev_sum_rate = float(metrics["sum_rate"])
    self.prev_mean_rate = float(metrics["mean_rate"])
    self.prev_served_bits = float(np.sum(served_bits_user))
    self.prev_boundary_penalty = float(np.sum(boundary_violation) / max(main_cfg.move_dist_each_step * main_cfg.uav_n, 1e-9))
    self.prev_nfz_penalty = float(np.sum(nfz_violation) / max(main_cfg.move_dist_each_step * main_cfg.uav_n, 1e-9))
    self.prev_nfz_violation = np.asarray(nfz_violation, dtype=np.float32).copy()
    self.prev_raw_action = raw_action.copy()
    self.prev_executed_action = executed_action.copy()
    self.prev_shield_flag = float(shield_info.get("shield_flag", 0.0))
    self.prev_shield_penalty = float(shield_info.get("shield_penalty", 0.0))
    self.prev_raw_boundary_penalty = float(shield_info.get("raw_boundary_penalty", 0.0))
    self.prev_raw_nfz_penalty = float(shield_info.get("raw_nfz_penalty", 0.0))
    self.prev_raw_boundary_violation = np.asarray(
        shield_info.get("raw_boundary_violation", np.zeros(main_cfg.uav_n)),
        dtype=np.float32,
    ).copy()
    self.prev_raw_nfz_violation = np.asarray(
        shield_info.get("raw_nfz_violation", np.zeros(main_cfg.uav_n)),
        dtype=np.float32,
    ).copy()
    self.prev_reward = reward
    self.prev_agent_reward = agent_reward.copy()
    self.prev_agent_sum_rate = per_uav_perf["sum_rate_each_uav"].copy()
    self.prev_agent_embb_utility = (
        per_uav_perf["sum_rate_each_uav"] / 1.0e6
    ).astype(np.float32)
    self.prev_agent_user_count = per_uav_perf["user_count_each_uav"].copy()
    self.prev_agent_illegal_flag = agent_reward_info["agent_illegal_flag"].copy()
    self.prev_terminated = terminated
    self.prev_truncated = truncated
    self.prev_done_reason = (
        "all_users_completed" if terminated else "time_limit" if truncated else ""
    )

    self.ep_reward += reward
    self.ep_agent_reward += agent_reward
    self.ep_total_rate += float(metrics["sum_rate"])
    self.ep_total_served_bits += float(np.sum(served_bits_user))

    self.reward_list.append(reward)
    self.agent_reward_list.append(agent_reward.copy())
    self.agent_sum_rate_list.append(per_uav_perf["sum_rate_each_uav"].copy())
    self.agent_embb_utility_list.append(self.prev_agent_embb_utility.copy())
    self.agent_user_count_list.append(per_uav_perf["user_count_each_uav"].copy())
    self.agent_boundary_penalty_list.append(agent_reward_info["agent_boundary_penalty"].copy())
    self.agent_nfz_penalty_list.append(agent_reward_info["agent_nfz_penalty"].copy())
    self.agent_shield_penalty_list.append(agent_reward_info["agent_shield_penalty"].copy())
    self.agent_raw_boundary_penalty_list.append(agent_reward_info["agent_raw_boundary_penalty"].copy())
    self.agent_raw_nfz_penalty_list.append(agent_reward_info["agent_raw_nfz_penalty"].copy())
    self.agent_illegal_flag_list.append(agent_reward_info["agent_illegal_flag"].copy())

    self.sum_rate_list.append(float(metrics["sum_rate"]))
    self.mean_rate_list.append(float(metrics["mean_rate"]))
    self.served_bits_list.append(float(np.sum(served_bits_user)))
    self.boundary_penalty_list.append(self.prev_boundary_penalty)
    self.nfz_penalty_list.append(self.prev_nfz_penalty)
    self.shield_flag_list.append(self.prev_shield_flag)
    self.shield_penalty_list.append(self.prev_shield_penalty)
    self.raw_boundary_penalty_list.append(self.prev_raw_boundary_penalty)
    self.raw_nfz_penalty_list.append(self.prev_raw_nfz_penalty)
    self.raw_boundary_violation_list.append(self.prev_raw_boundary_violation.copy())
    self.raw_nfz_violation_list.append(self.prev_raw_nfz_violation.copy())
    self.active_uav_n_list.append(main_cfg.uav_n)
    self.fairness_list.append(float(metrics.get("fairness", 1.0)))

    self.action_list.append(executed_action.copy())
    self.raw_action_list.append(raw_action.copy())
    self.executed_action_list.append(executed_action.copy())
    self.heading_list.append(heading.copy())
    self.slice_budget_ratio_list.append(slice_budget_ratio.copy())
    self.slice_budget_bandwidth_list.append(resource_result["slice_budget_bandwidth"].copy())
    self.slice_budget_power_list.append(resource_result["slice_budget_power"].copy())
    self.rb_budget_list.append(resource_result["rb_budget"].copy())
    self.rb_punctured_list.append(resource_result["rb_punctured"].copy())
    self.uav_pos_list.append(self.uav_pos.copy())
    self.user_pos_list.append(self.user_pos.copy())
    self.residual_energy_list.append(self.residual_energy.copy())
    self.active_mask_list.append(self.active_mask.copy())
    self.association_list.append(self.association.copy())
    self.associated_uav_list.append(self.associated_uav.copy())
    self.user_rate_list.append(self.last_user_rate.copy())

    self.remaining_data_bits_list.append(self.remaining_data_bits.copy())
    self.remaining_data_ratio_list.append(
        self.remaining_data_bits / np.maximum(self.initial_data_bits, 1.0)
    )
    self.unfinished_user_n_list.append(int(np.sum(~self.completed_mask)))
    self.completed_user_n_list.append(int(np.sum(self.completed_mask)))
    self.newly_completed_n_list.append(int(np.sum(newly_completed)))
    self.task_progress_list.append(float(progress))
    self.task_reward_list.append(float(task_reward))
    self.served_bits_user_list.append(served_bits_user.copy())
    self.completion_slot_snapshot_list.append(self.completion_slot.copy())

    if terminated:
        next_state = self._build_terminal_state(main_cfg)
    else:
        # For time-limit truncation this is the real final state, allowing the
        # replay transition to bootstrap because replay_done=False.
        next_state = self._build_state(main_cfg)
    self.state = next_state.copy()
    return next_state.copy(), reward, bool(episode_done)


Env.step = _finite_step


def _finite_get_ep_metric(self, main_cfg):
    slots = int(self.slot_count)
    completed_n = int(np.sum(self.completed_mask))
    unfinished_n = int(main_cfg.user_n - completed_n)
    all_completed = bool(completed_n == main_cfg.user_n)
    total_initial = float(np.sum(self.initial_data_bits))
    total_remaining = float(np.sum(self.remaining_data_bits))
    completion_ratio = float(completed_n / max(main_cfg.user_n, 1))
    data_completion_ratio = float(1.0 - total_remaining / max(total_initial, 1.0))

    completed_slots = self.completion_slot[self.completion_slot > 0]
    mean_completion_slot = float(np.mean(completed_slots)) if completed_slots.size > 0 else float(slots)
    p95_completion_slot = float(np.percentile(completed_slots, 95)) if completed_slots.size > 0 else float(slots)
    max_completion_slot = float(np.max(completed_slots)) if all_completed and completed_slots.size > 0 else float(main_cfg.max_step)

    def mean_list(x, default=0.0):
        return float(np.mean(np.asarray(x, dtype=np.float32))) if len(x) > 0 else float(default)

    m_n = int(main_cfg.uav_n)
    step_agent_reward = np.asarray(self.agent_reward_list, dtype=np.float32)
    step_agent_rate = np.asarray(self.agent_sum_rate_list, dtype=np.float32)
    step_agent_users = np.asarray(self.agent_user_count_list, dtype=np.float32)
    step_illegal = np.asarray(self.agent_illegal_flag_list, dtype=np.float32)
    if step_agent_reward.size == 0:
        step_agent_reward = np.zeros((0, m_n), dtype=np.float32)
        step_agent_rate = np.zeros((0, m_n), dtype=np.float32)
        step_agent_users = np.zeros((0, m_n), dtype=np.float32)
        step_illegal = np.zeros((0, m_n), dtype=np.float32)

    raw_b = np.asarray(self.raw_boundary_violation_list, dtype=np.float32)
    raw_n = np.asarray(self.raw_nfz_violation_list, dtype=np.float32)
    raw_b_ratio = float(np.mean(raw_b > 1e-8)) if raw_b.size > 0 else 0.0
    raw_n_ratio = float(np.mean(raw_n > 1e-8)) if raw_n.size > 0 else 0.0

    padded, counts = main_cfg.get_padded_nfz_polygons()
    avg_sum_rate = float(self.ep_total_rate / max(slots, 1))
    avg_mean_rate = float(avg_sum_rate / max(main_cfg.user_n, 1))

    metric = {
        "ep_reward": float(self.ep_reward),
        "ep_slot_count": slots,
        "ep_done_reason": str(self.prev_done_reason),
        "ep_terminated": bool(self.prev_terminated),
        "ep_truncated": bool(self.prev_truncated),
        "ep_all_users_completed": all_completed,
        "ep_time_limit_hit": bool(self.prev_truncated),
        "ep_completed_user_n": completed_n,
        "ep_unfinished_user_n": unfinished_n,
        "ep_user_completion_ratio": completion_ratio,
        "ep_data_completion_ratio": data_completion_ratio,
        "ep_total_initial_data_bits": total_initial,
        "ep_total_remaining_data_bits": total_remaining,
        "ep_total_served_bits": float(self.ep_total_served_bits),
        "ep_mean_completion_slot": mean_completion_slot,
        "ep_p95_completion_slot": p95_completion_slot,
        "ep_max_completion_slot": max_completion_slot,
        "ep_completion_slot": self.completion_slot.copy(),
        "ep_target_service_slots": self.target_service_slots.copy(),
        "ep_reference_user_rate": self.reference_user_rate.copy(),
        "ep_initial_data_bits": self.initial_data_bits.copy(),
        "ep_remaining_data_bits": self.remaining_data_bits.copy(),
        "ep_avg_sum_rate": avg_sum_rate,
        "ep_avg_mean_rate": avg_mean_rate,
        "ep_avg_fairness": mean_list(self.fairness_list, 1.0),
        "ep_final_fairness": float(self.fairness_list[-1]) if self.fairness_list else 1.0,
        "ep_avg_progress": mean_list(self.task_progress_list),
        "ep_avg_newly_completed_n": mean_list(self.newly_completed_n_list),
        "ep_agent_reward": np.sum(step_agent_reward, axis=0) if step_agent_reward.size else np.zeros(m_n, dtype=np.float32),
        "ep_avg_agent_reward": np.mean(step_agent_reward, axis=0) if step_agent_reward.size else np.zeros(m_n, dtype=np.float32),
        "ep_avg_agent_sum_rate": np.mean(step_agent_rate, axis=0) if step_agent_rate.size else np.zeros(m_n, dtype=np.float32),
        "ep_avg_agent_embb_utility": (np.mean(step_agent_rate, axis=0) / 1e6).astype(np.float32) if step_agent_rate.size else np.zeros(m_n, dtype=np.float32),
        "ep_avg_agent_user_count": np.mean(step_agent_users, axis=0) if step_agent_users.size else np.zeros(m_n, dtype=np.float32),
        "ep_agent_illegal_count": np.sum(step_illegal, axis=0) if step_illegal.size else np.zeros(m_n, dtype=np.float32),
        "ep_agent_illegal_ratio": np.mean(step_illegal, axis=0) if step_illegal.size else np.zeros(m_n, dtype=np.float32),
        "ep_raw_boundary_attempt_ratio": raw_b_ratio,
        "ep_raw_nfz_attempt_ratio": raw_n_ratio,
        "ep_avg_boundary_penalty": mean_list(self.boundary_penalty_list),
        "ep_avg_nfz_penalty": mean_list(self.nfz_penalty_list),
        "ep_avg_shield_penalty": mean_list(self.shield_penalty_list),
        "ep_shield_count": int(np.sum(np.asarray(self.shield_flag_list, dtype=np.float32))) if self.shield_flag_list else 0,
        "ep_shield_ratio": mean_list(self.shield_flag_list),
        "ep_avg_raw_boundary_penalty": mean_list(self.raw_boundary_penalty_list),
        "ep_avg_raw_nfz_penalty": mean_list(self.raw_nfz_penalty_list),
        "ep_nfz_count": int(len(main_cfg.nfz_polygons)),
        "ep_nfz_area_ratio": float(main_cfg.calc_nfz_area_ratio()),
        "ep_nfz_polygons_padded": padded.copy(),
        "ep_nfz_vertex_count": counts.copy(),
        "ep_uav_pos_list": np.asarray(self.uav_pos_list, dtype=np.float32),
        "ep_user_pos_list": np.asarray(self.user_pos_list, dtype=np.float32),
        "ep_active_mask_list": np.asarray(self.active_mask_list, dtype=bool),
        "ep_associated_uav_list": np.asarray(self.associated_uav_list, dtype=np.int32),
        "ep_remaining_data_bits_list": np.asarray(self.remaining_data_bits_list, dtype=np.float32),
        "ep_unfinished_user_n_list": np.asarray(self.unfinished_user_n_list, dtype=np.int32),
        "ep_completed_user_n_list": np.asarray(self.completed_user_n_list, dtype=np.int32),
        "ep_raw_action_list": np.asarray(self.raw_action_list, dtype=np.float32),
        "ep_executed_action_list": np.asarray(self.executed_action_list, dtype=np.float32),
        "ep_residual_energy": self.residual_energy.copy(),
        "ep_final_active_uav_n": int(main_cfg.uav_n),
        "ep_total_energy": 0.0,
        "ep_energy_efficiency": 0.0,
    }

    # Compatibility metrics for old result readers; they no longer define the task.
    metric.update({
        "ep_avg_embb_utility": avg_sum_rate / 1e6,
        "ep_avg_embb_sum_rate": avg_sum_rate,
        "ep_avg_embb_mean_rate": avg_mean_rate,
        "ep_avg_urllc_success_ratio": 1.0,
        "ep_avg_urllc_violation_ratio": 0.0,
        "ep_avg_urllc_constraint_gap": 0.0,
        "ep_avg_mmtc_success_ratio": 1.0,
        "ep_avg_mmtc_active_ratio": 0.0,
        "ep_avg_mmtc_constraint_gap": 0.0,
        "ep_avg_slice_rate_max_utility": avg_sum_rate / 1e6,
        "ep_avg_slice_guarantee_satisfaction": 1.0,
        "ep_avg_slice_fair_utility": 1.0,
        "ep_avg_rate_outage_ratio": 0.0,
        "ep_avg_rate_deficit_mean": 0.0,
        "ep_avg_puncture_ratio": 0.0,
        "ep_avg_punctured_rb": 0.0,
        "ep_avg_embb_base_sum_rate": avg_sum_rate,
        "ep_avg_embb_effective_sum_rate": avg_sum_rate,
        "ep_avg_embb_loss_rate": 0.0,
        "ep_urllc_arrivals": 0,
        "ep_urllc_violations": 0,
        "ep_mmtc_active_count": 0,
        "ep_mmtc_success_count": 0,
        "ep_agent_urllc_success_ratio": np.ones(m_n, dtype=np.float32),
        "ep_agent_mmtc_success_ratio": np.ones(m_n, dtype=np.float32),
        "ep_avg_slice_budget_ratio": np.zeros((m_n, main_cfg.slice_n), dtype=np.float32),
        "ep_final_slice_budget_ratio": np.zeros((m_n, main_cfg.slice_n), dtype=np.float32),
    })
    return metric


Env.get_ep_metric = _finite_get_ep_metric


def _finite_terminal_state(self, main_cfg):
    return np.zeros(main_cfg.state_dim, dtype=np.float32)


def _finite_terminal_obs(self, main_cfg):
    return np.zeros((main_cfg.uav_n, main_cfg.local_obs_dim), dtype=np.float32)


Env._build_terminal_state = _finite_terminal_state
Env._build_terminal_obs = _finite_terminal_obs

# ============================================================================
# Three-service constrained finite-completion-time runtime overrides
# ============================================================================
_PRE_SLICE_ENV_CLEAR_LISTS = Env._clear_episode_lists
_PRE_SLICE_ENV_RESET = Env.reset
_PRE_SLICE_ENV_GET_EP_METRIC = Env.get_ep_metric


def _sliced_clear_episode_lists(self):
    _PRE_SLICE_ENV_CLEAR_LISTS(self)
    self.urllc_window_reliability_list = []
    self.urllc_cumulative_reliability_list = []
    self.urllc_dense_gap_list = []
    self.urllc_fail_link_count_list = []
    self.urllc_fail_cap_count_list = []
    self.urllc_fail_budget_count_list = []
    self.urllc_fail_available_count_list = []
    self.urllc_fail_resource_count_list = []
    self.urllc_required_rb_samples_list = []
    self.urllc_tti_with_arrivals_list = []
    self.urllc_cap_hit_tti_count_list = []
    self.urllc_full_rb_hit_tti_count_list = []
    self.urllc_idle_rb_used_list = []
    self.urllc_total_rb_used_list = []
    self.urllc_max_punctured_rb_one_uav_tti_list = []
    self.urllc_max_total_rb_one_uav_tti_list = []
    self.mmtc_connection_ratio_list = []
    self.mmtc_episode_mean_ratio_list = []
    self.mmtc_dense_gap_list = []
    self.mmtc_floor_gap_list = []
    self.qos_penalty_list = []
    self.completion_potential_list = []
    self.potential_user_bottleneck_list = []
    self.potential_uav_bottleneck_list = []
    self.potential_mean_remaining_list = []
    self.agent_local_progress_contribution_list = []
    self.agent_credit_reward_list = []


def _current_qos_global_features(self, main_cfg):
    w = max(int(main_cfg.urllc_reward_window_slots), 1)
    arr_hist = list(getattr(self, "urllc_slot_arrival_history", []))
    suc_hist = list(getattr(self, "urllc_slot_success_history", []))
    win_arr = int(np.sum(arr_hist[-w:])) if arr_hist else 0
    win_suc = int(np.sum(suc_hist[-w:])) if suc_hist else 0
    win_rel = float(win_suc / win_arr) if win_arr > 0 else 1.0
    cum_arr = int(getattr(self, "urllc_total_arrivals", 0))
    cum_suc = int(getattr(self, "urllc_total_success", 0))
    cum_rel = float(cum_suc / cum_arr) if cum_arr > 0 else 1.0
    win_gap = max(float(main_cfg.urllc_reliability_target) - win_rel, 0.0)
    win_load = min(
        float(win_arr) / max(float(main_cfg.urllc_window_arrival_norm), 1.0), 1.0
    )

    m_hist = list(getattr(self, "mmtc_slot_ratio_history", []))
    m_last = float(m_hist[-1]) if m_hist else float(getattr(self, "mmtc_initial_ratio", 1.0))
    m_avg = float(np.mean(m_hist)) if m_hist else m_last
    m_gap = max(float(main_cfg.mmtc_connectivity_target) - m_last, 0.0)
    m_floor_gap = max(float(main_cfg.mmtc_connectivity_floor) - m_last, 0.0)
    return np.array(
        [
            np.clip(win_rel, 0.0, 1.0),
            np.clip(win_gap, 0.0, 1.0),
            np.clip(cum_rel, 0.0, 1.0),
            np.clip(win_load, 0.0, 1.0),
            np.clip(m_last, 0.0, 1.0),
            np.clip(m_avg, 0.0, 1.0),
            np.clip(m_gap, 0.0, 1.0),
            np.clip(m_floor_gap, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def _sliced_reset(self, main_cfg, ep_id=0):
    state = _PRE_SLICE_ENV_RESET(self, main_cfg, ep_id=ep_id)
    embb_mask = np.zeros(main_cfg.user_n, dtype=bool)
    embb_mask[main_cfg.embb_user_idx] = True

    # Only eMBB files are completion tasks. URLLC and mMTC are continuing QoS
    # services and are therefore excluded from completed/unfinished file masks.
    self.completed_mask = ~embb_mask
    self.completed_mask[main_cfg.embb_user_idx] = (
        self.remaining_data_bits[main_cfg.embb_user_idx]
        <= float(main_cfg.task_data_eps_bits)
    )
    self.completion_slot = -np.ones(main_cfg.user_n, dtype=np.int32)

    self.urllc_slot_arrival_history = []
    self.urllc_slot_success_history = []
    self.urllc_slot_violation_history = []
    self.mmtc_slot_ratio_history = []
    self.mmtc_total_connections = 0
    self.mmtc_total_opportunities = 0
    self.mmtc_active = np.zeros(main_cfg.user_n, dtype=np.float32)
    self.mmtc_active[main_cfg.mmtc_user_idx] = 1.0
    self.mmtc_pending_bits = np.zeros(main_cfg.user_n, dtype=np.float32)

    # Initialize mMTC geometry awareness before the first action.
    qos_link = main_cfg.build_qos_link_features(self.uav_pos, self.user_pos)
    if main_cfg.mmtc_user_n > 0:
        self.mmtc_initial_ratio = float(
            np.mean(qos_link[main_cfg.mmtc_user_idx, 1])
        )
    else:
        self.mmtc_initial_ratio = 1.0
    self.prev_mmtc_connection_ratio = self.mmtc_initial_ratio
    self.prev_urllc_window_reliability = 1.0
    self.prev_urllc_cumulative_reliability = 1.0
    self.prev_urllc_constraint_gap = 0.0
    self.prev_mmtc_constraint_gap = max(
        float(main_cfg.mmtc_connectivity_target) - self.mmtc_initial_ratio, 0.0
    )
    self.prev_qos_penalty = 0.0

    self.unfinished_user_n_list = [int(np.sum(~self.completed_mask))]
    self.completed_user_n_list = [0]
    self.remaining_data_bits_list = [self.remaining_data_bits.copy()]
    ratio = np.zeros(main_cfg.user_n, dtype=np.float32)
    idx = main_cfg.embb_user_idx
    ratio[idx] = self.remaining_data_bits[idx] / np.maximum(self.initial_data_bits[idx], 1.0)
    self.remaining_data_ratio_list = [ratio]
    self.completion_slot_snapshot_list = [self.completion_slot.copy()]

    # Cache Phi(s_0).  Subsequent steps only compute Phi(s_{t+1}), avoiding two
    # extra deterministic channel/allocation evaluations per environment step.
    decision_user_pos = self._decision_user_pos(main_cfg)
    self.current_completion_potential_info = estimate_embb_completion_potential(
        cfg=main_cfg,
        uav_pos=self.uav_pos,
        user_pos=decision_user_pos,
        remaining_data_bits=self.remaining_data_bits,
        initial_data_bits=self.initial_data_bits,
        active_mask=self.active_mask,
    )
    self.completion_potential_list = [
        float(self.current_completion_potential_info["potential"])
    ]
    self.potential_user_bottleneck_list = [
        float(self.current_completion_potential_info["user_bottleneck"])
    ]
    self.potential_uav_bottleneck_list = [
        float(self.current_completion_potential_info["uav_bottleneck"])
    ]
    self.potential_mean_remaining_list = [
        float(self.current_completion_potential_info["mean_remaining_ratio"])
    ]

    self.state = self._build_state(main_cfg)
    return self.state.copy()


def _sliced_build_state_env(self, main_cfg):
    decision_user_pos = self._decision_user_pos(main_cfg)
    return main_cfg.build_state(
        step_idx=self.step_id,
        uav_pos=self.uav_pos,
        user_pos=decision_user_pos,
        initial_data_bits=self.initial_data_bits,
        remaining_data_bits=self.remaining_data_bits,
        reference_rate=self.reference_user_rate,
        qos_global_features=_current_qos_global_features(self, main_cfg),
        ep_id=self.ep_id,
    )


def _sliced_get_obs(self, main_cfg):
    if self.prev_terminated:
        return self._build_terminal_obs(main_cfg)
    decision_user_pos = self._decision_user_pos(main_cfg)
    return main_cfg.build_local_obs_all(
        uav_pos=self.uav_pos,
        user_pos=decision_user_pos,
        step_idx=self.step_id,
        initial_data_bits=self.initial_data_bits,
        remaining_data_bits=self.remaining_data_bits,
        reference_rate=self.reference_user_rate,
        qos_global_features=_current_qos_global_features(self, main_cfg),
        ep_id=self.ep_id,
    )


def _sliced_task_reward(
    self,
    main_cfg,
    before_remaining,
    after_remaining,
    current_mmtc_ratio,
    phi_before,
    phi_after,
    terminated=False,
    truncated=False,
):
    """Minimum-slot reward with makespan-aligned potential and dual QoS cost.

    The 180-slot limit is a sampling truncation only.  Therefore ``truncated``
    does not create a terminal/cutoff penalty and Phi(s_next) remains the actual
    non-terminal potential so replay can bootstrap consistently.
    """
    idx = np.asarray(main_cfg.embb_user_idx, dtype=np.int32)
    init = np.maximum(self.initial_data_bits[idx], 1.0)
    before = np.maximum(np.asarray(before_remaining, dtype=np.float32)[idx], 0.0)
    after = np.maximum(np.asarray(after_remaining, dtype=np.float32)[idx], 0.0)
    before_ratio = float(np.mean(before / init)) if idx.size else 0.0
    after_ratio = float(np.mean(after / init)) if idx.size else 0.0
    progress = max(before_ratio - after_ratio, 0.0)

    shaping = (
        float(main_cfg.reward_potential_gamma) * float(phi_after)
        - float(phi_before)
    )
    reward = -float(main_cfg.reward_step_cost)
    reward += float(main_cfg.reward_progress_weight) * shaping

    # Dense constraint residuals.  Multipliers are updated from episode-level
    # performance in ctd4_train.py; the environment only applies current prices.
    w = max(int(main_cfg.urllc_reward_window_slots), 1)
    win_arr = int(np.sum(self.urllc_slot_arrival_history[-w:]))
    win_suc = int(np.sum(self.urllc_slot_success_history[-w:]))
    win_rel = float(win_suc / win_arr) if win_arr > 0 else 1.0
    if win_arr >= int(main_cfg.urllc_reward_min_window_arrivals):
        u_gap = max(float(main_cfg.urllc_reliability_target) - win_rel, 0.0)
    else:
        u_gap = 0.0
    m_gap = max(
        float(main_cfg.mmtc_connectivity_target) - float(current_mmtc_ratio), 0.0
    )
    m_floor_gap = max(
        float(main_cfg.mmtc_connectivity_floor) - float(current_mmtc_ratio), 0.0
    )

    lambda_u = float(getattr(
        main_cfg,
        "qos_lagrange_urllc",
        getattr(main_cfg, "reward_urllc_constraint_weight", 0.0),
    ))
    lambda_m = float(getattr(
        main_cfg,
        "qos_lagrange_mmtc",
        getattr(main_cfg, "reward_mmtc_constraint_weight", 0.0),
    ))
    qos_penalty = lambda_u * u_gap + lambda_m * m_gap
    reward -= qos_penalty

    # No time-limit or terminal QoS lump penalty.  True completion is already
    # represented by Phi(s_terminal)=0; truncation remains bootstrap-able.
    return float(reward), float(progress), {
        "urllc_window_reliability": float(win_rel),
        "urllc_dense_gap": float(u_gap),
        "mmtc_dense_gap": float(m_gap),
        "mmtc_floor_gap": float(m_floor_gap),
        "qos_penalty": float(qos_penalty),
        "qos_lambda_urllc": float(lambda_u),
        "qos_lambda_mmtc": float(lambda_m),
        "potential_before": float(phi_before),
        "potential_after": float(phi_after),
        "potential_shaping": float(shaping),
        "terminal_urllc_gap": 0.0,
        "terminal_mmtc_gap": 0.0,
    }


def _sliced_step(self, main_cfg, ep_id, action):
    if self.ep_id is None:
        raise RuntimeError("请先调用 env.reset()。")
    if int(ep_id) != int(self.ep_id):
        raise ValueError(f"ep_id不一致：环境={self.ep_id}, 输入={ep_id}")

    embb_idx = np.asarray(main_cfg.embb_user_idx, dtype=np.int32)
    if np.all(self.completed_mask[embb_idx]):
        self.prev_terminated = True
        self.prev_truncated = False
        self.prev_done_reason = "all_embb_files_completed"
        terminal = self._build_terminal_state(main_cfg)
        self.state = terminal.copy()
        return terminal, 0.0, True
    if self.step_id >= int(main_cfg.max_step):
        self.prev_terminated = False
        self.prev_truncated = True
        self.prev_done_reason = "time_limit"
        current = self._build_state(main_cfg)
        self.state = current.copy()
        return current, 0.0, True

    self.prev_done_reason = ""
    self.prev_terminated = False
    self.prev_truncated = False

    potential_before_info = getattr(self, "current_completion_potential_info", None)
    if potential_before_info is None:
        potential_before_info = estimate_embb_completion_potential(
            cfg=main_cfg,
            uav_pos=self.uav_pos,
            user_pos=self._decision_user_pos(main_cfg),
            remaining_data_bits=self.remaining_data_bits,
            initial_data_bits=self.initial_data_bits,
            active_mask=self.active_mask,
        )
    phi_before = float(potential_before_info["potential"])

    raw_action = self._parse_action(main_cfg, action)
    executed_action, shield_info = self._shield_action_by_nearest_safe_heading(
        main_cfg=main_cfg, raw_action=raw_action
    )
    heading, slice_budget_ratio = main_cfg.parse_joint_action(
        action_cont=executed_action, active_mask=self.active_mask
    )
    next_uav_pos, _, boundary_violation, nfz_violation = main_cfg.move_uav_one_slot(
        uav_pos=self.uav_pos, heading=heading, active_mask=self.active_mask
    )
    self.uav_pos = next_uav_pos.copy()
    self.user_pos, self.user_speed, self.user_heading = (
        self.preview_user_state_for_current_step(main_cfg)
    )

    before_remaining = self.remaining_data_bits.copy()
    unfinished_before = np.zeros(main_cfg.user_n, dtype=bool)
    unfinished_before[embb_idx] = (
        before_remaining[embb_idx] > float(main_cfg.task_data_eps_bits)
    )
    resource_result = serve_one_slot(
        cfg=main_cfg,
        uav_pos=self.uav_pos,
        user_pos=self.user_pos,
        residual_energy=self.residual_energy,
        active_mask=self.active_mask,
        ep_id=self.ep_id,
        step_idx=self.step_id,
        avg_user_rate=self.avg_user_rate,
        unfinished_mask=unfinished_before,
        remaining_data_bits=before_remaining,
    )

    user_rate = resource_result["user_rate"].astype(np.float32)
    served_bits_user = resource_result["served_bits_user"].astype(np.float32)
    self.remaining_data_bits = before_remaining.copy()
    self.remaining_data_bits[embb_idx] = np.maximum(
        before_remaining[embb_idx] - served_bits_user[embb_idx], 0.0
    )
    self.remaining_data_bits[np.setdiff1d(np.arange(main_cfg.user_n), embb_idx)] = 0.0

    completed_after = self.completed_mask.copy()
    completed_after[embb_idx] = (
        self.remaining_data_bits[embb_idx] <= float(main_cfg.task_data_eps_bits)
    )
    newly_completed = completed_after & (~self.completed_mask)
    self.completion_slot[newly_completed] = int(self.step_id + 1)
    self.completed_mask = completed_after

    service_association = resource_result["association"].astype(np.float32)
    service_associated_uav = resource_result["associated_uav"].astype(np.int32)
    self.association = service_association.copy()
    self.associated_uav = service_associated_uav.copy()
    self.association[:, self.completed_mask] = 0.0
    self.associated_uav[self.completed_mask] = -1
    self.last_user_rate = user_rate.copy()
    self.avg_user_rate = update_average_user_rate(
        self.avg_user_rate, user_rate, self.slot_count
    )

    traffic = resource_result["traffic_metrics"]
    arrivals = int(traffic["urllc_arrival_count"])
    successes = int(traffic["urllc_success_count"])
    violations = int(traffic["urllc_violation_count"])
    m_ratio = float(traffic["mmtc_success_ratio"])
    self.urllc_slot_arrival_history.append(arrivals)
    self.urllc_slot_success_history.append(successes)
    self.urllc_slot_violation_history.append(violations)
    self.urllc_total_arrivals += arrivals
    self.urllc_total_success += successes
    self.urllc_total_violations += violations
    self.mmtc_slot_ratio_history.append(m_ratio)
    self.mmtc_total_connections += int(traffic["mmtc_success_count"])
    self.mmtc_total_opportunities += int(traffic["mmtc_active_count"])
    self.mmtc_total_active += int(traffic["mmtc_active_count"])
    self.mmtc_total_success += int(traffic["mmtc_success_count"])

    self.step_id += 1
    self.slot_count += 1
    terminated = bool(np.all(self.completed_mask[embb_idx]))
    truncated = bool((self.step_id >= int(main_cfg.max_step)) and not terminated)
    episode_done = terminated or truncated

    if terminated:
        potential_after_info = {
            "potential": 0.0,
            "user_bottleneck": 0.0,
            "uav_bottleneck": 0.0,
            "mean_remaining_ratio": 0.0,
        }
    else:
        potential_after_info = estimate_embb_completion_potential(
            cfg=main_cfg,
            uav_pos=self.uav_pos,
            user_pos=self._decision_user_pos(main_cfg),
            remaining_data_bits=self.remaining_data_bits,
            initial_data_bits=self.initial_data_bits,
            active_mask=self.active_mask,
        )
    phi_after = float(potential_after_info["potential"])

    task_reward, progress, qos_info = _sliced_task_reward(
        self,
        main_cfg,
        before_remaining,
        self.remaining_data_bits,
        current_mmtc_ratio=m_ratio,
        phi_before=phi_before,
        phi_after=phi_after,
        terminated=terminated,
        truncated=truncated,
    )
    reward = float(task_reward)

    # Per-UAV normalized file progress.  The sum over UAVs equals the sum of
    # served_fraction across eMBB users; centering is done in _finite_agent_reward.
    local_progress_contribution = np.zeros(main_cfg.uav_n, dtype=np.float32)
    for k in embb_idx:
        m = int(service_associated_uav[k])
        if 0 <= m < int(main_cfg.uav_n):
            local_progress_contribution[m] += float(served_bits_user[k]) / max(
                float(self.initial_data_bits[k]), 1.0
            )

    agent_reward_info = _finite_agent_reward(
        self,
        main_cfg,
        task_reward=task_reward,
        boundary_violation=boundary_violation,
        nfz_violation=nfz_violation,
        shield_info=shield_info,
        local_progress_contribution=local_progress_contribution,
    )
    agent_reward = agent_reward_info["agent_reward"]
    per_uav_perf = self._calc_per_uav_performance(
        main_cfg=main_cfg,
        user_rate=user_rate,
        associated_uav=service_associated_uav,
    )
    metrics = resource_result["metrics"]

    qos_global = _current_qos_global_features(self, main_cfg)
    self.prev_heading = heading.copy()
    self.prev_sum_rate = float(metrics["sum_rate"])
    self.prev_mean_rate = float(metrics["mean_rate"])
    self.prev_served_bits = float(np.sum(served_bits_user))
    self.prev_boundary_penalty = float(
        np.sum(boundary_violation)
        / max(main_cfg.move_dist_each_step * main_cfg.uav_n, 1e-9)
    )
    self.prev_nfz_penalty = float(
        np.sum(nfz_violation)
        / max(main_cfg.move_dist_each_step * main_cfg.uav_n, 1e-9)
    )
    self.prev_nfz_violation = np.asarray(nfz_violation, dtype=np.float32).copy()
    self.prev_raw_action = raw_action.copy()
    self.prev_executed_action = executed_action.copy()
    self.prev_shield_flag = float(shield_info.get("shield_flag", 0.0))
    self.prev_shield_penalty = float(shield_info.get("shield_penalty", 0.0))
    self.prev_raw_boundary_penalty = float(shield_info.get("raw_boundary_penalty", 0.0))
    self.prev_raw_nfz_penalty = float(shield_info.get("raw_nfz_penalty", 0.0))
    self.prev_raw_boundary_violation = np.asarray(
        shield_info.get("raw_boundary_violation", np.zeros(main_cfg.uav_n)),
        dtype=np.float32,
    ).copy()
    self.prev_raw_nfz_violation = np.asarray(
        shield_info.get("raw_nfz_violation", np.zeros(main_cfg.uav_n)),
        dtype=np.float32,
    ).copy()
    self.prev_reward = reward
    self.prev_agent_reward = agent_reward.copy()
    self.prev_agent_sum_rate = per_uav_perf["sum_rate_each_uav"].copy()
    self.prev_agent_embb_utility = (
        per_uav_perf["sum_rate_each_uav"] / 1.0e6
    ).astype(np.float32)
    self.prev_agent_user_count = per_uav_perf["user_count_each_uav"].copy()
    self.prev_agent_illegal_flag = agent_reward_info["agent_illegal_flag"].copy()
    self.prev_terminated = terminated
    self.prev_truncated = truncated
    self.prev_done_reason = (
        "all_embb_files_completed" if terminated else "time_limit" if truncated else ""
    )
    self.prev_slice_metrics = dict(traffic)
    self.prev_puncture_ratio = float(traffic["puncture_ratio"])
    self.prev_embb_base_sum_rate = float(metrics["embb_base_sum_rate"])
    self.prev_embb_effective_sum_rate = float(metrics["embb_effective_sum_rate"])
    self.prev_embb_loss_rate = float(metrics["embb_loss_rate"])
    self.prev_mmtc_connection_ratio = m_ratio
    self.prev_urllc_window_reliability = float(qos_info["urllc_window_reliability"])
    self.prev_urllc_cumulative_reliability = float(qos_global[2])
    self.prev_urllc_constraint_gap = float(qos_info["urllc_dense_gap"])
    self.prev_mmtc_constraint_gap = float(qos_info["mmtc_dense_gap"])
    self.prev_qos_penalty = float(qos_info["qos_penalty"])
    self.prev_completion_potential = float(phi_after)
    self.prev_potential_user_bottleneck = float(
        potential_after_info["user_bottleneck"]
    )
    self.prev_potential_uav_bottleneck = float(
        potential_after_info["uav_bottleneck"]
    )
    self.prev_potential_mean_remaining = float(
        potential_after_info["mean_remaining_ratio"]
    )
    self.prev_agent_local_progress_contribution = agent_reward_info[
        "agent_local_progress_contribution"
    ].copy()
    self.prev_agent_credit_reward = agent_reward_info["agent_credit_reward"].copy()
    self.current_completion_potential_info = potential_after_info

    self.ep_reward += reward
    self.ep_agent_reward += agent_reward
    self.ep_total_rate += float(metrics["sum_rate"])
    self.ep_total_served_bits += float(np.sum(served_bits_user))

    self.reward_list.append(reward)
    self.agent_reward_list.append(agent_reward.copy())
    self.agent_sum_rate_list.append(per_uav_perf["sum_rate_each_uav"].copy())
    self.agent_embb_utility_list.append(self.prev_agent_embb_utility.copy())
    self.agent_user_count_list.append(per_uav_perf["user_count_each_uav"].copy())
    self.agent_boundary_penalty_list.append(agent_reward_info["agent_boundary_penalty"].copy())
    self.agent_nfz_penalty_list.append(agent_reward_info["agent_nfz_penalty"].copy())
    self.agent_shield_penalty_list.append(agent_reward_info["agent_shield_penalty"].copy())
    self.agent_raw_boundary_penalty_list.append(agent_reward_info["agent_raw_boundary_penalty"].copy())
    self.agent_raw_nfz_penalty_list.append(agent_reward_info["agent_raw_nfz_penalty"].copy())
    self.agent_illegal_flag_list.append(agent_reward_info["agent_illegal_flag"].copy())
    self.agent_urllc_success_count_list.append(traffic["urllc_success_each_uav"].copy())
    self.agent_urllc_violation_count_list.append(traffic["urllc_violation_each_uav"].copy())
    self.agent_mmtc_success_count_list.append(traffic["mmtc_success_each_uav"].copy())
    self.agent_mmtc_active_count_list.append(traffic["mmtc_active_each_uav"].copy())

    self.sum_rate_list.append(float(metrics["sum_rate"]))
    self.mean_rate_list.append(float(metrics["mean_rate"]))
    self.served_bits_list.append(float(np.sum(served_bits_user)))
    self.boundary_penalty_list.append(self.prev_boundary_penalty)
    self.nfz_penalty_list.append(self.prev_nfz_penalty)
    self.shield_flag_list.append(self.prev_shield_flag)
    self.shield_penalty_list.append(self.prev_shield_penalty)
    self.raw_boundary_penalty_list.append(self.prev_raw_boundary_penalty)
    self.raw_nfz_penalty_list.append(self.prev_raw_nfz_penalty)
    self.raw_boundary_violation_list.append(self.prev_raw_boundary_violation.copy())
    self.raw_nfz_violation_list.append(self.prev_raw_nfz_violation.copy())
    self.active_uav_n_list.append(main_cfg.uav_n)
    self.fairness_list.append(float(metrics.get("fairness", 1.0)))

    self.embb_utility_list.append(float(metrics["embb_effective_sum_rate"]) / 1e6)
    active_e_rates = user_rate[unfinished_before]
    self.embb_mean_rate_list.append(float(np.mean(active_e_rates)) if active_e_rates.size else 0.0)
    self.embb_sum_rate_list.append(float(metrics["embb_effective_sum_rate"]))
    slot_u_rel = float(successes / arrivals) if arrivals > 0 else 1.0
    self.urllc_success_ratio_list.append(slot_u_rel)
    self.urllc_violation_ratio_list.append(float(1.0 - slot_u_rel) if arrivals > 0 else 0.0)
    self.urllc_constraint_gap_list.append(float(qos_info["urllc_dense_gap"]))
    self.mmtc_success_ratio_list.append(m_ratio)
    self.mmtc_active_ratio_list.append(1.0)
    self.mmtc_constraint_gap_list.append(float(qos_info["mmtc_dense_gap"]))
    self.urllc_arrival_count_list.append(arrivals)
    self.urllc_success_count_list.append(successes)
    self.urllc_violation_count_list.append(violations)
    self.urllc_fail_link_count_list.append(
        int(traffic.get("urllc_fail_link_count", 0))
    )
    self.urllc_fail_cap_count_list.append(
        int(traffic.get("urllc_fail_cap_count", 0))
    )
    self.urllc_fail_budget_count_list.append(
        int(traffic.get("urllc_fail_budget_count", 0))
    )
    self.urllc_fail_available_count_list.append(
        int(traffic.get("urllc_fail_available_count", 0))
    )
    self.urllc_fail_resource_count_list.append(
        int(traffic.get("urllc_fail_resource_count", traffic.get("urllc_fail_budget_count", 0)))
    )
    self.urllc_required_rb_samples_list.append(
        np.asarray(traffic.get("urllc_required_rb_samples", np.zeros(0)), dtype=np.float32).copy()
    )
    self.urllc_tti_with_arrivals_list.append(
        int(traffic.get("urllc_tti_with_arrivals", 0))
    )
    self.urllc_cap_hit_tti_count_list.append(
        int(traffic.get("urllc_cap_hit_tti_count", 0))
    )
    self.urllc_full_rb_hit_tti_count_list.append(
        int(traffic.get("urllc_full_rb_hit_tti_count", traffic.get("urllc_cap_hit_tti_count", 0)))
    )
    self.urllc_idle_rb_used_list.append(
        float(np.sum(np.asarray(traffic.get("urllc_idle_rb_used_each_uav", np.zeros(main_cfg.uav_n)), dtype=np.float32)))
    )
    self.urllc_total_rb_used_list.append(
        float(np.sum(np.asarray(traffic.get("urllc_total_rb_used_each_uav", np.zeros(main_cfg.uav_n)), dtype=np.float32)))
    )
    self.urllc_max_punctured_rb_one_uav_tti_list.append(
        int(traffic.get("urllc_max_punctured_rb_one_uav_tti", 0))
    )
    self.urllc_max_total_rb_one_uav_tti_list.append(
        int(traffic.get("urllc_max_total_rb_one_uav_tti", 0))
    )
    self.mmtc_active_count_list.append(int(traffic["mmtc_active_count"]))
    self.mmtc_success_count_list.append(int(traffic["mmtc_success_count"]))
    self.mmtc_sinr_ok_count_list.append(int(traffic["mmtc_success_count"]))
    self.mmtc_sinr_ok_ratio_list.append(m_ratio)
    self.puncture_ratio_list.append(float(traffic["puncture_ratio"]))
    self.punctured_rb_list.append(float(np.sum(resource_result["rb_punctured"])))
    self.embb_base_sum_rate_list.append(float(metrics["embb_base_sum_rate"]))
    self.embb_effective_sum_rate_list.append(float(metrics["embb_effective_sum_rate"]))
    self.embb_loss_rate_list.append(float(metrics["embb_loss_rate"]))
    self.urllc_window_reliability_list.append(float(qos_info["urllc_window_reliability"]))
    self.urllc_cumulative_reliability_list.append(float(qos_global[2]))
    self.urllc_dense_gap_list.append(float(qos_info["urllc_dense_gap"]))
    self.mmtc_connection_ratio_list.append(m_ratio)
    self.mmtc_episode_mean_ratio_list.append(float(qos_global[5]))
    self.mmtc_dense_gap_list.append(float(qos_info["mmtc_dense_gap"]))
    self.mmtc_floor_gap_list.append(float(qos_info["mmtc_floor_gap"]))
    self.qos_penalty_list.append(float(qos_info["qos_penalty"]))
    self.completion_potential_list.append(float(phi_after))
    self.potential_user_bottleneck_list.append(
        float(potential_after_info["user_bottleneck"])
    )
    self.potential_uav_bottleneck_list.append(
        float(potential_after_info["uav_bottleneck"])
    )
    self.potential_mean_remaining_list.append(
        float(potential_after_info["mean_remaining_ratio"])
    )
    self.agent_local_progress_contribution_list.append(
        agent_reward_info["agent_local_progress_contribution"].copy()
    )
    self.agent_credit_reward_list.append(
        agent_reward_info["agent_credit_reward"].copy()
    )

    self.action_list.append(executed_action.copy())
    self.raw_action_list.append(raw_action.copy())
    self.executed_action_list.append(executed_action.copy())
    self.heading_list.append(heading.copy())
    self.slice_budget_ratio_list.append(slice_budget_ratio.copy())
    self.slice_budget_bandwidth_list.append(resource_result["slice_budget_bandwidth"].copy())
    self.slice_budget_power_list.append(resource_result["slice_budget_power"].copy())
    self.rb_budget_list.append(resource_result["rb_budget"].copy())
    self.rb_punctured_list.append(resource_result["rb_punctured"].copy())
    self.uav_pos_list.append(self.uav_pos.copy())
    self.user_pos_list.append(self.user_pos.copy())
    self.residual_energy_list.append(self.residual_energy.copy())
    self.active_mask_list.append(self.active_mask.copy())
    self.association_list.append(self.association.copy())
    self.associated_uav_list.append(self.associated_uav.copy())
    self.user_rate_list.append(self.last_user_rate.copy())
    self.urllc_queue_bits_list.append(np.zeros(main_cfg.user_n, dtype=np.float32))
    self.urllc_deadline_list.append(np.zeros(main_cfg.user_n, dtype=np.float32))
    self.mmtc_active_list.append(self.mmtc_active.copy())
    self.mmtc_pending_bits_list.append(self.mmtc_pending_bits.copy())

    self.remaining_data_bits_list.append(self.remaining_data_bits.copy())
    ratio = np.zeros(main_cfg.user_n, dtype=np.float32)
    ratio[embb_idx] = self.remaining_data_bits[embb_idx] / np.maximum(
        self.initial_data_bits[embb_idx], 1.0
    )
    self.remaining_data_ratio_list.append(ratio)
    self.unfinished_user_n_list.append(int(np.sum(~self.completed_mask)))
    self.completed_user_n_list.append(int(np.sum(self.completed_mask[embb_idx])))
    self.newly_completed_n_list.append(int(np.sum(newly_completed[embb_idx])))
    self.task_progress_list.append(float(progress))
    self.task_reward_list.append(float(task_reward))
    self.served_bits_user_list.append(served_bits_user.copy())
    self.completion_slot_snapshot_list.append(self.completion_slot.copy())

    if terminated:
        next_state = self._build_terminal_state(main_cfg)
    else:
        next_state = self._build_state(main_cfg)
    self.state = next_state.copy()
    return next_state.copy(), reward, bool(episode_done)


def _sliced_get_ep_metric(self, main_cfg):
    metric = _PRE_SLICE_ENV_GET_EP_METRIC(self, main_cfg)
    idx = np.asarray(main_cfg.embb_user_idx, dtype=np.int32)
    slots = int(self.slot_count)
    completed_n = int(np.sum(self.completed_mask[idx]))
    unfinished_n = int(main_cfg.embb_user_n - completed_n)
    all_completed = bool(completed_n == main_cfg.embb_user_n)
    total_initial = float(np.sum(self.initial_data_bits[idx]))
    total_remaining = float(np.sum(self.remaining_data_bits[idx]))
    user_completion = float(completed_n / max(main_cfg.embb_user_n, 1))
    data_completion = float(1.0 - total_remaining / max(total_initial, 1.0))
    completed_slots = self.completion_slot[idx]
    completed_slots = completed_slots[completed_slots > 0]

    ep_u_rel = (
        float(self.urllc_total_success / self.urllc_total_arrivals)
        if self.urllc_total_arrivals > 0 else 1.0
    )
    ep_m_rel = (
        float(self.mmtc_total_connections / self.mmtc_total_opportunities)
        if self.mmtc_total_opportunities > 0 else 1.0
    )
    u_gap = max(float(main_cfg.urllc_reliability_target) - ep_u_rel, 0.0)
    m_gap = max(float(main_cfg.mmtc_connectivity_target) - ep_m_rel, 0.0)
    qos_feasible = bool(u_gap <= 1e-12 and m_gap <= 1e-12)

    u_fail_link = int(np.sum(self.urllc_fail_link_count_list))
    u_fail_cap = int(np.sum(self.urllc_fail_cap_count_list))
    u_fail_budget = int(np.sum(self.urllc_fail_budget_count_list))
    u_fail_available = int(np.sum(self.urllc_fail_available_count_list))
    u_fail_resource = int(np.sum(self.urllc_fail_resource_count_list))
    u_arrivals = max(int(self.urllc_total_arrivals), 1)
    req_parts = [
        np.asarray(x, dtype=np.float32).reshape(-1)
        for x in self.urllc_required_rb_samples_list
        if np.asarray(x).size > 0
    ]
    req_samples = (
        np.concatenate(req_parts).astype(np.float32)
        if req_parts else np.zeros(0, dtype=np.float32)
    )
    u_tti = int(np.sum(self.urllc_tti_with_arrivals_list))
    u_cap_hit_tti = int(np.sum(self.urllc_cap_hit_tti_count_list))
    u_full_rb_hit_tti = int(np.sum(self.urllc_full_rb_hit_tti_count_list))

    metric.update({
        "ep_all_users_completed": all_completed,
        "ep_completed_user_n": completed_n,
        "ep_unfinished_user_n": unfinished_n,
        "ep_user_completion_ratio": user_completion,
        "ep_data_completion_ratio": data_completion,
        "ep_total_initial_data_bits": total_initial,
        "ep_total_remaining_data_bits": total_remaining,
        "ep_mean_completion_slot": float(np.mean(completed_slots)) if completed_slots.size else float(slots),
        "ep_p95_completion_slot": float(np.percentile(completed_slots, 95)) if completed_slots.size else float(slots),
        "ep_max_completion_slot": float(np.max(completed_slots)) if all_completed and completed_slots.size else float(main_cfg.max_step),
        "ep_avg_mean_rate": float(metric["ep_avg_sum_rate"] / max(main_cfg.embb_user_n, 1)),
        "ep_avg_embb_utility": float(metric["ep_avg_sum_rate"] / 1e6),
        "ep_avg_embb_sum_rate": float(metric["ep_avg_sum_rate"]),
        "ep_avg_embb_mean_rate": float(metric["ep_avg_sum_rate"] / max(main_cfg.embb_user_n, 1)),
        "ep_urllc_reliability": ep_u_rel,
        "ep_urllc_constraint_gap": u_gap,
        "ep_mmtc_connection_success_ratio": ep_m_rel,
        "ep_mmtc_constraint_gap": m_gap,
        "ep_qos_feasible": qos_feasible,
        "ep_avg_urllc_success_ratio": ep_u_rel,
        "ep_avg_urllc_violation_ratio": float(1.0 - ep_u_rel),
        "ep_avg_urllc_constraint_gap": u_gap,
        "ep_avg_mmtc_success_ratio": ep_m_rel,
        "ep_avg_mmtc_active_ratio": 1.0,
        "ep_avg_mmtc_constraint_gap": m_gap,
        "ep_avg_puncture_ratio": float(np.mean(self.puncture_ratio_list)) if self.puncture_ratio_list else 0.0,
        "ep_avg_punctured_rb": float(np.mean(self.punctured_rb_list)) if self.punctured_rb_list else 0.0,
        "ep_avg_embb_base_sum_rate": float(np.mean(self.embb_base_sum_rate_list)) if self.embb_base_sum_rate_list else 0.0,
        "ep_avg_embb_effective_sum_rate": float(np.mean(self.embb_effective_sum_rate_list)) if self.embb_effective_sum_rate_list else 0.0,
        "ep_avg_embb_loss_rate": float(np.mean(self.embb_loss_rate_list)) if self.embb_loss_rate_list else 0.0,
        "ep_urllc_arrivals": int(self.urllc_total_arrivals),
        "ep_urllc_successes": int(self.urllc_total_success),
        "ep_urllc_violations": int(self.urllc_total_violations),
        "ep_urllc_fail_link_count": u_fail_link,
        "ep_urllc_fail_cap_count": u_fail_cap,
        "ep_urllc_fail_budget_count": u_fail_budget,
        "ep_urllc_fail_available_count": u_fail_available,
        "ep_urllc_fail_link_ratio": float(u_fail_link / u_arrivals),
        "ep_urllc_fail_cap_ratio": float(u_fail_cap / u_arrivals),
        "ep_urllc_fail_budget_ratio": float(u_fail_budget / u_arrivals),
        "ep_urllc_fail_available_ratio": float(u_fail_available / u_arrivals),
        "ep_urllc_fail_resource_count": u_fail_resource,
        "ep_urllc_fail_resource_ratio": float(u_fail_resource / u_arrivals),
        "ep_urllc_avg_required_rb": float(np.mean(req_samples)) if req_samples.size else 0.0,
        "ep_urllc_p95_required_rb": float(np.percentile(req_samples, 95)) if req_samples.size else 0.0,
        "ep_urllc_cap_hit_tti_ratio": float(u_cap_hit_tti / max(u_tti, 1)),
        "ep_urllc_full_rb_hit_tti_ratio": float(u_full_rb_hit_tti / max(u_tti, 1)),
        "ep_urllc_avg_idle_rb_used": float(np.mean(self.urllc_idle_rb_used_list)) if self.urllc_idle_rb_used_list else 0.0,
        "ep_urllc_avg_total_rb_used": float(np.mean(self.urllc_total_rb_used_list)) if self.urllc_total_rb_used_list else 0.0,
        "ep_urllc_max_punctured_rb_one_uav_tti": float(
            np.max(self.urllc_max_punctured_rb_one_uav_tti_list)
        ) if self.urllc_max_punctured_rb_one_uav_tti_list else 0.0,
        "ep_urllc_max_total_rb_one_uav_tti": float(
            np.max(self.urllc_max_total_rb_one_uav_tti_list)
        ) if self.urllc_max_total_rb_one_uav_tti_list else 0.0,
        "ep_mmtc_active_count": int(self.mmtc_total_opportunities),
        "ep_mmtc_success_count": int(self.mmtc_total_connections),
        "ep_avg_qos_penalty": float(np.mean(self.qos_penalty_list)) if self.qos_penalty_list else 0.0,
        "ep_qos_lambda_urllc": float(getattr(main_cfg, "qos_lagrange_urllc", 0.0)),
        "ep_qos_lambda_mmtc": float(getattr(main_cfg, "qos_lagrange_mmtc", 0.0)),
        "ep_final_completion_potential": float(self.completion_potential_list[-1]) if self.completion_potential_list else 0.0,
        "ep_avg_potential_user_bottleneck": float(np.mean(self.potential_user_bottleneck_list)) if self.potential_user_bottleneck_list else 0.0,
        "ep_avg_potential_uav_bottleneck": float(np.mean(self.potential_uav_bottleneck_list)) if self.potential_uav_bottleneck_list else 0.0,
        "ep_avg_potential_mean_remaining": float(np.mean(self.potential_mean_remaining_list)) if self.potential_mean_remaining_list else 0.0,
        "ep_avg_urllc_window_reliability": float(np.mean(self.urllc_window_reliability_list)) if self.urllc_window_reliability_list else 1.0,
        "ep_min_mmtc_slot_connection_ratio": float(np.min(self.mmtc_connection_ratio_list)) if self.mmtc_connection_ratio_list else 1.0,
        "ep_low_mmtc_slot_ratio": float(np.mean(np.asarray(self.mmtc_connection_ratio_list) < float(main_cfg.mmtc_connectivity_floor))) if self.mmtc_connection_ratio_list else 0.0,
    })

    # Per-UAV QoS diagnostics.
    u_s = np.sum(np.asarray(self.agent_urllc_success_count_list, dtype=np.float32), axis=0) if self.agent_urllc_success_count_list else np.zeros(main_cfg.uav_n, dtype=np.float32)
    u_v = np.sum(np.asarray(self.agent_urllc_violation_count_list, dtype=np.float32), axis=0) if self.agent_urllc_violation_count_list else np.zeros(main_cfg.uav_n, dtype=np.float32)
    m_s = np.sum(np.asarray(self.agent_mmtc_success_count_list, dtype=np.float32), axis=0) if self.agent_mmtc_success_count_list else np.zeros(main_cfg.uav_n, dtype=np.float32)
    m_a = np.sum(np.asarray(self.agent_mmtc_active_count_list, dtype=np.float32), axis=0) if self.agent_mmtc_active_count_list else np.zeros(main_cfg.uav_n, dtype=np.float32)
    metric["ep_agent_urllc_success_ratio"] = np.divide(
        u_s, np.maximum(u_s + u_v, 1.0)
    ).astype(np.float32)
    metric["ep_agent_mmtc_success_ratio"] = np.divide(
        m_s, np.maximum(m_a, 1.0)
    ).astype(np.float32)
    metric["ep_avg_agent_local_progress_contribution"] = (
        np.mean(
            np.asarray(self.agent_local_progress_contribution_list, dtype=np.float32),
            axis=0,
        ).astype(np.float32)
        if self.agent_local_progress_contribution_list
        else np.zeros(main_cfg.uav_n, dtype=np.float32)
    )
    metric["ep_avg_agent_credit_reward"] = (
        np.mean(
            np.asarray(self.agent_credit_reward_list, dtype=np.float32), axis=0
        ).astype(np.float32)
        if self.agent_credit_reward_list
        else np.zeros(main_cfg.uav_n, dtype=np.float32)
    )
    budget = np.zeros((main_cfg.uav_n, main_cfg.slice_n), dtype=np.float32)
    budget[:, int(main_cfg.slice_embb)] = 1.0
    metric["ep_avg_slice_budget_ratio"] = budget
    metric["ep_final_slice_budget_ratio"] = budget.copy()
    return metric


Env._clear_episode_lists = _sliced_clear_episode_lists
Env.reset = _sliced_reset
Env._build_state = _sliced_build_state_env
Env.get_obs = _sliced_get_obs
Env.step = _sliced_step
Env.get_ep_metric = _sliced_get_ep_metric
