"""
ZHANG Wenqi

Finite eMBB completion-time configuration with URLLC puncturing and mMTC connectivity constraints.
The original geometry/channel/NFZ helper implementation is retained for
compatibility; the finite-data traffic, reward and state definitions are
installed in the explicit override section at the end of this file.
"""

from __future__ import annotations

import os
import datetime
import random
from statistics import NormalDist
import numpy as np


class MainConfig:
    def __init__(self):
        # ============================================================
        # 路径与时间
        # ============================================================
        self.curr_path = os.path.dirname(os.path.abspath(__file__))
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # ============================================================
        # 随机种子
        # ============================================================
        self.seed = 41
        np.random.seed(self.seed)
        random.seed(self.seed)

        # ============================================================
        # 训练 / 测试
        # ============================================================
        # True：运行所提DRL算法的训练、测试，并在测试阶段运行已启用的benchmark。
        # False：跳过所提DRL算法的训练和测试，只运行已启用的benchmark。
        # 保留原有开关名以兼容你之前的运行习惯：
        # True：运行下面 selected 的所提DRL；False：只运行benchmark。
        self.run_ctd4 = True

        # 固定无噪声验证开关：
        # False：训练过程中不再每100个EP额外运行30个fixed-validation场景；
        # True ：恢复原有fixed validation（具体间隔和回合数仍在ctd4_config.py中设置）。
        # 默认关闭，以减少训练额外耗时。
        self.use_fixed_validation = False

        # ============================================================
        # 所提DRL算法选择开关
        # 可选："CTD4" / "DDPG" / "TD3"
        # 三种算法共用完全相同的环境、state/obs/action、reward、随机探索阶段、
        # 行为探索噪声、网络隐藏层规模、学习率、gamma、tau、batch、replay、
        # 学习率衰减、训练/测试回合和固定验证设置。
        # 只保留算法定义本身必需的差异：
        #   DDPG: 单标量critic；每次更新actor和target。
        #   TD3 : twin scalar critics + min target + target smoothing + delayed actor。
        #   CTD4: distributional critic ensemble + Kalman fusion + delayed actor。
        self.proposed_algorithm = "CTD4"


        # ============================================================
        # Benchmark独立运行开关
        # True：运行对应benchmark；False：跳过对应benchmark。
        # 各开关彼此独立，也不受run_ctd4影响。
        # ============================================================
        self.run_benchmark_safe_nearest_unfinished = True
        self.run_benchmark_workload_distance = True
        self.run_benchmark_makespan_aware_one_step_greedy = True

        self.train_ep_n = 10000
        self.test_ep_n = 100

        # 一个RL step = 一个slot
        self.slot_time = 1.0

        # 基础默认值与文件末尾runtime override保持一致：180-slot仅作为
        # safety truncation；真实终止仍由全部eMBB文件完成触发。
        self.max_step = 300

        # ============================================================
        # 场景参数
        # ============================================================
        # 扩大区域，为多簇用户分布、随机UAV起点和较弱UAV间干扰提供空间。
        self.area_size = 1000.0

        self.uav_n = 3
        self.user_n = 120

        # 多簇用户空间模型：6个簇分布在区域不同位置，避免用户过度集中成3个小团。
        # 每个episode会对簇中心施加小幅随机漂移，增强场景泛化。
        self.service_cluster_centers = np.array(
            [
                [260.0, 260.0],
                [430.0, 370.0],
                [730.0, 260.0],
                [730.0, 490.0],
                [300.0, 720.0],
                [650.0, 720.0],
            ],
            dtype=np.float32,
        )
        self.user_cluster_n = int(self.service_cluster_centers.shape[0])

        # 每个簇覆盖较大的圆盘区域，圆盘内按面积均匀采样，并设置最小用户间距。
        # 这样总体仍呈现簇分布，但簇内用户更加分散，不会都挤在簇中心附近。
        self.user_cluster_radius = 155.0
        self.user_cluster_center_jitter = 35.0
        self.user_min_pair_dist = 12.0

        # 少量用户作为簇间散点，防止场景完全割裂。当前90用户下，
        # eMBB有3个散点，URLLC和mMTC各有1个散点。
        self.user_background_stride = 20

        # NFZ只避开各用户簇的核心，而不是避开整个大簇范围。
        self.nfz_cluster_core_clearance = 45.0

        # ============================================================
        # 网络切片参数
        # ============================================================
        # 0: eMBB  高吞吐增强切片，作为主优化目标
        # 1: URLLC 低时延高可靠切片，通过packet deadline violation建模
        # 2: mMTC  海量低速接入切片，通过active-device success建模
        self.slice_n = 3
        self.slice_embb = 0
        self.slice_urllc = 1
        self.slice_mmtc = 2

        # 兼容旧函数名，避免其他模块残留引用报错。
        self.slice_rate_max = self.slice_embb
        self.slice_rate_guarantee = self.slice_urllc
        self.slice_rate_fair = self.slice_mmtc

        self.slice_names = ["eMBB", "URLLC", "mMTC"]

        self.user_slice = self.build_user_slice()
        self.user_slice_onehot = self.build_user_slice_onehot(self.user_slice)

        # state/observation只保留真正会影响当前规则式资源分配的业务用户。
        # 用户切片编号在本问题中固定，因此不再把180维常量one-hot送入网络。
        self.embb_user_idx = np.where(self.user_slice == self.slice_embb)[0].astype(np.int32)
        self.urllc_user_idx = np.where(self.user_slice == self.slice_urllc)[0].astype(np.int32)
        self.mmtc_user_idx = np.where(self.user_slice == self.slice_mmtc)[0].astype(np.int32)
        self.embb_user_n = int(self.embb_user_idx.size)
        self.mmtc_user_n = int(self.mmtc_user_idx.size)

        # 仅用于诊断/归一化；URLLC与mMTC不再用简单速率门限定义业务成功。
        self.slice_rate_req = np.array([0.0e6, 0.0e6, 0.0e6], dtype=np.float32)
        self.user_rate_req = self.slice_rate_req[self.user_slice].astype(np.float32)

        # 切片级资源基础权重。新版中URLLC不参与静态预算，eMBB/mMTC由Actor决策RB比例。
        self.slice_resource_weight = np.array([1.00, 0.00, 1.20], dtype=np.float32)

        # eMBB为主目标，URLLC/mMTC作为QoS约束。
        # eMBB utility的数值单位相当于Mbps；URLLC gap通常很小，因此需要更大的乘子。
        self.reward_embb_weight = 1.0
        self.reward_urllc_constraint_weight = 400.0
        self.reward_mmtc_constraint_weight = 60.0

        # puncturing已经通过“puncturing后的eMBB有效速率”间接体现，
        # 不再重复加入独立惩罚；该字段保留用于兼容和日志。
        self.reward_puncture_weight = 0.0
        self.urllc_success_threshold = 0.99
        self.mmtc_success_threshold = 0.80

        # 兼容旧字段名；新reward不再使用三切片加权和。
        self.reward_slice_rate_max_weight = self.reward_embb_weight
        self.reward_slice_rate_guarantee_weight = 0.0
        self.reward_slice_rate_fair_weight = 0.0

        self.slice_user_count = np.array(
            [
                int(np.sum(self.user_slice == self.slice_embb)),
                int(np.sum(self.user_slice == self.slice_urllc)),
                int(np.sum(self.user_slice == self.slice_mmtc)),
            ],
            dtype=np.int32,
        )

        # URLLC业务模型：一个RL step为1 s UAV控制周期，内部包含多个1 ms TTI。
        # URLLC包在TTI级按Poisson过程突发到达，不参与Actor的静态切片预算；
        # 到达后只允许puncture已经分配给eMBB的RB，并须在deadline_tti内完成。
        self.tti_time = 1.0e-3
        self.tti_n_per_slot = int(round(self.slot_time / self.tti_time))
        self.urllc_arrival_rate = 2.0               # packets/s/user，调试阶段降低到达率以加快puncturing训练
        self.urllc_packet_bits = 512.0             # short-packet size
        self.urllc_deadline_tti = 1                # first version: same-TTI deadline
        self.urllc_arrival_prob = min(1.0, self.urllc_arrival_rate * self.tti_time)
        self.urllc_deadline_slots = self.urllc_deadline_tti
        self.urllc_queue_norm = max(1.0, 4.0 * self.urllc_packet_bits)
        self.urllc_deadline_norm = float(max(self.urllc_deadline_tti, 1))
        self.urllc_queue_priority_boost = 0.0
        self.urllc_deadline_priority_boost = 0.0

        # mMTC业务模型：mMTC设备低概率激活，成功标准为SINR门限且slot内完成小包。
        # 当前采用中等难度设置，让mMTC在训练初期更容易出现在约束附近波动。
        self.mmtc_active_prob = 0.25
        self.mmtc_packet_bits = 1536.0
        self.mmtc_pending_norm = self.mmtc_packet_bits
        self.mmtc_sinr_threshold_db = 1.0
        self.mmtc_sinr_threshold = float(10.0 ** (self.mmtc_sinr_threshold_db / 10.0))
        self.mmtc_active_priority_boost = 3.0

        # UAV固定飞行高度。三架UAV保留小幅高度差以避免同高度飞行，
        # 后续天线半波束宽度会根据“典型用户簇半径 + 中位UAV高度”自动匹配，
        # 而不是直接照搬其他场景的固定波束宽度。
        self.uav_height = np.array([100.0, 110.0, 120.0], dtype=np.float32)

        # UAV不同高度飞行，不设置水平碰撞约束。
        self.min_dist = 1.0

        # ============================================================
        # UAV运动参数
        # ============================================================
        # 文献中速度动作包括5/10/20 m/s；这里固定速度，只让CTD4决策方向。
        self.uav_fixed_speed = 15.0
        self.move_dist_each_step = self.uav_fixed_speed * self.slot_time

        # UAV是否允许飞出边界：
        # False表示动作导致越界时，将位置裁剪到区域内，并在env中给予边界惩罚。
        self.allow_out_of_area = False

        # ============================================================
        # 动态不规则禁飞区 NFZ 参数
        # ============================================================
        # True表示每个episode随机生成若干个区域内部不规则禁飞区。
        # 禁飞区在单个episode内固定，不同episode之间位置、形状和数量不同。
        self.use_nfz = True
        self.nfz_num_min = 2
        self.nfz_num_max = 4
        self.nfz_max_num = int(self.nfz_num_max)

        # 每个NFZ是不规则简单多边形。为了便于保存与绘图，最大顶点数固定为9。
        self.nfz_vertex_min = 5
        self.nfz_vertex_max = 9
        self.nfz_max_vertex_n = int(self.nfz_vertex_max)

        # 区域扩大后同步适度放大NFZ，使其仍具有实际几何意义，
        # 但不按面积比例完全放大，避免大量阻断用户簇。
        self.nfz_radius_min = 45.0
        self.nfz_radius_max = 90.0
        self.nfz_radius_jitter_min = 0.65
        self.nfz_radius_jitter_max = 1.25
        self.nfz_margin = 50.0
        self.nfz_min_center_dist_factor = 0.80
        self.nfz_min_area = 1000.0
        self.nfz_max_total_area_ratio = 0.15

        # 是否让用户/UAV初始位置避开NFZ。
        # UAV必须避开；用户是否避开取决于具体场景，这里默认避开，降低初始训练难度。
        self.avoid_nfz_for_user_init = True
        self.avoid_nfz_for_uav_init = True
        self.nfz_init_safe_margin = 5.0

        # NFZ安全投影参数。动作导致穿越/进入NFZ时，把UAV修正到碰撞前一点。
        self.nfz_projection_eps = 1e-3
        self.nfz_intersection_eps = 1e-9

        # ============================================================
        # Action shield：逐UAV将非法动作修正为最近合法方向
        # ============================================================
        # 若某架UAV的raw动作会越界或穿越/进入NFZ，只修正该UAV；
        # 同一slot内其他本身合法的UAV动作保持完全不变。
        # 修正动作只用于环境状态转移，replay buffer保存原始raw动作。
        self.use_action_shield = True

        # 最近安全方向的粗搜索分辨率：72个方向约为5度间隔。
        # 搜索仅做边界/NFZ几何检查，不再调用通信资源分配或一步reward评估。
        self.shield_candidate_n = 72

        # 在最近安全候选和原始非法方向之间进行二分细化的次数。
        self.shield_refine_iter = 12

        # 判断动作是否非法的数值阈值。
        self.shield_violation_eps = 1e-6

        # shield仅负责将raw非法动作修正为合法执行动作；
        # 非法行为由对应agent的hard negative reward处理，不再重复扣shield penalty。
        self.reward_shield_weight = 0.0

        # raw动作非法时，仅覆盖对应agent的reward为hard negative。
        # 该值应明显低于正常reward，但不宜比合法QoS失败惩罚极端得多。
        self.reward_illegal_action_fixed_penalty = 50.0

        # ============================================================
        # 原始动作的安全距离软惩罚
        # ============================================================
        # 无论raw动作最终是否非法，都根据该动作对应的raw下一位置，分别计算：
        # 1) 到矩形区域边界的最短距离；
        # 2) 到最近NFZ边界的最短距离。
        # 两者取最小值。距离小于50m时开始产生逐UAV软惩罚，越接近越大。
        # 最大软惩罚限制为15，避免合法但靠近边缘的动作压倒通信任务。
        self.safe_distance_penalty_margin = 50.0
        self.safe_distance_penalty_max = 15.0
        self.safe_distance_penalty_power = 2.0

        # NFZ observation：每架UAV使用固定维度几何特征，而不是直接输入多边形顶点。
        # 特征 = [nearest_dist, nearest_vec_x, nearest_vec_y, inside_flag, ray_dist_1...ray_dist_R]
        self.nfz_ray_n = 8
        self.nfz_ray_max_dist = float(np.sqrt(2.0 * self.area_size ** 2))
        self.nfz_feature_dim_per_uav = 4 + int(self.nfz_ray_n)
        self.nfz_ray_angles = np.linspace(
            -np.pi,
            np.pi,
            int(self.nfz_ray_n),
            endpoint=False,
            dtype=np.float32,
        )

        self.nfz_polygons = []
        self.nfz_polygon_radius = []
        self.set_episode_nfz(ep_id=0)

        # ============================================================
        # UAV初始位置
        # ============================================================
        # UAV起点与用户簇解耦：每个episode在整个有效区域内独立随机生成，
        # 只要求避开NFZ并保持足够的UAV间水平距离，不保证位于任何用户簇附近。
        self.random_uav_init_each_ep = True

        self.uav_init_margin = 100.0
        self.uav_init_min_pair_dist = 450.0
        self.uav_init_max_try = 5000

        # 三个宽范围初始化区域，仅用于让UAV初始时保持空间分离。
        # 每个区域都覆盖数百米范围，并非围绕某个具体用户簇设置；
        # 每个episode还会随机打乱“UAV编号-区域”的对应关系。
        self.uav_init_regions = np.array(
            [
                [[100.0, 420.0], [100.0, 520.0]],
                [[580.0, 900.0], [100.0, 520.0]],
                [[300.0, 700.0], [500.0, 900.0]],
            ],
            dtype=np.float32,
        )

        # 关闭随机初始化时使用的备用起点，也不与具体用户簇中心重合。
        self.fixed_uav_init_pos = np.array(
            [
                [180.0, 470.0],
                [820.0, 470.0],
                [500.0, 850.0],
            ],
            dtype=np.float32,
        )

        self.uav_init_pos = self.sample_uav_init_pos(ep_id=0)

        # ============================================================
        # 用户位置与移动参数
        # ============================================================
        # 用户以6个大簇为主并包含少量簇间散点；边界margin用于移动后的反射约束。
        self.user_margin = 70.0

        # 是否启用用户移动。文献中使用Gauss-Markov移动模型。
        # 如果你后面想进一步简化为静态用户，可以把它改成False。
        self.use_user_mobility = False
        # True：动作决策前直接提供本step通信/reward阶段实际采用的
        # 用户位置w_{t+1}（完美一步位置侧信息假设）。False用于消融，
        # 此时state/obs使用动作前用户位置w_t。
        self.use_perfect_next_user_position_observation = True

        # Gauss-Markov用户移动模型参数
        self.user_avg_speed = 1.3
        self.user_speed_std = 0.3
        self.user_speed_min = 0.0
        self.user_speed_max = 2.5

        self.user_mobility_c_speed = 0.90
        self.user_mobility_c_angle = 0.90

        self.user_speed_noise_std = 0.20
        self.user_angle_noise_std = 0.30

        self.user_pos = self.sample_user_pos(ep_id=0)
        self.user_speed, self.user_heading = self.sample_user_motion_state(ep_id=0)

        # ============================================================
        # 双瓣方向性天线模型（按当前场景几何自适应）
        # ============================================================
        # 保留常用双瓣模型：
        #   G_m(d)=2.285/Psi^2,  d <= H_m tan(Psi)  (main lobe)
        #          g,            otherwise          (side lobe)
        # 但不再照搬其他文献的 Psi=30 deg。当前场景为1000x1000 m、6个大用户簇，
        # 每个簇半径约155 m，而UAV高度仅100/110/120 m。若Psi=30 deg，
        # 主瓣半径只有约58--69 m，明显小于一个用户簇，导致绝大多数簇内用户长期
        # 只能依靠副瓣服务。这里令“中位高度UAV的主瓣半径≈一个典型用户簇半径”：
        #   H_ref*tan(Psi)=R_cluster.
        # 因此Psi≈54.64 deg，对应三架UAV主瓣半径约141/155/169 m。
        self.antenna_reference_height = float(np.median(self.uav_height))
        self.antenna_target_main_radius = float(self.user_cluster_radius)
        self.antenna_half_beamwidth = float(np.arctan(
            self.antenna_target_main_radius
            / max(self.antenna_reference_height, 1e-9)
        ))

        # 主瓣增益随波束展宽自然降低，避免“扩大覆盖却仍保持窄波束高增益”的不物理情况。
        self.antenna_main_gain = float(
            2.285 / (self.antenna_half_beamwidth ** 2)
        )
        self.antenna_main_gain_db = float(10.0 * np.log10(self.antenna_main_gain))

        # 用固定20 dB主/副瓣抑制度，而不是固定照搬某篇文献的绝对副瓣增益。
        # 这样波束变宽后，主瓣和副瓣增益会一起保持自洽。
        self.antenna_main_to_side_ratio_db = 20.0
        self.antenna_side_gain_db = float(
            self.antenna_main_gain_db - self.antenna_main_to_side_ratio_db
        )
        self.antenna_side_gain = float(10.0 ** (self.antenna_side_gain_db / 10.0))

        # 本问题有3架UAV但6个用户簇，而且mMTC要求每slot评估全部设备。即便把主瓣
        # 调到与单个用户簇相当，也不可能让3个主瓣同时覆盖6个簇。因此不能设置
        # “只有主瓣内才能建立连接”，否则mMTC 0.8连接率约束在几何上会先天过严。
        # 副瓣仍允许提供低增益服务，但主瓣会显著提高链路质量并驱动轨迹优化。
        self.require_main_lobe_for_service = False
        self.antenna_main_coverage_radius = (
            self.uav_height * np.tan(self.antenna_half_beamwidth)
        ).astype(np.float32)

        # ============================================================
        # 频谱与功率参数
        # ============================================================
        # RB级资源模型。每架UAV每个TTI拥有rb_n个RB，每个RB等功率。
        # URLLC puncturing直接抢占eMBB RB；抢占只在当前TTI生效，下一TTI释放回eMBB。
        self.rb_n = 50
        self.rb_bandwidth_hz = 180.0e3
        self.bandwidth_hz = float(self.rb_n * self.rb_bandwidth_hz)

        self.tx_power_dbm_each_uav = 20.0
        self.tx_power_w_each_uav = self.dbm_to_watt(self.tx_power_dbm_each_uav)
        self.rb_power_w_each_uav = self.tx_power_w_each_uav / float(max(self.rb_n, 1))

        # 资源分配方式：
        # "slice_aware": RB级eMBB/mMTC预算 + TTI级URLLC puncturing
        # "equal":       平均带宽 + 平均功率，作为消融/兼容选项
        self.resource_allocation_mode = "slice_aware"
        self.puncture_from = "embb_only"
        self.puncture_policy = "min_embb_loss"
        self.allow_puncture_mmtc = False
        self.embb_rb_allocation_policy = "log_utility"

        # 切片感知资源分配算法参数
        self.alloc_priority_eps = 1e-9
        self.alloc_spectral_eff_eps = 1e-6
        self.alloc_urllc_queue_boost = 2.0
        self.alloc_urllc_deadline_boost = 2.0
        self.alloc_mmtc_active_boost = 3.0
        self.alloc_min_priority = 1e-3

        # 多UAV共享频谱，考虑平均UAV间干扰
        self.full_frequency_reuse = True
        self.use_average_inter_uav_interference = True

        # 噪声功率谱密度。-174 dBm/Hz为热噪声底；地面终端接收机额外考虑7 dB
        # noise figure，避免原0 dB设置对弱副瓣链路过于乐观。
        self.noise_psd_dbm_per_hz = -174.0
        self.noise_figure_db = 7.0
        self.noise_psd_w_per_hz = self.dbm_per_hz_to_watt_per_hz(
            self.noise_psd_dbm_per_hz + self.noise_figure_db
        )

        # ============================================================
        # A2G LoS/NLoS信道参数（环境参数与几何参数分开处理）
        # ============================================================
        # 保留期望LoS/NLoS模型：
        #   h_mk = P_LoS*C0*d^-2 + (1-P_LoS)*kappa*C0*d^-2
        #   P_LoS = 1/[1+C1*exp(-C2*(phi-C1))].
        # C1/C2描述传播环境，不应因为区域从400 m变成1000 m就机械缩放；区域和高度
        # 已经通过距离d和仰角phi进入模型。因此这里采用更常见、变化更平滑的urban参数，
        # 而不是照搬参考文献中使P_LoS在约20 deg附近迅速饱和的(10,0.6)。
        self.use_small_scale_fading = False

        # 1 m参考增益由显式载频计算，而不是直接写死C0。2 GHz与当前9 MHz级
        # OFDMA蜂窝链路设定相符，且便于论文中解释和后续修改。
        self.carrier_frequency_hz = 2.0e9
        self.speed_of_light_mps = 299792458.0
        self.c0_linear = float((
            self.speed_of_light_mps
            / (4.0 * np.pi * self.carrier_frequency_hz)
        ) ** 2)
        self.c0_db = float(10.0 * np.log10(self.c0_linear))

        # Urban-type elevation-dependent LoS probability：随仰角平滑增加。
        self.los_c1 = 9.61
        self.los_c2 = 0.16

        # NLoS额外10 dB功率衰减。相比0.2(-7 dB)稍更保守，但不会把远端链路
        # 直接压到不可服务；其作用仍会与LoS概率连续融合。
        self.nlos_extra_loss_db = 10.0
        self.nlos_attenuation_kappa = float(10.0 ** (-self.nlos_extra_loss_db / 10.0))

        # ============================================================
        # 能量管理：关闭
        # ============================================================
        # 旧版固定时域代码（后续finite-data runtime override会替换）。所有UAV在整个episode内始终可服务，
        # 不计算飞行/通信能耗，不根据剩余能量失活，也不把能量送入state/obs。
        self.use_energy_management = False

        # 下列字段仅保留接口兼容；环境不会更新或使用它们进行决策/终止。
        self.uav_init_energy = np.zeros(self.uav_n, dtype=np.float32)
        self.uav_energy_threshold = -1.0
        self.uav_flight_power_w = 0.0
        self.include_comm_energy = False
        self.comm_power_if_active_w = 0.0
        self.energy_norm = 1.0
        self.energy_eps = 1e-9

        # ============================================================
        # 动作设计
        # ============================================================
        # CTD4连续联合动作维度 = UAV数量 * 每UAV动作维度。
        # 每个Traj-agent只输出：
        #   direction_xy_raw in [-1, 1]^2，作为二维方向向量，映射为
#   theta_m = atan2(direction_y, direction_x)。
        # 切片间RB不再作为动作：资源分配采用确定性规则，先保障可服务的active mMTC，
        # 剩余RB给eMBB；URLLC在TTI级到达后puncture eMBB RB。
        self.agent_n = self.uav_n
        self.local_traj_action_dim = 2
        self.budget_action_slice_ids = np.array([], dtype=np.int32)
        self.local_slice_budget_action_dim = 0
        self.local_action_dim = self.local_traj_action_dim
        self.action_dim = self.uav_n * self.local_action_dim
        self.action_low = -1.0
        self.action_high = 1.0

        # 轨迹-only版本不使用DRL切片预算。
        # 资源分配在resource_algorithms.serve_one_slot()中按规则完成：
        #   1) 对active且SINR达标的mMTC用户分配最小所需RB；
        #   2) 未被mMTC占用的RB全部给eMBB；
        #   3) URLLC从eMBB RB中puncture。
        self.use_drl_slice_budget = False
        self.resource_allocation_mode = "mmtc_min_rb_then_embb"

        # 兼容旧字段名。
        self.redistribute_empty_slice_budget = False
        self.slice_budget_eps = 1e-8

        # ============================================================
        # 状态设计：一步前视用户位置版本
        #
        # 动作a_t是在当前UAV位置q_t处选择；本step通信reward在UAV移动后的
        # q_{t+1}和用户移动后的w_{t+1}处计算。为使几何输入与reward对应，
        # state/obs中的“用户位置”由环境直接传入本step通信阶段实际使用的
        # w_{t+1}（完美一步位置预测/已知位置假设），而不是动作前的w_t。
        #
        # 已删除所有由动作前位置计算的瞬时链路结果特征：
        # 1) 当前用户关联UAV编号与局部关联mask；
        # 2) 当前eMBB频谱效率/SINR；
        # 3) 当前mMTC SINR裕量。
        # 这些量会在执行动作后，基于q_{t+1}, w_{t+1}重新计算并产生reward。
        #
        # state =
        # 1) 动作前UAV二维位置q_t                  2M
        # 2) 固定时域剩余比例                      1
        # 3) 本step通信时用户二维位置w_{t+1}       2K
        # 4) eMBB用户历史平均速率                  K_e
        # 5) mMTC用户当前active标志                K_m
        # 6) 每架UAV的NFZ几何感知特征              M*F_NFZ
        #
        # 能量特征已删除。加入剩余时域比例是为了让旧版固定时域保持Markov性：
        # 相同几何状态在第10步和第79步具有不同的剩余回报长度。
        # ============================================================
        self.state_dim = (
            2 * self.uav_n
            + 1
            + 2 * self.user_n
            + self.embb_user_n
            + self.mmtc_user_n
            + self.uav_n * self.nfz_feature_dim_per_uav
        )

        # ============================================================
        # multi-agent局部观测设计（共享Actor）
        #
        # 每个Actor看到当前UAV几何状态，以及本step通信阶段的已知用户位置。
        # 不再输入动作前关联、SINR、频谱效率或主/副瓣结果；Actor结合固定
        # 移动距离和所选方向，学习q_{t+1}相对w_{t+1}的通信价值。
        # ============================================================
        self.local_obs_dim = (
            2
            + 1  # 本UAV固定高度
            + 1  # 固定时域剩余比例（替代原剩余能量）
            + 3 * (self.uav_n - 1)
            + 2 * self.user_n
            + self.embb_user_n
            + self.mmtc_user_n
            + self.nfz_feature_dim_per_uav
        )

        # ============================================================
        # 归一化参数
        # ============================================================
        self.pos_norm = self.area_size
        self.height_norm = float(max(np.max(self.uav_height), 1.0))
        self.dist_2d_norm = np.sqrt(2.0 * self.area_size ** 2)

        # 历史速率改用log归一化，避免少数大速率维度支配第一层网络。
        self.state_rate_unit = 1.0e6
        self.state_rate_log_max_mbps = 20.0

        # 资源分配中的旧归一化字段继续保留，避免改变原资源算法。
        self.rate_norm = 1.0e6
        self.wait_time_norm = 50.0
        self.step_norm = float(max(self.max_step - 1, 1))

        # ============================================================
        # 奖励参数
        # ============================================================
        # Global task reward只衡量通信任务：
        # reward = eMBB utility
        #          - lambda_u [eta_u - URLLC_success]^+
        #          - lambda_m [eta_m - mMTC_success]^+
        # puncture、shield与安全违规单独作为诊断指标，不混入GlobalR。
        self.reward_unserved_weight = 0.0

        # 实际执行动作的boundary/NFZ惩罚只作为per-agent安全兜底项。
        # 在action shield正常工作时通常为0。
        self.reward_boundary_weight = 10.0
        self.reward_nfz_weight = 10.0

        # ============================================================
        # Per-agent reward参数
        # ============================================================
        # 每个UAV同时优化：
        #   30% 本UAV可归因eMBB贡献
        #   70% 全局通信任务 / UAV数量（含全局URLLC/mMTC约束）
        # 然后只扣本UAV自己的安全惩罚。
        self.agent_reward_local_ratio = 0.30
        self.agent_reward_global_ratio = 0.70

        # 兼容旧字段名；新版uav_env优先读取上面的ratio字段。
        self.agent_reward_global_weight = self.agent_reward_global_ratio
        self.agent_reward_local_embb_weight = self.agent_reward_local_ratio

        # eMBB速率奖励缩放。
        # 修正：eMBB utility 使用 eMBB sum rate，而不是 eMBB mean rate。
        # 因此 embb_utility = embb_sum_rate / reward_rate_scale。
        self.reward_rate_scale = 1.0e6

        self.rate_eps = 1e-12
        self.fairness_eps = 1e-12

        # ============================================================
        # 保存路径由 Ctd4Config.output_path / result_path / model_path 统一管理。

    # ================================================================
    # dBm / dBmHz 转换
    # ================================================================
    @staticmethod
    def dbm_to_watt(dbm):
        return float(10.0 ** ((float(dbm) - 30.0) / 10.0))

    @staticmethod
    def dbm_per_hz_to_watt_per_hz(dbm_per_hz):
        return float(10.0 ** ((float(dbm_per_hz) - 30.0) / 10.0))

    # ================================================================
    # 随机数器
    # ================================================================
    def make_rng(self, ep_id=0, offset=0):
        return np.random.default_rng(
            self.seed + int(offset) + int(ep_id) * 100000
        )

    # ================================================================
    # 用户切片初始化
    # ================================================================
    def build_user_slice(self):
        user_slice = np.zeros(self.user_n, dtype=np.int32)

        # 平衡划分三类切片；如果用户数不能整除3，余数按顺序分给前面的切片。
        base = self.user_n // self.slice_n
        rem = self.user_n % self.slice_n

        idx = 0
        for s in range(self.slice_n):
            count = base + (1 if s < rem else 0)
            user_slice[idx:idx + count] = s
            idx += count

        return user_slice.astype(np.int32)

    def build_user_slice_onehot(self, user_slice):
        user_slice = np.asarray(user_slice, dtype=np.int32).reshape(-1)

        onehot = np.zeros((user_slice.size, self.slice_n), dtype=np.float32)

        for k, s in enumerate(user_slice):
            if s < 0 or s >= self.slice_n:
                raise ValueError(f"用户{k}切片编号非法：{s}")
            onehot[k, s] = 1.0

        return onehot.astype(np.float32)

    # ================================================================
    # 动态不规则禁飞区 NFZ 生成与几何工具
    # ================================================================
    def set_episode_nfz(self, ep_id=0):
        """
        为当前episode生成一组不规则禁飞区。

        说明：
        1. 单个episode内NFZ固定；
        2. 不同episode通过ep_id改变随机种子，因此NFZ数量、位置、形状均变化；
        3. MainConfig保存当前episode的nfz_polygons，供状态构造、动作修正和绘图使用。
        """
        if not getattr(self, "use_nfz", False):
            self.nfz_polygons = []
            self.nfz_polygon_radius = []
            return []

        polygons, radius_list = self.sample_nfz_polygons(ep_id=ep_id)
        self.nfz_polygons = polygons
        self.nfz_polygon_radius = radius_list
        return self.nfz_polygons

    def sample_nfz_polygons(self, ep_id=0):
        rng = self.make_rng(ep_id=ep_id, offset=5000)

        nfz_num = int(rng.integers(self.nfz_num_min, self.nfz_num_max + 1))
        polygons = []
        radius_list = []
        total_area = 0.0
        max_total_area = float(self.nfz_max_total_area_ratio) * self.area_size ** 2

        max_try = 3000
        for _ in range(max_try):
            if len(polygons) >= nfz_num:
                break

            vertex_n = int(rng.integers(self.nfz_vertex_min, self.nfz_vertex_max + 1))
            base_radius = float(rng.uniform(self.nfz_radius_min, self.nfz_radius_max))

            low = float(self.nfz_margin + base_radius * self.nfz_radius_jitter_max)
            high = float(self.area_size - self.nfz_margin - base_radius * self.nfz_radius_jitter_max)
            if low >= high:
                continue

            center = rng.uniform(low=low, high=high, size=2).astype(np.float32)

            # 只保留各用户簇中心附近的小核心区，避免NFZ把整个簇中心完全覆盖。
            # 大簇外围仍允许出现NFZ，用户初始化会通过拒绝采样自行避开。
            cluster_clearance = (
                float(self.nfz_cluster_core_clearance)
                + base_radius * float(self.nfz_radius_jitter_max)
                + 10.0
            )
            center_to_clusters = np.linalg.norm(
                self.service_cluster_centers - center[None, :],
                axis=1,
            )
            if np.any(center_to_clusters < cluster_clearance):
                continue

            # 角度排序后生成星形扰动半径。排序保证顶点按极角连接，通常得到简单不规则多边形。
            angles = np.sort(rng.uniform(low=-np.pi, high=np.pi, size=vertex_n)).astype(np.float32)
            radii = base_radius * rng.uniform(
                low=self.nfz_radius_jitter_min,
                high=self.nfz_radius_jitter_max,
                size=vertex_n,
            )
            vertices = np.stack(
                [
                    center[0] + radii * np.cos(angles),
                    center[1] + radii * np.sin(angles),
                ],
                axis=1,
            ).astype(np.float32)

            if not self.is_polygon_inside_area(vertices):
                continue

            area = self.calc_polygon_area(vertices)
            if area < float(self.nfz_min_area):
                continue

            # 避免禁飞区之间过度拥挤。这里使用中心距近似控制，足够稳定且计算量低。
            too_close = False
            for old_poly, old_radius in zip(polygons, radius_list):
                old_center = np.mean(old_poly, axis=0)
                center_dist = float(np.linalg.norm(center - old_center))
                min_allowed = self.nfz_min_center_dist_factor * (base_radius + old_radius)
                if center_dist < min_allowed:
                    too_close = True
                    break
            if too_close:
                continue

            if total_area + area > max_total_area and len(polygons) >= 1:
                continue

            polygons.append(vertices.astype(np.float32))
            radius_list.append(float(base_radius))
            total_area += area

        return polygons, radius_list

    def is_polygon_inside_area(self, polygon):
        polygon = np.asarray(polygon, dtype=np.float32)
        return bool(
            np.all(polygon[:, 0] >= 0.0)
            and np.all(polygon[:, 0] <= self.area_size)
            and np.all(polygon[:, 1] >= 0.0)
            and np.all(polygon[:, 1] <= self.area_size)
        )

    @staticmethod
    def calc_polygon_area(polygon):
        polygon = np.asarray(polygon, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[0] < 3:
            return 0.0
        x = polygon[:, 0]
        y = polygon[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        return float(area)

    def calc_nfz_total_area(self):
        if not getattr(self, "use_nfz", False):
            return 0.0
        return float(sum(self.calc_polygon_area(poly) for poly in self.nfz_polygons))

    def calc_nfz_area_ratio(self):
        return float(self.calc_nfz_total_area() / max(self.area_size ** 2, 1e-9))

    def get_padded_nfz_polygons(self):
        padded = np.zeros(
            (self.nfz_max_num, self.nfz_max_vertex_n, 2),
            dtype=np.float32,
        )
        vertex_count = np.zeros(self.nfz_max_num, dtype=np.int32)

        for j, poly in enumerate(self.nfz_polygons[: self.nfz_max_num]):
            poly = np.asarray(poly, dtype=np.float32)
            n = min(poly.shape[0], self.nfz_max_vertex_n)
            padded[j, :n, :] = poly[:n]
            vertex_count[j] = n

        return padded, vertex_count

    def is_point_in_polygon(self, point, polygon):
        """Ray casting。边界点在后续距离判断中会被视为不安全。"""
        point = np.asarray(point, dtype=np.float32).reshape(2)
        polygon = np.asarray(polygon, dtype=np.float32)

        x, y = float(point[0]), float(point[1])
        inside = False
        n = polygon.shape[0]

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            cond = (float(y1) > y) != (float(y2) > y)
            if cond:
                x_inter = (float(x2) - float(x1)) * (y - float(y1)) / (float(y2) - float(y1) + 1e-12) + float(x1)
                if x < x_inter:
                    inside = not inside

        return bool(inside)

    def is_point_in_any_nfz(self, point):
        if not getattr(self, "use_nfz", False):
            return False
        for poly in self.nfz_polygons:
            if self.is_point_in_polygon(point, poly):
                return True
        return False

    @staticmethod
    def closest_point_on_segment(point, a, b):
        point = np.asarray(point, dtype=np.float32).reshape(2)
        a = np.asarray(a, dtype=np.float32).reshape(2)
        b = np.asarray(b, dtype=np.float32).reshape(2)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return a.copy()
        t = float(np.dot(point - a, ab) / denom)
        t = min(max(t, 0.0), 1.0)
        return (a + t * ab).astype(np.float32)

    def point_to_polygon_distance_and_closest(self, point, polygon):
        point = np.asarray(point, dtype=np.float32).reshape(2)
        polygon = np.asarray(polygon, dtype=np.float32)

        min_dist = np.inf
        closest = polygon[0].copy()
        n = polygon.shape[0]

        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            cand = self.closest_point_on_segment(point, a, b)
            dist = float(np.linalg.norm(point - cand))
            if dist < min_dist:
                min_dist = dist
                closest = cand.copy()

        if self.is_point_in_polygon(point, polygon):
            min_dist = 0.0

        return float(min_dist), closest.astype(np.float32)

    def point_to_any_nfz_distance_and_closest(self, point):
        if not getattr(self, "use_nfz", False) or len(self.nfz_polygons) == 0:
            return float(self.nfz_ray_max_dist), np.asarray(point, dtype=np.float32).reshape(2).copy(), False

        point = np.asarray(point, dtype=np.float32).reshape(2)
        best_dist = np.inf
        best_closest = point.copy()
        inside_any = False

        for poly in self.nfz_polygons:
            inside = self.is_point_in_polygon(point, poly)
            dist, closest = self.point_to_polygon_distance_and_closest(point, poly)
            if inside:
                inside_any = True
            if dist < best_dist:
                best_dist = dist
                best_closest = closest.copy()

        return float(best_dist), best_closest.astype(np.float32), bool(inside_any)

    def is_point_safe_from_nfz(self, point, margin=0.0):
        if not getattr(self, "use_nfz", False) or len(self.nfz_polygons) == 0:
            return True
        dist, _, inside = self.point_to_any_nfz_distance_and_closest(point)
        return bool((not inside) and dist >= float(margin))

    @staticmethod
    def _cross2d(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    def segment_intersection_t(self, p, p2, q, q2):
        """
        返回线段 p->p2 与 q->q2 的交点在第一条线段上的参数t。
        如果不相交，返回None。
        """
        p = np.asarray(p, dtype=np.float32).reshape(2)
        p2 = np.asarray(p2, dtype=np.float32).reshape(2)
        q = np.asarray(q, dtype=np.float32).reshape(2)
        q2 = np.asarray(q2, dtype=np.float32).reshape(2)

        r = p2 - p
        s = q2 - q
        denom = self._cross2d(r, s)
        qp = q - p

        if abs(denom) <= self.nfz_intersection_eps:
            return None

        t = self._cross2d(qp, s) / denom
        u = self._cross2d(qp, r) / denom

        if -1e-7 <= t <= 1.0 + 1e-7 and -1e-7 <= u <= 1.0 + 1e-7:
            return float(min(max(t, 0.0), 1.0))

        return None

    def first_segment_nfz_intersection_t(self, start_pos, end_pos):
        if not getattr(self, "use_nfz", False) or len(self.nfz_polygons) == 0:
            return None

        start_pos = np.asarray(start_pos, dtype=np.float32).reshape(2)
        end_pos = np.asarray(end_pos, dtype=np.float32).reshape(2)

        t_list = []
        for poly in self.nfz_polygons:
            n = poly.shape[0]
            for i in range(n):
                a = poly[i]
                b = poly[(i + 1) % n]
                t = self.segment_intersection_t(start_pos, end_pos, a, b)
                if t is not None:
                    # 起点恰好贴边时，忽略极小t，避免把贴边运动完全锁死。
                    if t > 1e-6:
                        t_list.append(float(t))

        if len(t_list) == 0:
            return None

        return float(min(t_list))

    def project_segment_out_of_nfz(self, start_pos, end_pos):
        start_pos = np.asarray(start_pos, dtype=np.float32).reshape(2)
        end_pos = np.asarray(end_pos, dtype=np.float32).reshape(2)

        if not getattr(self, "use_nfz", False) or len(self.nfz_polygons) == 0:
            return end_pos.astype(np.float32), 0.0

        move_vec = end_pos - start_pos
        move_len = float(np.linalg.norm(move_vec))
        if move_len <= 1e-12:
            if self.is_point_in_any_nfz(end_pos):
                return start_pos.astype(np.float32), 0.0
            return end_pos.astype(np.float32), 0.0

        end_inside = self.is_point_in_any_nfz(end_pos)
        hit_t = self.first_segment_nfz_intersection_t(start_pos, end_pos)

        if hit_t is None and not end_inside:
            return end_pos.astype(np.float32), 0.0

        if hit_t is None:
            # 极少数数值情况下：终点在NFZ内但没有检测到边交点，直接保持原位。
            safe_pos = start_pos.copy()
        else:
            eps_t = float(self.nfz_projection_eps) / max(move_len, 1e-12)
            safe_t = max(0.0, float(hit_t) - eps_t)
            safe_pos = start_pos + safe_t * move_vec

        violation = float(np.linalg.norm(end_pos - safe_pos))
        safe_pos = self.clip_pos(safe_pos)
        return safe_pos.astype(np.float32), float(violation)

    def ray_to_area_boundary_distance(self, point, direction):
        point = np.asarray(point, dtype=np.float32).reshape(2)
        direction = np.asarray(direction, dtype=np.float32).reshape(2)

        x, y = float(point[0]), float(point[1])
        dx, dy = float(direction[0]), float(direction[1])
        candidates = []

        if abs(dx) > 1e-12:
            t = (0.0 - x) / dx
            yy = y + t * dy
            if t >= 0.0 and 0.0 <= yy <= self.area_size:
                candidates.append(t)
            t = (self.area_size - x) / dx
            yy = y + t * dy
            if t >= 0.0 and 0.0 <= yy <= self.area_size:
                candidates.append(t)

        if abs(dy) > 1e-12:
            t = (0.0 - y) / dy
            xx = x + t * dx
            if t >= 0.0 and 0.0 <= xx <= self.area_size:
                candidates.append(t)
            t = (self.area_size - y) / dy
            xx = x + t * dx
            if t >= 0.0 and 0.0 <= xx <= self.area_size:
                candidates.append(t)

        if len(candidates) == 0:
            return float(self.nfz_ray_max_dist)
        return float(min(candidates))

    def ray_segment_intersection_distance(self, origin, direction, a, b):
        origin = np.asarray(origin, dtype=np.float32).reshape(2)
        direction = np.asarray(direction, dtype=np.float32).reshape(2)
        a = np.asarray(a, dtype=np.float32).reshape(2)
        b = np.asarray(b, dtype=np.float32).reshape(2)

        s = b - a
        denom = self._cross2d(direction, s)
        if abs(denom) <= self.nfz_intersection_eps:
            return None

        ao = a - origin
        t = self._cross2d(ao, s) / denom
        u = self._cross2d(ao, direction) / denom

        if t >= 0.0 and -1e-7 <= u <= 1.0 + 1e-7:
            return float(t)
        return None

    def ray_to_nfz_distance(self, point, direction):
        if not getattr(self, "use_nfz", False) or len(self.nfz_polygons) == 0:
            return float(self.nfz_ray_max_dist)

        point = np.asarray(point, dtype=np.float32).reshape(2)
        direction = np.asarray(direction, dtype=np.float32).reshape(2)

        if self.is_point_in_any_nfz(point):
            return 0.0

        best = float(self.nfz_ray_max_dist)
        for poly in self.nfz_polygons:
            n = poly.shape[0]
            for i in range(n):
                a = poly[i]
                b = poly[(i + 1) % n]
                d = self.ray_segment_intersection_distance(point, direction, a, b)
                if d is not None and d < best:
                    best = float(d)

        return float(best)

    def calc_nfz_features(self, uav_pos):
        uav_pos = np.asarray(uav_pos, dtype=np.float32)
        feat = np.zeros((self.uav_n, self.nfz_feature_dim_per_uav), dtype=np.float32)

        if not getattr(self, "use_nfz", False) or len(self.nfz_polygons) == 0:
            feat[:, 0] = 1.0
            feat[:, 4:] = 1.0
            return feat.astype(np.float32)

        ray_max = max(float(self.nfz_ray_max_dist), 1e-9)

        for m in range(self.uav_n):
            point = uav_pos[m]
            dist, closest, inside = self.point_to_any_nfz_distance_and_closest(point)
            rel_vec = (closest - point) / ray_max

            feat[m, 0] = float(np.clip(dist / ray_max, 0.0, 1.0))
            feat[m, 1:3] = np.clip(rel_vec, -1.0, 1.0)
            feat[m, 3] = 1.0 if inside else 0.0

            ray_feat = []
            for angle in self.nfz_ray_angles:
                direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
                area_dist = self.ray_to_area_boundary_distance(point, direction)
                nfz_dist = self.ray_to_nfz_distance(point, direction)
                ray_dist = min(area_dist, nfz_dist, ray_max)
                ray_feat.append(float(np.clip(ray_dist / ray_max, 0.0, 1.0)))

            feat[m, 4:] = np.asarray(ray_feat, dtype=np.float32)

        return feat.astype(np.float32)

    # ================================================================
    # 用户位置初始化
    # ================================================================
    def sample_user_pos(self, ep_id=0):
        """
        使用多簇、大范围、簇内分散的用户初始分布。

        1. 6个基础簇中心在每个episode内小幅随机漂移；
        2. 大多数用户在对应簇的155 m圆盘内按面积均匀分布；
        3. 每20个用户中有1个在全区域内作为簇间散点；
        4. 通过最小间距约束避免用户点过度重叠；
        5. 用户编号对簇取模，使三类业务都分布到多个簇中。
        """
        rng = self.make_rng(ep_id=ep_id, offset=1000)

        low = float(self.user_margin)
        high = float(self.area_size - self.user_margin)

        # 每个episode对簇中心做小幅二维漂移，但保持在有效区域内。
        cluster_centers = self.service_cluster_centers.copy().astype(np.float32)
        center_jitter = rng.uniform(
            low=-float(self.user_cluster_center_jitter),
            high=float(self.user_cluster_center_jitter),
            size=cluster_centers.shape,
        ).astype(np.float32)
        cluster_centers = np.clip(
            cluster_centers + center_jitter,
            low + float(self.user_cluster_radius),
            high - float(self.user_cluster_radius),
        ).astype(np.float32)

        user_pos = np.zeros((self.user_n, 2), dtype=np.float32)
        valid_count = 0

        for k in range(self.user_n):
            chosen = None
            is_background_user = (
                int(self.user_background_stride) > 0
                and k % int(self.user_background_stride) == 0
            )

            for _ in range(2000):
                if is_background_user:
                    candidate = rng.uniform(low=low, high=high, size=2).astype(np.float32)
                else:
                    cluster_id = int(k % self.user_cluster_n)
                    center = cluster_centers[cluster_id]
                    radius = float(self.user_cluster_radius) * np.sqrt(float(rng.uniform(0.0, 1.0)))
                    angle = float(rng.uniform(-np.pi, np.pi))
                    offset = radius * np.array(
                        [np.cos(angle), np.sin(angle)],
                        dtype=np.float32,
                    )
                    candidate = (center + offset).astype(np.float32)

                if np.any(candidate < low) or np.any(candidate > high):
                    continue

                if valid_count > 0:
                    min_user_dist = float(np.min(np.linalg.norm(
                        user_pos[:valid_count] - candidate[None, :],
                        axis=1,
                    )))
                    if min_user_dist < float(self.user_min_pair_dist):
                        continue

                if (
                    self.use_nfz
                    and self.avoid_nfz_for_user_init
                    and not self.is_point_safe_from_nfz(
                        candidate,
                        margin=self.nfz_init_safe_margin,
                    )
                ):
                    continue

                chosen = candidate
                break

            if chosen is None:
                # 极端情况下取消最小用户间距约束，但仍优先保证区域与NFZ合法。
                for _ in range(2000):
                    candidate = rng.uniform(low=low, high=high, size=2).astype(np.float32)
                    if (
                        self.use_nfz
                        and self.avoid_nfz_for_user_init
                        and not self.is_point_safe_from_nfz(
                            candidate,
                            margin=self.nfz_init_safe_margin,
                        )
                    ):
                        continue
                    chosen = candidate
                    break

            if chosen is None:
                chosen = np.array([self.area_size / 2.0, self.area_size / 2.0], dtype=np.float32)

            user_pos[k] = chosen
            valid_count += 1

        return user_pos.astype(np.float32)

    # ================================================================
    # 用户初始移动状态
    # ================================================================
    def sample_user_motion_state(self, ep_id=0):
        rng = self.make_rng(ep_id=ep_id, offset=2000)

        speed = rng.normal(
            loc=self.user_avg_speed,
            scale=self.user_speed_std,
            size=self.user_n
        )

        speed = np.clip(
            speed,
            self.user_speed_min,
            self.user_speed_max
        )

        heading = rng.uniform(
            low=-np.pi,
            high=np.pi,
            size=self.user_n
        )

        return speed.astype(np.float32), heading.astype(np.float32)

    # ================================================================
    # 用户移动一个slot
    # ================================================================
    def move_users_one_slot(self, user_pos, user_speed, user_heading, ep_id, step_idx):
        user_pos = np.asarray(user_pos, dtype=np.float32)
        user_speed = np.asarray(user_speed, dtype=np.float32).reshape(-1)
        user_heading = np.asarray(user_heading, dtype=np.float32).reshape(-1)

        if not self.use_user_mobility:
            return (
                user_pos.astype(np.float32),
                user_speed.astype(np.float32),
                user_heading.astype(np.float32)
            )

        rng_seed = (
            self.seed
            + 3000
            + int(ep_id) * 100000
            + int(step_idx)
        )
        rng = np.random.default_rng(rng_seed)

        speed_noise = rng.normal(
            loc=0.0,
            scale=self.user_speed_noise_std,
            size=self.user_n
        )

        angle_noise = rng.normal(
            loc=0.0,
            scale=self.user_angle_noise_std,
            size=self.user_n
        )

        new_speed = (
            self.user_mobility_c_speed * user_speed
            + (1.0 - self.user_mobility_c_speed) * self.user_avg_speed
            + np.sqrt(max(1.0 - self.user_mobility_c_speed ** 2, 0.0)) * speed_noise
        )

        new_speed = np.clip(
            new_speed,
            self.user_speed_min,
            self.user_speed_max
        )

        new_heading = (
            self.user_mobility_c_angle * user_heading
            + np.sqrt(max(1.0 - self.user_mobility_c_angle ** 2, 0.0)) * angle_noise
        )

        new_heading = self.wrap_angle(new_heading)

        dx = new_speed * np.cos(new_heading) * self.slot_time
        dy = new_speed * np.sin(new_heading) * self.slot_time

        new_pos = user_pos.copy()
        new_pos[:, 0] += dx
        new_pos[:, 1] += dy

        # 边界反弹
        for k in range(self.user_n):
            if new_pos[k, 0] < self.user_margin:
                new_pos[k, 0] = self.user_margin
                new_heading[k] = np.pi - new_heading[k]

            if new_pos[k, 0] > self.area_size - self.user_margin:
                new_pos[k, 0] = self.area_size - self.user_margin
                new_heading[k] = np.pi - new_heading[k]

            if new_pos[k, 1] < self.user_margin:
                new_pos[k, 1] = self.user_margin
                new_heading[k] = -new_heading[k]

            if new_pos[k, 1] > self.area_size - self.user_margin:
                new_pos[k, 1] = self.area_size - self.user_margin
                new_heading[k] = -new_heading[k]

        new_heading = self.wrap_angle(new_heading)

        return (
            new_pos.astype(np.float32),
            new_speed.astype(np.float32),
            new_heading.astype(np.float32)
        )

    # ================================================================
    # UAV初始位置
    # ================================================================
    def sample_uav_init_pos(self, ep_id=0):
        """
        UAV初始位置不绑定具体用户簇。

        随机模式下：
        1. 从3个数百米尺度的宽初始化区域分别采样1个位置；
        2. 每个episode随机打乱UAV编号与初始化区域的对应关系；
        3. 仅要求避开NFZ，并满足UAV间最小水平距离。

        因此UAV不一定在任何用户簇附近，但初始位置不会过度拥挤，
        从而兼顾轨迹学习难度与较弱的UAV间干扰。
        """
        if not self.random_uav_init_each_ep:
            pos = self.fixed_uav_init_pos.copy().astype(np.float32)
            if self.use_nfz and self.avoid_nfz_for_uav_init:
                rng = self.make_rng(ep_id=ep_id, offset=4050)
                low = float(self.uav_init_margin)
                high = float(self.area_size - self.uav_init_margin)

                for m in range(self.uav_n):
                    if self.is_point_safe_from_nfz(pos[m], margin=self.nfz_init_safe_margin):
                        continue

                    for _ in range(self.uav_init_max_try):
                        candidate = rng.uniform(low=low, high=high, size=2).astype(np.float32)
                        trial_pos = pos.copy()
                        trial_pos[m] = candidate
                        if not self.is_point_safe_from_nfz(
                            candidate,
                            margin=self.nfz_init_safe_margin,
                        ):
                            continue
                        if self.calc_min_pair_dist(trial_pos) < self.uav_init_min_pair_dist:
                            continue
                        pos[m] = candidate
                        break
            return pos.astype(np.float32)

        rng = self.make_rng(ep_id=ep_id, offset=4000)
        region_order = rng.permutation(self.uav_n)

        best_pos = self.fixed_uav_init_pos.copy().astype(np.float32)
        best_min_dist = -1.0

        for _ in range(self.uav_init_max_try):
            pos = np.zeros((self.uav_n, 2), dtype=np.float32)

            for m in range(self.uav_n):
                region = self.uav_init_regions[int(region_order[m])]
                pos[m, 0] = float(rng.uniform(region[0, 0], region[0, 1]))
                pos[m, 1] = float(rng.uniform(region[1, 0], region[1, 1]))

            if self.use_nfz and self.avoid_nfz_for_uav_init:
                safe_mask = [
                    self.is_point_safe_from_nfz(
                        point,
                        margin=self.nfz_init_safe_margin,
                    )
                    for point in pos
                ]
                if not all(safe_mask):
                    continue

            min_pair_dist = self.calc_min_pair_dist(pos)
            if min_pair_dist > best_min_dist:
                best_min_dist = min_pair_dist
                best_pos = pos.copy()

            if min_pair_dist >= float(self.uav_init_min_pair_dist):
                return pos.astype(np.float32)

        return best_pos.astype(np.float32)

    # ================================================================
    # 连续动作解析：联合动作 -> 飞行方向 + 切片预算
    # ================================================================
    def _reshape_joint_action(self, action_cont):
        action_cont = np.asarray(action_cont, dtype=np.float32).reshape(-1)

        if action_cont.size != self.action_dim:
            raise ValueError(
                f"动作维度错误：期望 {self.action_dim}，实际 {action_cont.size}"
            )

        action_cont = np.clip(
            action_cont,
            self.action_low,
            self.action_high
        )

        return action_cont.reshape(self.uav_n, self.local_action_dim).astype(np.float32)

    def cont_action_to_heading(self, action_cont):
        action_matrix = self._reshape_joint_action(action_cont)
        direction_xy = action_matrix[:, :2]
        heading = np.arctan2(direction_xy[:, 1], direction_xy[:, 0])

        zero_mask = np.linalg.norm(direction_xy, axis=1) <= 1.0e-8
        if np.any(zero_mask):
            heading = heading.astype(np.float32, copy=True)
            heading[zero_mask] = 0.0

        return heading.astype(np.float32)

    def heading_to_local_traj_action(self, heading):
        heading = np.asarray(heading, dtype=np.float32)
        direction_x = np.cos(heading)
        direction_y = np.sin(heading)
        action = np.stack([direction_x, direction_y], axis=-1).astype(np.float32)
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def cont_action_to_slice_budget(self, action_cont, active_mask=None):
        """
        兼容旧接口：轨迹-only版本中Actor不再输出切片预算。

        返回值仅用于日志/旧函数调用占位，真实RB预算由
        resource_algorithms.serve_one_slot()根据active mMTC和eMBB用户确定。
        """
        _ = self._reshape_joint_action(action_cont)

        budget_ratio = np.zeros((self.uav_n, self.slice_n), dtype=np.float32)
        # 名义上mMTC按需优先、剩余给eMBB；这里不代表实际动作决策。
        budget_ratio[:, int(self.slice_embb)] = 1.0
        budget_ratio[:, int(self.slice_mmtc)] = 0.0
        budget_ratio[:, int(self.slice_urllc)] = 0.0

        if active_mask is not None:
            active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if active_mask.size != self.uav_n:
                raise ValueError(
                    f"active_mask维度错误：期望 {self.uav_n}，实际 {active_mask.size}"
                )
            budget_ratio[~active_mask, :] = 0.0

        return budget_ratio.astype(np.float32)

    def parse_joint_action(self, action_cont, active_mask=None):
        heading = self.cont_action_to_heading(action_cont)
        slice_budget_ratio = self.cont_action_to_slice_budget(
            action_cont=action_cont,
            active_mask=active_mask,
        )

        return heading.astype(np.float32), slice_budget_ratio.astype(np.float32)

    # ================================================================
    # UAV按方向移动一个slot
    # ================================================================
    def move_uav_one_slot(self, uav_pos, heading, active_mask=None):
        uav_pos = np.asarray(uav_pos, dtype=np.float32)
        heading = np.asarray(heading, dtype=np.float32).reshape(-1)

        if active_mask is None:
            active_mask = np.ones(self.uav_n, dtype=bool)
        else:
            active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

        if uav_pos.shape != (self.uav_n, 2):
            raise ValueError(
                f"uav_pos维度错误：期望 {(self.uav_n, 2)}，实际 {uav_pos.shape}"
            )

        if heading.size != self.uav_n:
            raise ValueError(
                f"heading维度错误：期望 {self.uav_n}，实际 {heading.size}"
            )

        raw_next_pos = uav_pos.copy()

        for m in range(self.uav_n):
            if not active_mask[m]:
                continue

            raw_next_pos[m, 0] += (
                self.uav_fixed_speed * np.cos(heading[m]) * self.slot_time
            )
            raw_next_pos[m, 1] += (
                self.uav_fixed_speed * np.sin(heading[m]) * self.slot_time
            )

        # 与原始边界处理一致：先计算越界量；如果不允许出界，执行裁剪。
        boundary_violation = self.calc_boundary_violation(raw_next_pos)

        if self.allow_out_of_area:
            boundary_safe_pos = raw_next_pos.copy()
        else:
            boundary_safe_pos = self.clip_pos(raw_next_pos)

        # NFZ处理：若从当前位置到候选位置的线段穿越/进入NFZ，
        # 则把UAV修正到碰撞前的安全点，并把被修正距离作为NFZ violation。
        next_pos = boundary_safe_pos.copy()
        nfz_violation = np.zeros(self.uav_n, dtype=np.float32)

        if self.use_nfz:
            for m in range(self.uav_n):
                if not active_mask[m]:
                    continue

                safe_pos_m, violation_m = self.project_segment_out_of_nfz(
                    start_pos=uav_pos[m],
                    end_pos=boundary_safe_pos[m],
                )
                next_pos[m] = safe_pos_m
                nfz_violation[m] = float(violation_m)

        return (
            next_pos.astype(np.float32),
            raw_next_pos.astype(np.float32),
            boundary_violation.astype(np.float32),
            nfz_violation.astype(np.float32),
        )

    # ================================================================
    # 位置裁剪
    # ================================================================
    def clip_pos(self, pos):
        pos = np.asarray(pos, dtype=np.float32)
        return np.clip(pos, 0.0, self.area_size).astype(np.float32)

    def clip_uav_pos(self, uav_pos):
        return self.clip_pos(uav_pos)

    def clip_user_pos(self, user_pos):
        low = self.user_margin
        high = self.area_size - self.user_margin
        return np.clip(user_pos, low, high).astype(np.float32)

    # ================================================================
    # 边界越界量
    # ================================================================
    def calc_boundary_violation(self, pos):
        pos = np.asarray(pos, dtype=np.float32)

        low_violation = np.maximum(0.0, -pos)
        high_violation = np.maximum(0.0, pos - self.area_size)

        violation = low_violation + high_violation
        violation_dist = np.linalg.norm(violation, axis=1)

        return violation_dist.astype(np.float32)

    # ================================================================
    # UAV-user二维 / 三维距离
    # ================================================================
    def calc_uav_user_distance_2d(self, uav_pos, user_pos):
        uav_pos = np.asarray(uav_pos, dtype=np.float32)
        user_pos = np.asarray(user_pos, dtype=np.float32)

        diff = uav_pos[:, None, :] - user_pos[None, :, :]
        dist_2d = np.linalg.norm(diff, axis=2)

        return np.maximum(dist_2d, self.min_dist).astype(np.float32)

    def calc_uav_user_distance_3d(self, uav_pos, user_pos):
        dist_2d = self.calc_uav_user_distance_2d(uav_pos, user_pos)

        height = self.uav_height.reshape(self.uav_n, 1)

        dist_3d = np.sqrt(dist_2d ** 2 + height ** 2)

        return np.maximum(dist_3d, self.min_dist).astype(np.float32)

    # 兼容旧命名
    def calc_distance(self, uav_pos, user_pos):
        return self.calc_uav_user_distance_3d(uav_pos, user_pos)

    # ================================================================
    # UAV主瓣覆盖矩阵和天线增益矩阵
    # ================================================================
    def calc_main_lobe_mask(self, uav_pos, user_pos):
        dist_2d = self.calc_uav_user_distance_2d(uav_pos, user_pos)

        coverage_radius = (
            self.uav_height.reshape(self.uav_n, 1)
            * np.tan(self.antenna_half_beamwidth)
        )

        main_lobe_mask = dist_2d <= coverage_radius

        return main_lobe_mask.astype(bool)

    def calc_antenna_gain_matrix(self, uav_pos, user_pos):
        main_lobe_mask = self.calc_main_lobe_mask(uav_pos, user_pos)

        gain = np.where(
            main_lobe_mask,
            self.antenna_main_gain,
            self.antenna_side_gain
        )

        return gain.astype(np.float32), main_lobe_mask

    # ================================================================
    # LoS概率和A2G期望信道增益
    # ================================================================
    def calc_los_probability(self, uav_pos, user_pos):
        dist_3d = self.calc_uav_user_distance_3d(uav_pos, user_pos)

        height = self.uav_height.reshape(self.uav_n, 1)

        ratio = np.clip(height / np.maximum(dist_3d, self.min_dist), 0.0, 1.0)
        elevation_rad = np.arcsin(ratio)
        elevation_deg = elevation_rad * 180.0 / np.pi

        p_los = 1.0 / (
            1.0
            + self.los_c1 * np.exp(-self.los_c2 * (elevation_deg - self.los_c1))
        )

        return np.clip(p_los, 0.0, 1.0).astype(np.float32)

    def calc_channel_gain(self, uav_pos, user_pos, ep_id=0, step_idx=0):
        dist_3d = self.calc_uav_user_distance_3d(uav_pos, user_pos)
        p_los = self.calc_los_probability(uav_pos, user_pos)

        los_gain = self.c0_linear * (dist_3d ** -2.0)
        nlos_gain = self.nlos_attenuation_kappa * self.c0_linear * (dist_3d ** -2.0)

        large_scale_gain = p_los * los_gain + (1.0 - p_los) * nlos_gain

        if self.use_small_scale_fading:
            small_scale = self.sample_small_scale_gain(ep_id=ep_id, step_idx=step_idx)
            channel_gain = large_scale_gain * small_scale
        else:
            channel_gain = large_scale_gain

        return channel_gain.astype(np.float32)

    def sample_small_scale_gain(self, ep_id=0, step_idx=0):
        rng_seed = (
            self.seed
            + 5000
            + int(ep_id) * 100000
            + int(step_idx)
        )
        rng = np.random.default_rng(rng_seed)

        shape = (self.uav_n, self.user_n)

        g = (
            rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)
        ) / np.sqrt(2.0)

        g_power = np.abs(g) ** 2

        return g_power.astype(np.float32)

    # ================================================================
    # UAV飞行功率
    # ================================================================
    def calc_flight_power(self, speed):
        speed = np.asarray(speed, dtype=np.float32)

        term_1 = self.P0 * (
            1.0 + 3.0 * speed ** 2 / self.U_tip ** 2
        )

        inner = (
            np.sqrt(1.0 + speed ** 4 / (4.0 * self.v0 ** 4))
            - speed ** 2 / (2.0 * self.v0 ** 2)
        )
        inner = np.maximum(inner, 0.0)

        term_2 = self.Pi * np.sqrt(inner)

        term_3 = (
            0.5
            * self.d0
            * self.rho
            * self.s
            * self.A
            * speed ** 3
        )

        power = term_1 + term_2 + term_3

        return np.asarray(power, dtype=np.float32)

    # ================================================================
    # 单个slot能耗
    # ================================================================
    def calc_uav_energy_one_slot(self, active_mask=None):
        # finite-data版本不建模能量；保留该函数仅兼容旧统计接口。
        if not bool(getattr(self, "use_energy_management", False)):
            return 0.0, np.zeros(self.uav_n, dtype=np.float32)

        if active_mask is None:
            active_mask = np.ones(self.uav_n, dtype=bool)
        else:
            active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

        energy_each_uav = np.zeros(self.uav_n, dtype=np.float32)

        for m in range(self.uav_n):
            if not active_mask[m]:
                continue

            power = self.uav_flight_power_w

            if self.include_comm_energy:
                power += self.comm_power_if_active_w

            energy_each_uav[m] = power * self.slot_time

        total_energy = float(np.sum(energy_each_uav))

        return total_energy, energy_each_uav.astype(np.float32)

    # ================================================================
    # Jain公平性
    # ================================================================
    def calc_jain_fairness(self, user_value):
        user_value = np.asarray(user_value, dtype=np.float32).reshape(-1)

        numerator = float(np.sum(user_value) ** 2)
        denominator = float(
            user_value.size * np.sum(user_value ** 2)
            + self.fairness_eps
        )

        return numerator / denominator

    # ================================================================
    # 切片速率指标
    # ================================================================
    def calc_slice_metrics(
        self,
        user_rate,
        avg_user_rate,
        urllc_success_ratio=1.0,
        urllc_violation_ratio=0.0,
        mmtc_success_ratio=1.0,
        mmtc_active_ratio=0.0,
        mmtc_coverage_ratio=0.0,
        puncture_ratio=0.0,
        embb_base_sum_rate=None,
        embb_loss_rate=0.0,
        urllc_arrival_count=0,
        urllc_success_count=0,
        urllc_violation_count=0,
    ):
        user_rate = np.asarray(user_rate, dtype=np.float32).reshape(-1)
        avg_user_rate = np.asarray(avg_user_rate, dtype=np.float32).reshape(-1)

        metric = {}

        idx_e = np.where(self.user_slice == self.slice_embb)[0]
        idx_u = np.where(self.user_slice == self.slice_urllc)[0]
        idx_m = np.where(self.user_slice == self.slice_mmtc)[0]

        if idx_e.size > 0:
            embb_mean_rate = float(np.mean(user_rate[idx_e]))
            embb_sum_rate = float(np.sum(user_rate[idx_e]))
            embb_avg_rate = float(np.mean(avg_user_rate[idx_e]))
            # eMBB主目标使用puncturing后的有效总吞吐量。
            embb_utility = embb_sum_rate / max(self.reward_rate_scale, self.rate_eps)
        else:
            embb_mean_rate = 0.0
            embb_sum_rate = 0.0
            embb_avg_rate = 0.0
            embb_utility = 0.0

        urllc_success_ratio = float(np.clip(urllc_success_ratio, 0.0, 1.0))
        urllc_violation_ratio = float(np.clip(urllc_violation_ratio, 0.0, 1.0))
        urllc_constraint_gap = float(max(self.urllc_success_threshold - urllc_success_ratio, 0.0))

        mmtc_success_ratio = float(np.clip(mmtc_success_ratio, 0.0, 1.0))
        mmtc_active_ratio = float(np.clip(mmtc_active_ratio, 0.0, 1.0))
        mmtc_coverage_ratio = float(np.clip(mmtc_coverage_ratio, 0.0, 1.0))
        mmtc_constraint_gap = float(max(self.mmtc_success_threshold - mmtc_success_ratio, 0.0))

        if embb_base_sum_rate is None:
            embb_base_sum_rate = embb_sum_rate
        embb_base_sum_rate = float(embb_base_sum_rate)
        embb_loss_rate = float(max(embb_loss_rate, 0.0))
        puncture_ratio = float(np.clip(puncture_ratio, 0.0, 1.0))

        metric["embb_mean_rate"] = float(embb_mean_rate)
        metric["embb_sum_rate"] = float(embb_sum_rate)
        metric["embb_avg_rate"] = float(embb_avg_rate)
        metric["embb_utility"] = float(embb_utility)
        metric["embb_base_sum_rate"] = float(embb_base_sum_rate)
        metric["embb_effective_sum_rate"] = float(embb_sum_rate)
        metric["embb_loss_rate"] = float(embb_loss_rate)
        metric["puncture_ratio"] = float(puncture_ratio)

        metric["urllc_success_ratio"] = float(urllc_success_ratio)
        metric["urllc_violation_ratio"] = float(urllc_violation_ratio)
        metric["urllc_constraint_gap"] = float(urllc_constraint_gap)
        metric["urllc_arrival_count"] = int(urllc_arrival_count)
        metric["urllc_success_count"] = int(urllc_success_count)
        metric["urllc_violation_count"] = int(urllc_violation_count)

        metric["mmtc_success_ratio"] = float(mmtc_success_ratio)
        metric["mmtc_active_ratio"] = float(mmtc_active_ratio)
        metric["mmtc_coverage_ratio"] = float(mmtc_coverage_ratio)
        metric["mmtc_constraint_gap"] = float(mmtc_constraint_gap)

        # 兼容旧字段名，便于旧画图/保存逻辑不报错。
        metric["slice_rate_max_mean_rate"] = metric["embb_mean_rate"]
        metric["slice_rate_max_sum_rate"] = metric["embb_sum_rate"]
        metric["slice_rate_max_base_sum_rate"] = metric["embb_base_sum_rate"]
        metric["slice_rate_max_loss_rate"] = metric["embb_loss_rate"]
        metric["slice_rate_max_avg_rate"] = metric["embb_avg_rate"]
        metric["slice_rate_max_utility"] = metric["embb_utility"]
        metric["slice_guarantee_satisfaction"] = metric["urllc_success_ratio"]
        metric["slice_guarantee_avg_satisfaction"] = metric["urllc_success_ratio"]
        metric["slice_guarantee_deficit"] = metric["urllc_constraint_gap"]
        metric["slice_guarantee_outage"] = metric["urllc_violation_ratio"]
        metric["slice_fair_jain"] = metric["mmtc_success_ratio"]
        metric["slice_fair_target_satisfaction"] = metric["mmtc_success_ratio"]
        metric["slice_fair_min_avg_rate"] = 0.0
        metric["slice_fair_utility"] = metric["mmtc_success_ratio"]
        metric["slice_fair_deficit"] = metric["mmtc_constraint_gap"]
        metric["slice_fair_outage"] = 1.0 - metric["mmtc_success_ratio"]
        metric["rate_outage_ratio"] = metric["urllc_violation_ratio"]
        metric["rate_deficit_mean"] = 0.5 * (metric["urllc_constraint_gap"] + metric["mmtc_constraint_gap"])

        return metric

    # ================================================================
    # 构造状态
    # ================================================================
    def _normalize_embb_avg_rate_for_state(self, avg_user_rate):
        avg_user_rate = np.asarray(avg_user_rate, dtype=np.float32).reshape(-1)
        rate_mbps = np.maximum(avg_user_rate[self.embb_user_idx], 0.0) / max(
            float(self.state_rate_unit), float(self.rate_eps)
        )
        denom = np.log1p(max(float(self.state_rate_log_max_mbps), 1.0))
        feat = np.log1p(rate_mbps) / max(float(denom), float(self.rate_eps))
        return np.clip(feat, 0.0, 1.5).astype(np.float32)

    def build_state(
        self,
        step_idx,
        uav_pos,
        user_pos,
        residual_energy=None,
        avg_user_rate=None,
        urllc_queue_bits=None,
        urllc_deadline=None,
        mmtc_active=None,
        mmtc_pending_bits=None,
        prev_heading=None,
        unserved_time=None,
        associated_uav=None,
        ep_id=0,
    ):
        uav_pos = np.asarray(uav_pos, dtype=np.float32)
        user_pos = np.asarray(user_pos, dtype=np.float32)

        if avg_user_rate is None:
            avg_user_rate = np.zeros(self.user_n, dtype=np.float32)
        avg_user_rate = np.asarray(avg_user_rate, dtype=np.float32).reshape(-1)

        if mmtc_active is None:
            mmtc_active = np.zeros(self.user_n, dtype=np.float32)
        mmtc_active = np.asarray(mmtc_active, dtype=np.float32).reshape(-1)

        state = np.concatenate(
            [
                uav_pos.reshape(-1) / self.pos_norm,
                np.array(
                    [np.clip((self.max_step - int(step_idx)) / float(max(self.max_step, 1)), 0.0, 1.0)],
                    dtype=np.float32,
                ),
                user_pos.reshape(-1) / self.pos_norm,
                self._normalize_embb_avg_rate_for_state(avg_user_rate),
                np.clip(mmtc_active[self.mmtc_user_idx], 0.0, 1.0),
                self.calc_nfz_features(uav_pos).reshape(-1).astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        if state.shape[0] != self.state_dim:
            raise ValueError(f"状态维度错误：期望 {self.state_dim}，实际 {state.shape[0]}")
        return state

    # ================================================================
    # 构造multi-agent局部观测
    # ================================================================
    def build_local_obs_all(
        self,
        uav_pos,
        user_pos,
        residual_energy=None,
        avg_user_rate=None,
        urllc_queue_bits=None,
        urllc_deadline=None,
        mmtc_active=None,
        mmtc_pending_bits=None,
        prev_heading=None,
        unserved_time=None,
        associated_uav=None,
        ep_id=0,
        step_idx=0,
    ):
        obs_list = []
        for m in range(self.uav_n):
            obs_list.append(
                self.build_local_obs(
                    agent_idx=m,
                    uav_pos=uav_pos,
                    user_pos=user_pos,
                    residual_energy=residual_energy,
                    avg_user_rate=avg_user_rate,
                    mmtc_active=mmtc_active,
                    associated_uav=associated_uav,
                    step_idx=step_idx,
                )
            )
        return np.asarray(obs_list, dtype=np.float32)

    def build_local_obs(
        self,
        agent_idx,
        uav_pos,
        user_pos,
        residual_energy=None,
        avg_user_rate=None,
        urllc_queue_bits=None,
        urllc_deadline=None,
        mmtc_active=None,
        mmtc_pending_bits=None,
        prev_heading=None,
        unserved_time=None,
        associated_uav=None,  # 兼容旧接口；本版本不作为观测输入
        link_features=None,    # 兼容旧接口；本版本不作为观测输入
        step_idx=0,
    ):
        m = int(agent_idx)
        if m < 0 or m >= self.uav_n:
            raise ValueError(f"agent_idx越界：{m}")

        uav_pos = np.asarray(uav_pos, dtype=np.float32)
        user_pos = np.asarray(user_pos, dtype=np.float32)

        if avg_user_rate is None:
            avg_user_rate = np.zeros(self.user_n, dtype=np.float32)
        avg_user_rate = np.asarray(avg_user_rate, dtype=np.float32).reshape(-1)

        if mmtc_active is None:
            mmtc_active = np.zeros(self.user_n, dtype=np.float32)
        mmtc_active = np.asarray(mmtc_active, dtype=np.float32).reshape(-1)

        # 按固定物理高度排序，保持共享Actor输入顺序稳定。
        other_idx = sorted(
            [i for i in range(self.uav_n) if i != m],
            key=lambda i: (float(self.uav_height[i]), int(i)),
        )
        if len(other_idx) > 0:
            other_feat_rows = []
            for j in other_idx:
                rel_xy = (uav_pos[j] - uav_pos[m]) / self.pos_norm
                rel_h = (self.uav_height[j] - self.uav_height[m]) / self.height_norm
                other_feat_rows.append(
                    np.array([rel_xy[0], rel_xy[1], rel_h], dtype=np.float32)
                )
            other_uav_feat = np.concatenate(other_feat_rows, axis=0)
        else:
            other_uav_feat = np.zeros(0, dtype=np.float32)

        obs = np.concatenate(
            [
                uav_pos[m] / self.pos_norm,
                np.array(
                    [np.clip(self.uav_height[m] / self.height_norm, 0.0, 1.5)],
                    dtype=np.float32,
                ),
                np.array(
                    [np.clip((self.max_step - int(step_idx)) / float(max(self.max_step, 1)), 0.0, 1.0)],
                    dtype=np.float32,
                ),
                other_uav_feat,
                (user_pos - uav_pos[m][None, :]).reshape(-1) / self.pos_norm,
                self._normalize_embb_avg_rate_for_state(avg_user_rate),
                np.clip(mmtc_active[self.mmtc_user_idx], 0.0, 1.0),
                self.calc_nfz_features(uav_pos)[m].astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        if obs.shape[0] != self.local_obs_dim:
            raise ValueError(f"局部观测维度错误：期望 {self.local_obs_dim}，实际 {obs.shape[0]}")
        return obs

    # ================================================================
    # 工具函数
    # ================================================================
    @staticmethod
    def wrap_angle(angle):
        angle = np.asarray(angle, dtype=np.float32)
        return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

    @staticmethod
    def calc_min_pair_dist(pos):
        pos = np.asarray(pos, dtype=np.float32)

        if pos.shape[0] <= 1:
            return float("inf")

        min_dist = float("inf")

        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                min_dist = min(min_dist, dist)

        return float(min_dist)

# ============================================================================
# Finite-data completion-time task overrides
# ============================================================================
# The original geometry, mobility, channel, NFZ and action-safety utilities above
# are intentionally retained.  The overrides below replace only the traffic,
# state and episode-design portions of MainConfig.

_ORIGINAL_MAINCONFIG_INIT = MainConfig.__init__


def _finite_build_user_slice(self):
    """All users are homogeneous finite-data downlink users."""
    return np.zeros(self.user_n, dtype=np.int32)


# Install before __init__ is called so the original initializer also sees all
# users as the same service class.
MainConfig.build_user_slice = _finite_build_user_slice


def _finite_mainconfig_init(self):
    _ORIGINAL_MAINCONFIG_INIT(self)

    # ------------------------------------------------------------------
    # Finite-data task definition
    # ------------------------------------------------------------------
    self.task_name = "MultiUAV_FiniteData_MinCompletionTime"
    self.max_step = 180                 # safety time limit, not a true terminal
    self.max_episode_slots = self.max_step
    self.expected_episode_slots = 105    # target converged episode length for training schedules

    # Each UAV is sampled independently and uniformly over the whole area in
    # every episode.  No UAV is tied to a prescribed sub-region or user cluster.
    # NFZ rejection remains active, and a small pair-distance guard only prevents
    # nearly coincident starts; it does not spatially partition the UAVs.
    self.uav_init_uniform_low = 0.0
    self.uav_init_uniform_high = float(self.area_size)
    self.uav_init_margin = 0.0
    self.uav_init_min_pair_dist = 20.0

    # Legacy compact-start fields are retained as harmless compatibility aliases.
    self.uav_start_cluster_center_low = self.uav_init_uniform_low
    self.uav_start_cluster_center_high = self.uav_init_uniform_high
    self.uav_start_cluster_radius = 0.0
    self.uav_start_cluster_min_pair_dist = self.uav_init_min_pair_dist
    self.uav_start_cluster_fallback_low = self.uav_init_uniform_low
    self.uav_start_cluster_fallback_high = self.uav_init_uniform_high

    # File sizes are sampled directly and independently of geometry/channel.
    # Unit: Mbit.  The mixture creates heterogeneous workloads and a meaningful
    # long tail without making a remote user's file artificially smaller.
    self.file_size_short_mbit_range = (13.0, 26.0)
    self.file_size_medium_mbit_range = (30.0, 61.0)
    self.file_size_long_mbit_range = (69.0, 113.0)
    self.file_size_mix_prob = np.array([0.20, 0.50, 0.30], dtype=np.float32)
    self.file_size_unit_bits = 1.0e6
    self.file_size_min_bits = self.file_size_short_mbit_range[0] * self.file_size_unit_bits
    self.file_size_max_bits = self.file_size_long_mbit_range[1] * self.file_size_unit_bits

    # Compatibility/diagnostic quantities only.  File generation no longer
    # uses any reference channel rate.
    self.nominal_workload_rate_bps = 1.50e6
    self.reference_rate_min_bps = self.nominal_workload_rate_bps
    self.reference_rate_max_bps = self.nominal_workload_rate_bps
    self.reference_rate_eps = 1.0
    self.task_data_eps_bits = 1.0

    # Equal-RB allocation among unfinished users associated with each UAV.
    self.resource_allocation_mode = "finite_equal_rb"

    # No continuing URLLC/mMTC arrivals in a finite workload problem.  Keep the
    # three legacy indices only so old plotting/checkpoint code remains usable.
    self.slice_names = ["FiniteData", "Unused-1", "Unused-2"]
    self.user_slice = np.zeros(self.user_n, dtype=np.int32)
    self.user_slice_onehot = self.build_user_slice_onehot(self.user_slice)
    self.embb_user_idx = np.arange(self.user_n, dtype=np.int32)
    self.urllc_user_idx = np.zeros(0, dtype=np.int32)
    self.mmtc_user_idx = np.zeros(0, dtype=np.int32)
    self.embb_user_n = self.user_n
    self.mmtc_user_n = 0
    self.slice_user_count = np.array([self.user_n, 0, 0], dtype=np.int32)
    self.urllc_arrival_rate = 0.0
    self.urllc_arrival_prob = 0.0
    self.mmtc_active_prob = 0.0

    # ------------------------------------------------------------------
    # Completion-time reward
    # ------------------------------------------------------------------
    # Every executed slot costs one unit.  The dense term is strict
    # potential-based shaping: gamma * Phi(s') - Phi(s), where
    # Phi(s) = -mean_k(remaining_k / initial_k).  Using the same gamma as CTD4
    # preserves the optimal policy of the original minimum-makespan objective.
    self.reward_step_cost = 1.0
    self.reward_progress_weight = 8.0
    self.reward_potential_gamma = 0.995
    self.reward_time_limit_penalty = 20.0

    # Existing safety mechanism is retained unchanged.
    self.reward_boundary_weight = 10.0
    self.reward_nfz_weight = 10.0

    # Legacy reward fields are harmless compatibility aliases.
    self.reward_embb_weight = 0.0
    self.reward_urllc_constraint_weight = 0.0
    self.reward_mmtc_constraint_weight = 0.0
    self.reward_puncture_weight = 0.0

    # ------------------------------------------------------------------
    # State / local observation
    # ------------------------------------------------------------------
    # Per user: remaining ratio, unfinished mask, normalized absolute remaining bits.
    self.task_feature_dim_per_user = 3
    self.state_dim = (
        2 * self.uav_n
        + 2                              # total progress + unfinished ratio
        + 2 * self.user_n
        + self.task_feature_dim_per_user * self.user_n
        + self.uav_n * self.nfz_feature_dim_per_uav
    )
    self.local_obs_dim = (
        2                                # own horizontal position
        + 1                              # own height
        + 3 * (self.uav_n - 1)           # other UAV relative geometry
        + 2                              # total progress + unfinished ratio
        + 2 * self.user_n                # user relative positions
        + self.task_feature_dim_per_user * self.user_n
        + self.nfz_feature_dim_per_uav
    )

    self.task_slot_norm = float(max(self.max_step, 1))
    self.data_norm_bits = float(self.file_size_max_bits)
    self.step_norm = float(max(self.max_step - 1, 1))


MainConfig.__init__ = _finite_mainconfig_init


def _calc_large_scale_channel_gain(self, uav_pos, user_pos):
    """A2G expected gain without small-scale fading, used for file sizing."""
    dist_3d = self.calc_uav_user_distance_3d(uav_pos, user_pos)
    p_los = self.calc_los_probability(uav_pos, user_pos)
    los_gain = self.c0_linear * (dist_3d ** -2.0)
    nlos_gain = self.nlos_attenuation_kappa * self.c0_linear * (dist_3d ** -2.0)
    return (p_los * los_gain + (1.0 - p_los) * nlos_gain).astype(np.float32)


MainConfig.calc_large_scale_channel_gain = _calc_large_scale_channel_gain


def _sample_initial_data_bits(self, ep_id=0):
    """Sample deterministic heterogeneous file sizes independent of geometry."""
    rng = self.make_rng(ep_id=ep_id, offset=43117)
    cls = rng.choice(3, size=self.user_n, p=self.file_size_mix_prob)
    ranges = [
        self.file_size_short_mbit_range,
        self.file_size_medium_mbit_range,
        self.file_size_long_mbit_range,
    ]
    size_mbit = np.zeros(self.user_n, dtype=np.float32)
    for c, (lo, hi) in enumerate(ranges):
        idx = np.where(cls == c)[0]
        if idx.size > 0:
            size_mbit[idx] = rng.uniform(float(lo), float(hi), size=idx.size)
    return (size_mbit * float(self.file_size_unit_bits)).astype(np.float32)


MainConfig.sample_initial_data_bits = _sample_initial_data_bits


def _sample_target_service_slots(self, ep_id=0, initial_data_bits=None):
    """Compatibility diagnostic: nominal slots at a fixed 1.5-Mbps service rate."""
    if initial_data_bits is None:
        initial_data_bits = self.sample_initial_data_bits(ep_id=ep_id)
    slots = np.ceil(
        np.asarray(initial_data_bits, dtype=np.float32)
        / max(float(self.nominal_workload_rate_bps) * float(self.slot_time), 1.0)
    )
    return np.maximum(slots, 1.0).astype(np.int32)


MainConfig.sample_target_service_slots = _sample_target_service_slots


def _finite_task_features(self, initial_data_bits, remaining_data_bits, reference_rate):
    init_bits = np.maximum(
        np.asarray(initial_data_bits, dtype=np.float32).reshape(-1),
        float(self.task_data_eps_bits),
    )
    remain = np.maximum(
        np.asarray(remaining_data_bits, dtype=np.float32).reshape(-1),
        0.0,
    )
    remaining_ratio = np.clip(remain / init_bits, 0.0, 1.0)
    unfinished = (remain > float(self.task_data_eps_bits)).astype(np.float32)
    absolute_remaining = np.clip(
        remain / max(float(self.data_norm_bits), 1.0),
        0.0,
        1.0,
    )
    return np.stack(
        [remaining_ratio, unfinished, absolute_remaining], axis=1
    ).astype(np.float32)


MainConfig.build_task_features = _finite_task_features


def _finite_global_progress(self, initial_data_bits, remaining_data_bits):
    init_bits = np.maximum(
        np.asarray(initial_data_bits, dtype=np.float32).reshape(-1), 0.0
    )
    remain = np.maximum(
        np.asarray(remaining_data_bits, dtype=np.float32).reshape(-1), 0.0
    )
    progress = 1.0 - float(np.sum(remain) / max(float(np.sum(init_bits)), 1.0))
    unfinished_ratio = float(
        np.mean(remain > float(self.task_data_eps_bits))
    ) if remain.size > 0 else 0.0
    return np.array(
        [np.clip(progress, 0.0, 1.0), np.clip(unfinished_ratio, 0.0, 1.0)],
        dtype=np.float32,
    )


MainConfig.build_global_progress_features = _finite_global_progress


def _finite_build_state(
    self,
    step_idx,
    uav_pos,
    user_pos,
    residual_energy=None,
    avg_user_rate=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
    prev_heading=None,
    unserved_time=None,
    associated_uav=None,
    ep_id=0,
    initial_data_bits=None,
    remaining_data_bits=None,
    reference_rate=None,
):
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    if initial_data_bits is None:
        initial_data_bits = np.ones(self.user_n, dtype=np.float32)
    if remaining_data_bits is None:
        remaining_data_bits = np.asarray(initial_data_bits, dtype=np.float32).copy()
    if reference_rate is None:
        reference_rate = np.ones(self.user_n, dtype=np.float32)

    task_feat = self.build_task_features(
        initial_data_bits, remaining_data_bits, reference_rate
    )
    global_feat = self.build_global_progress_features(
        initial_data_bits, remaining_data_bits
    )
    state = np.concatenate(
        [
            uav_pos.reshape(-1) / self.pos_norm,
            global_feat,
            user_pos.reshape(-1) / self.pos_norm,
            task_feat.reshape(-1),
            self.calc_nfz_features(uav_pos).reshape(-1).astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if state.size != int(self.state_dim):
        raise ValueError(f"状态维度错误：期望 {self.state_dim}，实际 {state.size}")
    return state


MainConfig.build_state = _finite_build_state


def _finite_build_local_obs(
    self,
    agent_idx,
    uav_pos,
    user_pos,
    residual_energy=None,
    avg_user_rate=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
    prev_heading=None,
    unserved_time=None,
    associated_uav=None,
    link_features=None,
    step_idx=0,
    initial_data_bits=None,
    remaining_data_bits=None,
    reference_rate=None,
):
    m = int(agent_idx)
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    if initial_data_bits is None:
        initial_data_bits = np.ones(self.user_n, dtype=np.float32)
    if remaining_data_bits is None:
        remaining_data_bits = np.asarray(initial_data_bits, dtype=np.float32).copy()
    if reference_rate is None:
        reference_rate = np.ones(self.user_n, dtype=np.float32)

    other_idx = sorted(
        [i for i in range(self.uav_n) if i != m],
        key=lambda i: (float(self.uav_height[i]), int(i)),
    )
    other_rows = []
    for j in other_idx:
        rel_xy = (uav_pos[j] - uav_pos[m]) / self.pos_norm
        rel_h = (self.uav_height[j] - self.uav_height[m]) / self.height_norm
        other_rows.append(np.array([rel_xy[0], rel_xy[1], rel_h], dtype=np.float32))
    other_feat = np.concatenate(other_rows) if other_rows else np.zeros(0, dtype=np.float32)

    task_feat = self.build_task_features(
        initial_data_bits, remaining_data_bits, reference_rate
    )
    global_feat = self.build_global_progress_features(
        initial_data_bits, remaining_data_bits
    )
    obs = np.concatenate(
        [
            uav_pos[m] / self.pos_norm,
            np.array([self.uav_height[m] / self.height_norm], dtype=np.float32),
            other_feat,
            global_feat,
            (user_pos - uav_pos[m][None, :]).reshape(-1) / self.pos_norm,
            task_feat.reshape(-1),
            self.calc_nfz_features(uav_pos)[m].astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if obs.size != int(self.local_obs_dim):
        raise ValueError(f"局部观测维度错误：期望 {self.local_obs_dim}，实际 {obs.size}")
    return obs


MainConfig.build_local_obs = _finite_build_local_obs


def _finite_build_local_obs_all(
    self,
    uav_pos,
    user_pos,
    residual_energy=None,
    avg_user_rate=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
    prev_heading=None,
    unserved_time=None,
    associated_uav=None,
    ep_id=0,
    step_idx=0,
    initial_data_bits=None,
    remaining_data_bits=None,
    reference_rate=None,
):
    return np.asarray(
        [
            self.build_local_obs(
                agent_idx=m,
                uav_pos=uav_pos,
                user_pos=user_pos,
                step_idx=step_idx,
                initial_data_bits=initial_data_bits,
                remaining_data_bits=remaining_data_bits,
                reference_rate=reference_rate,
            )
            for m in range(self.uav_n)
        ],
        dtype=np.float32,
    )


MainConfig.build_local_obs_all = _finite_build_local_obs_all


# ============================================================================
# Whole-area uniform UAV-start override
# ============================================================================
def _sample_uniform_uav_init_pos(self, ep_id=0):
    """Sample all UAV starts uniformly over the whole valid service area.

    Every UAV uses the same two-dimensional uniform distribution and no UAV is
    assigned to a particular sub-region.  Different episode IDs produce different
    deterministic random draws through ``make_rng``.  Samples inside an NFZ are
    rejected, and a small minimum pair distance avoids nearly identical starts.
    """
    if not bool(getattr(self, "random_uav_init_each_ep", True)):
        return np.asarray(self.fixed_uav_init_pos, dtype=np.float32).copy()

    rng = self.make_rng(ep_id=ep_id, offset=4000)
    low = float(getattr(self, "uav_init_uniform_low", 0.0))
    high = float(getattr(self, "uav_init_uniform_high", self.area_size))
    min_pair = float(getattr(self, "uav_init_min_pair_dist", 20.0))
    attempts = int(getattr(self, "uav_init_max_try", 5000))

    if not (0.0 <= low < high <= float(self.area_size)):
        raise ValueError(
            f"Invalid whole-area UAV initialization range: [{low}, {high}] "
            f"for area_size={self.area_size}"
        )

    best = None
    best_min_pair = -1.0
    for _ in range(max(attempts, 1)):
        # Joint i.i.d. uniform draw: every UAV has exactly the same sampling law.
        pos = rng.uniform(low=low, high=high, size=(self.uav_n, 2)).astype(np.float32)

        if bool(getattr(self, "use_nfz", False)) and bool(
            getattr(self, "avoid_nfz_for_uav_init", True)
        ):
            if not all(
                self.is_point_safe_from_nfz(p, margin=self.nfz_init_safe_margin)
                for p in pos
            ):
                continue

        pair_dist = float(self.calc_min_pair_dist(pos))
        if pair_dist > best_min_pair:
            best = pos.copy()
            best_min_pair = pair_dist
        if pair_dist >= min_pair:
            return pos.astype(np.float32)

    if best is None:
        raise RuntimeError(
            "Unable to sample whole-area UAV initial positions outside the NFZs"
        )
    return np.asarray(best, dtype=np.float32)


MainConfig.sample_uav_init_pos = _sample_uniform_uav_init_pos

# ============================================================================
# eMBB finite files + pure-URLLC-puncturing + mMTC connectivity overrides
# ============================================================================
_PRE_SLICE_MAINCONFIG_INIT = MainConfig.__init__
_PRE_SLICE_SAMPLE_INITIAL_DATA_BITS = MainConfig.sample_initial_data_bits
_PRE_SLICE_SAMPLE_TARGET_SERVICE_SLOTS = MainConfig.sample_target_service_slots


def _sliced_finite_mainconfig_init(self):
    """Install the three-service finite-completion-time task.

    eMBB users have finite files and determine episode termination. URLLC uses
    no reserved RBs and can only puncture eMBB RBs. mMTC uses a fixed common
    access/reference channel and is evaluated only through access SINR.
    """
    _PRE_SLICE_MAINCONFIG_INIT(self)

    # With 60 finite-file eMBB users, the per-user workloads below are calibrated
    # so a converged policy should finish in roughly 65--75 slots. Noise and
    # training schedules therefore use 70 environment steps per episode.
    self.expected_episode_slots = 105

    self.task_name = "MultiUAV_eMBB_Finite_URLLC_Puncture_mMTC_Connectivity"
    self.slice_names = ["eMBB-FiniteData", "URLLC-Puncture", "mMTC-Connectivity"]

    # Deterministic service classes: 60 eMBB, 15 URLLC, 15 mMTC.
    if int(self.user_n) != 120:
        raise ValueError(f"当前任务要求总用户数为90，实际为{self.user_n}")
    self.user_slice = np.full(self.user_n, self.slice_embb, dtype=np.int32)
    self.user_slice[90:105] = self.slice_urllc
    self.user_slice[105:120] = self.slice_mmtc
    self.user_slice_onehot = self.build_user_slice_onehot(self.user_slice)
    self.embb_user_idx = np.where(self.user_slice == self.slice_embb)[0].astype(np.int32)
    self.urllc_user_idx = np.where(self.user_slice == self.slice_urllc)[0].astype(np.int32)
    self.mmtc_user_idx = np.where(self.user_slice == self.slice_mmtc)[0].astype(np.int32)
    self.embb_user_n = int(self.embb_user_idx.size)
    self.urllc_user_n = int(self.urllc_user_idx.size)
    self.mmtc_user_n = int(self.mmtc_user_idx.size)
    self.slice_user_count = np.array(
        [self.embb_user_n, self.urllc_user_n, self.mmtc_user_n], dtype=np.int32
    )
    self.user_rate_req = np.zeros(self.user_n, dtype=np.float32)

    # ------------------------------------------------------------------
    # Resource/traffic model
    # ------------------------------------------------------------------
    # All dynamic data RBs are first allocated to unfinished eMBB users.
    self.resource_allocation_mode = "embb_workload_aware_then_urllc_puncture"
    self.embb_rb_allocation_policy = "minimax_remaining_workload"
    self.use_drl_slice_budget = False

    # URLLC: no reservation; packets can only puncture already assigned eMBB RBs.
    self.urllc_arrival_rate = 80.0 / 15.0  # 5.333/s/user; aggregate remains 80 packets/s
    self.urllc_packet_bits = 512.0
    self.urllc_deadline_tti = 1

    # URLLC physical layer: finite-blocklength (FBL) normal approximation.
    # eMBB remains Shannon-rate based.  The FBL error probability is the
    # target decoding error probability used when converting URLLC SINR and
    # allocated RB-time into reliably deliverable short-packet bits.
    self.urllc_fbl_enabled = True
    self.urllc_fbl_error_prob = 1e-5
    self.urllc_fbl_qinv = float(
        NormalDist().inv_cdf(1.0 - self.urllc_fbl_error_prob)
    )
    # One complex channel use per Hz-second, hence 180 channel uses for a
    # 180-kHz RB over a 1-ms TTI.  Multiple punctured RBs form one FBL block.
    self.urllc_fbl_channel_uses_per_rb_tti = float(
        self.rb_bandwidth_hz * self.tti_time
    )

    self.urllc_arrival_prob = min(1.0, self.urllc_arrival_rate * self.tti_time)
    # URLLC resource rule: no static reservation and no additional artificial
    # puncturing cap.  In every TTI, URLLC first consumes RBs that are idle
    # after eMBB allocation; if those are insufficient, it punctures eMBB RBs.
    # The only per-UAV upper bound is the physical RB pool itself (rb_n=50).
    self.urllc_idle_rb_first = True
    self.urllc_physical_rb_limit_per_tti = int(self.rb_n)
    # Legacy compatibility fields: they now equal the physical RB pool rather
    # than imposing an extra 20%/10-RB puncturing restriction.
    self.urllc_max_puncture_ratio_per_tti = 1.0
    self.urllc_max_puncture_rb_per_tti = int(self.rb_n)
    self.urllc_reliability_target = 0.99
    self.urllc_success_threshold = self.urllc_reliability_target
    self.urllc_reward_window_slots = 10
    self.urllc_reward_min_window_arrivals = 10
    self.urllc_window_arrival_norm = max(
        1.0,
        self.urllc_arrival_rate
        * max(self.urllc_user_n, 1)
        * self.urllc_reward_window_slots,
    )

    # mMTC: all devices are evaluated every slot; no activation and no data RB.
    # A common narrowband access/reference channel is assumed, with one-RB power
    # and bandwidth. Connectivity uses deterministic large-scale A2G SINR so the
    # trajectory has a direct, low-variance influence on the constraint.
    self.mmtc_active_prob = 1.0
    self.mmtc_access_bandwidth_hz = float(self.rb_bandwidth_hz)
    self.mmtc_access_power_w_each_uav = float(self.rb_power_w_each_uav)
    self.mmtc_sinr_threshold_db = -1.5
    self.mmtc_sinr_threshold = float(10.0 ** (self.mmtc_sinr_threshold_db / 10.0))
    self.mmtc_connectivity_target = 0.80
    self.mmtc_connectivity_floor = 0.60
    self.mmtc_success_threshold = self.mmtc_connectivity_target

    # ------------------------------------------------------------------
    # Completion-time reward and constrained-QoS optimization
    # ------------------------------------------------------------------
    # max_step remains the original 180-slot safety truncation.  It is not a
    # task terminal: truncated replay samples continue to bootstrap and no
    # special cutoff reward is added.
    self.max_step = 180
    self.max_episode_slots = self.max_step
    self.task_slot_norm = float(max(self.max_step, 1))
    self.step_norm = float(max(self.max_step - 1, 1))

    # Potential Phi(s) is aligned with makespan rather than one-step average
    # progress.  It combines: hardest user's estimated remaining time, heaviest
    # UAV load, and mean remaining file ratio.  The weights sum to one.
    self.reward_potential_user_bottleneck_weight = 0.70
    self.reward_potential_uav_bottleneck_weight = 0.20
    self.reward_potential_mean_remaining_weight = 0.10
    self.reward_potential_softmax_temperature = 10.0
    self.reward_potential_component_clip = 2.0

    # Centered per-UAV credit: beta * (local normalized progress - team mean).
    # The centered term sums to zero across UAVs and therefore preserves the
    # shared team objective while improving multi-agent credit assignment.
    self.reward_agent_credit_weight = 2.0

    # Adaptive Lagrange multipliers replace fixed QoS reward weights.  They are
    # updated once per training episode from episode-level constraint residuals.
    self.use_adaptive_qos_lagrange = True
    self.qos_lagrange_urllc_init = 10.0
    self.qos_lagrange_mmtc_init = 1.0
    self.qos_lagrange_urllc = float(self.qos_lagrange_urllc_init)
    self.qos_lagrange_mmtc = float(self.qos_lagrange_mmtc_init)
    self.qos_lagrange_urllc_lr = 5.0
    self.qos_lagrange_mmtc_lr = 0.5
    self.qos_lagrange_urllc_min = 0.0
    self.qos_lagrange_mmtc_min = 0.0
    self.qos_lagrange_urllc_max = 100.0
    self.qos_lagrange_mmtc_max = 20.0

    # Compatibility aliases retained for result readers.  The runtime reward
    # uses qos_lagrange_urllc/qos_lagrange_mmtc instead of these fixed weights.
    self.reward_urllc_constraint_weight = self.qos_lagrange_urllc_init
    self.reward_mmtc_constraint_weight = self.qos_lagrange_mmtc_init
    self.reward_mmtc_floor_weight = 0.0
    self.reward_terminal_urllc_weight = 0.0
    self.reward_terminal_mmtc_weight = 0.0
    self.reward_time_limit_penalty = 0.0
    self.reward_puncture_weight = 0.0  # eMBB loss is already reflected in progress.

    # Safety reward scale: shield remains unchanged, but reward magnitudes are
    # kept comparable with the one-unit slot cost.
    self.reward_illegal_action_fixed_penalty = 15.0
    self.safe_distance_penalty_margin = 30.0
    self.safe_distance_penalty_max = 2.0

    # Validation uses lexicographic constrained selection rather than a single
    # weighted score: completion -> data progress -> QoS feasibility/gap -> slots.
    self.validation_required_success_ratio = 1.0
    self.validation_required_qos_feasible_ratio = 1.0
    # Legacy fields are retained only for backward-compatible logs.
    self.validation_urllc_gap_multiplier = 100.0
    self.validation_mmtc_gap_multiplier = 5.0

    # ------------------------------------------------------------------
    # State / local observation
    # ------------------------------------------------------------------
    # Per user:
    #   3 finite-file features (zero for non-eMBB),
    #   3-dimensional service one-hot,
    #   2 large-scale QoS link features.
    # Global QoS history:
    #   URLLC window reliability/gap/cumulative reliability/window load,
    #   mMTC last ratio/episode mean/gap/severe gap.
    self.task_feature_dim_per_user = 3
    self.slice_feature_dim_per_user = int(self.slice_n)
    self.qos_link_feature_dim_per_user = 2
    self.qos_global_feature_dim = 8

    per_user_dim = (
        self.task_feature_dim_per_user
        + self.slice_feature_dim_per_user
        + self.qos_link_feature_dim_per_user
    )
    self.state_dim = (
        2 * self.uav_n
        + 2
        + self.qos_global_feature_dim
        + 2 * self.user_n
        + per_user_dim * self.user_n
        + self.uav_n * self.nfz_feature_dim_per_uav
    )
    self.local_obs_dim = (
        2
        + 1
        + 3 * (self.uav_n - 1)
        + 2
        + self.qos_global_feature_dim
        + 2 * self.user_n
        + per_user_dim * self.user_n
        + self.nfz_feature_dim_per_uav
    )


def _sliced_sample_initial_data_bits(self, ep_id=0):
    bits = _PRE_SLICE_SAMPLE_INITIAL_DATA_BITS(self, ep_id=ep_id)
    mask = np.zeros(self.user_n, dtype=bool)
    mask[self.embb_user_idx] = True
    bits[~mask] = 0.0
    return bits.astype(np.float32)


def _sliced_sample_target_service_slots(self, ep_id=0, initial_data_bits=None):
    if initial_data_bits is None:
        initial_data_bits = self.sample_initial_data_bits(ep_id=ep_id)
    bits = np.asarray(initial_data_bits, dtype=np.float32)
    slots = np.zeros(self.user_n, dtype=np.int32)
    idx = self.embb_user_idx
    slots[idx] = np.maximum(
        np.ceil(
            bits[idx]
            / max(float(self.nominal_workload_rate_bps) * float(self.slot_time), 1.0)
        ),
        1.0,
    ).astype(np.int32)
    return slots


def _sliced_task_features(self, initial_data_bits, remaining_data_bits, reference_rate):
    init_bits = np.asarray(initial_data_bits, dtype=np.float32).reshape(-1)
    remain = np.maximum(np.asarray(remaining_data_bits, dtype=np.float32).reshape(-1), 0.0)
    feat = np.zeros((self.user_n, 3), dtype=np.float32)
    idx = self.embb_user_idx
    if idx.size > 0:
        denom = np.maximum(init_bits[idx], float(self.task_data_eps_bits))
        feat[idx, 0] = np.clip(remain[idx] / denom, 0.0, 1.0)
        feat[idx, 1] = (remain[idx] > float(self.task_data_eps_bits)).astype(np.float32)
        feat[idx, 2] = np.clip(
            remain[idx] / max(float(self.data_norm_bits), 1.0), 0.0, 1.0
        )
    return feat


def _sliced_global_progress(self, initial_data_bits, remaining_data_bits):
    init_bits = np.asarray(initial_data_bits, dtype=np.float32).reshape(-1)
    remain = np.maximum(np.asarray(remaining_data_bits, dtype=np.float32).reshape(-1), 0.0)
    idx = self.embb_user_idx
    if idx.size == 0:
        return np.zeros(2, dtype=np.float32)
    init_e = np.maximum(init_bits[idx], 0.0)
    rem_e = remain[idx]
    progress = 1.0 - float(np.sum(rem_e) / max(float(np.sum(init_e)), 1.0))
    unfinished_ratio = float(np.mean(rem_e > float(self.task_data_eps_bits)))
    return np.array(
        [np.clip(progress, 0.0, 1.0), np.clip(unfinished_ratio, 0.0, 1.0)],
        dtype=np.float32,
    )


def _sliced_urllc_fbl_bits(self, sinr, rb_count=1):
    """Finite-blocklength URLLC payload bits for a given SINR and RB count.

    The normal approximation is
        L = n log2(1+gamma)
            - sqrt(n V(gamma)) Q^{-1}(epsilon) / ln(2),
    where n = N_RB * B_RB * T_TTI and
    V(gamma) = 1 - (1+gamma)^(-2).

    The packet is jointly coded across all RBs punctured for it in the TTI.
    eMBB throughput is intentionally unchanged and continues to use Shannon
    capacity.
    """
    gamma = np.maximum(np.asarray(sinr, dtype=np.float64), 0.0)
    n_rb = np.maximum(np.asarray(rb_count, dtype=np.float64), 0.0)
    n = n_rb * float(self.urllc_fbl_channel_uses_per_rb_tti)
    dispersion = np.maximum(1.0 - np.power(1.0 + gamma, -2.0), 0.0)
    first_order = n * np.log2(1.0 + gamma)
    penalty = (
        np.sqrt(np.maximum(n * dispersion, 0.0))
        * float(self.urllc_fbl_qinv)
        / np.log(2.0)
    )
    bits = np.maximum(first_order - penalty, 0.0)
    bits = np.where(n_rb > 0.0, bits, 0.0)
    if np.ndim(bits) == 0:
        return float(bits)
    return bits.astype(np.float64)


def _sliced_urllc_required_rb(self, sinr, packet_bits=None, max_rb=None):
    """Minimum same-TTI RB count for one URLLC FBL packet.

    For the active normal approximation,

        L(N) = a N - b sqrt(N),

    where ``N`` is the number of RBs.  Setting ``x=sqrt(N)`` turns the
    feasibility boundary ``L(N) >= packet_bits`` into a quadratic inequality,
    so the continuous crossing point is available in closed form.  We round up
    to an integer RB count and then make a tiny local correction using the
    original ``calc_urllc_fbl_bits`` function.  Therefore this routine returns
    exactly the same minimum feasible integer RB count as the former 1..max_rb
    linear scan, but needs only O(1) work instead of up to ``max_rb`` FBL calls.
    """
    if packet_bits is None:
        packet_bits = float(self.urllc_packet_bits)
    if max_rb is None:
        max_rb = int(self.rb_n)

    max_rb = max(int(max_rb), 0)
    target = max(float(packet_bits), 0.0)
    gamma = float(sinr)

    if target <= 0.0:
        return 0
    if max_rb <= 0 or (not np.isfinite(gamma)) or gamma <= 0.0:
        return np.inf

    n0 = float(self.urllc_fbl_channel_uses_per_rb_tti)
    if (not np.isfinite(n0)) or n0 <= 0.0:
        return np.inf

    # L(N) = a*N - b*sqrt(N), before the final max(., 0) clipping.
    dispersion = max(1.0 - (1.0 + gamma) ** (-2.0), 0.0)
    a = n0 * float(np.log2(1.0 + gamma))
    b = (
        float(np.sqrt(max(n0 * dispersion, 0.0)))
        * float(self.urllc_fbl_qinv)
        / float(np.log(2.0))
    )
    if (not np.isfinite(a)) or a <= 0.0 or (not np.isfinite(b)):
        return np.inf

    # Positive root of a*x^2 - b*x - target = 0, with x = sqrt(N).
    disc = b * b + 4.0 * a * target
    if (not np.isfinite(disc)) or disc < 0.0:
        return np.inf
    x_req = (b + float(np.sqrt(disc))) / (2.0 * a)
    n_cont = x_req * x_req
    if not np.isfinite(n_cont):
        return np.inf

    # ceil gives the analytical integer candidate.  A two-sided local check
    # preserves the exact old ``+1e-12 >= target`` integer-boundary semantics.
    candidate = max(1, int(np.ceil(n_cont)))

    # If floating-point roundoff pushed the candidate one RB too high, step
    # down while the previous RB count is still feasible.  In normal operation
    # this loop executes zero times (occasionally once near an exact boundary).
    while candidate > 1:
        prev_bits = float(self.calc_urllc_fbl_bits(gamma, candidate - 1))
        if prev_bits + 1e-12 < target:
            break
        candidate -= 1

    # Conversely, correct a rare downward rounding error.  This is still O(1)
    # around the analytical root rather than a scan from RB=1.
    while candidate <= max_rb:
        cand_bits = float(self.calc_urllc_fbl_bits(gamma, candidate))
        if cand_bits + 1e-12 >= target:
            return int(candidate)
        candidate += 1

    return np.inf


def _sliced_qos_link_features(self, uav_pos, user_pos):
    """Large-scale link hints for URLLC and mMTC users.

    These features are deterministic for a geometry and do not expose the
    instantaneous small-scale fading used by eMBB/URLLC packet service.
    """
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    gain = self.calc_large_scale_channel_gain(uav_pos, user_pos)
    antenna, _ = self.calc_antenna_gain_matrix(uav_pos, user_pos)
    rx = float(self.rb_power_w_each_uav) * gain * antenna
    noise = float(self.rb_bandwidth_hz) * float(self.noise_psd_w_per_hz)
    total_rx = np.sum(rx, axis=0, keepdims=True)
    sinr = rx / np.maximum(noise + total_rx - rx, 1e-30)
    best_m = np.argmax(rx, axis=0)
    best_sinr = sinr[best_m, np.arange(self.user_n)]

    feat = np.zeros((self.user_n, 2), dtype=np.float32)
    if self.urllc_user_idx.size > 0:
        idx = self.urllc_user_idx
        s = np.maximum(best_sinr[idx], 0.0)
        feat[idx, 0] = np.clip(np.log1p(s) / np.log1p(100.0), 0.0, 1.0)
        # The second URLLC link feature uses the same FBL model as packet
        # service, so the actor observes a geometry-dependent indication of
        # how many punctured RBs a 512-bit short packet would require.
        req = np.asarray(
            [
                self.calc_urllc_required_rb(
                    float(gamma),
                    packet_bits=float(self.urllc_packet_bits),
                    max_rb=int(self.rb_n),
                )
                for gamma in s
            ],
            dtype=np.float64,
        )
        physical_rb = max(float(self.urllc_physical_rb_limit_per_tti), 1.0)
        req[~np.isfinite(req)] = 2.0 * physical_rb
        feat[idx, 1] = np.clip(req / physical_rb, 0.0, 2.0) / 2.0
    if self.mmtc_user_idx.size > 0:
        idx = self.mmtc_user_idx
        ratio = np.maximum(best_sinr[idx], 1e-12) / max(
            float(self.mmtc_sinr_threshold), 1e-12
        )
        feat[idx, 0] = 0.5 + 0.5 * np.tanh(np.log10(ratio))
        feat[idx, 1] = (best_sinr[idx] >= float(self.mmtc_sinr_threshold)).astype(np.float32)
    return feat.astype(np.float32)


def _sliced_build_state(
    self,
    step_idx,
    uav_pos,
    user_pos,
    residual_energy=None,
    avg_user_rate=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
    prev_heading=None,
    unserved_time=None,
    associated_uav=None,
    ep_id=0,
    initial_data_bits=None,
    remaining_data_bits=None,
    reference_rate=None,
    qos_global_features=None,
):
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    if initial_data_bits is None:
        initial_data_bits = np.zeros(self.user_n, dtype=np.float32)
    if remaining_data_bits is None:
        remaining_data_bits = np.asarray(initial_data_bits, dtype=np.float32).copy()
    if reference_rate is None:
        reference_rate = np.ones(self.user_n, dtype=np.float32)
    if qos_global_features is None:
        qos_global_features = np.zeros(self.qos_global_feature_dim, dtype=np.float32)
    qos_global_features = np.asarray(qos_global_features, dtype=np.float32).reshape(-1)
    if qos_global_features.size != self.qos_global_feature_dim:
        raise ValueError(
            f"qos_global_features维度错误：期望{self.qos_global_feature_dim}，"
            f"实际{qos_global_features.size}"
        )

    task_feat = self.build_task_features(initial_data_bits, remaining_data_bits, reference_rate)
    global_feat = self.build_global_progress_features(initial_data_bits, remaining_data_bits)
    qos_link = self.build_qos_link_features(uav_pos, user_pos)
    state = np.concatenate(
        [
            uav_pos.reshape(-1) / self.pos_norm,
            global_feat,
            qos_global_features,
            user_pos.reshape(-1) / self.pos_norm,
            task_feat.reshape(-1),
            self.user_slice_onehot.reshape(-1),
            qos_link.reshape(-1),
            self.calc_nfz_features(uav_pos).reshape(-1).astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if state.size != int(self.state_dim):
        raise ValueError(f"状态维度错误：期望 {self.state_dim}，实际 {state.size}")
    return state


def _sliced_build_local_obs(
    self,
    agent_idx,
    uav_pos,
    user_pos,
    residual_energy=None,
    avg_user_rate=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
    prev_heading=None,
    unserved_time=None,
    associated_uav=None,
    link_features=None,
    step_idx=0,
    initial_data_bits=None,
    remaining_data_bits=None,
    reference_rate=None,
    qos_global_features=None,
):
    m = int(agent_idx)
    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)
    if initial_data_bits is None:
        initial_data_bits = np.zeros(self.user_n, dtype=np.float32)
    if remaining_data_bits is None:
        remaining_data_bits = np.asarray(initial_data_bits, dtype=np.float32).copy()
    if reference_rate is None:
        reference_rate = np.ones(self.user_n, dtype=np.float32)
    if qos_global_features is None:
        qos_global_features = np.zeros(self.qos_global_feature_dim, dtype=np.float32)
    qos_global_features = np.asarray(qos_global_features, dtype=np.float32).reshape(-1)

    other_idx = sorted(
        [i for i in range(self.uav_n) if i != m],
        key=lambda i: (float(self.uav_height[i]), int(i)),
    )
    other_rows = []
    for j in other_idx:
        rel_xy = (uav_pos[j] - uav_pos[m]) / self.pos_norm
        rel_h = (self.uav_height[j] - self.uav_height[m]) / self.height_norm
        other_rows.append(np.array([rel_xy[0], rel_xy[1], rel_h], dtype=np.float32))
    other_feat = np.concatenate(other_rows) if other_rows else np.zeros(0, dtype=np.float32)

    task_feat = self.build_task_features(initial_data_bits, remaining_data_bits, reference_rate)
    global_feat = self.build_global_progress_features(initial_data_bits, remaining_data_bits)
    qos_link = self.build_qos_link_features(uav_pos, user_pos)
    obs = np.concatenate(
        [
            uav_pos[m] / self.pos_norm,
            np.array([self.uav_height[m] / self.height_norm], dtype=np.float32),
            other_feat,
            global_feat,
            qos_global_features,
            (user_pos - uav_pos[m][None, :]).reshape(-1) / self.pos_norm,
            task_feat.reshape(-1),
            self.user_slice_onehot.reshape(-1),
            qos_link.reshape(-1),
            self.calc_nfz_features(uav_pos)[m].astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if obs.size != int(self.local_obs_dim):
        raise ValueError(f"局部观测维度错误：期望 {self.local_obs_dim}，实际 {obs.size}")
    return obs


def _sliced_build_local_obs_all(
    self,
    uav_pos,
    user_pos,
    residual_energy=None,
    avg_user_rate=None,
    urllc_queue_bits=None,
    urllc_deadline=None,
    mmtc_active=None,
    mmtc_pending_bits=None,
    prev_heading=None,
    unserved_time=None,
    associated_uav=None,
    ep_id=0,
    step_idx=0,
    initial_data_bits=None,
    remaining_data_bits=None,
    reference_rate=None,
    qos_global_features=None,
):
    return np.asarray(
        [
            self.build_local_obs(
                agent_idx=m,
                uav_pos=uav_pos,
                user_pos=user_pos,
                step_idx=step_idx,
                initial_data_bits=initial_data_bits,
                remaining_data_bits=remaining_data_bits,
                reference_rate=reference_rate,
                qos_global_features=qos_global_features,
            )
            for m in range(self.uav_n)
        ],
        dtype=np.float32,
    )


MainConfig.__init__ = _sliced_finite_mainconfig_init
MainConfig.sample_initial_data_bits = _sliced_sample_initial_data_bits
MainConfig.sample_target_service_slots = _sliced_sample_target_service_slots
MainConfig.build_task_features = _sliced_task_features
MainConfig.build_global_progress_features = _sliced_global_progress
MainConfig.calc_urllc_fbl_bits = _sliced_urllc_fbl_bits
MainConfig.calc_urllc_required_rb = _sliced_urllc_required_rb
MainConfig.build_qos_link_features = _sliced_qos_link_features
MainConfig.build_state = _sliced_build_state
MainConfig.build_local_obs = _sliced_build_local_obs
MainConfig.build_local_obs_all = _sliced_build_local_obs_all

# Keep the public helper consistent with the active three-slice task.
def _sliced_build_user_slice(self):
    if int(self.user_n) != 120:
        raise ValueError(f"当前任务要求总用户数为90，实际为{self.user_n}")
    user_slice = np.full(self.user_n, self.slice_embb, dtype=np.int32)
    user_slice[90:105] = self.slice_urllc
    user_slice[105:120] = self.slice_mmtc
    return user_slice.astype(np.int32)


MainConfig.build_user_slice = _sliced_build_user_slice
