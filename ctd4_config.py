"""
ZHANG Wenqi

CTD4 hyperparameters for the finite-data minimum-completion-time task.
The original distributional critics and Kalman fusion are unchanged.
"""

from __future__ import annotations

import os
import torch


class Ctd4Config:
    def __init__(self, main_cfg):
        self.algorithm = str(getattr(main_cfg, "proposed_algorithm", "CTD4")).strip().upper()
        if self.algorithm not in ("CTD4", "DDPG", "TD3"):
            raise ValueError(
                f"proposed_algorithm必须是 CTD4/DDPG/TD3，当前为：{self.algorithm}"
            )
        self.algorithm_key = self.algorithm.lower()
        self.algo_name = f"SharedActor-MA-{self.algorithm}-Finite-eMBB-URLLC-mMTC"
        self.env_name = "Finite_eMBB_URLLC_Puncture_mMTC_Connectivity_MultiUAV"
        self.model_checkpoint_name = f"{self.algorithm_key}_checkpoint.pt"
        self.model_actor_name = f"{self.algorithm_key}_actor.pt"

        # ============================================================
        # 随机种子
        # ============================================================
        self.seed = int(main_cfg.seed)

        # ============================================================
        # device
        # ============================================================
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ============================================================
        # 维度
        # ============================================================
        self.state_dim = int(main_cfg.state_dim)
        self.agent_n = int(main_cfg.agent_n)
        self.local_obs_dim = int(main_cfg.local_obs_dim)
        self.local_action_dim = int(main_cfg.local_action_dim)
        self.action_dim = int(main_cfg.action_dim)

        self.action_low = float(main_cfg.action_low)
        self.action_high = float(main_cfg.action_high)

        # ============================================================
        # 折扣因子与软更新
        #
        # 当前问题以全部eMBB文件完成作为真实终止，180 slot仅为安全截断。
        # gamma不宜太小，否则只关注局部速率。
        # ============================================================
        self.gamma = 0.995
        if abs(float(getattr(main_cfg, "reward_potential_gamma", self.gamma)) - self.gamma) > 1e-12:
            raise ValueError(
                "Potential-based reward shaping gamma must equal DRL gamma: "
                f"reward={main_cfg.reward_potential_gamma}, DRL={self.gamma}"
            )
        self.soft_tau = 0.005

        # ============================================================
        # 学习率
        # ============================================================
        # 初始学习率
        self.actor_lr = 1e-4
        self.critic_lr = 3e-4

        # 中间/最终学习率
        self.actor_lr_mid = 3e-5
        self.actor_lr_min = 1e-5

        self.critic_lr_mid = 1e-4
        self.critic_lr_min = 3e-5

        self.actor_weight_decay = 0.0
        self.critic_weight_decay = 0.0

        # ============================================================
        # CTD4 critic设置
        #
        # num_critics=3:
        #   比TD3的双critic更稳，但计算量仍然可接受。
        #
        # critic_fusion可选：
        #   "minimum"
        #   "mean" / "average"
        #   "median"
        #   "truncated_mean"
        #   "kalman"
        #
        # 注意：修改后的ctd4_agent中，所有UAV共享一个actor；
        # 每个UAV仍拥有自己的centralized critic ensemble。
        # 每个critic输出(mean, std)，按原论文使用 D_KL(Z_current || Z_target) 更新。
        # ============================================================
        self.num_critics = 3

        self.critic_fusion = "kalman"

        # ============================================================
        # CTD4 distributional critic / Kalman数值稳定参数
        #
        # 原论文的critic用Softplus保证sigma>0；作者代码的Kalman variance
        # 递推只加入1e-6量级的数值jitter。这里使用同量级下界，避免此前
        # 1e-3被直接加到variance中而人为把融合std抬到约sqrt(1e-3)=0.0316。
        # ============================================================
        self.critic_std_min = 1e-6
        self.target_std_min = 1e-6
        self.kalman_variance_eps = 1e-6

        # 下面两个参数仅为兼容旧版配置；当前Kalman融合直接使用critic输出的std。
        self.kalman_init_var = 1.0
        self.kalman_min_var = 1e-6

        # ============================================================
        # TD3风格目标策略平滑
        # ============================================================
        # 单位为弧度：target动作在角度域平滑，并随训练衰减。
        self.policy_noise = 0.10
        self.policy_noise_min = 0.02
        self.noise_clip = 0.20
        self.policy_freq = 2

        # ============================================================
        # 网络结构
        #
        # 小规模调试版使用更小网络，加快训练。
        # 如果后续恢复大规模问题，可以改回256或512。
        # ============================================================
        self.hidden_dim = 192
        self.actor_hidden_dim = 192
        self.critic_hidden_dim = 192

        self.use_layer_norm = True
        self.dropout_p = 0.0

        # 公平对比：三种算法使用完全相同的共享Actor结构。
        # scalar critic（DDPG/TD3）也使用与CTD4 critic相同的隐藏层宽度、
        # LayerNorm/dropout设置；输出头差异仅由算法定义决定。
        self.critic_init_w = 3e-4

        # ============================================================
        # Replay Buffer
        # ============================================================
        self.memory_capacity = 200000
        self.batch_size = 128

        # ============================================================
        # 训练参数与步数计划
        # ============================================================
        self.train_ep_n = int(main_cfg.train_ep_n)
        self.test_ep_n = int(main_cfg.test_ep_n)
        self.max_step = int(main_cfg.max_step)

        # 用收敛后每个episode的预计slot数估算训练调度，而不是直接使用
        # time-limit上限。当前预计约70 slot/episode。
        expected_steps = max(
            int(getattr(main_cfg, "expected_episode_slots", 70)),
            1,
        )
        expected_total_env_steps = max(
            self.train_ep_n * expected_steps,
            1,
        )

        # 前3000个环境步使用完全随机方向；之后改用Actor动作。
        self.random_steps = 10000

        # Replay Buffer至少积累2000个样本后开始网络更新。
        self.update_after = 10000
        self.update_every = 1
        self.update_times = 1

        # ============================================================
        # 行为探索噪声：按episode统一衰减
        #
        # 噪声加在方向角上，而不是直接加到二维动作分量上。
        # 为保证CTD4/DDPG/TD3在相同episode具有完全相同的探索强度，
        # 行为探索噪声不再依赖各算法实际经历的slot/environment step数量。
        # noise_decay_fraction表示占总训练episode数的比例。
        # 例如6000 EP、0.53 -> 在前3180 EP内由0.80 rad线性降到0.003 rad。
        # ============================================================
        self.use_exploration_noise = True
        self.exploration_noise_init = 0.80
        self.exploration_noise_min = 0.003
        self.noise_decay_fraction = 0.53
        self.exploration_noise_decay_episodes = max(
            int(self.noise_decay_fraction * self.train_ep_n),
            1,
        )

        # Target policy smoothing noise仍属于TD3/CTD4算法内部机制，
        # 继续按agent.train()调用次数衰减，不改成episode-based。
        # 当前update_every=1、update_times=1，因此其更新次数近似等于
        # update_after之后的环境步数。
        expected_update_steps = max(
            expected_total_env_steps - self.update_after,
            1,
        )
        self.policy_noise_decay_steps = max(
            int(self.noise_decay_fraction * expected_update_steps),
            1,
        )

        # ============================================================
        # Actor / Critic学习率衰减：按episode统一
        #
        # 三种DRL算法使用相同的episode-wise LR schedule：
        # 前25% episode保持初始学习率；55%处到达中间学习率；80%处到达
        # 最小学习率；最后20%用小学习率精调。
        # 这样算法当前slot性能不会反过来改变其LR衰减速度。
        # ============================================================
        self.use_lr_decay = True
        self.lr_decay_total_episodes = int(self.train_ep_n)
        self.lr_decay_hold_episodes = max(int(0.25 * self.train_ep_n), 1)
        self.lr_decay_mid_episodes = max(
            int(0.55 * self.train_ep_n),
            self.lr_decay_hold_episodes + 1,
        )
        self.lr_decay_end_episodes = max(
            int(0.80 * self.train_ep_n),
            self.lr_decay_mid_episodes + 1,
        )

        # 兼容旧代码读取字段；其单位现在也是episode，而非environment step。
        self.exploration_noise_decay_steps = int(self.exploration_noise_decay_episodes)
        self.noise_decay_steps = int(self.exploration_noise_decay_episodes)

        # 兼容旧字段名，但LR调度单位已明确改为episode。
        self.lr_decay_total_steps = int(self.lr_decay_total_episodes)
        self.lr_decay_hold_steps = int(self.lr_decay_hold_episodes)
        self.lr_decay_mid_steps = int(self.lr_decay_mid_episodes)
        self.lr_decay_end_steps = int(self.lr_decay_end_episodes)

        # ============================================================
        # reward clip
        #
        # 新reward包含自适应QoS乘子与非法动作惩罚。开启适度裁剪，避免少量
        # 极端样本扩大distributional critic方差，同时保留正常slot差异。
        # ============================================================
        self.use_reward_clip = True
        self.reward_clip_low = -20.0
        self.reward_clip_high = 5.0

        # ============================================================
        # 梯度裁剪
        # ============================================================
        self.use_grad_clip = True
        self.grad_clip_norm = 10.0

        # ============================================================
        # Actor imitation warm-start
        # ============================================================
        # 先用强规则teacher生成所有UAV的局部观测-动作样本，对共享Actor做监督预训练；
        # 然后再进入正常CTD4训练。这样可以显著提高初始策略质量。
        self.use_actor_imitation_pretrain = False

        self.imitation_teacher = "RemainingWorkloadGreedy"

        # 为了避免每次训练前过慢，默认规模设为中等。
        # 如果想更强的warm-start，可以改为 200-500 episodes 和 20-30 epochs。
        self.imitation_ep_n = 150
        self.imitation_max_step = int(main_cfg.max_step)
        self.imitation_train_epoch = 12
        self.imitation_batch_size = 512
        self.imitation_lr = 1e-4

        # loss = direction MSE + beta * KL(teacher_budget || actor_budget)
        self.imitation_budget_loss_weight = 0.5
        self.imitation_grad_clip_norm = 10.0

        # 采样过多会占内存。None表示不限制。
        self.imitation_max_samples = None

        # 预训练后是否保存一个warm-start actor，方便单独测试。
        self.save_imitation_actor = True

        # ============================================================
        # 测试
        # ============================================================
        self.test_add_noise = False

        # Benchmark是否运行由main_config.py中的独立开关控制。
        self.benchmark_switches = {
            "SafeNearestUnfinished": bool(getattr(
                main_cfg, "run_benchmark_safe_nearest_unfinished", True
            )),
            "WorkloadDistance": bool(getattr(
                main_cfg, "run_benchmark_workload_distance", True
            )),
            "MakespanAwareOneStepGreedy": bool(getattr(
                main_cfg, "run_benchmark_makespan_aware_one_step_greedy", True
            )),
        }
        self.benchmark_names = [
            name for name, enabled in self.benchmark_switches.items() if enabled
        ]
        self.run_benchmarks = len(self.benchmark_names) > 0

        # 固定无噪声验证：由main_config.py中的开关控制。
        # True时，每100个训练episode复用同一组30个独立场景；False时完全跳过。
        self.use_fixed_validation = bool(getattr(main_cfg, "use_fixed_validation", False))
        self.validation_interval = 100
        self.validation_ep_n = 30
        self.validation_ep_id_start = 1000000
        self.validation_plot_all_episodes = False
        self.validation_representative_n = 3
        self.validation_save_best = True
        # 训练结束后最终测试直接使用最后一个训练episode得到的actor；
        # 固定验证best仍照常保存，仅不再恢复为最终测试模型。
        self.validation_use_best_for_final_test = False

        # 最终测试和benchmark保存每个episode的轨迹。
        self.test_plot_all_trajectories = True
        self.benchmark_plot_all_trajectories = True

        # 轨迹图中每隔多少step标一个数字。
        self.trajectory_annotate_every = 20

        # ============================================================
        # 保存与日志
        # ============================================================
        self.save = True
        self.save_freq = 100
        self.print_interval = 10

        self.output_path = os.path.join(
            main_cfg.curr_path,
            "outputs",
            self.env_name,
            main_cfg.curr_time
        )

        self.result_path = os.path.join(self.output_path, "results")
        self.model_path = os.path.join(self.output_path, "models")
        self.log_path = os.path.join(self.output_path, "logs")

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        # ============================================================
        # 加载模型
        #
        # load_model_path可以是：
        #   1) 某个模型目录，例如 outputs/.../models/final
        #   2) 某个actor文件，例如 ctd4_actor.pt / ddpg_actor.pt / td3_actor.pt
        # ============================================================
        self.load_model = False
        self.load_model_path = None
        self.load_actor_path = None


# 兼容旧命名
DqnConfig = Ctd4Config
