"""
ZHANG Wenqi

Shared-actor multi-agent CTD4 for finite-data multi-UAV trajectory control.
Each UAV observes a local task-aware observation and outputs a 2-D heading
action.  Per-agent centralized distributional critic ensembles are retained,
including the original Kalman fusion implementation.
"""

from __future__ import annotations

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# 全局随机种子
# ============================================================
def set_torch_seed(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MLP模块
# ============================================================
def build_mlp(
    input_dim,
    hidden_dim,
    output_dim,
    output_activation=None,
    use_layer_norm=False,
    dropout_p=0.0,
):
    layers = []

    layers.append(nn.Linear(input_dim, hidden_dim))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.ReLU())
    if dropout_p > 0.0:
        layers.append(nn.Dropout(p=float(dropout_p)))

    layers.append(nn.Linear(hidden_dim, hidden_dim))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.ReLU())
    if dropout_p > 0.0:
        layers.append(nn.Dropout(p=float(dropout_p)))

    layers.append(nn.Linear(hidden_dim, output_dim))

    if output_activation == "tanh":
        layers.append(nn.Tanh())
    elif output_activation is None:
        pass
    else:
        raise ValueError(f"未知output_activation: {output_activation}")

    return nn.Sequential(*layers)


# ============================================================
# Actor：单个UAV局部观测 -> 单个UAV局部动作
# ============================================================
class Actor(nn.Module):
    def __init__(
        self,
        local_obs_dim,
        local_action_dim,
        hidden_dim,
        use_layer_norm=False,
        dropout_p=0.0,
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=int(local_obs_dim),
            hidden_dim=int(hidden_dim),
            output_dim=int(local_action_dim),
            output_activation="tanh",
            use_layer_norm=use_layer_norm,
            dropout_p=dropout_p,
        )

    def forward(self, obs):
        action = self.net(obs)
        # 环境只使用atan2(dy, dx)，因此将二维动作投影到单位圆，
        # 消除同一方向对应无穷多个动作模长造成的Critic伪梯度。
        if action.shape[-1] == 2:
            norm = torch.sqrt(torch.sum(action * action, dim=-1, keepdim=True) + 1e-8)
            action = action / norm
        return action


# 兼容旧命名：当前多UAV实现仍使用一个共享Actor处理各UAV局部观测。
SharedActor = Actor


# ============================================================
# Centralized Distributional Critic：全局状态 + 联合动作 -> (mean, std)
# ============================================================
class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        joint_action_dim,
        hidden_dim,
        use_layer_norm=False,
        dropout_p=0.0,
        init_w=3e-4,
        std_min=1e-6,
    ):
        super().__init__()

        self.std_min = float(std_min)

        input_dim = int(state_dim) + int(joint_action_dim)
        hidden_dim = int(hidden_dim)

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim) if use_layer_norm else None

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else None

        self.dropout = nn.Dropout(p=float(dropout_p)) if float(dropout_p) > 0.0 else None

        self.mean_head = nn.Linear(hidden_dim, 1)
        self.std_head = nn.Linear(hidden_dim, 1)

        self.mean_head.weight.data.uniform_(-init_w, init_w)
        self.mean_head.bias.data.uniform_(-init_w, init_w)
        self.std_head.weight.data.uniform_(-init_w, init_w)
        self.std_head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, joint_action):
        x = torch.cat([state, joint_action], dim=1)

        x = self.linear1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.linear2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)

        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + self.std_min

        return mean, std


# ============================================================
# Scalar Centralized Critic：DDPG/TD3使用
# 与CTD4 critic保持相同隐藏层深度/宽度/LayerNorm/dropout，
# 仅输出标量Q，这是算法定义本身的必要差异。
# ============================================================
class ScalarCritic(nn.Module):
    def __init__(
        self, state_dim, joint_action_dim, hidden_dim,
        use_layer_norm=False, dropout_p=0.0, init_w=3e-4,
    ):
        super().__init__()
        input_dim = int(state_dim) + int(joint_action_dim)
        hidden_dim = int(hidden_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.dropout = nn.Dropout(p=float(dropout_p)) if float(dropout_p) > 0.0 else None
        self.q_head = nn.Linear(hidden_dim, 1)
        self.q_head.weight.data.uniform_(-init_w, init_w)
        self.q_head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, joint_action):
        x = torch.cat([state, joint_action], dim=1)
        x = self.linear1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.q_head(x)


# ============================================================
# 多Critic ensemble：对应一个agent的多个distributional critics
# ============================================================
class EnsembleCritic(nn.Module):
    def __init__(
        self,
        num_critics,
        state_dim,
        joint_action_dim,
        hidden_dim,
        use_layer_norm=False,
        dropout_p=0.0,
        init_w=3e-4,
        std_min=1e-6,
    ):
        super().__init__()

        self.num_critics = int(num_critics)

        self.critics = nn.ModuleList(
            [
                Critic(
                    state_dim=state_dim,
                    joint_action_dim=joint_action_dim,
                    hidden_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                    dropout_p=dropout_p,
                    init_w=init_w,
                    std_min=std_min,
                )
                for _ in range(self.num_critics)
            ]
        )

    def forward(self, state, joint_action):
        means = []
        stds = []

        for critic in self.critics:
            mean, std = critic(state, joint_action)
            means.append(mean)
            stds.append(std)

        return means, stds


# ============================================================
# Replay Buffer：保存每个agent独立reward向量
# ============================================================
class ReplayBuffer:
    def __init__(
        self,
        state_dim,
        agent_n,
        local_obs_dim,
        action_dim,
        capacity,
    ):
        self.capacity = int(capacity)
        self.agent_n = int(agent_n)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.capacity, int(state_dim)), dtype=np.float32)
        self.obs = np.zeros((self.capacity, int(agent_n), int(local_obs_dim)), dtype=np.float32)
        self.action = np.zeros((self.capacity, int(action_dim)), dtype=np.float32)
        self.reward = np.zeros((self.capacity, int(agent_n)), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, int(state_dim)), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, int(agent_n), int(local_obs_dim)), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

    def push(self, state, obs, action, reward, next_state, next_obs, done):
        self.state[self.ptr] = np.asarray(state, dtype=np.float32).reshape(-1)
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.action[self.ptr] = np.asarray(action, dtype=np.float32).reshape(-1)

        reward_arr = np.asarray(reward, dtype=np.float32).reshape(-1)
        if reward_arr.size == 1:
            reward_arr = np.repeat(reward_arr, self.agent_n)
        if reward_arr.size != self.agent_n:
            raise ValueError(
                f"reward维度错误：期望 {self.agent_n} 或 1，实际 {reward_arr.size}。"
            )
        self.reward[self.ptr] = reward_arr.astype(np.float32)

        self.next_state[self.ptr] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.done[self.ptr, 0] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        batch_size = int(batch_size)

        if self.size < batch_size:
            raise ValueError(
                f"ReplayBuffer样本数量不足：当前{self.size}，需要{batch_size}。"
            )

        idx = np.random.randint(low=0, high=self.size, size=batch_size)

        return (
            self.state[idx],
            self.obs[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.next_obs[idx],
            self.done[idx],
        )

    def __len__(self):
        return int(self.size)


# ============================================================
# Independent-actor Multi-agent CTD4
# ============================================================
class CTD4:
    def __init__(self, cfg):
        self.cfg = cfg
        self.algorithm = "CTD4"
        self.algorithm_key = "ctd4"
        self.checkpoint_filename = "ctd4_checkpoint.pt"
        self.actor_filename = "ctd4_actor.pt"

        seed = int(getattr(cfg, "seed", 42))
        set_torch_seed(seed)

        self.device = torch.device(cfg.device)

        self.state_dim = int(cfg.state_dim)
        self.agent_n = int(cfg.agent_n)
        self.local_obs_dim = int(cfg.local_obs_dim)
        self.local_action_dim = int(cfg.local_action_dim)
        self.action_dim = int(cfg.action_dim)

        self.action_low = float(cfg.action_low)
        self.action_high = float(cfg.action_high)

        if self.action_dim != self.agent_n * self.local_action_dim:
            raise ValueError("当前实现要求 action_dim = agent_n * local_action_dim。")

        use_layer_norm = bool(getattr(cfg, "use_layer_norm", False))
        dropout_p = float(getattr(cfg, "dropout_p", 0.0))
        critic_init_w = float(getattr(cfg, "critic_init_w", 3e-4))
        critic_std_min = float(getattr(cfg, "critic_std_min", 1e-6))

        # ========================================================
        # 所有UAV共享一个Actor
        #
        # 同一个网络分别处理每架UAV的局部观测。由于局部观测中包含
        # 本机高度、位置、剩余时域、相对用户位置和相对其他UAV位置，
        # 参数共享不会要求三架UAV输出相同动作。
        # ========================================================
        self.actor = Actor(
            local_obs_dim=self.local_obs_dim,
            local_action_dim=self.local_action_dim,
            hidden_dim=cfg.actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            dropout_p=dropout_p,
        ).to(self.device)

        self.target_actor = Actor(
            local_obs_dim=self.local_obs_dim,
            local_action_dim=self.local_action_dim,
            hidden_dim=cfg.actor_hidden_dim,
            use_layer_norm=use_layer_norm,
            dropout_p=dropout_p,
        ).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())

        # ========================================================
        # 每个agent一个centralized distributional critic ensemble
        # ========================================================
        self.critics = nn.ModuleList(
            [
                EnsembleCritic(
                    num_critics=cfg.num_critics,
                    state_dim=self.state_dim,
                    joint_action_dim=self.action_dim,
                    hidden_dim=cfg.critic_hidden_dim,
                    use_layer_norm=use_layer_norm,
                    dropout_p=dropout_p,
                    init_w=critic_init_w,
                    std_min=critic_std_min,
                ).to(self.device)
                for _ in range(self.agent_n)
            ]
        )

        self.target_critics = nn.ModuleList(
            [
                EnsembleCritic(
                    num_critics=cfg.num_critics,
                    state_dim=self.state_dim,
                    joint_action_dim=self.action_dim,
                    hidden_dim=cfg.critic_hidden_dim,
                    use_layer_norm=use_layer_norm,
                    dropout_p=dropout_p,
                    init_w=critic_init_w,
                    std_min=critic_std_min,
                ).to(self.device)
                for _ in range(self.agent_n)
            ]
        )

        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

        # 兼容旧代码中agent.critic / agent.target_critic的访问。
        self.critic = self.critics[0]
        self.target_critic = self.target_critics[0]

        # ========================================================
        # Optimizer
        # ========================================================
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=float(cfg.actor_lr),
            weight_decay=float(getattr(cfg, "actor_weight_decay", 0.0)),
        )

        self.critic_optimizers = []
        for ensemble in self.critics:
            opts_m = [
                optim.Adam(
                    critic.parameters(),
                    lr=float(cfg.critic_lr),
                    weight_decay=float(getattr(cfg, "critic_weight_decay", 0.0)),
                )
                for critic in ensemble.critics
            ]
            self.critic_optimizers.append(opts_m)

        # ========================================================
        # Replay Buffer
        # ========================================================
        self.memory = ReplayBuffer(
            state_dim=self.state_dim,
            agent_n=self.agent_n,
            local_obs_dim=self.local_obs_dim,
            action_dim=self.action_dim,
            capacity=cfg.memory_capacity,
        )

        # ========================================================
        # 超参数
        # ========================================================
        self.gamma = float(cfg.gamma)
        self.soft_tau = float(cfg.soft_tau)

        self.batch_size = int(cfg.batch_size)

        self.policy_noise = float(cfg.policy_noise)
        self.policy_noise_min = float(getattr(cfg, "policy_noise_min", self.policy_noise))
        self.policy_noise_decay_steps = max(
            int(getattr(cfg, "policy_noise_decay_steps", 1)),
            1,
        )
        self.noise_clip = float(cfg.noise_clip)
        self.policy_freq = int(cfg.policy_freq)

        self.num_critics = int(cfg.num_critics)
        self.critic_fusion = str(cfg.critic_fusion)

        self.use_reward_clip = bool(getattr(cfg, "use_reward_clip", False))
        self.reward_clip_low = float(getattr(cfg, "reward_clip_low", -10.0))
        self.reward_clip_high = float(getattr(cfg, "reward_clip_high", 10.0))

        self.use_grad_clip = bool(getattr(cfg, "use_grad_clip", True))
        self.grad_clip_norm = float(getattr(cfg, "grad_clip_norm", 10.0))

        self.critic_std_min = critic_std_min
        self.target_std_min = float(getattr(cfg, "target_std_min", critic_std_min))
        # 原论文/作者代码的Kalman插值仅加入很小的variance jitter（1e-6量级），
        # 不能把critic的std下界直接加到variance上，否则会人为放大融合不确定度。
        self.kalman_variance_eps = float(getattr(cfg, "kalman_variance_eps", 1e-6))

        self.total_it = 0

        # ========================================================
        # Learning-rate decay
        # ========================================================
        self.use_lr_decay = bool(getattr(cfg, "use_lr_decay", False))
        self.actor_lr_init = float(cfg.actor_lr)
        self.actor_lr_mid = float(getattr(cfg, "actor_lr_mid", self.actor_lr_init))
        self.actor_lr_min = float(getattr(cfg, "actor_lr_min", self.actor_lr_mid))
        self.critic_lr_init = float(cfg.critic_lr)
        self.critic_lr_mid = float(getattr(cfg, "critic_lr_mid", self.critic_lr_init))
        self.critic_lr_min = float(getattr(cfg, "critic_lr_min", self.critic_lr_mid))
        self.lr_decay_hold_episodes = max(int(getattr(cfg, "lr_decay_hold_episodes", 1)), 1)
        self.lr_decay_mid_episodes = max(
            int(getattr(cfg, "lr_decay_mid_episodes", self.lr_decay_hold_episodes + 1)),
            self.lr_decay_hold_episodes + 1,
        )
        self.lr_decay_end_episodes = max(
            int(getattr(cfg, "lr_decay_end_episodes", self.lr_decay_mid_episodes + 1)),
            self.lr_decay_mid_episodes + 1,
        )
        self.current_episode = 0
        self.current_actor_lr = self.actor_lr_init
        self.current_critic_lr = self.critic_lr_init
        self._apply_learning_rate_schedule(episode=0)

    # ========================================================
    # 选择联合动作
    # ========================================================
    def choose_action(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32)

        if obs.shape != (self.agent_n, self.local_obs_dim):
            raise ValueError(
                f"obs维度错误：期望 {(self.agent_n, self.local_obs_dim)}，实际 {obs.shape}"
            )

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            # [M,D] -> [M,A]；每架UAV一组局部动作，但共享同一套参数。
            local_actions = self.actor(obs_t).cpu().numpy()
        if was_training:
            self.actor.train()

        action = local_actions.reshape(-1)
        action = np.clip(action, self.action_low, self.action_high)
        return action.astype(np.float32)

    def select_action(self, obs):
        return self.choose_action(obs, deterministic=True)

    # ========================================================
    # 存储经验
    # ========================================================
    def store_transition(self, state, obs, action, reward, next_state, next_obs, done):
        self.memory.push(
            state=state,
            obs=obs,
            action=action,
            reward=reward,
            next_state=next_state,
            next_obs=next_obs,
            done=done,
        )

    # ========================================================
    # 共享Actor批量前向：obs [B,M,D] -> joint_action [B,M*A]
    #
    # 将B个联合样本中的M组局部观测展平为B*M组Actor样本，
    # 因而一个联合transition可向共享Actor提供M份局部训练信号。
    # ========================================================
    def _actors_forward_batch(self, obs, use_target=False):
        actor = self.target_actor if use_target else self.actor
        batch_size = int(obs.shape[0])
        flat_obs = obs.reshape(batch_size * self.agent_n, self.local_obs_dim)
        flat_action = actor(flat_obs)
        local_action = flat_action.reshape(
            batch_size, self.agent_n, self.local_action_dim
        )
        return local_action.reshape(batch_size, self.action_dim)

    def _current_policy_noise(self):
        frac = min(float(self.total_it) / float(self.policy_noise_decay_steps), 1.0)
        return float(
            self.policy_noise
            - frac * (self.policy_noise - self.policy_noise_min)
        )

    @staticmethod
    def _linear_interpolate(start_value, end_value, step, start_step, end_step):
        if int(end_step) <= int(start_step):
            return float(end_value)
        frac = (float(step) - float(start_step)) / float(end_step - start_step)
        frac = min(max(frac, 0.0), 1.0)
        return float(start_value + frac * (end_value - start_value))

    def _current_learning_rates(self, episode=None):
        """Return the common episode-wise actor/critic learning rates."""
        episode = int(self.current_episode if episode is None else episode)
        if not self.use_lr_decay:
            return float(self.actor_lr_init), float(self.critic_lr_init)

        if episode <= self.lr_decay_hold_episodes:
            actor_lr = self.actor_lr_init
            critic_lr = self.critic_lr_init
        elif episode <= self.lr_decay_mid_episodes:
            actor_lr = self._linear_interpolate(
                self.actor_lr_init,
                self.actor_lr_mid,
                episode,
                self.lr_decay_hold_episodes,
                self.lr_decay_mid_episodes,
            )
            critic_lr = self._linear_interpolate(
                self.critic_lr_init,
                self.critic_lr_mid,
                episode,
                self.lr_decay_hold_episodes,
                self.lr_decay_mid_episodes,
            )
        elif episode <= self.lr_decay_end_episodes:
            actor_lr = self._linear_interpolate(
                self.actor_lr_mid,
                self.actor_lr_min,
                episode,
                self.lr_decay_mid_episodes,
                self.lr_decay_end_episodes,
            )
            critic_lr = self._linear_interpolate(
                self.critic_lr_mid,
                self.critic_lr_min,
                episode,
                self.lr_decay_mid_episodes,
                self.lr_decay_end_episodes,
            )
        else:
            actor_lr = self.actor_lr_min
            critic_lr = self.critic_lr_min

        return float(actor_lr), float(critic_lr)

    def _apply_learning_rate_schedule(self, episode=None):
        actor_lr, critic_lr = self._current_learning_rates(episode=episode)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_lr
        for opts_m in self.critic_optimizers:
            for optimizer in opts_m:
                for group in optimizer.param_groups:
                    group["lr"] = critic_lr
        self.current_actor_lr = float(actor_lr)
        self.current_critic_lr = float(critic_lr)
        return self.current_actor_lr, self.current_critic_lr

    def set_training_episode(self, episode):
        """Set 1-based training episode and apply the common episode-wise LR schedule."""
        self.current_episode = max(int(episode), 0)
        return self._apply_learning_rate_schedule(episode=self.current_episode)

    def _normalize_joint_action_tensor(self, joint_action):
        if self.local_action_dim != 2:
            return torch.clamp(joint_action, self.action_low, self.action_high)
        action_mat = joint_action.reshape(-1, self.agent_n, 2)
        norm = torch.sqrt(torch.sum(action_mat * action_mat, dim=-1, keepdim=True) + 1e-8)
        return (action_mat / norm).reshape(-1, self.action_dim)

    def _add_heading_noise_tensor(self, joint_action, sigma, clip_value):
        """在角度域执行TD3 target-policy smoothing。"""
        if self.local_action_dim != 2:
            noise = torch.randn_like(joint_action) * float(sigma)
            noise = torch.clamp(noise, -float(clip_value), float(clip_value))
            return torch.clamp(
                joint_action + noise,
                self.action_low,
                self.action_high,
            )

        action_mat = joint_action.reshape(-1, self.agent_n, 2)
        theta = torch.atan2(action_mat[..., 1], action_mat[..., 0])
        angle_noise = torch.randn_like(theta) * float(sigma)
        angle_noise = torch.clamp(
            angle_noise,
            -float(clip_value),
            float(clip_value),
        )
        theta = theta + angle_noise
        noisy = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        return noisy.reshape(-1, self.action_dim)

    @staticmethod
    def _ctd4_kl(current_dist, target_dist):
        """CTD4 critic loss from the paper/code: D_KL(Z_current || Z_target)."""
        return torch.distributions.kl.kl_divergence(current_dist, target_dist)

    # ========================================================
    # 训练一步
    # ========================================================
    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        self.total_it += 1
        current_actor_lr = float(self.current_actor_lr)
        current_critic_lr = float(self.current_critic_lr)

        state, obs, action, reward, next_state, next_obs, done = self.memory.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        # Replay中的done只表示“全部用户数据传完”的真实终止；time-limit截断写入done=0。
        # 按CTD4原论文/作者代码：done只关闭Bellman均值的bootstrap；
        # target分布的std仍由 gamma * fused_std 给出，并对所有样本使用distributional KL。
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        not_terminal = 1.0 - done

        if self.use_reward_clip:
            reward = torch.clamp(reward, self.reward_clip_low, self.reward_clip_high)

        batch_size = state.shape[0]

        critic_loss_by_agent = []
        target_q_mean_by_agent = []
        target_q_std_by_agent = []

        # ====================================================
        # 1) 计算target joint action
        # ====================================================
        with torch.no_grad():
            next_action = self._actors_forward_batch(next_obs, use_target=True)
            current_policy_noise = self._current_policy_noise()
            next_action = self._add_heading_noise_tensor(
                next_action,
                sigma=current_policy_noise,
                clip_value=self.noise_clip,
            )

        # ====================================================
        # 2) 每个agent分别更新自己的centralized critics
        # ====================================================
        for m in range(self.agent_n):
            with torch.no_grad():
                target_means, target_stds = self.target_critics[m](next_state, next_action)
                fusion_mean, fusion_std = self._fuse_critic_outputs(
                    target_means,
                    target_stds,
                    batch_size=batch_size,
                )

                reward_m = reward[:, m:m + 1]

                # CTD4 distributional Bellman target（与原论文/作者代码一致）：
                #   mu_target    = r + gamma * (1-done) * mu_fused
                #   sigma_target = gamma * sigma_fused
                # 注意：done只作用于均值bootstrap，不把sigma_target压成0。
                target_mean = reward_m + self.gamma * not_terminal * fusion_mean
                target_std = torch.clamp(
                    self.gamma * fusion_std,
                    min=self.target_std_min,
                )
                target_dist = torch.distributions.Normal(target_mean, target_std)

            loss_list_m = []
            for critic, optimizer in zip(self.critics[m].critics, self.critic_optimizers[m]):
                current_mean, current_std = critic(state, action)
                current_std = torch.clamp(current_std, min=self.critic_std_min)
                current_dist = torch.distributions.Normal(current_mean, current_std)

                # 原论文Eq.(8)-(10)与作者代码均使用：
                #     D_KL(Z_current || Z_target)
                # 而不是反向的 D_KL(Z_target || Z_current)。
                critic_kl = self._ctd4_kl(
                    current_dist,
                    target_dist,
                )
                if critic_kl.dim() == 1:
                    critic_kl = critic_kl.unsqueeze(1)
                loss_i = critic_kl.mean()

                optimizer.zero_grad()
                loss_i.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip_norm)
                optimizer.step()

                loss_list_m.append(float(loss_i.item()))

            critic_loss_by_agent.append(float(np.mean(loss_list_m)))
            target_q_mean_by_agent.append(float(target_mean.mean().item()))
            # CTD4原始distributional target对terminal样本同样保留sigma_target。
            target_q_std_by_agent.append(float(target_std.mean().item()))

        critic_loss_value = float(np.mean(critic_loss_by_agent))

        actor_loss_by_agent = [np.nan for _ in range(self.agent_n)]
        actor_q_by_agent = [np.nan for _ in range(self.agent_n)]

        # ====================================================
        # 3) 延迟更新共享Actor和目标网络
        #
        # 对一个batch中的每条联合transition：
        #   - 共享Actor产生M组局部动作；
        #   - 第m个critic只给第m组动作提供策略梯度；
        #   - 其他UAV动作在该项loss中detach；
        #   - M个loss求平均后只执行一次backward/optimizer.step。
        # 这样既获得M份局部样本，又避免同一次更新中连续修改共享参数。
        # ====================================================
        if self.total_it % self.policy_freq == 0:
            flat_obs = obs.reshape(
                batch_size * self.agent_n, self.local_obs_dim
            )
            pred_local_actions = self.actor(flat_obs).reshape(
                batch_size, self.agent_n, self.local_action_dim
            )

            # Actor更新只需要critic对动作的梯度，不更新critic参数。
            for ensemble in self.critics:
                for param in ensemble.parameters():
                    param.requires_grad_(False)

            actor_loss_terms = []
            for m in range(self.agent_n):
                joint_parts = []
                for j in range(self.agent_n):
                    action_j = pred_local_actions[:, j, :]
                    joint_parts.append(action_j if j == m else action_j.detach())

                pred_joint_action_m = self._normalize_joint_action_tensor(
                    torch.cat(joint_parts, dim=1)
                )

                means_for_actor, stds_for_actor = self.critics[m](
                    state, pred_joint_action_m
                )
                q_for_actor, _ = self._fuse_critic_outputs(
                    means_for_actor,
                    stds_for_actor,
                    batch_size=batch_size,
                )

                loss_m = -q_for_actor.mean()
                actor_loss_terms.append(loss_m)
                actor_loss_by_agent[m] = float(loss_m.detach().item())
                actor_q_by_agent[m] = float(q_for_actor.detach().mean().item())

            shared_actor_loss = torch.stack(actor_loss_terms).mean()

            self.actor_optimizer.zero_grad()
            shared_actor_loss.backward()
            if self.use_grad_clip:
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.grad_clip_norm
                )
            self.actor_optimizer.step()

            for ensemble in self.critics:
                for param in ensemble.parameters():
                    param.requires_grad_(True)

            self._soft_update(self.actor, self.target_actor)
            for m in range(self.agent_n):
                self._soft_update(self.critics[m], self.target_critics[m])

        actor_loss_value = float(np.nanmean(actor_loss_by_agent)) if np.any(np.isfinite(actor_loss_by_agent)) else None
        actor_q_value = float(np.nanmean(actor_q_by_agent)) if np.any(np.isfinite(actor_q_by_agent)) else None

        return {
            "critic_loss": critic_loss_value,
            "critic_loss_by_agent": np.asarray(critic_loss_by_agent, dtype=np.float32),
            "actor_loss": actor_loss_value,
            "actor_loss_by_agent": np.asarray(actor_loss_by_agent, dtype=np.float32),
            "actor_q": actor_q_value,
            "actor_q_by_agent": np.asarray(actor_q_by_agent, dtype=np.float32),
            "target_q_mean": float(np.mean(target_q_mean_by_agent)),
            "target_q_mean_by_agent": np.asarray(target_q_mean_by_agent, dtype=np.float32),
            "target_q_std_mean": float(np.mean(target_q_std_by_agent)),
            "target_q_std_by_agent": np.asarray(target_q_std_by_agent, dtype=np.float32),
            "q_backup_mean": float(np.mean(target_q_mean_by_agent)),
            "reward_mean": float(reward.mean().item()),
            "reward_mean_by_agent": reward.mean(dim=0).detach().cpu().numpy().astype(np.float32),
            # 兼容旧日志字段；这里表示batch中的固定时域terminal比例。
            "done_mean": float(done.mean().item()),
            "terminal_mean": float(done.mean().item()),
            "total_it": int(self.total_it),
            "buffer_size": int(len(self.memory)),
            "policy_noise": float(self._current_policy_noise()),
            "actor_lr": float(current_actor_lr),
            "critic_lr": float(current_critic_lr),
        }

    # ========================================================
    # 多critic分布融合
    # ========================================================
    def _fuse_critic_outputs(self, means, stds, batch_size=None):
        method = str(self.critic_fusion).lower()
        if method == "mean":
            method = "average"

        mean_stack = torch.stack(means, dim=0)  # [N, B, 1]
        std_stack = torch.stack(stds, dim=0)    # [N, B, 1]
        std_stack = torch.clamp(std_stack, min=self.critic_std_min)

        if batch_size is None:
            batch_size = int(mean_stack.shape[1])

        if method == "average":
            fusion_mean = torch.mean(mean_stack, dim=0)
            fusion_std = torch.mean(std_stack, dim=0)
            return fusion_mean, fusion_std

        if method == "minimum":
            fusion_mean, idx = torch.min(mean_stack, dim=0)
            idx = idx.squeeze(1)
            b_idx = torch.arange(batch_size, device=fusion_mean.device)
            fusion_std = std_stack[:, :, 0].transpose(0, 1)[b_idx, idx].unsqueeze(1)
            return fusion_mean, fusion_std

        if method == "median":
            fusion_mean, idx = torch.median(mean_stack, dim=0)
            idx = idx.squeeze(1)
            b_idx = torch.arange(batch_size, device=fusion_mean.device)
            fusion_std = std_stack[:, :, 0].transpose(0, 1)[b_idx, idx].unsqueeze(1)
            return fusion_mean, fusion_std

        if method == "truncated_mean":
            if mean_stack.shape[0] <= 2:
                fusion_mean, idx = torch.min(mean_stack, dim=0)
                idx = idx.squeeze(1)
                b_idx = torch.arange(batch_size, device=fusion_mean.device)
                fusion_std = std_stack[:, :, 0].transpose(0, 1)[b_idx, idx].unsqueeze(1)
                return fusion_mean, fusion_std

            sorted_mean, sorted_idx = torch.sort(mean_stack, dim=0)
            kept_mean = sorted_mean[:-1]
            kept_idx = sorted_idx[:-1, :, 0]
            std_by_critic = std_stack[:, :, 0]
            gathered_std = torch.gather(std_by_critic, dim=0, index=kept_idx).unsqueeze(-1)
            fusion_mean = torch.mean(kept_mean, dim=0)
            fusion_std = torch.mean(gathered_std, dim=0)
            return fusion_mean, fusion_std

        if method == "kalman":
            # 原论文Eq.(1)-(3) / 作者代码的 sequential interpolated Kalman fusion：
            # K = var_fused / (var_fused + var_next),
            # mu <- mu + K(mu_next-mu),
            # var <- (1-K)var + K var_next (+ tiny numerical jitter).
            fusion_mean = means[0]
            fusion_std = torch.clamp(stds[0], min=self.critic_std_min)
            for i in range(1, len(means)):
                next_mean = means[i]
                next_std = torch.clamp(stds[i], min=self.critic_std_min)
                k_gain = fusion_std.pow(2) / (fusion_std.pow(2) + next_std.pow(2) + 1e-12)
                fusion_mean = fusion_mean + k_gain * (next_mean - fusion_mean)
                fusion_std = torch.sqrt(
                    (1.0 - k_gain) * fusion_std.pow(2)
                    + k_gain * next_std.pow(2)
                    + self.kalman_variance_eps
                )
            fusion_std = torch.clamp(fusion_std, min=self.critic_std_min)
            return fusion_mean, fusion_std

        raise ValueError(f"未知critic_fusion: {self.critic_fusion}")

    # ========================================================
    # 兼容旧版：只融合mean，返回标量Q估计
    # ========================================================
    def _fuse_q(self, q_stack):
        fake_means = [q_stack[i] for i in range(q_stack.shape[0])]
        fake_stds = [torch.ones_like(q_stack[i]) for i in range(q_stack.shape[0])]
        fusion_mean, _ = self._fuse_critic_outputs(
            fake_means,
            fake_stds,
            batch_size=int(q_stack.shape[1]),
        )
        return fusion_mean

    # ========================================================
    # 软更新
    # ========================================================
    def _soft_update(self, source_net, target_net):
        for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.soft_tau * source_param.data + (1.0 - self.soft_tau) * target_param.data
            )

    # ========================================================
    # 保存完整模型
    # ========================================================
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, "ctd4_checkpoint.pt")

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "critics": [critic.state_dict() for critic in self.critics],
                "target_critics": [critic.state_dict() for critic in self.target_critics],
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizers": [
                    [optimizer.state_dict() for optimizer in opts_m]
                    for opts_m in self.critic_optimizers
                ],
                "total_it": int(self.total_it),
                "current_episode": int(self.current_episode),
                "current_actor_lr": float(self.current_actor_lr),
                "current_critic_lr": float(self.current_critic_lr),
                "state_dim": int(self.state_dim),
                "agent_n": int(self.agent_n),
                "local_obs_dim": int(self.local_obs_dim),
                "local_action_dim": int(self.local_action_dim),
                "action_dim": int(self.action_dim),
                "critic_fusion": str(self.critic_fusion),
                "actor_mode": "shared",
                "critic_mode": "per_agent_centralized",
            },
            save_path,
        )
        return save_path

    @staticmethod
    def _average_actor_state_dicts(state_dict_list):
        """将旧版多个独立Actor参数取平均，作为共享Actor初始化。"""
        if len(state_dict_list) == 0:
            raise ValueError("旧checkpoint中的actors为空。")

        averaged = {}
        first = state_dict_list[0]
        for key, value in first.items():
            values = [sd[key] for sd in state_dict_list]
            if torch.is_floating_point(value):
                averaged[key] = torch.stack(values, dim=0).mean(dim=0)
            else:
                averaged[key] = value.clone()
        return averaged

    def _load_actor_state_checked(self, state_dict, source_name):
        try:
            self.actor.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"无法加载{source_name}的Actor参数。当前共享Actor局部观测维度为"
                f"{self.local_obs_dim}，并新增了UAV高度特征；旧维度模型通常不能直接复用，"
                f"请从头训练当前版本。原始错误：{exc}"
            ) from exc

    # ========================================================
    # 加载完整模型
    # ========================================================
    def load(self, path):
        load_path = os.path.join(path, "ctd4_checkpoint.pt")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"找不到模型文件：{load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        if "actor" in checkpoint:
            self._load_actor_state_checked(checkpoint["actor"], "共享Actor checkpoint")
            self.target_actor.load_state_dict(
                checkpoint.get("target_actor", checkpoint["actor"])
            )
        elif "actors" in checkpoint:
            averaged_actor = self._average_actor_state_dicts(checkpoint["actors"])
            self._load_actor_state_checked(averaged_actor, "旧版独立Actor checkpoint")
            self.target_actor.load_state_dict(self.actor.state_dict())
        else:
            raise KeyError("checkpoint中未找到actor或actors参数。")

        if "critics" in checkpoint:
            try:
                for critic, state_dict in zip(self.critics, checkpoint["critics"]):
                    critic.load_state_dict(state_dict)
                for target_critic, state_dict in zip(
                    self.target_critics,
                    checkpoint.get("target_critics", checkpoint["critics"]),
                ):
                    target_critic.load_state_dict(state_dict)
            except RuntimeError as exc:
                raise RuntimeError(
                    "无法加载旧Critic参数。当前全局状态新增了UAV高度，state_dim已变化，"
                    f"请从头训练当前版本。原始错误：{exc}"
                ) from exc

        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizers" in checkpoint:
            for opts_m, opt_states_m in zip(
                self.critic_optimizers, checkpoint["critic_optimizers"]
            ):
                for optimizer, opt_state in zip(opts_m, opt_states_m):
                    optimizer.load_state_dict(opt_state)

        self.total_it = int(checkpoint.get("total_it", 0))
        self.current_episode = int(checkpoint.get("current_episode", 0))
        # Optimizer checkpoint may contain an old param-group LR. Recompute it
        # from the saved episode so resume training follows the episode-wise plan.
        self._apply_learning_rate_schedule(episode=self.current_episode)

    # ========================================================
    # 只保存Actor
    # ========================================================
    def save_actor(self, path):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, "ctd4_actor.pt")

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "state_dim": int(self.state_dim),
                "agent_n": int(self.agent_n),
                "local_obs_dim": int(self.local_obs_dim),
                "local_action_dim": int(self.local_action_dim),
                "action_dim": int(self.action_dim),
                "actor_mode": "shared",
            },
            save_path,
        )
        return save_path

    # ========================================================
    # 只加载Actor
    # ========================================================
    def load_actor(self, actor_path):
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"找不到Actor文件：{actor_path}")

        checkpoint = torch.load(actor_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "actor" in checkpoint:
            state_dict = checkpoint["actor"]
            source_name = "共享Actor文件"
        elif isinstance(checkpoint, dict) and "actors" in checkpoint:
            state_dict = self._average_actor_state_dicts(checkpoint["actors"])
            source_name = "旧版独立Actor文件"
        else:
            state_dict = checkpoint
            source_name = "裸Actor state_dict"

        self._load_actor_state_checked(state_dict, source_name)
        self.target_actor.load_state_dict(self.actor.state_dict())



# ============================================================
# DDPG / TD3：与当前CTD4共享完全相同的multi-agent外壳
# ============================================================
class _ScalarMAAgentBase(CTD4):
    """Common shared-actor / per-agent centralized-critic shell.

    This class intentionally reuses CTD4's actor, replay-buffer, action
    normalization, LR schedule and Polyak update helpers.  DDPG/TD3 differ only
    in the critic/target/policy-update mechanisms required by the algorithms.
    """
    algorithm = None
    critic_n = None

    def __init__(self, cfg):
        self.cfg = cfg
        if self.algorithm is None or self.critic_n is None:
            raise RuntimeError("_ScalarMAAgentBase must be subclassed.")
        self.algorithm_key = self.algorithm.lower()
        self.checkpoint_filename = f"{self.algorithm_key}_checkpoint.pt"
        self.actor_filename = f"{self.algorithm_key}_actor.pt"

        seed = int(getattr(cfg, "seed", 42))
        set_torch_seed(seed)
        self.device = torch.device(cfg.device)
        self.state_dim = int(cfg.state_dim)
        self.agent_n = int(cfg.agent_n)
        self.local_obs_dim = int(cfg.local_obs_dim)
        self.local_action_dim = int(cfg.local_action_dim)
        self.action_dim = int(cfg.action_dim)
        self.action_low = float(cfg.action_low)
        self.action_high = float(cfg.action_high)
        if self.action_dim != self.agent_n * self.local_action_dim:
            raise ValueError("当前实现要求 action_dim = agent_n * local_action_dim。")

        use_layer_norm = bool(getattr(cfg, "use_layer_norm", False))
        dropout_p = float(getattr(cfg, "dropout_p", 0.0))
        critic_init_w = float(getattr(cfg, "critic_init_w", 3e-4))

        # Exact same actor as CTD4 for fair comparison.
        self.actor = Actor(
            self.local_obs_dim, self.local_action_dim, cfg.actor_hidden_dim,
            use_layer_norm=use_layer_norm, dropout_p=dropout_p,
        ).to(self.device)
        self.target_actor = Actor(
            self.local_obs_dim, self.local_action_dim, cfg.actor_hidden_dim,
            use_layer_norm=use_layer_norm, dropout_p=dropout_p,
        ).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Per-agent centralized scalar critic(s): one for DDPG, two for TD3.
        self.critics = nn.ModuleList()
        self.target_critics = nn.ModuleList()
        self.critic_optimizers = []
        for _ in range(self.agent_n):
            online = nn.ModuleList([
                ScalarCritic(
                    self.state_dim, self.action_dim, cfg.critic_hidden_dim,
                    use_layer_norm=use_layer_norm, dropout_p=dropout_p,
                    init_w=critic_init_w,
                ).to(self.device)
                for _ in range(int(self.critic_n))
            ])
            target = nn.ModuleList([
                ScalarCritic(
                    self.state_dim, self.action_dim, cfg.critic_hidden_dim,
                    use_layer_norm=use_layer_norm, dropout_p=dropout_p,
                    init_w=critic_init_w,
                ).to(self.device)
                for _ in range(int(self.critic_n))
            ])
            target.load_state_dict(online.state_dict())
            self.critics.append(online)
            self.target_critics.append(target)
            self.critic_optimizers.append([
                optim.Adam(
                    critic.parameters(), lr=float(cfg.critic_lr),
                    weight_decay=float(getattr(cfg, "critic_weight_decay", 0.0)),
                ) for critic in online
            ])

        self.critic = self.critics[0][0]
        self.target_critic = self.target_critics[0][0]
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=float(cfg.actor_lr),
            weight_decay=float(getattr(cfg, "actor_weight_decay", 0.0)),
        )
        self.memory = ReplayBuffer(
            self.state_dim, self.agent_n, self.local_obs_dim,
            self.action_dim, cfg.memory_capacity,
        )

        self.gamma = float(cfg.gamma)
        self.soft_tau = float(cfg.soft_tau)
        self.batch_size = int(cfg.batch_size)
        self.policy_noise = float(cfg.policy_noise)
        self.policy_noise_min = float(getattr(cfg, "policy_noise_min", self.policy_noise))
        self.policy_noise_decay_steps = max(int(getattr(cfg, "policy_noise_decay_steps", 1)), 1)
        self.noise_clip = float(cfg.noise_clip)
        self.policy_freq = int(cfg.policy_freq)
        self.use_reward_clip = bool(getattr(cfg, "use_reward_clip", False))
        self.reward_clip_low = float(getattr(cfg, "reward_clip_low", -10.0))
        self.reward_clip_high = float(getattr(cfg, "reward_clip_high", 10.0))
        self.use_grad_clip = bool(getattr(cfg, "use_grad_clip", True))
        self.grad_clip_norm = float(getattr(cfg, "grad_clip_norm", 10.0))
        self.total_it = 0

        # Same LR schedule as CTD4.
        self.use_lr_decay = bool(getattr(cfg, "use_lr_decay", False))
        self.actor_lr_init = float(cfg.actor_lr)
        self.actor_lr_mid = float(getattr(cfg, "actor_lr_mid", self.actor_lr_init))
        self.actor_lr_min = float(getattr(cfg, "actor_lr_min", self.actor_lr_mid))
        self.critic_lr_init = float(cfg.critic_lr)
        self.critic_lr_mid = float(getattr(cfg, "critic_lr_mid", self.critic_lr_init))
        self.critic_lr_min = float(getattr(cfg, "critic_lr_min", self.critic_lr_mid))
        self.lr_decay_hold_episodes = max(int(getattr(cfg, "lr_decay_hold_episodes", 1)), 1)
        self.lr_decay_mid_episodes = max(
            int(getattr(cfg, "lr_decay_mid_episodes", self.lr_decay_hold_episodes + 1)),
            self.lr_decay_hold_episodes + 1,
        )
        self.lr_decay_end_episodes = max(
            int(getattr(cfg, "lr_decay_end_episodes", self.lr_decay_mid_episodes + 1)),
            self.lr_decay_mid_episodes + 1,
        )
        self.current_episode = 0
        self.current_actor_lr = self.actor_lr_init
        self.current_critic_lr = self.critic_lr_init
        self._apply_learning_rate_schedule(episode=0)

    def _freeze_critics(self, freeze=True):
        for group in self.critics:
            for critic in group:
                for param in critic.parameters():
                    param.requires_grad_(not freeze)

    def _actor_update(self, state, obs, critic_selector=0):
        batch_size = int(state.shape[0])
        pred_local = self.actor(
            obs.reshape(batch_size * self.agent_n, self.local_obs_dim)
        ).reshape(batch_size, self.agent_n, self.local_action_dim)
        self._freeze_critics(True)
        losses, qs = [], []
        for m in range(self.agent_n):
            parts = [
                pred_local[:, j, :] if j == m else pred_local[:, j, :].detach()
                for j in range(self.agent_n)
            ]
            joint = self._normalize_joint_action_tensor(torch.cat(parts, dim=1))
            q = self.critics[m][int(critic_selector)](state, joint)
            loss_m = -q.mean()
            losses.append(loss_m)
            qs.append(float(q.detach().mean().item()))
        loss = torch.stack(losses).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()
        self._freeze_critics(False)
        return float(loss.detach().item()), np.asarray(qs, dtype=np.float32)

    def _sample_tensors(self):
        state, obs, action, reward, next_state, next_obs, done = self.memory.sample(self.batch_size)
        tensors = [
            torch.tensor(x, dtype=torch.float32, device=self.device)
            for x in (state, obs, action, reward, next_state, next_obs, done)
        ]
        state, obs, action, reward, next_state, next_obs, done = tensors
        if self.use_reward_clip:
            reward = torch.clamp(reward, self.reward_clip_low, self.reward_clip_high)
        return state, obs, action, reward, next_state, next_obs, done

    def _common_info(self, critic_losses, actor_loss, actor_q_by_agent, reward, done, policy_noise):
        actor_q = None if actor_q_by_agent is None else float(np.mean(actor_q_by_agent))
        return {
            "critic_loss": float(np.mean(critic_losses)),
            "critic_loss_by_agent": np.asarray(critic_losses, dtype=np.float32),
            "actor_loss": actor_loss,
            "actor_loss_by_agent": np.full(self.agent_n, np.nan, dtype=np.float32)
                if actor_loss is None else np.full(self.agent_n, float(actor_loss), dtype=np.float32),
            "actor_q": actor_q,
            "actor_q_by_agent": np.full(self.agent_n, np.nan, dtype=np.float32)
                if actor_q_by_agent is None else np.asarray(actor_q_by_agent, dtype=np.float32),
            "reward_mean": float(reward.mean().item()),
            "reward_mean_by_agent": reward.mean(dim=0).detach().cpu().numpy().astype(np.float32),
            "done_mean": float(done.mean().item()),
            "terminal_mean": float(done.mean().item()),
            "total_it": int(self.total_it),
            "buffer_size": int(len(self.memory)),
            "policy_noise": float(policy_noise),
            "actor_lr": float(self.current_actor_lr),
            "critic_lr": float(self.current_critic_lr),
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, self.checkpoint_filename)
        torch.save({
            "algorithm": self.algorithm,
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critics": [group.state_dict() for group in self.critics],
            "target_critics": [group.state_dict() for group in self.target_critics],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizers": [[o.state_dict() for o in opts] for opts in self.critic_optimizers],
            "total_it": int(self.total_it),
            "current_episode": int(self.current_episode),
            "state_dim": self.state_dim,
            "agent_n": self.agent_n,
            "local_obs_dim": self.local_obs_dim,
            "local_action_dim": self.local_action_dim,
            "action_dim": self.action_dim,
            "actor_mode": "shared",
            "critic_mode": "per_agent_centralized_scalar",
        }, save_path)
        return save_path

    def load(self, path):
        load_path = os.path.join(path, self.checkpoint_filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"找不到模型文件：{load_path}")
        ckpt = torch.load(load_path, map_location=self.device)
        if str(ckpt.get("algorithm", self.algorithm)).upper() != self.algorithm:
            raise ValueError(f"模型算法与当前选择不一致：checkpoint={ckpt.get('algorithm')} current={self.algorithm}")
        self._load_actor_state_checked(ckpt["actor"], f"{self.algorithm}共享Actor checkpoint")
        self.target_actor.load_state_dict(ckpt.get("target_actor", ckpt["actor"]))
        for group, sd in zip(self.critics, ckpt["critics"]):
            group.load_state_dict(sd)
        for group, sd in zip(self.target_critics, ckpt.get("target_critics", ckpt["critics"])):
            group.load_state_dict(sd)
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizers" in ckpt:
            for opts, states in zip(self.critic_optimizers, ckpt["critic_optimizers"]):
                for opt, state in zip(opts, states):
                    opt.load_state_dict(state)
        self.total_it = int(ckpt.get("total_it", 0))
        self.current_episode = int(ckpt.get("current_episode", 0))
        self._apply_learning_rate_schedule(episode=self.current_episode)

    def save_actor(self, path):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, self.actor_filename)
        torch.save({
            "algorithm": self.algorithm,
            "actor": self.actor.state_dict(),
            "state_dim": self.state_dim,
            "agent_n": self.agent_n,
            "local_obs_dim": self.local_obs_dim,
            "local_action_dim": self.local_action_dim,
            "action_dim": self.action_dim,
            "actor_mode": "shared",
        }, save_path)
        return save_path

    def load_actor(self, actor_path):
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"找不到Actor文件：{actor_path}")
        ckpt = torch.load(actor_path, map_location=self.device)
        if isinstance(ckpt, dict) and "actor" in ckpt:
            if "algorithm" in ckpt and str(ckpt["algorithm"]).upper() != self.algorithm:
                raise ValueError(f"Actor算法与当前选择不一致：file={ckpt['algorithm']} current={self.algorithm}")
            sd = ckpt["actor"]
        else:
            sd = ckpt
        self._load_actor_state_checked(sd, f"{self.algorithm}共享Actor文件")
        self.target_actor.load_state_dict(self.actor.state_dict())


class DDPG(_ScalarMAAgentBase):
    algorithm = "DDPG"
    critic_n = 1

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        self.total_it += 1
        state, obs, action, reward, next_state, next_obs, done = self._sample_tensors()
        not_terminal = 1.0 - done

        with torch.no_grad():
            next_action = self._actors_forward_batch(next_obs, use_target=True)

        critic_losses = []
        target_means = []
        for m in range(self.agent_n):
            with torch.no_grad():
                target_q = reward[:, m:m+1] + self.gamma * not_terminal * self.target_critics[m][0](next_state, next_action)
            q = self.critics[m][0](state, action)
            loss = F.mse_loss(q, target_q)
            opt = self.critic_optimizers[m][0]
            opt.zero_grad(); loss.backward()
            if self.use_grad_clip:
                nn.utils.clip_grad_norm_(self.critics[m][0].parameters(), self.grad_clip_norm)
            opt.step()
            critic_losses.append(float(loss.item()))
            target_means.append(float(target_q.mean().item()))

        # DDPG intrinsic behavior: actor and all target networks update every train step.
        actor_loss, actor_q_by_agent = self._actor_update(state, obs, critic_selector=0)
        self._soft_update(self.actor, self.target_actor)
        for m in range(self.agent_n):
            self._soft_update(self.critics[m][0], self.target_critics[m][0])

        info = self._common_info(critic_losses, actor_loss, actor_q_by_agent, reward, done, 0.0)
        info.update({
            "target_q_mean": float(np.mean(target_means)),
            "target_q_std_mean": 0.0,
            "q_backup_mean": float(np.mean(target_means)),
        })
        return info


class TD3(_ScalarMAAgentBase):
    algorithm = "TD3"
    critic_n = 2

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        self.total_it += 1
        state, obs, action, reward, next_state, next_obs, done = self._sample_tensors()
        not_terminal = 1.0 - done
        current_noise = self._current_policy_noise()

        with torch.no_grad():
            next_action = self._actors_forward_batch(next_obs, use_target=True)
            next_action = self._add_heading_noise_tensor(next_action, current_noise, self.noise_clip)

        critic_losses = []
        target_means = []
        for m in range(self.agent_n):
            with torch.no_grad():
                tq1 = self.target_critics[m][0](next_state, next_action)
                tq2 = self.target_critics[m][1](next_state, next_action)
                target_q = reward[:, m:m+1] + self.gamma * not_terminal * torch.min(tq1, tq2)
            losses_m = []
            for i in range(2):
                q = self.critics[m][i](state, action)
                loss = F.mse_loss(q, target_q)
                opt = self.critic_optimizers[m][i]
                opt.zero_grad(); loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.critics[m][i].parameters(), self.grad_clip_norm)
                opt.step()
                losses_m.append(float(loss.item()))
            critic_losses.append(float(np.mean(losses_m)))
            target_means.append(float(target_q.mean().item()))

        actor_loss = None
        actor_q_by_agent = None
        if self.total_it % self.policy_freq == 0:
            actor_loss, actor_q_by_agent = self._actor_update(state, obs, critic_selector=0)
            self._soft_update(self.actor, self.target_actor)
            for m in range(self.agent_n):
                for i in range(2):
                    self._soft_update(self.critics[m][i], self.target_critics[m][i])

        info = self._common_info(critic_losses, actor_loss, actor_q_by_agent, reward, done, current_noise)
        info.update({
            "target_q_mean": float(np.mean(target_means)),
            "target_q_std_mean": 0.0,
            "q_backup_mean": float(np.mean(target_means)),
        })
        return info


def build_agent(cfg):
    algorithm = str(getattr(cfg, "algorithm", "CTD4")).strip().upper()
    if algorithm == "CTD4":
        return CTD4(cfg)
    if algorithm == "DDPG":
        return DDPG(cfg)
    if algorithm == "TD3":
        return TD3(cfg)
    raise ValueError(f"未知DRL算法：{algorithm}")


# 兼容命名
CTD4Agent = CTD4
