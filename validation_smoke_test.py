import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import tempfile
import numpy as np
import torch
torch.set_num_threads(1)

from main_config import MainConfig
from ctd4_config import Ctd4Config
from uav_env import Env
from ctd4_agent import CTD4, DDPG, TD3, build_agent
from ctd4_train import _run_fixed_validation
from resource_algorithms import _apply_limited_urllc_puncturing
from ctd4_test import (
    _safe_nearest_unfinished,
    _workload_distance,
    _makespan_aware_one_step_greedy,
    _evaluate_action_global_reward,
)

print('=== CONFIG / SHAPE ===')
cfg = MainConfig()
cc = Ctd4Config(cfg)
configured_memory_capacity = int(cc.memory_capacity)
# Smoke test avoids allocating the full production replay buffer.
cc.memory_capacity = 32
env = Env()
agent = CTD4(cc)
state = env.reset(cfg, ep_id=0)
obs = env.get_obs(cfg)
print('slice_user_count', cfg.slice_user_count.tolist())
print('state_shape', state.shape, 'obs_shape', obs.shape, 'action_dim', cc.action_dim)
print('URLLC', cfg.urllc_arrival_rate, cfg.urllc_packet_bits, cfg.urllc_max_puncture_rb_per_tti, cfg.urllc_reliability_target)
print('URLLC_FBL', cfg.urllc_fbl_error_prob, cfg.urllc_fbl_qinv, cfg.urllc_fbl_channel_uses_per_rb_tti)
assert bool(cfg.urllc_fbl_enabled)
assert abs(float(cfg.urllc_fbl_error_prob) - 1e-5) < 1e-15
assert abs(float(cfg.urllc_fbl_channel_uses_per_rb_tti) - cfg.rb_bandwidth_hz * cfg.tti_time) < 1e-9
# FBL capacity must be below the Shannon first-order term for finite n and epsilon<0.5.
test_sinr = 10.0
shannon_bits = cfg.rb_bandwidth_hz * cfg.tti_time * np.log2(1.0 + test_sinr)
fbl_bits = cfg.calc_urllc_fbl_bits(test_sinr, 1)
assert 0.0 < fbl_bits < shannon_bits
assert cfg.calc_urllc_required_rb(test_sinr, cfg.urllc_packet_bits, cfg.rb_n) >= 1
assert bool(cfg.urllc_idle_rb_first)
assert int(cfg.urllc_physical_rb_limit_per_tti) == int(cfg.rb_n)
assert int(cfg.urllc_max_puncture_rb_per_tti) == int(cfg.rb_n)

print('\n=== URLLC IDLE-FIRST / NO ARTIFICIAL PUNCTURE CAP ===')
# One idle RB is available on UAV0.  The first 1-RB URLLC packet must consume
# that idle RB without puncturing eMBB; a second simultaneous packet then
# punctures exactly one eMBB RB.
m_n, k_n = cfg.uav_n, cfg.user_n
active = np.array([True] + [False] * (m_n - 1), dtype=bool)
embb_alloc = np.zeros((m_n, k_n), dtype=np.int32)
embb_alloc[0, int(cfg.embb_user_idx[0])] = cfg.rb_n - 1
sinr_hi = np.zeros((m_n, k_n), dtype=np.float32)
u_k = int(cfg.urllc_user_idx[0])
sinr_hi[0, u_k] = 100.0
rb_rate_fake = np.ones((m_n, k_n), dtype=np.float32) * 1.0e6
rb_bits_fake = rb_rate_fake * cfg.tti_time
p1 = {"user": u_k, "bits": cfg.urllc_packet_bits, "arrival_tti": 0}
out1 = _apply_limited_urllc_puncturing(
    cfg, [p1], sinr_hi, rb_bits_fake, rb_rate_fake, embb_alloc, active
)
assert out1['success'] == 1
assert int(np.sum(out1['punctured_rb_user'])) == 0
assert int(np.sum(out1['urllc_idle_rb_used_each_uav'])) == 1

out2 = _apply_limited_urllc_puncturing(
    cfg, [p1, dict(p1)], sinr_hi, rb_bits_fake, rb_rate_fake, embb_alloc, active
)
assert out2['success'] == 2
assert int(np.sum(out2['urllc_idle_rb_used_each_uav'])) == 1
assert int(np.sum(out2['punctured_rb_user'])) == 1

# At -6 dB the 512-bit FBL packet needs more than the old 10-RB cap, but it
# must now remain serviceable because the only upper bound is the 50-RB pool.
req12 = cfg.calc_urllc_required_rb(10.0 ** (-6.0 / 10.0), cfg.urllc_packet_bits, cfg.rb_n)
assert np.isfinite(req12) and 10 < int(req12) <= cfg.rb_n
embb_full = np.zeros((m_n, k_n), dtype=np.int32)
embb_full[0, int(cfg.embb_user_idx[0])] = cfg.rb_n
sinr_low = np.zeros((m_n, k_n), dtype=np.float32)
sinr_low[0, u_k] = 10.0 ** (-6.0 / 10.0)
out3 = _apply_limited_urllc_puncturing(
    cfg, [p1], sinr_low, rb_bits_fake, rb_rate_fake, embb_full, active
)
assert out3['success'] == 1
assert out3['urllc_fail_resource_count'] == 0
assert out3['urllc_fail_cap_count'] == 0
assert int(np.sum(out3['punctured_rb_user'])) == int(req12)
print('idle_first_ok', out1['urllc_idle_rb_used_each_uav'].tolist())
print('old_cap_exceeded_req_rb', req12, 'served=', out3['success'])
print('URLLC_FULL_PHYSICAL_POOL_VALIDATED')
print('mMTC', cfg.mmtc_sinr_threshold_db, cfg.mmtc_connectivity_target, cfg.mmtc_connectivity_floor)
assert cfg.slice_user_count.tolist() == [90,15,15]
assert state.shape == (cfg.state_dim,)
assert obs.shape == (cfg.uav_n, cfg.local_obs_dim)
assert int(np.sum(env.initial_data_bits > 0)) == cfg.embb_user_n
assert int(np.sum(~env.completed_mask)) == cfg.embb_user_n

print('\n=== SCENARIO-MATCHED A2G CHANNEL / TWO-LOBE ANTENNA ===')
expected_fc = 2.0e9
expected_c0 = (cfg.speed_of_light_mps / (4.0 * np.pi * expected_fc)) ** 2
expected_psi = np.arctan(cfg.user_cluster_radius / np.median(cfg.uav_height))
expected_main_gain = 2.285 / expected_psi ** 2
expected_main_gain_db = 10.0 * np.log10(expected_main_gain)
assert abs(float(cfg.carrier_frequency_hz) - expected_fc) < 1e-6
assert abs(float(cfg.c0_linear) - expected_c0) < 1e-15
assert abs(float(cfg.los_c1) - 9.61) < 1e-12
assert abs(float(cfg.los_c2) - 0.16) < 1e-12
assert abs(float(cfg.nlos_extra_loss_db) - 10.0) < 1e-12
assert abs(float(cfg.nlos_attenuation_kappa) - 0.1) < 1e-12
assert abs(float(cfg.noise_figure_db) - 7.0) < 1e-12
assert not bool(cfg.use_small_scale_fading)
assert abs(float(cfg.antenna_half_beamwidth) - expected_psi) < 1e-12
assert abs(float(cfg.antenna_main_gain) - expected_main_gain) < 1e-10
assert abs(float(cfg.antenna_side_gain_db) - (expected_main_gain_db - 20.0)) < 1e-10
assert np.allclose(
    cfg.antenna_main_coverage_radius,
    cfg.uav_height * np.tan(expected_psi),
    rtol=0.0,
    atol=1e-5,
)
# Three UAVs cannot simultaneously place six user clusters inside main lobes;
# low-gain side-lobe service therefore remains available.
assert not bool(cfg.require_main_lobe_for_service)
print('fc_GHz/C0_dB', cfg.carrier_frequency_hz / 1e9, cfg.c0_db)
print('C1/C2/kappa/NF', cfg.los_c1, cfg.los_c2, cfg.nlos_attenuation_kappa, cfg.noise_figure_db)
print('antenna_half_beam_deg', np.degrees(cfg.antenna_half_beamwidth))
print('antenna_main_dBi', cfg.antenna_main_gain_db, 'side_dBi', cfg.antenna_side_gain_db)
print('main_lobe_radius_m', cfg.antenna_main_coverage_radius.tolist())
print('SCENARIO_MATCHED_CHANNEL_MODEL_VALIDATED')

print('\n=== ONE STEP / RESOURCE RULES ===')
action = np.tile(np.array([1.0,0.0], dtype=np.float32), cfg.uav_n)
next_state, reward, done = env.step(cfg, 0, action)
print('reward', reward, 'done', done)
print('rb_budget', env.rb_budget_list[-1].tolist())
print('puncture_ratio', env.prev_puncture_ratio)
print('urllc_slot', env.prev_slice_metrics['urllc_success_ratio'])
print('mmtc_slot', env.prev_slice_metrics['mmtc_success_ratio'])
assert np.all(env.rb_budget_list[-1][:, cfg.slice_urllc] == 0)
assert np.all(env.rb_budget_list[-1][:, cfg.slice_mmtc] == 0)
assert np.all(env.rb_budget_list[-1][:, cfg.slice_embb] == cfg.rb_n)
assert env.prev_slice_metrics['mmtc_active_count'] == cfg.mmtc_user_n
assert cfg.max_step == 180
assert abs(float(np.sum(env.prev_agent_credit_reward))) < 1e-6
assert np.isfinite(env.prev_completion_potential)
assert cfg.reward_time_limit_penalty == 0.0

print('\n=== CRITIC UPDATE ===')
cc2 = Ctd4Config(cfg)
cc2.memory_capacity = 32
cc2.batch_size = 2
env2 = Env(); agent2 = CTD4(cc2)
s = env2.reset(cfg, 1); o = env2.get_obs(cfg)
rng = np.random.default_rng(1)
for t in range(2):
    th = rng.uniform(-np.pi, np.pi, cfg.uav_n)
    a = np.stack([np.cos(th), np.sin(th)], axis=1).reshape(-1).astype(np.float32)
    ns, r, d = env2.step(cfg, 1, a)
    no = np.zeros_like(o) if env2.prev_terminated else env2.get_obs(cfg)
    agent2.store_transition(s, o, env2.prev_raw_action, env2.prev_agent_reward, ns, no, env2.prev_terminated)
    s, o = ns, no
info = agent2.train()
print('critic_loss', info['critic_loss'], 'policy_noise', info['policy_noise'], 'actor_lr', info['actor_lr'], 'critic_lr', info['critic_lr'])
assert np.isfinite(info['critic_loss'])

print('\n=== CTD4 PAPER/CODE ALIGNMENT ===')
# 1) KL direction must be D_KL(current || target), not the reverse.
cur = torch.distributions.Normal(
    torch.tensor([[0.0]], dtype=torch.float32),
    torch.tensor([[0.5]], dtype=torch.float32),
)
tgt = torch.distributions.Normal(
    torch.tensor([[1.0]], dtype=torch.float32),
    torch.tensor([[1.5]], dtype=torch.float32),
)
kl_impl = agent2._ctd4_kl(cur, tgt)
kl_forward = torch.distributions.kl.kl_divergence(cur, tgt)
kl_reverse = torch.distributions.kl.kl_divergence(tgt, cur)
print('KL current||target', float(kl_impl.item()), 'reverse', float(kl_reverse.item()))
assert torch.allclose(kl_impl, kl_forward, atol=1e-8, rtol=1e-8)
assert not torch.allclose(kl_impl, kl_reverse, atol=1e-6, rtol=1e-6)

# 2) Kalman fusion must follow paper Eq.(1)-(3) with only tiny variance jitter.
means = [
    torch.tensor([[0.0]], dtype=torch.float32, device=agent2.device),
    torch.tensor([[2.0]], dtype=torch.float32, device=agent2.device),
]
stds = [
    torch.tensor([[1.0]], dtype=torch.float32, device=agent2.device),
    torch.tensor([[2.0]], dtype=torch.float32, device=agent2.device),
]
f_mu, f_std = agent2._fuse_critic_outputs(means, stds, batch_size=1)
k = 1.0 / (1.0 + 4.0)
expected_mu = 0.0 + k * (2.0 - 0.0)
expected_var = (1.0 - k) * 1.0 + k * 4.0 + cc2.kalman_variance_eps
print('Kalman fused mu/std', float(f_mu.item()), float(f_std.item()))
assert abs(float(f_mu.item()) - expected_mu) < 1e-6
assert abs(float(f_std.item()) - float(np.sqrt(expected_var))) < 1e-6
assert abs(cc2.critic_std_min - 1e-6) < 1e-15
assert abs(cc2.target_std_min - 1e-6) < 1e-15
assert abs(cc2.kalman_variance_eps - 1e-6) < 1e-15

print('\n=== SAVE / LOAD ===')
with tempfile.TemporaryDirectory() as td:
    agent2.save(td)
    agent3 = CTD4(cc2)
    agent3.load(td)
    test_obs = env2.get_obs(cfg)
    a2 = agent2.choose_action(test_obs)
    a3 = agent3.choose_action(test_obs)
    diff = float(np.max(np.abs(a2-a3)))
    print('max_action_diff', diff)
    assert diff < 1e-6

print('\n=== FULL HEURISTIC EPISODE ===')
env3 = Env(); env3.reset(cfg, 123)
while True:
    a = _safe_nearest_unfinished(cfg, env3)
    _, _, d = env3.step(cfg, 123, a)
    if d:
        break
metric = env3.get_ep_metric(cfg)
print('slots', metric['ep_slot_count'], 'embb_complete', metric['ep_all_users_completed'])
print('URLLC_ep', metric['ep_urllc_reliability'], 'mMTC_ep', metric['ep_mmtc_connection_success_ratio'], 'QoS', metric['ep_qos_feasible'])
assert int(metric['ep_slot_count']) <= int(cfg.max_step)
assert metric['ep_done_reason'] in ('all_embb_files_completed', 'time_limit')
if bool(metric['ep_all_users_completed']):
    assert metric['ep_completed_user_n'] == cfg.embb_user_n
    assert metric['ep_done_reason'] == 'all_embb_files_completed'
else:
    assert int(metric['ep_slot_count']) == int(cfg.max_step)
    assert metric['ep_done_reason'] == 'time_limit'

print('\n=== FIXED VALIDATION INTERFACE ===')
cfgv = MainConfig(); cfgv.max_step = 2; cfgv.train_ep_n = 1
ccv = Ctd4Config(cfgv); ccv.memory_capacity = 32; ccv.validation_ep_n = 1; ccv.validation_representative_n = 1; ccv.validation_plot_all_episodes = False
envv = Env(); agentv = CTD4(ccv)
h = {k: [] for k in ['train_episode','mean_slots','std_slots','success_ratio','mean_reward','data_completion_ratio','user_completion_ratio','urllc_reliability','mmtc_connectivity','qos_feasible_ratio','score']}
summary = _run_fixed_validation(ccv, envv, agentv, cfgv, 1, h)
print('validation_summary', summary)
assert 'urllc_reliability' in summary and 'mmtc_connectivity' in summary

print('\nALL_VALIDATIONS_PASSED')

print('\n=== EPISODE-BASED NOISE / LR SCHEDULE ===')
assert cfg.max_step == 180
assert cfg.file_size_short_mbit_range == (13.0, 26.0)
assert cfg.file_size_medium_mbit_range == (30.0, 61.0)
assert cfg.file_size_long_mbit_range == (69.0, 113.0)
assert abs(cfg.data_norm_bits - 113.0e6) < 1.0
expected_total = cfg.train_ep_n * cfg.expected_episode_slots
expected_updates = max(expected_total - cc.update_after, 1)
assert cc.exploration_noise_decay_episodes == max(int(cc.noise_decay_fraction * cfg.train_ep_n), 1)
assert cc.policy_noise_decay_steps == max(int(cc.noise_decay_fraction * expected_updates), 1)
print('expected_episode_slots_for_internal_update_estimate', cfg.expected_episode_slots)
print('file_ranges_Mbit', cfg.file_size_short_mbit_range, cfg.file_size_medium_mbit_range, cfg.file_size_long_mbit_range)
print('exploration_noise_decay_episodes', cc.exploration_noise_decay_episodes)
print('policy_noise_decay_steps', cc.policy_noise_decay_steps)

print('\n=== EPISODE-BASED LR DECAY / REPLAY ===')
assert configured_memory_capacity == 200000
assert cc.lr_decay_hold_episodes == max(int(0.25 * cfg.train_ep_n), 1)
assert cc.lr_decay_mid_episodes == max(int(0.55 * cfg.train_ep_n), cc.lr_decay_hold_episodes + 1)
assert cc.lr_decay_end_episodes == max(int(0.80 * cfg.train_ep_n), cc.lr_decay_mid_episodes + 1)
assert 0.0 < cc.actor_lr_min <= cc.actor_lr_mid <= cc.actor_lr
assert 0.0 < cc.critic_lr_min <= cc.critic_lr_mid <= cc.critic_lr
agent_lr = CTD4(cc)
for episode, expected_a, expected_c in [
    (0, cc.actor_lr, cc.critic_lr),
    (cc.lr_decay_hold_episodes, cc.actor_lr, cc.critic_lr),
    (cc.lr_decay_mid_episodes, cc.actor_lr_mid, cc.critic_lr_mid),
    (cc.lr_decay_end_episodes, cc.actor_lr_min, cc.critic_lr_min),
    (cc.lr_decay_end_episodes + 1000, cc.actor_lr_min, cc.critic_lr_min),
]:
    a_lr, c_lr = agent_lr.set_training_episode(episode)
    print('episode', episode, 'actor_lr', a_lr, 'critic_lr', c_lr)
    assert abs(a_lr - expected_a) < 1e-12
    assert abs(c_lr - expected_c) < 1e-12
print('EPISODE_LR_DECAY_VALIDATED')

print('\n=== REWARD OPTIMIZATION SETTINGS ===')
assert cfg.use_adaptive_qos_lagrange
assert cfg.reward_agent_credit_weight == 2.0
assert cfg.reward_illegal_action_fixed_penalty == 15.0
assert cfg.safe_distance_penalty_max == 2.0
assert cc.use_reward_clip and cc.reward_clip_low == -20.0 and cc.reward_clip_high == 5.0
print('potential_weights', cfg.reward_potential_user_bottleneck_weight, cfg.reward_potential_uav_bottleneck_weight, cfg.reward_potential_mean_remaining_weight)
print('lagrange_init', cfg.qos_lagrange_urllc, cfg.qos_lagrange_mmtc)
print('REWARD_SETTINGS_VALIDATED')


print('\n=== BENCHMARK SWITCHES / MAKESPAN GREEDY ===')
assert cc.benchmark_names == [
    'SafeNearestUnfinished',
    'WorkloadDistance',
    'MakespanAwareOneStepGreedy',
]
bench_env = Env()
bench_env.reset(cfg, ep_id=9)
bench_action = _makespan_aware_one_step_greedy(cfg, bench_env)
assert bench_action.shape == (cfg.action_dim,)
assert np.all(np.isfinite(bench_action))

print('\n=== REWARD-ALIGNED GREEDY ===')
import copy
env_g = Env(); env_g.reset(cfg, ep_id=77)
rng_g = np.random.default_rng(77)
for j in range(6):
    th = rng_g.uniform(-np.pi, np.pi, cfg.uav_n)
    a = np.stack([np.cos(th), np.sin(th)], axis=1).reshape(-1).astype(np.float32)
    score = _evaluate_action_global_reward(cfg, env_g, a)
    shadow = copy.deepcopy(env_g)
    _, direct_reward, _ = shadow.step(cfg, 77, a)
    diff = abs(float(score) - float(direct_reward))
    print('case', j, 'diff', diff)
    assert diff < 1e-8
print('REWARD_ALIGNED_GREEDY_VALIDATED')

print('enabled_benchmarks', cc.benchmark_names)
print('makespan_greedy_action', bench_action.tolist())
print('BENCHMARK_SETTINGS_VALIDATED')

print('\n=== DRL ALGORITHM SWITCH / FAIR SHARED SETTINGS ===')
for algo_name, algo_cls in [('CTD4', CTD4), ('DDPG', DDPG), ('TD3', TD3)]:
    cfg_a = MainConfig(); cfg_a.proposed_algorithm = algo_name; cfg_a.train_ep_n = 1
    cc_a = Ctd4Config(cfg_a); cc_a.memory_capacity = 32; cc_a.batch_size = 2
    a = build_agent(cc_a)
    assert isinstance(a, algo_cls)
    assert a.actor.net[0].in_features == cc_a.local_obs_dim
    assert a.actor.net[0].out_features == cc_a.actor_hidden_dim
    assert abs(a.gamma - cc_a.gamma) < 1e-12
    assert abs(a.soft_tau - cc_a.soft_tau) < 1e-12
    assert a.batch_size == cc_a.batch_size
    e = Env(); st = e.reset(cfg_a, 9000); ob = e.get_obs(cfg_a)
    rng_a = np.random.default_rng(123)
    for _ in range(2):
        th = rng_a.uniform(-np.pi, np.pi, cfg_a.uav_n)
        ac = np.stack([np.cos(th), np.sin(th)], axis=1).reshape(-1).astype(np.float32)
        ns, rr, dd = e.step(cfg_a, 9000, ac)
        nob = np.zeros_like(ob) if e.prev_terminated else e.get_obs(cfg_a)
        a.store_transition(st, ob, e.prev_raw_action, e.prev_agent_reward, ns, nob, e.prev_terminated)
        st, ob = ns, nob
        if dd:
            break
    if len(a.memory) >= 2:
        inf = a.train()
        assert inf is not None and np.isfinite(inf['critic_loss'])
    print(algo_name, 'OK', 'checkpoint=', a.checkpoint_filename)
