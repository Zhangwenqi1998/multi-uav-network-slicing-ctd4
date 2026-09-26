"""Main entry: finite eMBB completion time with URLLC puncturing and mMTC connectivity."""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import datetime
import random
import numpy as np
import torch

from main_config import MainConfig
from ctd4_config import Ctd4Config
from uav_env import Env
from ctd4_agent import CTD4, DDPG, TD3, build_agent
from ctd4_train import ctd4_train_func
from ctd4_test import ctd4_test_func, evaluate_benchmarks, safe_mean


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_system_info(main_cfg, ctd4_cfg):
    print("\n================ 系统参数 ================")
    print(
        f"区域={main_cfg.area_size:.0f}x{main_cfg.area_size:.0f}m | "
        f"UAV={main_cfg.uav_n} | 用户={main_cfg.user_n} | slot={main_cfg.slot_time:.1f}s"
    )
    print(
        f"切片用户=eMBB/URLLC/mMTC={main_cfg.slice_user_count.tolist()} | "
        f"URLLC包={main_cfg.urllc_packet_bits:.0f}bit, λ={main_cfg.urllc_arrival_rate:.1f}/s/user, "
        f"FBL ε={main_cfg.urllc_fbl_error_prob:.0e} | "
        f"mMTC门限={main_cfg.mmtc_sinr_threshold_db:.1f}dB"
    )
    print(
        f"URLLC PHY=finite blocklength normal approximation | "
        f"epsilon={main_cfg.urllc_fbl_error_prob:.1e} | "
        f"TTI={1e3 * main_cfg.tti_time:.1f}ms | "
        f"channel uses/RB/TTI={main_cfg.urllc_fbl_channel_uses_per_rb_tti:.0f}"
    )
    print(
        f"固定速度={main_cfg.uav_fixed_speed:.1f}m/s | 高度={main_cfg.uav_height.tolist()}m | "
        f"最大slot={main_cfg.max_step}（仅time-limit truncation）"
    )
    print(
        f"A2G信道=期望LoS/NLoS(fc={main_cfg.carrier_frequency_hz/1e9:.2f}GHz, "
        f"C0={main_cfg.c0_db:.2f}dB, C1={main_cfg.los_c1:.2f}, "
        f"C2={main_cfg.los_c2:.2f}, kappa={main_cfg.nlos_attenuation_kappa:.2f}, "
        f"NF={main_cfg.noise_figure_db:.1f}dB)"
    )
    print(
        f"双瓣天线Psi={np.degrees(main_cfg.antenna_half_beamwidth):.1f}deg, "
        f"Gmain={main_cfg.antenna_main_gain_db:.2f}dBi, "
        f"Gside={main_cfg.antenna_side_gain_db:.2f}dBi | "
        f"主瓣半径={np.round(main_cfg.antenna_main_coverage_radius, 1).tolist()}m | "
        f"副瓣低增益服务={'允许' if not main_cfg.require_main_lobe_for_service else '关闭'}"
    )
    print(
        "终止条件=全部eMBB有限文件传完；URLLC和mMTC为持续QoS业务，不参与终止。"
    )
    print(
        "文件大小与位置/信道独立随机生成（Mbit）："
        f"短{main_cfg.file_size_short_mbit_range}/"
        f"中{main_cfg.file_size_medium_mbit_range}/"
        f"长{main_cfg.file_size_long_mbit_range}，"
        f"比例={main_cfg.file_size_mix_prob.tolist()}"
    )
    print(
        f"UAV全区域独立随机起飞：x,y∈[{main_cfg.uav_init_uniform_low:.0f},"
        f"{main_cfg.uav_init_uniform_high:.0f}]m | 避开NFZ | "
        f"最小间距={main_cfg.uav_init_min_pair_dist:.0f}m"
    )
    print(
        f"资源分配=每UAV的{main_cfg.rb_n}个RB先按预计剩余完成时间分给eMBB；"
        f"URLLC每TTI先用空闲RB、不足再穿孔eMBB（仅受{main_cfg.rb_n}RB物理池限制）；"
        f"mMTC不占数据RB"
    )
    print("关联=eMBB按剩余工作量/每RB速率进行负载均衡关联；URLLC按可穿孔链路服务；mMTC按公共接入信道最大接收信号关联。")
    print(
        f"Reward=-{main_cfg.reward_step_cost:.1f}/slot + "
        f"{main_cfg.reward_progress_weight:.1f}×[gamma*Phi(s')-Phi(s)]，"
        f"Phi=-(用户完成时间瓶颈+UAV负载瓶颈+平均剩余比例)，"
        f"gamma={main_cfg.reward_potential_gamma:.3f}"
    )
    print(
        f"QoS约束=自适应Lagrange：lambda_U={main_cfg.qos_lagrange_urllc:.2f}, "
        f"lambda_M={main_cfg.qos_lagrange_mmtc:.2f}；"
        f"URLLC目标={main_cfg.urllc_reliability_target:.3f}，"
        f"mMTC目标={main_cfg.mmtc_connectivity_target:.2f}"
    )
    print(
        f"Per-UAV credit={main_cfg.reward_agent_credit_weight:.1f}×"
        "(本UAV归一化任务进度-团队平均进度)；"
        f"非法动作固定惩罚={main_cfg.reward_illegal_action_fixed_penalty:.1f}"
    )
    print(
        f"NFZ={main_cfg.use_nfz} | num={main_cfg.nfz_num_min}-{main_cfg.nfz_num_max} | "
        f"ActionShield={main_cfg.use_action_shield}"
    )
    if ctd4_cfg.algorithm == "CTD4":
        algo_detail = (
            f"distributional critic ensemble | fusion={ctd4_cfg.critic_fusion} | "
            f"critics={ctd4_cfg.num_critics} | delayed policy={ctd4_cfg.policy_freq}"
        )
    elif ctd4_cfg.algorithm == "TD3":
        algo_detail = (
            f"twin scalar critics | min target | target smoothing | "
            f"delayed policy={ctd4_cfg.policy_freq}"
        )
    else:
        algo_detail = "single scalar critic | actor/target updated every gradient step"
    print(
        f"所提算法={ctd4_cfg.algorithm}: shared actor + per-agent centralized {algo_detail} | "
        f"gamma={ctd4_cfg.gamma}"
    )
    enabled_benchmarks = list(getattr(ctd4_cfg, "benchmark_names", []))
    print(
        "启用的benchmark="
        + (", ".join(enabled_benchmarks) if enabled_benchmarks else "无")
    )
    print(
        f"state={ctd4_cfg.state_dim} | local_obs={ctd4_cfg.local_obs_dim} | "
        f"joint_action={ctd4_cfg.action_dim} | device={ctd4_cfg.device}"
    )
    print(
        f"探索：random_steps={ctd4_cfg.random_steps} | update_after={ctd4_cfg.update_after} | "
        f"behavior-noise decay={ctd4_cfg.exploration_noise_decay_episodes} EP | "
        f"LR schedule=episode-based"
    )
    if ctd4_cfg.use_fixed_validation:
        print(
            f"固定验证：每{ctd4_cfg.validation_interval}EP × "
            f"{ctd4_cfg.validation_ep_n}个无噪声固定场景；"
            f"每次输出best/median/worst轨迹并保存best模型。"
        )
    print(f"输出目录={ctd4_cfg.output_path}")
    print("==========================================")


def print_test_summary(name, ave_reward, metric_dict, ave_decision_time_us):
    print(f"\n================ {name} 测试摘要 ================")
    print(f"Average reward: {ave_reward:.4f}")
    print(
        f"Average slots: {safe_mean(metric_dict['ep_slot_count']):.2f} | "
        f"All-complete ratio: {safe_mean(metric_dict['ep_all_users_completed']):.4f}"
    )
    print(
        f"User completion ratio: {safe_mean(metric_dict['ep_user_completion_ratio']):.4f} | "
        f"Data completion ratio: {safe_mean(metric_dict['ep_data_completion_ratio']):.4f}"
    )
    print(
        f"Mean/P95 user completion slot: "
        f"{safe_mean(metric_dict['ep_mean_completion_slot']):.2f}/"
        f"{safe_mean(metric_dict['ep_p95_completion_slot']):.2f}"
    )
    print(
        f"Average eMBB sum rate: {safe_mean(metric_dict['ep_avg_sum_rate'])/1e6:.3f} Mbps | "
        f"Decision time: {ave_decision_time_us:.3f} us"
    )
    print(
        f"URLLC reliability: {safe_mean(metric_dict['ep_urllc_reliability']):.4f} | "
        f"mMTC connectivity: {safe_mean(metric_dict['ep_mmtc_connection_success_ratio']):.4f} | "
        f"QoS feasible ratio: {safe_mean(metric_dict['ep_qos_feasible']):.4f}"
    )
    urllc_gap = np.asarray(metric_dict["ep_urllc_constraint_gap"], dtype=np.float64).reshape(-1)
    urllc_gap = urllc_gap[np.isfinite(urllc_gap)]
    mmtc_gap = np.asarray(metric_dict["ep_mmtc_constraint_gap"], dtype=np.float64).reshape(-1)
    mmtc_gap = mmtc_gap[np.isfinite(mmtc_gap)]
    urllc_ok = int(np.sum(urllc_gap <= 1e-12))
    mmtc_ok = int(np.sum(mmtc_gap <= 1e-12))
    urllc_total = int(urllc_gap.size)
    mmtc_total = int(mmtc_gap.size)
    print(
        f"URLLC QoS satisfied episodes: {urllc_ok}/{urllc_total} "
        f"({(100.0 * urllc_ok / urllc_total) if urllc_total else 0.0:.2f}%) | "
        f"mMTC QoS satisfied episodes: {mmtc_ok}/{mmtc_total} "
        f"({(100.0 * mmtc_ok / mmtc_total) if mmtc_total else 0.0:.2f}%)"
    )
    print(
        f"URLLC fail ratios (FBL/resource): "
        f"{safe_mean(metric_dict['ep_urllc_fail_link_ratio']):.4f}/"
        f"{safe_mean(metric_dict['ep_urllc_fail_resource_ratio']):.4f} | "
        f"ReqRB avg/P95: {safe_mean(metric_dict['ep_urllc_avg_required_rb']):.2f}/"
        f"{safe_mean(metric_dict['ep_urllc_p95_required_rb']):.2f} | "
        f"Full-RB-hit TTI ratio: {safe_mean(metric_dict['ep_urllc_full_rb_hit_tti_ratio']):.4f} | "
        f"Idle RB used avg/slot: {safe_mean(metric_dict['ep_urllc_avg_idle_rb_used']):.2f}"
    )
    print("================================================")


def save_main_summary(ctd4_cfg, main_cfg, ave_reward, metric_dict, ave_decision_time_us):
    os.makedirs(ctd4_cfg.result_path, exist_ok=True)
    path = os.path.join(ctd4_cfg.result_path, "main_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Finite eMBB completion time with URLLC puncturing and mMTC connectivity constraints\n")
        f.write(f"average_reward: {ave_reward:.8f}\n")
        f.write(f"average_slot_count: {safe_mean(metric_dict['ep_slot_count']):.8f}\n")
        f.write(f"all_complete_ratio: {safe_mean(metric_dict['ep_all_users_completed']):.8f}\n")
        f.write(f"user_completion_ratio: {safe_mean(metric_dict['ep_user_completion_ratio']):.8f}\n")
        f.write(f"data_completion_ratio: {safe_mean(metric_dict['ep_data_completion_ratio']):.8f}\n")
        f.write(f"average_sum_rate_bps: {safe_mean(metric_dict['ep_avg_sum_rate']):.8f}\n")
        f.write(f"urllc_reliability: {safe_mean(metric_dict['ep_urllc_reliability']):.8f}\n")
        f.write(f"urllc_fail_link_ratio: {safe_mean(metric_dict['ep_urllc_fail_link_ratio']):.8f}\n")
        f.write(f"urllc_fail_cap_ratio: {safe_mean(metric_dict['ep_urllc_fail_cap_ratio']):.8f}\n")
        f.write(f"urllc_fail_budget_ratio: {safe_mean(metric_dict['ep_urllc_fail_budget_ratio']):.8f}\n")
        f.write(f"urllc_fail_available_ratio: {safe_mean(metric_dict['ep_urllc_fail_available_ratio']):.8f}\n")
        f.write(f"urllc_fail_resource_ratio: {safe_mean(metric_dict['ep_urllc_fail_resource_ratio']):.8f}\n")
        f.write(f"urllc_avg_required_rb: {safe_mean(metric_dict['ep_urllc_avg_required_rb']):.8f}\n")
        f.write(f"urllc_p95_required_rb: {safe_mean(metric_dict['ep_urllc_p95_required_rb']):.8f}\n")
        f.write(f"urllc_cap_hit_tti_ratio: {safe_mean(metric_dict['ep_urllc_cap_hit_tti_ratio']):.8f}\n")
        f.write(f"urllc_full_rb_hit_tti_ratio: {safe_mean(metric_dict['ep_urllc_full_rb_hit_tti_ratio']):.8f}\n")
        f.write(f"urllc_avg_idle_rb_used: {safe_mean(metric_dict['ep_urllc_avg_idle_rb_used']):.8f}\n")
        f.write(f"urllc_avg_total_rb_used: {safe_mean(metric_dict['ep_urllc_avg_total_rb_used']):.8f}\n")
        f.write(f"mmtc_connectivity: {safe_mean(metric_dict['ep_mmtc_connection_success_ratio']):.8f}\n")
        f.write(f"qos_feasible_ratio: {safe_mean(metric_dict['ep_qos_feasible']):.8f}\n")
        f.write(f"average_decision_time_us: {ave_decision_time_us:.8f}\n")
        f.write(f"time_limit_slots: {main_cfg.max_step}\n")
        f.write(f"algorithm: {ctd4_cfg.algorithm}\n")
        if ctd4_cfg.algorithm == "CTD4":
            f.write(f"critic_fusion: {ctd4_cfg.critic_fusion}\n")
    print(f"主结果摘要已保存：{path}")


def get_final_model_path(ctd4_cfg):
    if ctd4_cfg.load_model_path is not None:
        return ctd4_cfg.load_model_path
    if ctd4_cfg.load_actor_path is not None:
        return ctd4_cfg.load_actor_path
    # `final` is the actual last training iterate used for testing;
    # `final_last` is retained as an explicit backup/fallback.
    for name in ("final", "final_last"):
        model_dir = os.path.join(ctd4_cfg.model_path, name)
        if os.path.exists(os.path.join(model_dir, ctd4_cfg.model_checkpoint_name)) or os.path.exists(
            os.path.join(model_dir, ctd4_cfg.model_actor_name)
        ):
            return model_dir
    raise FileNotFoundError(
        f"未找到最后训练模型：{os.path.join(ctd4_cfg.model_path, 'final')}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train_test",
        choices=["train_test", "train_only", "test_only", "train", "test", "all"],
    )
    parser.add_argument("--algorithm", type=str, default=None, choices=["CTD4", "DDPG", "TD3", "ctd4", "ddpg", "td3"])
    parser.add_argument("--train_ep_n", type=int, default=None)
    parser.add_argument("--test_ep_n", type=int, default=None)
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--no_benchmarks", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--validation_interval", type=int, default=None)
    parser.add_argument("--validation_ep_n", type=int, default=None)
    parser.add_argument("--validation_plot_all", action="store_true")
    return parser.parse_args()


def main():
    print("开始：有限数据多UAV最短完成时间轨迹优化")
    print("当前时间：", datetime.datetime.now().strftime("%Y年%m月%d日 %H点%M分%S秒"))
    args = parse_args()
    mode = {"all": "train_test", "train": "train_only", "test": "test_only"}.get(args.mode, args.mode)

    main_cfg = MainConfig()
    if args.algorithm is not None:
        main_cfg.proposed_algorithm = str(args.algorithm).upper()
    if args.train_ep_n is not None:
        main_cfg.train_ep_n = int(args.train_ep_n)
    if args.test_ep_n is not None:
        main_cfg.test_ep_n = int(args.test_ep_n)
    ctd4_cfg = Ctd4Config(main_cfg)
    if args.load_model_path is not None:
        ctd4_cfg.load_model = True
        ctd4_cfg.load_model_path = args.load_model_path
    if args.no_benchmarks:
        ctd4_cfg.run_benchmarks = False
        ctd4_cfg.benchmark_names = []
    if args.no_save:
        ctd4_cfg.save = False
    if args.validation_interval is not None:
        ctd4_cfg.validation_interval = max(int(args.validation_interval), 1)
    if args.validation_ep_n is not None:
        ctd4_cfg.validation_ep_n = max(int(args.validation_ep_n), 1)
    if args.validation_plot_all:
        ctd4_cfg.validation_plot_all_episodes = True
    set_global_seed(main_cfg.seed)
    print_system_info(main_cfg, ctd4_cfg)

    env = Env()
    # 保留原有run_ctd4开关语义：现在它控制“当前所选的所提DRL算法”。
    run_proposed = bool(getattr(main_cfg, "run_ctd4", True))

    if run_proposed:
        agent = build_agent(ctd4_cfg)

        if ctd4_cfg.load_model and mode != "test_only":
            path = ctd4_cfg.load_model_path or ctd4_cfg.load_actor_path
            if os.path.isfile(path):
                agent.load_actor(path)
            else:
                agent.load(path)

        if mode in ("train_only", "train_test"):
            ctd4_train_func(ctd4_cfg, env, agent, main_cfg)

        if mode in ("test_only", "train_test"):
            load_path = get_final_model_path(ctd4_cfg) if mode == "test_only" else None
            ave_reward, metric_dict, decision_us = ctd4_test_func(
                ctd4_cfg, env, agent, main_cfg, load_model_path=load_path
            )
            print_test_summary(ctd4_cfg.algorithm, ave_reward, metric_dict, decision_us)
            save_main_summary(ctd4_cfg, main_cfg, ave_reward, metric_dict, decision_us)
            if ctd4_cfg.run_benchmarks:
                evaluate_benchmarks(
                    ctd4_cfg=ctd4_cfg,
                    env=env,
                    main_cfg=main_cfg,
                    benchmark_names=ctd4_cfg.benchmark_names,
                    ctd4_result={"rewards": metric_dict["ep_reward"], "metric_dict": metric_dict},
                )
    else:
        if ctd4_cfg.run_benchmarks and len(ctd4_cfg.benchmark_names) > 0:
            print("\nrun_ctd4=False：跳过当前所选所提DRL算法的训练和测试，只运行已启用的benchmark。")
            evaluate_benchmarks(
                ctd4_cfg=ctd4_cfg,
                env=env,
                main_cfg=main_cfg,
                benchmark_names=ctd4_cfg.benchmark_names,
                ctd4_result=None,
            )
        else:
            print("\nrun_ctd4=False，且未启用任何benchmark；无算法需要运行。")
    print("\n程序结束。")


if __name__ == "__main__":
    main()
