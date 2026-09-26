"""
ZHANG Wenqi

some_functions.py

通用保存与画图函数
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    """确保文件夹存在"""
    os.makedirs(path, exist_ok=True)


def moving_average(data, beta=0.9):
    """指数滑动平均"""
    data = np.asarray(data, dtype=np.float32).reshape(-1)

    ma = []
    for x in data:
        if len(ma) == 0:
            ma.append(float(x))
        else:
            ma.append(beta * ma[-1] + (1.0 - beta) * float(x))

    return np.asarray(ma, dtype=np.float32)


def save_array(array, path, name):
    """保存数组为npy"""
    ensure_dir(path)
    arr = np.asarray(array)
    np.save(os.path.join(path, name + ".npy"), arr)


def save_metric_dict(metric_dict, path, prefix=""):
    """保存指标字典"""
    ensure_dir(path)

    for key, value in metric_dict.items():
        if value is None:
            continue

        try:
            arr = np.asarray(value)
        except Exception:
            continue

        if arr.dtype == object:
            continue

        np.save(os.path.join(path, prefix + key + ".npy"), arr)


def plot_curve(y, path, save_name, title=None, xlabel="Episode", ylabel="Value"):
    """画单条曲线"""
    ensure_dir(path)

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return

    plt.figure()
    plt.plot(np.arange(len(y)), y, label="Raw")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, save_name + ".png"), dpi=300)
    plt.close()


def plot_curve_with_ma(
        y,
        path,
        save_name,
        title=None,
        xlabel="Episode",
        ylabel="Value",
        beta=0.9
):
    """
    原始曲线和MA曲线画在同一张图上
    """
    ensure_dir(path)

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return

    ma = moving_average(y, beta=beta)

    x = np.arange(len(y))

    plt.figure()
    plt.plot(x, y, label="Raw")
    plt.plot(x, ma, label="MA")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, save_name + ".png"), dpi=300)
    plt.close()


def plot_rewards(rewards, ma_rewards, cfg, tag="train"):
    """
    画reward曲线
    原始reward和MA reward在同一张图上
    """
    ensure_dir(cfg.result_path)

    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    ma_rewards = np.asarray(ma_rewards, dtype=np.float32).reshape(-1)

    if rewards.size == 0:
        return

    x = np.arange(len(rewards))

    plt.figure()
    plt.plot(x, rewards, label="Raw")
    if ma_rewards.size == rewards.size:
        plt.plot(x, ma_rewards, label="MA")
    else:
        plt.plot(x, moving_average(rewards), label="MA")

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(f"{cfg.algo_name} Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_path, f"{tag}_reward.png"), dpi=300)
    plt.close()


def plot_metric_curve(metric, cfg, ylabel, save_name, title=None, tag="train", use_ma=True):
    """
    画指标曲线
    默认原始曲线和MA曲线在同一张图上
    """
    if title is None:
        title = cfg.algo_name

    if use_ma:
        plot_curve_with_ma(
            y=metric,
            path=cfg.result_path,
            save_name=f"{tag}_{save_name}",
            title=title,
            ylabel=ylabel
        )
    else:
        plot_curve(
            y=metric,
            path=cfg.result_path,
            save_name=f"{tag}_{save_name}",
            title=title,
            ylabel=ylabel
        )


def plot_uav_trajectory(uav_pos, user_pos, cfg, path, save_name, title=None):
    """
    画单个episode的UAV轨迹图

    uav_pos: [T+1, uav_n, 2]
    user_pos: [user_n, 2]
    """
    ensure_dir(path)

    uav_pos = np.asarray(uav_pos, dtype=np.float32)
    user_pos = np.asarray(user_pos, dtype=np.float32)

    if uav_pos.ndim != 3 or uav_pos.shape[-1] != 2:
        return

    if title is None:
        title = "UAV Trajectory"

    plt.figure()

    # 用户位置
    if user_pos.size > 0:
        plt.scatter(
            user_pos[:, 0],
            user_pos[:, 1],
            marker="x",
            label="Users"
        )

    # UAV轨迹
    uav_n = uav_pos.shape[1]
    for m in range(uav_n):
        traj = uav_pos[:, m, :]

        plt.plot(
            traj[:, 0],
            traj[:, 1],
            marker="o",
            markersize=2,
            label=f"UAV {m + 1}"
        )

        # 起点
        plt.scatter(
            traj[0, 0],
            traj[0, 1],
            marker="s",
            label=f"UAV {m + 1} Start"
        )

        # 终点
        plt.scatter(
            traj[-1, 0],
            traj[-1, 1],
            marker="^",
            label=f"UAV {m + 1} End"
        )

    plt.xlim(0, cfg.area_size)
    plt.ylim(0, cfg.area_size)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(path, save_name + ".png"), dpi=300)
    plt.close()


def save_train_outputs(rewards, ma_rewards, metric_dict, cfg):
    """
    保存训练结果并画图
    reward和各指标均为 raw + MA 同图
    """
    ensure_dir(cfg.result_path)

    save_array(rewards, cfg.result_path, "train_rewards")
    save_array(ma_rewards, cfg.result_path, "train_ma_rewards")
    save_metric_dict(metric_dict, cfg.result_path, prefix="train_")

    plot_rewards(
        rewards=rewards,
        ma_rewards=ma_rewards,
        cfg=cfg,
        tag="train"
    )

    if "ep_energy_efficiency" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_energy_efficiency"],
            cfg=cfg,
            ylabel="Energy Efficiency (bit/J)",
            save_name="energy_efficiency",
            tag="train"
        )

    if "ep_total_bits" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_total_bits"],
            cfg=cfg,
            ylabel="Total Bits",
            save_name="total_bits",
            tag="train"
        )

    if "ep_total_energy" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_total_energy"],
            cfg=cfg,
            ylabel="Total Energy (J)",
            save_name="total_energy",
            tag="train"
        )

    if "ep_avg_slot_rate" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_avg_slot_rate"],
            cfg=cfg,
            ylabel="Average Slot Rate (bit/s)",
            save_name="avg_slot_rate",
            tag="train"
        )

    if "ep_avg_slot_ee" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_avg_slot_ee"],
            cfg=cfg,
            ylabel="Average Slot EE (bit/J)",
            save_name="avg_slot_ee",
            tag="train"
        )

    if "ep_avg_speed" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_avg_speed"],
            cfg=cfg,
            ylabel="Average UAV Speed (m/s)",
            save_name="avg_speed",
            tag="train"
        )

    if "ep_avg_power" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_avg_power"],
            cfg=cfg,
            ylabel="Average UAV Power (W)",
            save_name="avg_power",
            tag="train"
        )

    if "critic_loss" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["critic_loss"],
            cfg=cfg,
            ylabel="Critic Loss",
            save_name="critic_loss",
            tag="train"
        )

    if "actor_loss" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["actor_loss"],
            cfg=cfg,
            ylabel="Actor Loss",
            save_name="actor_loss",
            tag="train"
        )


def save_test_outputs(metric_dict, cfg):
    """
    保存测试结果并画图
    测试指标也默认 raw + MA 同图
    """
    ensure_dir(cfg.result_path)
    save_metric_dict(metric_dict, cfg.result_path, prefix="test_")

    if "ep_energy_efficiency" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_energy_efficiency"],
            cfg=cfg,
            ylabel="Energy Efficiency (bit/J)",
            save_name="energy_efficiency",
            tag="test"
        )

    if "ep_total_bits" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_total_bits"],
            cfg=cfg,
            ylabel="Total Bits",
            save_name="total_bits",
            tag="test"
        )

    if "ep_total_energy" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_total_energy"],
            cfg=cfg,
            ylabel="Total Energy (J)",
            save_name="total_energy",
            tag="test"
        )

    if "ep_avg_slot_rate" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_avg_slot_rate"],
            cfg=cfg,
            ylabel="Average Slot Rate (bit/s)",
            save_name="avg_slot_rate",
            tag="test"
        )

    if "ep_avg_slot_ee" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_avg_slot_ee"],
            cfg=cfg,
            ylabel="Average Slot EE (bit/J)",
            save_name="avg_slot_ee",
            tag="test"
        )

    if "ep_avg_speed" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_avg_speed"],
            cfg=cfg,
            ylabel="Average UAV Speed (m/s)",
            save_name="avg_speed",
            tag="test"
        )

    if "ep_avg_power" in metric_dict:
        plot_metric_curve(
            metric=metric_dict["ep_avg_power"],
            cfg=cfg,
            ylabel="Average UAV Power (W)",
            save_name="avg_power",
            tag="test"
        )


def safe_mean(x):
    """安全求均值"""
    if x is None:
        return 0.0

    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0

    return float(np.mean(arr))


def print_train_summary(rewards, metric_dict):
    """打印训练结果摘要"""
    print("\n================ 训练结果摘要 ================")
    print(f"平均训练reward：{safe_mean(rewards):.4f}")

    if "ep_energy_efficiency" in metric_dict:
        print(f"平均长期能效(bit/J)：{safe_mean(metric_dict['ep_energy_efficiency']):.4e}")

    if "ep_total_bits" in metric_dict:
        print(f"平均累计bits：{safe_mean(metric_dict['ep_total_bits']):.4e}")

    if "ep_total_energy" in metric_dict:
        print(f"平均累计能耗(J)：{safe_mean(metric_dict['ep_total_energy']):.4e}")

    if "ep_avg_speed" in metric_dict:
        print(f"平均UAV速度(m/s)：{safe_mean(metric_dict['ep_avg_speed']):.4f}")

    print("==============================================")