import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 这里按你的工程实际情况修改导入路径
from policy_trainer import (   # 👈 换成你原来那份RNN脚本的文件名（不带.py）
    load_trajectory,
    compute_dt,
    find_learn_segment,
    simulate_step,
)


def test_ppo_policy(args):
    """
    使用经 RL 微调后的 PPO 模型，在一条给定轨迹上进行 rollout，
    并与原始轨迹做对比（结构参考你原来的 test_policy）。
    """

    # 1. 加载 PPO 模型（里面已经包含 RNN 特征提取器 + 微调后的 GRU 权重）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PPO model from: {args.model_path}")
    model = PPO.load(args.model_path, device=device)

    n_hist = args.n_hist

    # 2. 读取测试轨迹
    states_abs, h, t = load_trajectory(args.traj_csv)
    dt = compute_dt(t)
    T = len(states_abs)

    lam0, phi0, r0 = states_abs[0, 1:4]
    goal_abs = states_abs[-1, 1:4].copy()
    goal_rel = np.array(
        [goal_abs[0] - lam0, goal_abs[1] - phi0, goal_abs[2] - r0],
        dtype=np.float32,
    )

    states_rel = states_abs.copy()
    states_rel[:, 1] -= lam0
    states_rel[:, 2] -= phi0
    states_rel[:, 3] -= r0

    learn_start = find_learn_segment(states_abs[:, 0], h)
    if learn_start is None:
        print("This trajectory has no learning segment; nothing to predict.")
        return

    print(f"Learning segment starts at index {learn_start} (time={t[learn_start]}s)")

    # 3. 构造预测轨迹 (绝对/相对状态)
    pred_abs = np.zeros_like(states_abs)
    pred_abs[: learn_start + 1] = states_abs[: learn_start + 1].copy()
    pred_rel = np.zeros_like(states_rel)
    pred_rel[: learn_start + 1] = states_rel[: learn_start + 1].copy()

    # 4. 初始化历史窗口（从真实段开始）
    #    与原 test_policy 完全一致
    start_idx = learn_start + 1 - n_hist
    hist_states = [states_rel[i].copy() for i in range(start_idx, learn_start + 1)]
    if learn_start + 1 < n_hist:
        pad_num = n_hist - (learn_start + 1)
        pad = [states_rel[0].copy()] * pad_num
        hist_states = pad + hist_states
    hist_states = hist_states[-n_hist:]

    # 5. 从 learn_start 开始循环预测
    for t_idx in range(learn_start, T - 1):
        current_rel = pred_rel[t_idx]
        current_abs = pred_abs[t_idx]

        # 历史窗口更新：用当前“预测”的状态
        hist_states.append(current_rel.copy())
        if len(hist_states) > n_hist:
            hist_states.pop(0)

        hist_arr = np.stack(hist_states, axis=0)  # (n_hist, 6)
        goal_tile = np.repeat(goal_rel[None, :], n_hist, axis=0)
        seq_input = np.concatenate([hist_arr, goal_tile], axis=-1)  # (n_hist, 9)

        # 注意：PPO 模型内部的 RNNFeatureExtractor 会自己做标准化，
        # 这里直接给“原始”的 seq_input 即可，不要再手动 (x-mean)/std。
        obs = seq_input.astype(np.float32)

        # stable-baselines3 的 predict 接受一个单个 obs，内部会自动加 batch 维
        action, _ = model.predict(obs, deterministic=True)  # action: (3,)

        # 用简化动力学滚一步（与你原来的 test_policy 一致）
        next_rel, next_abs = simulate_step(current_rel, current_abs, action, dt, lam0, phi0, r0)
        pred_rel[t_idx + 1] = next_rel
        pred_abs[t_idx + 1] = next_abs

    # 6. 画图对比：λ / φ / r
    lam_pred = pred_abs[:, 1]
    lam_true = states_abs[:, 1]
    phi_pred = pred_abs[:, 2]
    phi_true = states_abs[:, 2]
    r_pred = pred_abs[:, 3]
    r_true = states_abs[:, 3]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].plot(lam_pred, color="blue", label="Predicted", linestyle="-")
    axs[0].plot(lam_true, color="red", label="Target", linestyle="--")
    axs[0].axvline(learn_start, color="black", linestyle="--")
    axs[0].set_title("Longitude")
    axs[0].legend()

    axs[1].plot(phi_pred, color="blue", label="Predicted", linestyle="-")
    axs[1].plot(phi_true, color="red", label="Target", linestyle="--")
    axs[1].axvline(learn_start, color="black", linestyle="--")
    axs[1].set_title("Latitude")
    axs[1].legend()

    axs[2].plot(r_pred, color="blue", label="Predicted", linestyle="-")
    axs[2].plot(r_true, color="red", label="Target", linestyle="--")
    axs[2].axvline(learn_start, color="black", linestyle="--")
    axs[2].set_title("Earth Center Distance")
    axs[2].legend()

    plt.tight_layout()

    os.makedirs("./results", exist_ok=True)
    save_path = "./results/finetuned_policy.png"
    fig.savefig(save_path)

    # ---- 3D 轨迹对比 ----
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # 使用预测和真实轨迹的 λ/φ/r 作为三维坐标
    ax.plot(lam_pred, phi_pred, r_pred, 'b-', label='Predicted')
    ax.plot(lam_true, phi_true, r_true, 'r--', label='Target')
    ax.set_title('3D Trajectory')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Earth Center Distance')
    ax.legend()

    plt.tight_layout()
    save_path = "./results/finetuned_policy_3d.png"
    fig.savefig(save_path)

    # 7. 终点误差
    final_err = np.linalg.norm(pred_abs[-1, 1:4] - states_abs[-1, 1:4])
    print(f"Final endpoint error (λ,φ,r Euclidean) = {final_err:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Test RL-finetuned PPO policy on a single trajectory")
    parser.add_argument("--model_path", type=str, required=True, help="PPO RL 微调后的模型路径（.zip）")
    parser.add_argument("--traj_csv", type=str, required=True, help="用来测试的轨迹 CSV 文件路径")
    parser.add_argument("--n_hist", type=int, default=10, help="历史窗口长度（需要和训练时一致）")

    args = parser.parse_args()
    test_ppo_policy(args)


if __name__ == "__main__":
    main()
