import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ===============================
# 1. 轨迹读取 & 数据处理
# ===============================

STATE_COLS = ["v", "lambda(rad)", "phi(rad)", "r", "psi(rad)", "gamma(rad)"]
H_COL = "h(m)"
TIME_COL = "time(s)"


def load_trajectory(csv_path):
    """
    从单个 CSV 读取轨迹:
    返回:
      states_abs: (T, 6) [v, lambda, phi, r, psi, gamma]
      h: (T,) 高度
      time: (T,) 秒
    """
    df = pd.read_csv(csv_path)
    # 兼容你给的列名
    v = df["v"].values.astype(np.float32)
    lam = df["lambda(rad)"].values.astype(np.float32)
    phi = df["phi(rad)"].values.astype(np.float32)
    r = df["r"].values.astype(np.float32)
    psi = df["psi(rad)"].values.astype(np.float32)
    gamma = df["gamma(rad)"].values.astype(np.float32)

    h = df["h(m)"].values.astype(np.float32)
    t = df[TIME_COL].values.astype(np.float32)

    states_abs = np.stack([v, lam, phi, r, psi, gamma], axis=-1)
    return states_abs, h, t


def find_learn_segment(v, h, v_thr=1200.0, h_thr=20000.0):
    """
    找到“需要学习的后半段”的起始索引:
    满足: 不再同时满足 (v>v_thr & h>h_thr) 的第一帧。
    若全部满足，则返回 None (跳过此轨迹)。
    """
    mask_direct = (v > v_thr) & (h > h_thr)
    # 第一个 mask_direct=False 的位置
    idxs = np.where(~mask_direct)[0]
    if len(idxs) == 0:
        return None
    return int(idxs[0])


def compute_dt(time_arr):
    if len(time_arr) < 2:
        return 1.0
    diffs = np.diff(time_arr)
    return float(np.mean(diffs))


# ===============================
# 2. 根据两帧状态反推 expert action
# ===============================

def infer_action_from_states(s_t, s_tp1):
    """
    根据连续两帧绝对状态，反推出 env 中的动作:
    我们使用的动力学缩放是:
        V_change   = a[0] * 5.0
        psi_change = a[1] * 0.005
        gamma_chg  = a[2] * 0.005
    即:
        a[0] = dv/5, a[1]=dpsi/0.005, a[2]=dgamma/0.005
    再 clip 到 [-1,0]x[-1,1]x[-1,1]
    """
    v_t, lam_t, phi_t, r_t, psi_t, gamma_t = s_t
    v_tp1, lam_tp1, phi_tp1, r_tp1, psi_tp1, gamma_tp1 = s_tp1

    dv = v_tp1 - v_t
    dpsi = psi_tp1 - psi_t
    dgamma = gamma_tp1 - gamma_t

    a0 = dv / 5.0
    a1 = dpsi / 0.005
    a2 = dgamma / 0.005

    a = np.array([a0, a1, a2], dtype=np.float32)
    # env 中 action 空间: [Δv, Δpsi, Δgamma] ∈ [-1,0]×[-1,1]×[-1,1]
    a[0] = np.clip(a[0], -1.0, 0.0)
    a[1:] = np.clip(a[1:], -1.0, 1.0)
    return a


# ===============================
# 3. Dataset: 历史窗口 + 终点 -> 动作
# ===============================

class TrajDataset(Dataset):
    def __init__(self, csv_paths, n_hist=10):
        self.samples = []  # 每个样本: (seq_input, action)
        self.n_hist = n_hist

        for path in csv_paths:
            print(f"Loading trajectory: {path}")
            states_abs, h, t = load_trajectory(path)
            T = len(states_abs)
            if T < n_hist + 2:
                continue

            dt = compute_dt(t)
            # 参考: 起点 & 终点 (绝对)
            lam0, phi0, r0 = states_abs[0, 1:4]
            goal_abs = states_abs[-1, 1:4].copy()
            goal_rel = np.array(
                [goal_abs[0] - lam0, goal_abs[1] - phi0, goal_abs[2] - r0],
                dtype=np.float32,
            )

            # 计算相对坐标状态
            states_rel = states_abs.copy()
            states_rel[:, 1] -= lam0
            states_rel[:, 2] -= phi0
            states_rel[:, 3] -= r0

            learn_start = find_learn_segment(states_abs[:, 0], h)
            if learn_start is None or learn_start >= T - 1:
                print(f"  Skip (no learning segment) for {path}")
                continue

            # 从 learn_start 到 T-2 作为训练样本 (最后一帧没有 next)
            for t_idx in range(learn_start, T - 1):
                s_t_abs = states_abs[t_idx]
                s_tp1_abs = states_abs[t_idx + 1]

                action = infer_action_from_states(s_t_abs, s_tp1_abs)

                # 历史窗口索引
                start_idx = max(0, t_idx - n_hist + 1)
                hist = states_rel[start_idx : t_idx + 1]  # (<=n_hist, 6)
                # pad 到 n_hist (前面重复第一帧)
                if hist.shape[0] < n_hist:
                    pad_num = n_hist - hist.shape[0]
                    pad = np.repeat(hist[0:1], pad_num, axis=0)
                    hist = np.concatenate([pad, hist], axis=0)

                assert hist.shape[0] == n_hist
                # 对每个时间步拼上相同的 goal_rel
                goal_tile = np.repeat(goal_rel[None, :], n_hist, axis=0)
                seq_input = np.concatenate([hist, goal_tile], axis=-1)  # (n_hist, 9)

                self.samples.append((seq_input.astype(np.float32), action.astype(np.float32)))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, act = self.samples[idx]
        return seq, act


# ===============================
# 4. GRU Policy 网络
# ===============================

class Policy(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, act_dim=3, policy_type="gru"):
        super().__init__()
        if policy_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif policy_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),  # 输出在 [-1, 1]
        )

    def forward(self, seq):
        """
        seq: (B, T, input_dim)
        """
        out, h = self.rnn(seq)  # out: (B, T, H)
        last = out[:, -1, :]    # 取最后一个时间步
        act = self.fc(last)     # (B, act_dim)
        return act


# ===============================
# 5. 训练函数
# ===============================

def train_policy(args):
    csv_paths = sorted(glob.glob(os.path.join(args.csv_path, "*.csv")))
    if not csv_paths:
        print("No csv files found!")
        return

    dataset = TrajDataset(csv_paths, n_hist=args.n_hist)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # 计算输入规范化（方便训练）
    all_inputs = np.concatenate(
        [s[0].reshape(-1, s[0].shape[-1]) for s in dataset.samples], axis=0
    )  # (N * n_hist, input_dim)
    mean = all_inputs.mean(axis=0, keepdims=True)
    std = all_inputs.std(axis=0, keepdims=True) + 1e-6

    # 包一层 Dataset 做归一化
    class NormWrapper(Dataset):
        def __init__(self, base, mean, std):
            self.base = base
            self.mean = mean
            self.std = std

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            seq, act = self.base[idx]
            seq = (seq - self.mean) / self.std
            return seq, act

    norm_dataset = NormWrapper(dataset, mean.astype(np.float32), std.astype(np.float32))
    loader = DataLoader(norm_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(input_dim=9, hidden_dim=args.hidden_dim, num_layers=2, act_dim=3, policy_type=args.policy).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_loss = 0.0
        for batch_seq, batch_act in loader:
            batch_seq = batch_seq.to(device)          # (B, T, 9)
            batch_act = batch_act.to(device)          # (B, 3)
            pred_act = policy(batch_seq)
            loss = loss_fn(pred_act, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_seq.size(0)
        avg_loss = total_loss / len(norm_dataset)
        print(f"Epoch {epoch}/{args.epochs}, loss={avg_loss:.6f}")

    # 保存模型 + 归一化参数 + n_hist
    ckpt = {
        "model_state": policy.state_dict(),
        "input_mean": mean,
        "input_std": std,
        "n_hist": args.n_hist,
        "hidden_dim": args.hidden_dim,
        "policy_type": args.policy,
    }
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    save_name = f"{os.path.splitext(args.save_path)[0]}_{args.policy}_epoch{args.epochs}_lr{args.lr:.0e}.pt"
    torch.save(ckpt, save_name)
    print(f"Saved policy to {save_name}")


# ===============================
# 6. 用 policy 在一条轨迹上 rollout
# ===============================

def simulate_step(state_rel, state_abs, action, dt, lam0, phi0, r0):
    """
    使用和 env 一致的简化动力学，从当前状态 + 动作得到下一状态。
    state_rel: 当前相对状态 [v, λ_rel, φ_rel, r_rel, psi, gamma]
    state_abs: 当前绝对状态 [v, λ, φ, r, psi, gamma]
    返回: (next_state_rel, next_state_abs)
    """
    v, lam_rel, phi_rel, r_rel, psi, gamma = state_rel
    v_abs, lam_abs, phi_abs, r_abs, psi_abs, gamma_abs = state_abs

    # 动作缩放
    a0, a1, a2 = action
    V_change = a0 * 5.0
    psi_change = a1 * 0.005
    gamma_change = a2 * 0.005

    new_v = np.clip(v + V_change, 100.0, 10000.0)
    new_psi = psi + psi_change
    new_gamma = gamma + gamma_change
    new_gamma = np.clip(new_gamma, -0.3, 0.1)

    horizontal_V = new_v * np.cos(new_gamma)
    vertical_V = new_v * np.sin(new_gamma)

    abs_phi = phi_rel + phi0
    abs_r = r_rel + r0

    dlambda = horizontal_V * dt * np.sin(new_psi) / (abs_r * np.cos(abs_phi))
    dphi = horizontal_V * dt * np.cos(new_psi) / abs_r
    dr = vertical_V * dt

    new_lam_rel = lam_rel + dlambda
    new_phi_rel = phi_rel + dphi
    new_r_rel = r_rel + dr

    new_state_rel = np.array(
        [new_v, new_lam_rel, new_phi_rel, new_r_rel, new_psi, new_gamma],
        dtype=np.float32,
    )
    new_state_abs = np.array(
        [
            new_v,
            new_lam_rel + lam0,
            new_phi_rel + phi0,
            new_r_rel + r0,
            new_psi,
            new_gamma,
        ],
        dtype=np.float32,
    )
    return new_state_rel, new_state_abs


def test_policy(args):
    # 读取模型
    ckpt = torch.load(args.model_path, map_location="cpu")
    n_hist = ckpt["n_hist"]
    input_mean = ckpt["input_mean"].astype(np.float32)
    input_std = ckpt["input_std"].astype(np.float32)
    hidden_dim = ckpt["hidden_dim"]
    policy_type = ckpt.get("policy_type", args.policy)

    policy = Policy(input_dim=9, hidden_dim=hidden_dim, num_layers=2, act_dim=3, policy_type=policy_type)
    policy.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    # 读取测试轨迹
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

    # 构造预测轨迹 (绝对状态)
    pred_abs = np.zeros_like(states_abs)
    pred_abs[: learn_start + 1] = states_abs[: learn_start + 1].copy()  # 直接拷贝前段
    pred_rel = np.zeros_like(states_rel)
    pred_rel[: learn_start + 1] = states_rel[: learn_start + 1].copy()

    # 维护一个历史窗口（从真实段开始）
    hist_states = [states_rel[i].copy() for i in range(learn_start + 1 - n_hist, learn_start + 1)]
    # 若不足 n_hist，前面用第一帧填充
    if learn_start + 1 < n_hist:
        pad_num = n_hist - (learn_start + 1)
        pad = [states_rel[0].copy()] * pad_num
        hist_states = pad + hist_states
    hist_states = hist_states[-n_hist:]

    for t_idx in range(learn_start, T - 1):
        current_rel = pred_rel[t_idx]
        current_abs = pred_abs[t_idx]

        # 更新历史：用当前预测状态
        hist_states.append(current_rel.copy())
        if len(hist_states) > n_hist:
            hist_states.pop(0)

        hist_arr = np.stack(hist_states, axis=0)  # (n_hist, 6)
        goal_tile = np.repeat(goal_rel[None, :], n_hist, axis=0)
        seq_input = np.concatenate([hist_arr, goal_tile], axis=-1)  # (n_hist, 9)

        seq_norm = (seq_input - input_mean) / input_std
        seq_tensor = torch.from_numpy(seq_norm[None, ...]).float().to(device)
        with torch.no_grad():
            action_tensor = policy(seq_tensor)
        action = action_tensor.cpu().numpy()[0]  # (3,)

        # 用简化动力学滚一步
        next_rel, next_abs = simulate_step(current_rel, current_abs, action, dt, lam0, phi0, r0)
        pred_rel[t_idx + 1] = next_rel
        pred_abs[t_idx + 1] = next_abs

    # ============ 画图对比 ============
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
    plt.show()
    fig.savefig("./results/rnn_policy.png")

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
    save_path = "./results/rnn_policy_3d.png"
    fig.savefig(save_path)

    # 打印终点误差
    final_err = np.linalg.norm(pred_abs[-1, 1:4] - states_abs[-1, 1:4])
    print(f"Final endpoint error (λ,φ,r Euclidean) = {final_err:.3f}")


# ===============================
# 7. CLI
# ===============================

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--csv_path", type=str, default="/home/star/helong/repos/drones/drones-MetaActions-new/data/raw_data1025/",
                         help="Path to training trajectory CSV file")
    p_train.add_argument("--save_path", type=str, default="./ckpt/glider_policy/glider_policy.pt")
    p_train.add_argument("--n_hist", type=int, default=10)
    p_train.add_argument("--hidden_dim", type=int, default=128)
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--policy", type=str, default="gru", choices=["gru", "lstm"], help="Policy type: gru or lstm")

    # test
    p_test = subparsers.add_parser("test")
    p_test.add_argument("--model_path", type=str, required=True)
    p_test.add_argument("--traj_csv", type=str, required=True)
    p_test.add_argument("--n_hist", type=int, default=10)
    p_test.add_argument("--policy", type=str, default="gru", choices=["gru", "lstm"], help="Policy type: gru or lstm")

    args = parser.parse_args()

    if args.cmd == "train":
        train_policy(args)
    elif args.cmd == "test":
        test_policy(args)


if __name__ == "__main__":
    main()
