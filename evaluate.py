import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from drone_env import DroneEnv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import argparse

matplotlib.use('Agg')
np.random.seed(42)

# 你的新数据目录
DATA_DIR = "/home/star/helong/repos/drones/drones-MetaActions-new/data/processed_data"


def find_traj_files(data_dir):
    """
    在指定目录中查找所有 *_X.npy 和对应的 *_y.npy 轨迹文件。
    返回: [(x_path, y_path), ...] （按文件名排序）
    """
    all_files = os.listdir(data_dir)
    x_files = sorted([f for f in all_files if f.endswith("_X.npy")])

    traj_pairs = []
    for x_file in x_files:
        x_path = os.path.join(data_dir, x_file)
        y_file = x_file.replace("_X.npy", "_y.npy")
        y_path = os.path.join(data_dir, y_file)
        if os.path.exists(y_path):
            traj_pairs.append((x_path, y_path))
        else:
            print(f"WARNING: y file not found for {x_file}, expected: {y_file}, skip this trajectory.")
    return traj_pairs


def evaluate_model(model, env, simulate=False, num_steps=None):
    """
    评估模型：
    - simulate=False: 一步预测 (per-step reset)，用于看“对下一时刻位置的拟合情况”
    - simulate=True : 从当前 env.current_episode 开始闭环模拟轨迹
    """

    if num_steps is None:
        num_steps = env.trajectory_length

    assert len(env.X) == len(env.y)
    predictions = []
    targets = []

    # 方便使用：起点的绝对坐标偏移
    offset = np.array([env.longitude_start, env.latitude_start, env.r_start], dtype=np.float32)

    if not simulate:
        # 一步预测：对每个时间步 step，reset 到该步，然后只向前滚动一小步
        num_steps = min(num_steps, env.trajectory_length - 1)

        for step in range(num_steps):
            # 让当前 episode 从该索引开始
            env.current_episode = step
            obs = env.reset()

            # 目标位置 = y[step] (绝对坐标: [lambda_next, phi_next, r_next])
            target_abs = env.y[step].copy()

            action, _ = model.predict(obs, deterministic=True)
            obs_next, _, done, _ = env.step(action)

            # 预测位置：state 中是相对坐标，需要加回起点偏移
            pred_abs = env.state[1:4] + offset

            predictions.append(pred_abs)
            targets.append(target_abs)

    else:
        # 闭环模拟：不再每步 reset，而是从当前 env.current_episode 持续滚动
        # 常见做法：从 0 开始，也可以在外面先设 env.current_episode = start_obs 再传进来
        obs = env.reset()
        num_steps = min(num_steps, env.trajectory_length - 1)

        for t in range(num_steps):
            # 这一步要拟合的目标位置：从当前 idx 出发的“下一时刻绝对位置”
            idx = env.current_episode + env.current_step
            if idx >= env.trajectory_length:
                break

            target_abs = env.y[idx].copy()

            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            pred_abs = env.state[1:4] + offset

            predictions.append(pred_abs)
            targets.append(target_abs)


    return np.array(predictions), np.array(targets)


def plot_trajectories(predictions, targets, save_prefix="prediction_results"):
    os.makedirs("./visualize/testing_plots", exist_ok=True)

    # ---- 1D 曲线对比 ----
    plt.figure(figsize=(15, 18))

    # 经度
    plt.subplot(3, 1, 1)
    plt.plot(predictions[:, 0], 'b-', label='Predicted')
    plt.plot(targets[:, 0], 'r--', label='Target')
    plt.title('Longitude')
    plt.xlabel('Time Step')
    plt.ylabel('Longitude')
    plt.legend()

    # 纬度
    plt.subplot(3, 1, 2)
    plt.plot(predictions[:, 1], 'b-', label='Predicted')
    plt.plot(targets[:, 1], 'r--', label='Target')
    plt.title('Latitude')
    plt.xlabel('Time Step')
    plt.ylabel('Latitude')
    plt.legend()

    # 地心距
    plt.subplot(3, 1, 3)
    plt.plot(predictions[:, 2], 'b-', label='Predicted')
    plt.plot(targets[:, 2], 'r--', label='Target')
    plt.title('Earth Center Distance')
    plt.xlabel('Time Step')
    plt.ylabel('Earth Center Distance')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./visualize/testing_plots/{save_prefix}_1d.png')
    plt.close()

    # ---- 3D 轨迹对比 ----
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], 'b-', label='Predicted')
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], 'r--', label='Target')
    ax.set_title('3D Trajectory')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Radius')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'./visualize/testing_plots/{save_prefix}_3d.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true', help='是否闭环仿真完整轨迹')
    parser.add_argument('--cuda', type=int, default=0, help='GPU设备编号 (默认: 0)')
    parser.add_argument('--n_obs', type=int, default=10, help='历史观测窗口大小 (需与训练时一致)')
    parser.add_argument('--max_steps', type=int, default=100, help='环境内最大步数 (需与训练时一致)')
    parser.add_argument('--dt', type=float, default=1.0, help='时间步长，与time(s)间隔对应')
    parser.add_argument('--traj_index', type=int, default=-1,
                        help='选择第几条轨迹作为测试集 (按文件名排序, -1 表示最后一条)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='PPO模型路径，如 ./ckpt/20250702-125155/model_ppo_trace8.zip')
    args = parser.parse_args()

    # 设置CUDA设备（如果你训练在CPU，可以忽略）
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
        print(f"使用 CUDA 设备 {args.cuda}")
        device = f"cuda:{args.cuda}"
    else:
        print("CUDA不可用，使用CPU")
        device = "cpu"

    # 选一条轨迹作为测试集（默认用最后一条）
    traj_pairs = find_traj_files(DATA_DIR)
    if len(traj_pairs) == 0:
        raise RuntimeError(f"No trajectory files found in {DATA_DIR}")

    idx = args.traj_index if args.traj_index >= 0 else (len(traj_pairs) - 1)
    idx = max(0, min(idx, len(traj_pairs) - 1))
    x_path, y_path = traj_pairs[idx]

    print(f"使用测试轨迹: {os.path.basename(x_path)} / {os.path.basename(y_path)}")

    X_test = np.load(x_path)
    y_test = np.load(y_path)

    test_env = DroneEnv(X_test, y_test, dt=args.dt, test=True,
                        n_obs=args.n_obs, max_steps=args.max_steps)

    print(f"加载模型: {args.model_path}")
    model = PPO.load(args.model_path, device=device)

    predictions, targets = evaluate_model(
        model,
        test_env,
        simulate=args.simulate,
        num_steps=X_test.shape[0]
    )

    # 简单打印几个评价指标（可选）
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")

    # 画图
    base_name = os.path.splitext(os.path.basename(x_path))[0]
    save_prefix = f"{base_name}_{'simulate' if args.simulate else 'onestep'}"
    plot_trajectories(predictions, targets, save_prefix=save_prefix)
