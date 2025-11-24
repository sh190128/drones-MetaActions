import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_env import DroneEnv
import matplotlib
from datetime import datetime
from utils import plot_training_curves
import json
import os
import argparse
import torch

matplotlib.use('Agg')
np.random.seed(42)

# 新的数据目录（你的新路径）
DATA_DIR = "/home/star/helong/repos/drones/drones-MetaActions-new/data/processed_data"


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.reset_stats()
    
    def reset_stats(self):
        """每读取一条新的训练轨迹，重置所有统计数据"""
        self.rewards = []
        self.temp_rewards = []
        self.lambda_errors = []
        self.phi_errors = []
        self.r_errors = []
        self.policy_losses = []
        self.value_losses = []
        self.temp_lambda_errors = []
        self.temp_phi_errors = []
        self.temp_r_errors = []
        self.temp_next_state_errors = []

        self.distance_rewards = []
        self.temp_distance_rewards = []

        self.temp_end_rewards = []
    
    def _on_step(self):
        """每一步调用一次"""
        if self.locals is not None and 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]  # 获取第一个环境的信息
            self.temp_lambda_errors.append(info['lambda_error'])
            self.temp_phi_errors.append(info['phi_error'])
            self.temp_r_errors.append(info['r_error'])
            self.temp_rewards.append(info['reward'])
            self.temp_next_state_errors.append(info['next_state_error'])
            
            self.temp_distance_rewards.append(info['distance_reward'])
            self.temp_end_rewards.append(info['end_reward'])
            
            self.logger.record("train/reward", info['reward'])
            self.logger.record("train/lambda_error", info['lambda_error'])
            self.logger.record("train/phi_error", info['phi_error'])
            self.logger.record("train/r_error", info['r_error'])
            self.logger.record("train/next_state_error", info['next_state_error'])
            self.logger.record("train/distance_reward", info['distance_reward'])
            self.logger.record("train/end_reward", info['end_reward'])

        return True
    
    def on_rollout_end(self):
        """一个rollout结束后调用"""
        if len(self.temp_rewards) > 0:
            self.rewards.append(np.mean(self.temp_rewards))
            self.lambda_errors.append(np.mean(self.temp_lambda_errors))
            self.phi_errors.append(np.mean(self.temp_phi_errors))
            self.r_errors.append(np.mean(self.temp_r_errors))
            self.distance_rewards.append(np.mean(self.temp_distance_rewards))
            
            self.logger.record("rollout/avg_reward", np.mean(self.temp_rewards))
            self.logger.record("rollout/avg_lambda_error", np.mean(self.temp_lambda_errors))
            self.logger.record("rollout/avg_phi_error", np.mean(self.temp_phi_errors))
            self.logger.record("rollout/avg_r_error", np.mean(self.temp_r_errors))
            self.logger.record("rollout/avg_next_state_error", np.mean(self.temp_next_state_errors))
            self.logger.record("rollout/avg_distance_reward", np.mean(self.temp_distance_rewards))
            self.logger.record("rollout/avg_end_reward", np.mean(self.temp_end_rewards))
            
            self.temp_lambda_errors = []
            self.temp_phi_errors = []
            self.temp_r_errors = []
            self.temp_rewards = []
            self.temp_distance_rewards = []

            self.reset_stats()
            self.logger.dump(self.num_timesteps)
        return True


callback = TrainingCallback()

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

# 尝试加载最优参数
try:
    with open("optimization_results/best_parameters.json", "r") as f:
        params = json.load(f)
    print("Successfully loaded best training parameters:", params)
except FileNotFoundError: 
    print("WARNING: Best parameters file not found, using default training parameters")
    params = {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.5,
        "ent_coef": 0.01,
        "vf_coef": 0.5
    }


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


def train(cuda_id=0, model_save_path=None, n_obs=5, max_steps=100, dt=1.0):
    """
    Args:
        cuda_id (int): GPU设备编号
        model_save_path (str): 模型保存路径
        n_obs (int): 历史观测窗口大小
        max_steps (int): 每条轨迹的最大步数
        dt (float): 时间步长（建议与你数据中 time(s) 的间隔一致）
    """
    traj_pairs = find_traj_files(DATA_DIR)
    if len(traj_pairs) == 0:
        raise RuntimeError(f"No trajectory files found in {DATA_DIR}")

    print(f"在 {DATA_DIR} 中找到 {len(traj_pairs)} 条轨迹数据。")

    # 先用第一条轨迹初始化 env 和 model
    first_x, first_y = traj_pairs[0]
    X_train = np.load(first_x)
    y_train = np.load(first_y)

    print(f"初始化环境使用轨迹: {os.path.basename(first_x)} (X shape={X_train.shape}, y shape={y_train.shape})")

    env = DummyVecEnv([lambda: DroneEnv(X_train, y_train, dt=dt, n_obs=n_obs, max_steps=max_steps)])
    
    if model_save_path is None:
        model_save_path = run_id
        
    # 设置设备（你之前强制用CPU，这里沿用）
    device = "cpu"
    print(f"\n{'='*50}")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(cuda_id)}")
    print(f"历史观测窗口大小: {n_obs}")
    print(f"最大步数: {max_steps}")
    print(f"{'='*50}\n")
    
    # 单个状态6维，历史n_obs个状态，终点3维，步数比例1维
    obs_dim = 6 * n_obs + 3 + 1
    print(f"观测空间理论维度（用于检查）: {obs_dim}")

    # 调整网络架构以适应更大的观测空间
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./drone_ppo_tensorboard/{run_id}",
        device=device,
        policy_kwargs=policy_kwargs,
        **params
    )
    
    # 遍历每一条轨迹进行训练
    for trace_idx, (x_path, y_path) in enumerate(traj_pairs, start=1):
        X_train = np.load(x_path)
        y_train = np.load(y_path)
        print(f"\n训练数据集 {trace_idx}:")
        print(f"  X: {os.path.basename(x_path)}, shape={X_train.shape}")
        print(f"  y: {os.path.basename(y_path)}, shape={y_train.shape}")
        
        env = DummyVecEnv([lambda X=X_train, Y=y_train: DroneEnv(X, Y, dt=dt, n_obs=n_obs, max_steps=max_steps)])
        model.set_env(env)
        
        # 如果以后你在 DroneEnv 里实现 .to(device) 再启用这一段
        if hasattr(env.envs[0], 'to'):
            env.envs[0].to(device)
        
        callback.reset_stats()
        
        model.learn(
            total_timesteps=50000, 
            callback=callback, 
            reset_num_timesteps=False,
        )
        
        save_dir = os.path.join("./ckpt", model_save_path)
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, f"model_ppo_trace{trace_idx}"))
        
        plot_training_curves(callback, './visualize/new_training_plots', trace_num=trace_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练无人机轨迹规划模型')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA设备编号')
    parser.add_argument('--save_path', type=str, default=None, help='模型保存路径名')
    parser.add_argument('--n_obs', type=int, default=10, help='历史观测窗口大小')
    parser.add_argument('--max_steps', type=int, default=300, help='每条轨迹的最大步数')
    parser.add_argument('--dt', type=float, default=1.0, help='时间步长，与time(s)间隔对应')
    
    args = parser.parse_args()
    
    train(
        cuda_id=args.cuda,
        model_save_path=args.save_path,
        n_obs=args.n_obs,
        max_steps=args.max_steps,
        dt=args.dt
    )
