import os
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')

import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl_env import GliderRNNEnv
from policy_trainer import Policy 


class RNNFeatureExtractor(BaseFeaturesExtractor):
    """
    使用你 supervised 训练好的 GRU 作为 PPO 的特征提取器：
    - 输入: obs (B, n_hist, 9)
    - 输出: (B, hidden_dim)
    - 内部包含: 标准化 + GRU
    """

    def __init__(self, observation_space, n_hist, ckpt_path, policy_type="gru"):
        # 先加载 checkpoint，拿到 hidden_dim
        ckpt = torch.load(ckpt_path, map_location="cpu")
        hidden_dim = int(ckpt["hidden_dim"])
        n_hist_ckpt = int(ckpt["n_hist"])

        if n_hist_ckpt != n_hist:
            print(f"[WARN] ckpt 中的 n_hist={n_hist_ckpt} 与当前 env 的 n_hist={n_hist} 不一致，"
                  f"请确认是否有问题。这里仍然强行使用 n_hist={n_hist} 的 obs。")

        super().__init__(observation_space, features_dim=hidden_dim)

        # 记录 n_hist & 输入维度
        self.n_hist = n_hist
        self.input_dim = 9
        self.policy_type = ckpt.get("policy_type", policy_type)

        # ====== 1) 输入标准化参数 ======
        input_mean = ckpt["input_mean"].astype(np.float32)  # (1, 9)
        input_std = ckpt["input_std"].astype(np.float32)

        # 用 buffer 存，这样会自动跟着模型到 GPU / CPU
        self.register_buffer("input_mean", torch.from_numpy(input_mean))  # (1, 9)
        self.register_buffer("input_std", torch.from_numpy(input_std))

        # ====== 2) 构建与 supervised 完全相同的 Policy，然后拷贝参数 ======
        self.core_policy = Policy(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            act_dim=3,
            policy_type=self.policy_type,
        )

        # 原始 ckpt 里的整个 state_dict
        pretrained_state = ckpt["model_state"]
        current_state = self.core_policy.state_dict()

        # 只覆盖匹配的键（包括 rnn 和 fc，其实都可以，提取器只用 rnn）
        for k in current_state.keys():
            if k in pretrained_state:
                current_state[k] = pretrained_state[k]
        self.core_policy.load_state_dict(current_state)

        # 取出 RNN 部分，之后 forward 只用它
        self.rnn = self.core_policy.rnn

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (B, n_hist, 9)
        返回: (B, hidden_dim)
        """
        # SB3 给的 obs 是 (B, n_hist, 9)，确保一下 dtype
        x = observations.float()

        # broadcast (1,9) 到 (1,1,9) 再到 (B, n_hist, 9)
        mean = self.input_mean.view(1, 1, -1)
        std = self.input_std.view(1, 1, -1)

        x_norm = (x - mean) / std

        # 直接用 GRU
        out, h = self.rnn(x_norm)   # out: (B, n_hist, hidden_dim)
        last = out[:, -1, :]        # (B, hidden_dim)

        return last


class RNNTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.reset_stats()

    def reset_stats(self):
        self.episode_rewards = []
        self.episode_distances = []
        self.temp_rewards = []
        self.temp_distances = []

    def _on_step(self):
        if self.locals is not None and 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if "reward" in info:
                self.temp_rewards.append(info["reward"])
                self.logger.record("train/reward", info["reward"])
            if "distance_to_goal_m" in info:
                self.temp_distances.append(info["distance_to_goal_m"])
                self.logger.record("train/distance_to_goal_m", info["distance_to_goal_m"])
        return True

    def on_rollout_end(self):
        if len(self.temp_rewards) > 0:
            mean_r = float(np.mean(self.temp_rewards))
            mean_d = float(np.mean(self.temp_distances)) if len(self.temp_distances) > 0 else np.nan

            self.episode_rewards.append(mean_r)
            self.episode_distances.append(mean_d)

            self.logger.record("rollout/avg_reward", mean_r)
            self.logger.record("rollout/avg_distance_to_goal_m", mean_d)

        self.temp_rewards = []
        self.temp_distances = []

        self.logger.dump(self.num_timesteps)
        return True


callback = RNNTrainingCallback()
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")


try:
    with open("optimization_results/best_parameters.json", "r") as f:
        params = json.load(f)
    print("Successfully loaded best training parameters:", params)
except FileNotFoundError:
    print("WARNING: Best parameters file not found, using default training parameters")
    params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    }


CSV_DIR = "/home/star/helong/repos/drones/drones-MetaActions-new/data/raw_data1025/"


def find_csv_files(data_dir):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    csv_paths = [os.path.join(data_dir, f) for f in files]
    if len(csv_paths) == 0:
        raise RuntimeError(f"No CSV trajectory files found in {data_dir}")
    return csv_paths


def train(
    ckpt_path,                # 👈 你 supervised 的 .pt
    cuda_id=0,
    model_save_path=None,
    n_hist=10,
    max_steps=200,
    arrival_threshold_m=1_000.0,
    distance_scale_m=50_000.0,
    imitation_weight=0.0,
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    csv_paths = find_csv_files(CSV_DIR)
    print(f"在 {CSV_DIR} 中找到 {len(csv_paths)} 条 CSV 轨迹数据。")

    # 先用第一条轨迹初始化 env 和 PPO 模型
    first_csv = csv_paths[0]
    print(f"初始化环境使用轨迹: {os.path.basename(first_csv)}")

    def make_env(path):
        return GliderRNNEnv(
            csv_path=path,
            n_hist=n_hist,
            max_steps=max_steps,
            start_from_learn_segment=True,
            arrival_threshold_m=arrival_threshold_m,
            distance_scale_m=distance_scale_m,
            imitation_weight=imitation_weight,
        )

    env = DummyVecEnv([lambda: make_env(first_csv)])

    if model_save_path is None:
        model_save_path = run_id

    device = "cpu"  # 如果后面想用 GPU，可以改成 f"cuda:{cuda_id}"
    print(f"\n{'='*50}")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"检测到 GPU: {torch.cuda.get_device_name(cuda_id)}")
    print(f"历史窗口 n_hist: {n_hist}")
    print(f"max_steps: {max_steps}")
    print(f"{'='*50}\n")

    obs_dim = n_hist * 9
    print(f"观测空间 flatten 后维度（用于检查）: {obs_dim}")

    # 这里的 net_arch 只作用于 PPO 顶层的 policy/value MLP，
    # GRU 部分在 RNNFeatureExtractor 里已经来自你的 checkpoint。
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])],
        features_extractor_class=RNNFeatureExtractor,
        features_extractor_kwargs=dict(
            n_hist=n_hist,
            ckpt_path=ckpt_path,
            policy_type="gru",   # 或从 ckpt 中读
        ),
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./glider_rnn_ppo_ft_tensorboard/{run_id}",
        device=device,
        policy_kwargs=policy_kwargs,
        **params,
    )

    # ========= 遍历所有轨迹做 finetune =========
    total_traces = len(csv_paths)
    for trace_idx, csv_path in enumerate(csv_paths, start=1):
        print(f"\n微调轨迹 {trace_idx}/{total_traces}: {os.path.basename(csv_path)}")

        env = DummyVecEnv([lambda p=csv_path: make_env(p)])
        model.set_env(env)

        callback.reset_stats()

        # 每条轨迹上跑固定 step 数，可以按需要调
        model.learn(
            total_timesteps=10_000,
            callback=callback,
            reset_num_timesteps=False,
        )

        save_dir = os.path.join("./ckpt/rnn_ppo_finetune", model_save_path)
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, f"ppo_glider_rnn_trace{trace_idx}"))

    final_path = os.path.join("./ckpt/rnn_ppo_finetune", model_save_path, "ppo_glider_rnn_final")
    model.save(final_path)
    print(f"\nRL 微调结束，最终模型已保存至: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 PPO 对 supervised 训练好的 GRU Policy 进行 RL 微调")
    parser.add_argument("--ckpt_path", type=str, required=True, help="supervised 训练得到的 .pt checkpoint 路径")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA 设备编号（如果使用 GPU）")
    parser.add_argument("--save_path", type=str, default=None, help="模型保存目录名")
    parser.add_argument("--n_hist", type=int, default=10, help="历史窗口长度 n_hist（需与 ckpt 大致一致）")
    parser.add_argument("--max_steps", type=int, default=200, help="每个 episode 最大步数")
    parser.add_argument("--arrival_threshold", type=float, default=1_000.0, help="判定到达终点的距离阈值 (m)")
    parser.add_argument("--distance_scale", type=float, default=50_000.0, help="距离 shaping 的尺度 (m)")
    parser.add_argument("--imitation_weight", type=float, default=0.0, help="模仿原始轨迹的权重（0 表示不模仿）")

    args = parser.parse_args()

    train(
        ckpt_path=args.ckpt_path,
        cuda_id=args.cuda,
        model_save_path=args.save_path,
        n_hist=args.n_hist,
        max_steps=args.max_steps,
        arrival_threshold_m=args.arrival_threshold,
        distance_scale_m=args.distance_scale,
        imitation_weight=args.imitation_weight,
    )
