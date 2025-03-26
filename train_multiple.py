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


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.reset_stats()
    
    def reset_stats(self):
        """每读取一条新的训练轨迹，重置所有统计数据"""
        self.rewards = []
        self.lambda_errors = []
        self.phi_errors = []
        self.r_errors = []
        self.policy_losses = []
        self.value_losses = []
        self.temp_lambda_errors = []
        self.temp_phi_errors = []
        self.temp_r_errors = []
        self.temp_rewards = []

    def _on_step(self):
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if 'lambda_error' in info:
                # 每个step数据存储到临时列表
                self.temp_lambda_errors.append(info['lambda_error'])
                self.temp_phi_errors.append(info['phi_error'])
                self.temp_r_errors.append(info['r_error'])
                if 'reward' in info:
                    self.temp_rewards.append(info['reward'])

        # 每n_steps记录一次平均值
        if len(self.temp_lambda_errors) >= self.model.n_steps:
            avg_lambda = np.mean(self.temp_lambda_errors)
            avg_phi = np.mean(self.temp_phi_errors)
            avg_r = np.mean(self.temp_r_errors)
            avg_reward = np.mean(self.temp_rewards) if self.temp_rewards else 0

            self.lambda_errors.append(avg_lambda)
            self.phi_errors.append(avg_phi)
            self.r_errors.append(avg_r)
            self.rewards.append(avg_reward)

            self.logger.record("custom/lambda_error", avg_lambda)
            self.logger.record("custom/phi_error", avg_phi)
            self.logger.record("custom/r_error", avg_r)
            self.logger.record("custom/reward", avg_reward)

            self.temp_lambda_errors = []
            self.temp_phi_errors = []
            self.temp_r_errors = []
            self.temp_rewards = []
            
            # 获取SB3记录的 policy / value loss
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logs = self.model.logger.name_to_value
                if 'train/policy_gradient_loss' in logs:
                    self.policy_losses.append(logs['train/policy_gradient_loss'])
                if 'train/value_loss' in logs:
                    self.value_losses.append(logs['train/value_loss'])

            self.logger.dump(step=self.num_timesteps)
        
        return True

callback = TrainingCallback()


run_id = datetime.now().strftime("%Y%m%d-%H%M%S")


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


def train(cuda_id=0, model_save_path=None):
    """
    训练函数
    Args:
        cuda_id (int): GPU设备编号
        model_save_path (str): 模型保存路径
    """
    
    X_train = np.load('./data/processed_data/output1-ode1_X.npy')
    y_train = np.load('./data/processed_data/output1-ode1_y.npy')
    env = DummyVecEnv([lambda: DroneEnv(X_train, y_train, dt=1)])
    
    if model_save_path is None:
        model_save_path = run_id
        
    # 设置CUDA设备
    device = f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(cuda_id)
    print(f"\n{'='*50}")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(cuda_id)}")
        # print(f"当前GPU编号: {torch.cuda.current_device()}")
    print(f"{'='*50}\n")
    
    model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=f"./drone_ppo_tensorboard/{run_id}",
    device=device,
    **params
)
    
    for i in range(1, 9):
        X_train = np.load(f'./data/processed_data/output{i}-ode1_X.npy')
        y_train = np.load(f'./data/processed_data/output{i}-ode1_y.npy')
        print(f"\n训练数据集 {i}, 训练集大小:", X_train.shape[0])
        
        env = DummyVecEnv([lambda: DroneEnv(X_train, y_train, dt=1)])
        model.set_env(env)
        
        # 确保环境中的张量也在正确的设备上
        if hasattr(env.envs[0], 'to'):
            env.envs[0].to(device)
        
        callback.reset_stats()
        
        model.learn(
            total_timesteps=100000, 
            callback=callback, 
            reset_num_timesteps=False,
        )
        
        # 保存模型到指定路径
        save_path = os.path.join("./ckpt", model_save_path)
        os.makedirs(save_path, exist_ok=True)
        model.save(os.path.join(save_path, f"model_ppo_trace{i}"))
        
        plot_training_curves(callback, './visualize/training_plots', trace_num=i)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='无人机轨迹跟踪PPO训练程序')
    parser.add_argument('--cuda', type=int, default=0,
                      help='GPU设备编号 (默认: 0)')
    parser.add_argument('--model_save_path', type=str, default=None,
                      help='模型保存路径 (默认: 使用时间戳)')
    
    args = parser.parse_args()
    
    train(cuda_id=args.cuda, model_save_path=args.model_save_path)