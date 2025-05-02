import numpy as np
import matplotlib.pyplot as plt
import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import argparse
import matplotlib
from drone_env_IL import DroneEnvIL

matplotlib.use('Agg')
np.random.seed(42)
torch.manual_seed(42)

class FeatureExpectation:
    """特征期望计算类，用于MaxEnt IRL算法"""
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.reset()
        
    def reset(self):
        self.feature_expectations = np.zeros(self.feature_dim)
        self.count = 0
        
    def update(self, features, gamma=0.99):
        """更新特征期望"""
        for i, feature in enumerate(features):
            # 使用衰减因子累加特征
            self.feature_expectations += (gamma ** i) * feature
            
        self.count += 1
        
    def get(self):
        """获取平均特征期望"""
        if self.count == 0:
            return self.feature_expectations
        return self.feature_expectations / self.count


class RewardNetwork(nn.Module):
    """奖励函数网络，将状态特征映射到奖励值"""
    def __init__(self, input_dim):
        super(RewardNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.model(x)


class CustomRewardCallback(BaseCallback):
    """自定义奖励回调函数，用于IRL训练"""
    def __init__(self, reward_net, feature_expectations, verbose=0):
        super(CustomRewardCallback, self).__init__(verbose)
        self.reward_net = reward_net
        self.feature_expectations = feature_expectations
        self.rewards = []
        self.features_buffer = []
        
    def _on_step(self):
        # 获取当前状态特征
        features = self.locals['infos'][0]['features']
        # 累积特征以计算特征期望
        self.features_buffer.append(features)
        
        # 使用奖励网络预测奖励
        with torch.no_grad():
            reward = self.reward_net(torch.FloatTensor(features)).item()
            
        self.rewards.append(reward)
        return True
    
    def on_rollout_end(self):
        # 在rollout结束时更新特征期望
        if self.features_buffer:
            self.feature_expectations.update(self.features_buffer)
            self.features_buffer = []


class TrainingCallback(BaseCallback):
    """训练回调函数，用于记录训练过程"""
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.rewards = []
        self.position_errors = []
        self.imitation_rewards = []
        
    def _on_step(self):
        info = self.locals['infos'][0]
        self.rewards.append(self.locals['rewards'][0])
        self.position_errors.append(info['position_error'])
        self.imitation_rewards.append(info['imitation_reward'])
        return True


def maxent_irl(env, expert_feature_expectations, learning_rate=0.01, n_iters=10, 
              gamma=0.99, n_samples=100, feature_dim=10):
    """
    最大熵逆强化学习算法
    Args:
        env: 环境
        expert_feature_expectations: 专家轨迹的特征期望
        learning_rate: 学习率
        n_iters: 迭代次数
        gamma: 折扣因子
        n_samples: 每次迭代的采样数
        feature_dim: 特征维度
    Returns:
        reward_net: 训练好的奖励网络
    """
    # 初始化奖励网络
    reward_net = RewardNetwork(feature_dim)
    optimizer = optim.Adam(reward_net.parameters(), lr=learning_rate)
    
    # 初始化策略模型
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]
    )
    
    policy = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=gamma,
        policy_kwargs=policy_kwargs
    )
    
    # 训练迭代
    for i in range(n_iters):
        print(f"\nIRL迭代 {i+1}/{n_iters}")
        
        # 使用当前奖励函数进行策略优化
        feature_expectations = FeatureExpectation(feature_dim)
        callback = CustomRewardCallback(reward_net, feature_expectations)
        
        policy.learn(total_timesteps=n_samples, callback=callback)
        
        # 计算梯度
        learner_feature_expectations = feature_expectations.get()
        gradient = expert_feature_expectations - learner_feature_expectations
        
        # 更新奖励函数
        optimizer.zero_grad()
        
        # 计算特征的加权和
        weights = torch.nn.Parameter(torch.FloatTensor(gradient))
        features_tensor = torch.FloatTensor(np.eye(feature_dim))
        rewards = reward_net(features_tensor)
        
        loss = -torch.mean(weights * rewards.squeeze())
        loss.backward()
        optimizer.step()
        
        print(f"  Loss: {loss.item():.4f}")
        print(f"  特征期望差距: {np.linalg.norm(gradient):.4f}")
        
    return reward_net


def train_with_learned_reward(env, reward_net, total_timesteps=1000, log_dir="./drone_il_tensorboard"):
    """
    使用学习到的奖励函数训练策略
    Args:
        env: 环境
        reward_net: 训练好的奖励网络
        total_timesteps: 训练步数
        log_dir: tensorboard日志目录
    Returns:
        model: 训练好的策略模型
    """
    # 创建自定义奖励环境包装器
    class CustomRewardEnvWrapper(gym.Wrapper):
        def __init__(self, env, reward_net):
            super(CustomRewardEnvWrapper, self).__init__(env)
            self.reward_net = reward_net
            
        def step(self, action):
            obs, _, done, info = self.env.step(action)
            # 使用奖励网络计算奖励
            with torch.no_grad():
                reward = self.reward_net(torch.FloatTensor(info['features'])).item()
            return obs, reward, done, info
    
    # 包装环境
    wrapped_env = CustomRewardEnvWrapper(env, reward_net)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
    # 初始化策略模型
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]
    )
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir
    )
    
    # 训练模型
    callback = TrainingCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # 绘制训练曲线
    plot_training_curves(callback)
    
    return model


def plot_training_curves(callback, save_dir="./visualize/il_training_plots"):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # 绘制奖励曲线
    plt.subplot(1, 3, 1)
    plt.plot(callback.rewards)
    plt.title('奖励')
    plt.xlabel('步数')
    plt.ylabel('奖励值')
    
    # 绘制位置误差曲线
    plt.subplot(1, 3, 2)
    plt.plot(callback.position_errors)
    plt.title('位置误差')
    plt.xlabel('步数')
    plt.ylabel('误差值')
    
    # 绘制模仿奖励曲线
    plt.subplot(1, 3, 3)
    plt.plot(callback.imitation_rewards)
    plt.title('模仿奖励')
    plt.xlabel('步数')
    plt.ylabel('奖励值')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


def get_expert_feature_expectations(env, samples=100, trajs_per_sample=5, traj_length=None, gamma=0.99):
    """获取专家轨迹的特征期望"""
    print("计算专家轨迹的特征期望...")
    
    feature_dim = 10  # 特征维度
    expert_fe = FeatureExpectation(feature_dim)
    
    for i in range(samples):
        start_idx = np.random.randint(0, len(env.X) - (traj_length or env.max_steps) - 1)
        
        for j in range(trajs_per_sample):
            # 获取专家轨迹片段
            expert_traj = env.get_expert_trajectory(start_idx=start_idx, length=traj_length)
            
            # 提取特征
            features = [traj[3] for traj in expert_traj]
            
            # 更新特征期望
            expert_fe.update(features, gamma)
        
        if i % 10 == 0:
            print(f"  已处理 {i}/{samples} 个样本")
    
    return expert_fe.get()


def evaluate_model(model, env, num_episodes=10):
    """评估模型性能"""
    print("\n评估模型性能...")
    
    position_errors = []
    imitation_rewards = []
    
    for i in range(num_episodes):
        done = False
        obs = env.reset()
        
        episode_position_errors = []
        episode_imitation_rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            
            episode_position_errors.append(info['position_error'])
            episode_imitation_rewards.append(info['imitation_reward'])
        
        position_errors.append(np.mean(episode_position_errors))
        imitation_rewards.append(np.mean(episode_imitation_rewards))
        
        print(f"  Episode {i+1}: 平均位置误差 = {position_errors[-1]:.4f}, 平均模仿奖励 = {imitation_rewards[-1]:.4f}")
    
    print(f"总评估结果: 平均位置误差 = {np.mean(position_errors):.4f}, 平均模仿奖励 = {np.mean(imitation_rewards):.4f}")
    return np.mean(position_errors), np.mean(imitation_rewards)


def main(args):
    # 设置CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
        print(f"使用CUDA设备 {args.cuda}")
    
    # 创建保存目录
    os.makedirs("./ckpt/il_models", exist_ok=True)
    os.makedirs("./visualize/il_training_plots", exist_ok=True)
    
    # 加载数据
    print("加载训练数据...")
    X_train = np.load(f'./data/processed_data/output{args.trace_id}-ode1_X.npy')
    y_train = np.load(f'./data/processed_data/output{args.trace_id}-ode1_y.npy')
    print(f"训练数据大小: {len(X_train)}")
    
    # 创建环境
    env = DroneEnvIL(X_train, y_train, dt=1, n_obs=args.n_obs, max_steps=args.max_steps)
    
    # 获取专家轨迹的特征期望
    expert_feature_expectations = get_expert_feature_expectations(
        env, 
        samples=args.fe_samples, 
        trajs_per_sample=args.trajs_per_sample,
        traj_length=args.traj_length,
        gamma=args.gamma
    )
    
    # 使用MaxEnt IRL算法学习奖励函数
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./drone_il_tensorboard/{run_id}"
    
    print("\n开始MaxEnt IRL训练...")
    reward_net = maxent_irl(
        env,
        expert_feature_expectations,
        learning_rate=args.lr,
        n_iters=args.irl_iters,
        gamma=args.gamma,
        n_samples=args.irl_samples,
        feature_dim=10
    )
    
    # 保存奖励网络
    torch.save(reward_net.state_dict(), f"./ckpt/il_models/reward_net_{run_id}.pt")
    
    # 使用学习到的奖励函数训练策略
    print("\n使用学习到的奖励函数训练策略...")
    model = train_with_learned_reward(
        env,
        reward_net,
        total_timesteps=args.policy_timesteps,
        log_dir=log_dir
    )
    
    # 保存模型
    model.save(f"./ckpt/il_models/drone_il_model_{run_id}")
    
    # 评估模型
    evaluate_model(model, env, num_episodes=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于MaxEnt IRL的无人机模仿学习训练")
    parser.add_argument('--cuda', type=int, default=0, help='CUDA设备编号')
    parser.add_argument('--n_obs', type=int, default=5, help='历史观测窗口大小')
    parser.add_argument('--max_steps', type=int, default=100, help='每个episode的最大步数')
    parser.add_argument('--trace_id', type=int, default=1, help='使用哪条轨迹数据(1-8)')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--lr', type=float, default=0.01, help='IRL学习率')
    parser.add_argument('--irl_iters', type=int, default=10, help='IRL迭代次数')
    parser.add_argument('--irl_samples', type=int, default=2000, help='每次IRL迭代的样本数')
    parser.add_argument('--fe_samples', type=int, default=100, help='计算特征期望的样本数')
    parser.add_argument('--trajs_per_sample', type=int, default=5, help='每个样本的轨迹数')
    parser.add_argument('--traj_length', type=int, default=None, help='每条轨迹的长度')
    parser.add_argument('--policy_timesteps', type=int, default=100000, help='策略训练总步数')
    
    args = parser.parse_args()
    main(args) 