import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import sys
import os
import time
import datetime
import argparse
import pickle
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import glob
from tqdm import tqdm

from drone_env_IL import DroneEnvIL

def load_trajectory_data(data_dir, num_trajectories=8):
    """
    加载多条轨迹数据
    
    参数:
    data_dir: 数据文件所在目录
    num_trajectories: 要加载的轨迹数量
    
    返回:
    X_list: 包含所有轨迹状态的列表
    """
    X_list = []
    
    # 查找所有匹配的轨迹X文件
    trajectory_files_X = []
    for i in range(1, num_trajectories + 1):
        file_path = os.path.join(data_dir, f"output{i}-ode1_X.npy")
        if os.path.exists(file_path):
            trajectory_files_X.append((i, file_path))
    
    if not trajectory_files_X:
        raise FileNotFoundError(f"在 {data_dir} 中未找到轨迹X数据文件")
    
    print(f"找到 {len(trajectory_files_X)} 条轨迹X数据")
    
    # 加载每条轨迹数据
    for idx, file_path in trajectory_files_X:
        # 加载轨迹X状态
        trajectory_X = np.load(file_path)
        print(f"加载轨迹: {file_path}, 形状: {trajectory_X.shape}")
        
        X_list.append(trajectory_X)
    
    return X_list

class IRLFeatureExtractor(nn.Module):
    """特征提取器，用于MaxEnt IRL算法"""
    def __init__(self, input_dim, hidden_dim=64, feature_dim=32):
        super(IRLFeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class MaxEntIRL:
    """最大熵逆强化学习算法实现"""
    def __init__(self, env, feature_extractor, lr=0.001):
        self.env = env
        self.feature_extractor = feature_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor.to(self.device)
        self.optimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr)
        
        # 奖励权重
        self.reward_weights = nn.Parameter(torch.randn(feature_extractor.net[-1].out_features).to(self.device))
        self.reward_optimizer = optim.Adam([self.reward_weights], lr=lr)
        
    def compute_reward(self, states):
        """计算给定状态的奖励"""
        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states).to(self.device)
        features = self.feature_extractor(states)
        rewards = torch.matmul(features, self.reward_weights)
        return rewards
    
    def compute_expert_feature_expectations(self, demos):
        """计算专家轨迹的特征期望"""
        expert_fe = torch.zeros(self.feature_extractor.net[-1].out_features).to(self.device)
        demo_count = 0
        
        pbar = tqdm(demos, desc="计算专家特征期望")
        for states, _ in pbar:
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            features = self.feature_extractor(states_tensor)
            expert_fe += features.sum(dim=0)
            demo_count += len(states)
            
        return expert_fe / demo_count if demo_count > 0 else expert_fe
    
    def compute_policy_feature_expectations(self, policy, num_episodes=10):
        """计算当前策略的特征期望"""
        policy_fe = torch.zeros(self.feature_extractor.net[-1].out_features).to(self.device)
        total_steps = 0
        
        pbar = tqdm(total=num_episodes, desc="计算策略特征期望")
        for episode in range(num_episodes):
            try:
                obs = self.env.reset()
                done = False
                episode_steps = 0
                max_episode_steps = 500  # 设置单个回合的最大步数
                
                while not done and episode_steps < max_episode_steps:
                    try:
                        action, _ = policy.predict(obs, deterministic=True)
                        obs, _, done, _ = self.env.step(action)
                        
                        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        features = self.feature_extractor(state_tensor)
                        policy_fe += features.squeeze(0)
                        total_steps += 1
                        episode_steps += 1
                    except Exception as e:
                        print(f"步骤执行错误: {e}")
                        break
            except Exception as e:
                print(f"回合 {episode+1} 错误: {e}")
                continue
            pbar.update(1)
        pbar.close()
                
        return policy_fe / max(total_steps, 1)
    
    def train(self, demos, n_iterations=5, policy_train_steps=10000):
        """训练IRL算法并返回学习到的奖励函数和策略"""
        # 计算专家特征期望
        expert_fe = self.compute_expert_feature_expectations(demos)
        
        # 初始化策略（PPO）
        def make_env():
            return lambda: self.env
        
        vec_env = DummyVecEnv([make_env()])
        policy = PPO("MlpPolicy", vec_env, verbose=0)
        
        for iteration in tqdm(range(n_iterations), desc="IRL训练迭代"):
            print(f"IRL Iteration {iteration+1}/{n_iterations}")
            
            # 1. 使用当前奖励函数训练策略
            print("  Training policy...")
            
            # 创建奖励模型回调
            class CustomRewardCallback(BaseCallback):
                def __init__(self, irl_model, verbose=0):
                    super(CustomRewardCallback, self).__init__(verbose)
                    self.irl_model = irl_model
                    self.pbar = None
                
                def _on_training_start(self):
                    self.pbar = tqdm(total=policy_train_steps, desc="训练策略进度")
                
                def _on_training_end(self):
                    if self.pbar is not None:
                        self.pbar.close()
                
                def _on_step(self):
                    # 获取当前观察
                    obs = self.locals['new_obs']
                    # 计算自定义奖励
                    custom_rewards = self.irl_model.compute_reward(obs).detach().cpu().numpy()
                    # 替换环境给出的奖励
                    self.locals['rewards'] = custom_rewards
                    # 更新进度条
                    if self.pbar is not None:
                        self.pbar.update(1)
                    return True
            
            callback = CustomRewardCallback(self)
            policy.learn(total_timesteps=policy_train_steps, callback=callback)
            
            # 2. 计算策略的特征期望
            print("  Computing policy feature expectations...")
            policy_fe = self.compute_policy_feature_expectations(policy)
            
            # 3. 更新奖励权重
            print("  Updating reward weights...")
            feature_diff = expert_fe - policy_fe
            
            # 梯度下降更新权重
            self.reward_optimizer.zero_grad()
            loss = -torch.norm(feature_diff)  # 最大化特征差异
            loss.backward(retain_graph=True)  # 添加retain_graph=True以保留计算图
            self.reward_optimizer.step()
            
            print(f"  Loss: {loss.item()}")
            
            # 评估当前策略
            eval_env = self.env
            mean_reward, std_reward = evaluate_policy(policy, eval_env, n_eval_episodes=5)
            print(f"  Evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        
        return policy

class EvaluationCallback(BaseCallback):
    """用于记录评估结果的回调"""
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, log_path="./logs/", verbose=1):
        super(EvaluationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.best_mean_reward = -np.inf
        self.rewards = []
        self.timesteps = []
        
        # 创建日志目录
        os.makedirs(log_path, exist_ok=True)
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )
            self.rewards.append(mean_reward)
            self.timesteps.append(self.n_calls)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"{self.log_path}/best_model")
                
            # 保存评估结果
            np.savez(
                f"{self.log_path}/eval_results.npz",
                timesteps=np.array(self.timesteps),
                rewards=np.array(self.rewards)
            )
            
            # 绘制学习曲线
            plt.figure(figsize=(10, 6))
            plt.plot(self.timesteps, self.rewards)
            plt.xlabel("Timesteps")
            plt.ylabel("Mean Reward")
            plt.title("Learning Curve")
            plt.savefig(f"{self.log_path}/learning_curve.png")
            plt.close()
            
            if self.verbose > 0:
                print(f"Timestep {self.n_calls} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                
        return True

def train_with_irl(X_list, dt, max_steps=100, n_obs=5, save_path="./models/", trajectory_idx=0):
    """使用IRL训练无人机策略"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 选择一条轨迹作为训练环境的基础
    X = X_list[trajectory_idx]
    
    # 初始化环境
    env = DroneEnvIL(X, dt, max_steps=max_steps, n_obs=n_obs)
    
    # 计算观察空间维度
    obs_dim = env.observation_space.shape[0]
    
    # 初始化特征提取器
    feature_extractor = IRLFeatureExtractor(input_dim=obs_dim, hidden_dim=128, feature_dim=64)
    
    # 初始化IRL算法
    irl = MaxEntIRL(env, feature_extractor, lr=3e-4)
    
    # 获取专家示范数据（从所有轨迹中）
    demos = []
    for i in range(len(X_list)):
        # 为每条轨迹创建一个临时环境以获取示范
        temp_env = DroneEnvIL(X_list[i], dt, max_steps=max_steps, n_obs=n_obs)
        # 获取该轨迹的示范数据
        traj_demos = temp_env.get_expert_demonstration(num_trajectories=3)  # 每条轨迹生成3个示范
        demos.extend(traj_demos)
    
    print(f"共收集了 {len(demos)} 条专家示范轨迹")
    
    # 使用IRL训练
    policy = irl.train(demos, n_iterations=10, policy_train_steps=100000)
    
    # 保存训练后的模型
    policy.save(f"{save_path}/irl_policy")
    
    # 评估训练后的策略
    mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10)
    print(f"Final Evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return policy, irl

def train_with_bc(X_list, dt, max_steps=100, n_obs=5, save_path="./models/"):
    """使用行为克隆（Behavior Cloning）训练无人机策略"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 获取专家示范数据（从所有轨迹中）
    all_states = []
    all_actions = []
    
    for i in range(len(X_list)):
        # 为每条轨迹创建一个临时环境以获取示范
        temp_env = DroneEnvIL(X_list[i], dt, max_steps=max_steps, n_obs=n_obs)
        # 获取该轨迹的示范数据
        traj_demos = temp_env.get_expert_demonstration(num_trajectories=3)  # 每条轨迹生成3个示范
        
        # 收集状态和动作
        for states, actions in traj_demos:
            all_states.extend(states)
            all_actions.extend(actions)
    
    # 转换为numpy数组
    X_train = np.array(all_states)
    y_train = np.array(all_actions)
    
    print(f"行为克隆训练数据: 状态 {X_train.shape}, 动作 {y_train.shape}")
    
    # 选择第一条轨迹作为评估环境
    env = DroneEnvIL(X_list[0], dt, max_steps=max_steps, n_obs=n_obs)
    
    # 定义行为克隆模型
    class BCPolicy(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(BCPolicy, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()  # 输出范围为[-1, 1]
            )
            
        def forward(self, x):
            return self.net(x)
    
    # 初始化模型
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    bc_model = BCPolicy(state_dim, action_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bc_model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(bc_model.parameters(), lr=1e-3)
    
    # 训练模型
    batch_size = 64
    num_epochs = 100000
    
    # 将数据转换为张量
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)
    
    # 创建数据集
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    bc_model.train()
    for epoch in tqdm(range(num_epochs), desc="BC训练进度"):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = bc_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.6f}")
    
    # 保存训练好的模型
    torch.save(bc_model.state_dict(), f"{save_path}/bc_model.pt")
    
    # 创建基于BC模型的策略
    class BCPolicyForEnv:
        def __init__(self, bc_model, device):
            self.bc_model = bc_model
            self.device = device
            self.bc_model.eval()
            
        def predict(self, obs, deterministic=True):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.bc_model(obs_tensor).squeeze(0).cpu().numpy()
                return action, None
    
    bc_policy = BCPolicyForEnv(bc_model, device)
    
    # 评估BC策略（在所有轨迹上）
    all_rewards = []
    for i in tqdm(range(len(X_list)), desc="评估BC策略"):
        eval_env = DroneEnvIL(X_list[i], dt, max_steps=max_steps, n_obs=n_obs)
        rewards = []
        for _ in range(3):  # 每条轨迹评估3次
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = bc_policy.predict(obs)
                obs, reward, done, _ = eval_env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        print(f"轨迹 {i+1} 评估: 平均奖励 = {mean_reward:.2f}")
        all_rewards.extend(rewards)
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"BC 总体评估: 平均奖励 = {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return bc_policy, bc_model

def main(args):
    # 设置固定的时间步长
    dt = 1.0
    
    # 加载轨迹数据
    X_list = load_trajectory_data(args.data_dir, args.num_trajectories)
    
    # 训练模型
    if args.method == 'irl':
        print("使用最大熵IRL算法训练...")
        policy, irl_model = train_with_irl(
            X_list, dt, 
            max_steps=args.max_steps, 
            n_obs=args.n_obs, 
            save_path=args.save_path,
            trajectory_idx=args.trajectory_idx
        )
    elif args.method == 'bc':
        print("使用行为克隆算法训练...")
        policy, bc_model = train_with_bc(
            X_list, dt, 
            max_steps=args.max_steps, 
            n_obs=args.n_obs, 
            save_path=args.save_path
        )
    else:
        print(f"未知方法: {args.method}")
        return
    
    print("训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="无人机模仿学习训练")
    parser.add_argument("--data_dir", type=str, default="./data/processed_data", help="轨迹数据文件所在目录")
    parser.add_argument("--num_trajectories", type=int, default=8, help="要加载的轨迹数量")
    parser.add_argument("--trajectory_idx", type=int, default=0, help="用于IRL训练环境的轨迹索引")
    parser.add_argument("--method", type=str, default="irl", choices=["irl", "bc"], help="模仿学习方法：irl (MaxEnt IRL) 或 bc (Behavior Cloning)")
    parser.add_argument("--max_steps", type=int, default=100, help="每个轨迹的最大步数")
    parser.add_argument("--n_obs", type=int, default=5, help="历史观测窗口大小")
    parser.add_argument("--save_path", type=str, default="./ckpt/il_models", help="模型保存路径")
    
    args = parser.parse_args()
    main(args) 