import gym
import numpy as np
from gym import spaces

class DroneEnvIL(gym.Env):
    def __init__(self, X, dt, max_steps=100, test=False, n_obs=5):
        super(DroneEnvIL, self).__init__()
        
        self.dt = dt
        self.test = test
        self.X = X  # 专家轨迹状态数据
        self.n_obs = n_obs  # 历史观测窗口大小
        
        self.trajectory_length = len(X)
        self.max_steps = max_steps
        
        self.current_step = 0
        self.current_episode = 0
        
        # 记录起始点
        self.longitude_start = self.X[0][1]
        self.latitude_start = self.X[0][2]
        self.r_start = self.X[0][3]
        
        
        # Action Space：速度、航向角和弹道倾角
        # 每个动作维度的范围为[-1, 1]，将在step中映射到实际调整量
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        
        # 每个状态包含6个变量：速度、经度、纬度、地心距、航向角、弹道倾角
        single_state_dim = 6
        
        # 为历史状态创建边界数组
        # 每个状态的边界：[速度, 经度, 纬度, 地心距, 航向角, 弹道倾角]
        state_low = np.array([0, -np.pi, -np.pi/2, 6370000, 0, -np.pi])
        state_high = np.array([10000, np.pi, np.pi/2, 6500000, 2*np.pi, np.pi])
        
        # 重复n_obs次以适应历史状态
        history_low = np.tile(state_low, n_obs)
        history_high = np.tile(state_high, n_obs)
        
        # 终点坐标的边界（经度、纬度、地心距）
        endpoint_low = np.array([-np.pi, -np.pi/2, 6370000])
        endpoint_high = np.array([np.pi, np.pi/2, 6500000])
        
        # 步数比例的边界
        steps_low = np.array([0.0])  # 剩余步数比例最小为0
        steps_high = np.array([1.0])  # 剩余步数比例最大为1
        
        # 组合所有边界
        obs_low = np.concatenate([history_low, endpoint_low, steps_low])
        obs_high = np.concatenate([history_high, endpoint_high, steps_high])
        
        # 定义observation space
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
        self.state = None
        self.expert_next_state = None
        self.trajectory_end = self.X[-1][1:4].copy()
        
        # 存储历史观测状态
        self.history_states = []
        
        # 专家轨迹动作缓存
        self.expert_actions = self._compute_expert_actions()
    
    def _compute_expert_actions(self):
        """计算专家轨迹的动作"""
        expert_actions = []
        
        for i in range(len(self.X) - 1):
            current_state = self.X[i]
            next_state = self.X[i + 1]
            
            # 计算动作（简化版本，实际应该基于物理模型反推）
            # 速度变化
            V_change = (next_state[0] - current_state[0]) / 100  # 归一化到[-1,1]
            
            # 航向角变化
            psi_change = (next_state[4] - current_state[4]) / 0.1
            
            # 弹道倾角变化
            theta_change = (next_state[5] - current_state[5]) / 0.1
            
            # 裁剪动作到[-1,1]范围
            action = np.clip(np.array([V_change, psi_change, theta_change]), -1, 1)
            expert_actions.append(action)
            
        return expert_actions
    
    def reset(self):
        if not self.test:
            self.current_episode = np.random.randint(0, self.trajectory_length - self.max_steps)
        else:
            self.current_episode = 0
            
        self.state = self.X[self.current_episode].copy()
        
        # 确保有足够的状态数据用于expert_next_state
        if self.current_episode + 1 < len(self.X):
            self.expert_next_state = self.X[self.current_episode + 1].copy()
        else:
            self.expert_next_state = self.state.copy()
        
        # 保存轨迹终点信息
        self.trajectory_end = self.X[-1][1:4].copy()
        
        # 转换为相对坐标系
        self.state[1] -= self.longitude_start  # 经度相对值
        self.state[2] -= self.latitude_start   # 纬度相对值
        self.state[3] -= self.r_start          # 地心距相对值
        
        # 初始化历史状态列表
        self.history_states = []
        
        # 如果当前步数小于n_obs，用当前状态填充历史
        # history_states=[states_t-5, states_t-4, ......]
        while len(self.history_states) < self.n_obs:
            # 如果有更早的状态可用，则使用
            if self.current_episode - (self.n_obs - len(self.history_states)) >= 0:
                prev_idx = self.current_episode - (self.n_obs - len(self.history_states))
                prev_state = self.X[prev_idx].copy()
                # 转换为相对坐标
                prev_state[1] -= self.longitude_start
                prev_state[2] -= self.latitude_start
                prev_state[3] -= self.r_start
                self.history_states.append(prev_state)  # 直接append，保持时序
            else:
                # 没有更早的状态，使用当前状态填充
                self.history_states.append(self.state.copy())
        
        # 将终点转换为相对坐标
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ])
        
        # 计算剩余步数与最大步数的比例（用于给模型提供时间信息）
        steps_remaining_ratio = (self.max_steps - self.current_step) / self.max_steps
        
        # 组合观测状态：历史状态 + 终点坐标 + 步数比例
        flat_history = np.concatenate(self.history_states)
        observation = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]])
    
        self.current_step = 0
        return observation
    
    def step(self, action):
        # 获取专家在当前步骤的动作
        # print(len(self.expert_actions))
        # print(f'current_episode: {self.current_episode}, current_step: {self.current_step}')
        expert_action = self.expert_actions[self.current_episode + self.current_step]
        
        # 获取专家下一步状态
        next_idx = self.current_episode + self.current_step + 1
        if next_idx < len(self.X):
            self.expert_next_state = self.X[next_idx].copy()
        else:
            # 如果已经到达轨迹末尾，使用最后一个状态
            self.expert_next_state = self.X[-1].copy()
            
        expert_next_position = self.expert_next_state[1:4] - np.array([self.longitude_start, self.latitude_start, self.r_start])
    
        # 动作调整
        V_change = action[0] * 100  # 速度调整范围 ±100
        psi_change = action[1] * 0.1  # 航向角调整范围 ±0.1 弧度
        theta_change = action[2] * 0.1  # 弹道倾角调整范围 ±0.1 弧度

        # 更新速度
        new_V = self.state[0] + V_change
        new_V = max(0, min(new_V, 10000))
    
        # 更新航向角和弹道倾角
        new_psi = self.state[4] + psi_change
        new_theta = self.state[5] + theta_change
        
        # 计算新的经纬度，地心距
        horizontal_V = new_V * np.cos(new_theta)
        vertical_V = new_V * np.sin(new_theta)
    
        earth_radius = self.state[3] + self.r_start  # 恢复绝对地心距
        dlambda = horizontal_V * self.dt * np.sin(new_psi) / (earth_radius * np.cos(self.state[2] + self.latitude_start))
        dphi = horizontal_V * self.dt * np.cos(new_psi) / earth_radius
        dr = vertical_V * self.dt
    
        # 更新状态
        new_state = np.array([
            new_V,
            self.state[1] + dlambda,  # 经度相对值
            self.state[2] + dphi,      # 纬度相对值
            self.state[3] + dr,        # 地心距相对值
            new_psi,
            new_theta
        ])
    
        # 更新历史状态
        self.history_states.append(new_state.copy())
        if len(self.history_states) > self.n_obs:
            self.history_states.pop(0)  # 移除最早的状态
        
        self.state = new_state
    
        # 计算与专家轨迹的偏差
        predicted_position = new_state[1:4]  # 经度、纬度、地心距（相对值）
        
        # 计算与专家下一步状态的欧氏距离
        deviation = np.sqrt(np.sum((predicted_position - expert_next_position)**2))
        
        # 计算动作与专家动作的偏差
        action_deviation = np.sqrt(np.sum((action - expert_action)**2))
        
        # 计算轨迹终点
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ])
        
        # 计算与终点的欧式距离
        euclidean_distance_to_end = np.sqrt(np.sum((predicted_position - trajectory_end_relative)**2))
        
        # 计算奖励，主要基于与专家轨迹的偏差
        reward = -deviation * 0.5 - action_deviation * 0.5
        
        # 更新步数
        self.current_step += 1
        
        # 达到最大步数或非常接近终点时结束
        done = (self.current_step >= self.max_steps) or (euclidean_distance_to_end < 0.1)
        
        # 计算剩余步数与最大步数的比例
        steps_remaining_ratio = (self.max_steps - self.current_step) / self.max_steps
        
        # 组合观测状态：历史状态 + 终点坐标 + 步数比例
        flat_history = np.concatenate(self.history_states)
        observation = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]])
        
        # 信息字典
        info = {
            'state_deviation': deviation,
            'action_deviation': action_deviation,
            'expert_action': expert_action,
            'agent_action': action,
            'distance_to_end': euclidean_distance_to_end
        }
        
        return observation, reward, done, info
    
    def get_expert_demonstration(self, num_trajectories=10):
        """获取专家示范数据，用于模仿学习"""
        demos = []
        
        for _ in range(num_trajectories):
            # 随机选择一个起始点
            start_idx = np.random.randint(0, self.trajectory_length - self.max_steps)
            
            states = []
            actions = []
            
            # 收集轨迹
            for i in range(self.max_steps):
                if start_idx + i >= len(self.X) - 1:
                    break
                    
                # 获取当前状态
                current_state = self.X[start_idx + i].copy()
                
                # 获取专家动作
                expert_action = self.expert_actions[start_idx + i]
                
                # 转换为相对坐标
                current_state_rel = current_state.copy()
                current_state_rel[1] -= self.longitude_start
                current_state_rel[2] -= self.latitude_start
                current_state_rel[3] -= self.r_start
                
                # 创建状态（包括历史、终点等）
                history_states = []
                for j in range(self.n_obs):
                    if start_idx + i - j >= 0:
                        prev_state = self.X[start_idx + i - j].copy()
                        prev_state[1] -= self.longitude_start
                        prev_state[2] -= self.latitude_start
                        prev_state[3] -= self.r_start
                        history_states.append(prev_state)
                    else:
                        history_states.append(current_state_rel)

                # 轨迹终点
                trajectory_end_relative = np.array([
                    self.trajectory_end[0] - self.longitude_start,
                    self.trajectory_end[1] - self.latitude_start,
                    self.trajectory_end[2] - self.r_start
                ])
                
                # 剩余步数比例
                steps_remaining_ratio = (self.max_steps - i) / self.max_steps
                
                # 组合状态
                flat_history = np.concatenate(history_states)
                complete_state = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]])
                
                states.append(complete_state)
                actions.append(expert_action)
            
            demos.append((states, actions))
            
        return demos 