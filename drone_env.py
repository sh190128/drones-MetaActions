import gym 
import numpy as np
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self, X, y, dt, max_steps=100, test=False, n_obs=5):
        super(DroneEnv, self).__init__()
        
        self.dt = dt
        self.test = test
        self.X = X
        self.y = y
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
        
        
        single_state_dim = 6
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
        self.target = None
        self.trajectory_end = None  # 添加终点信息
        
        # 存储历史观测状态
        self.history_states = []
    
    def reset(self):
        if not self.test:
            self.current_episode = np.random.randint(0, self.trajectory_length - self.max_steps)
        self.state = self.X[self.current_episode].copy()
        self.target = self.y[self.current_episode].copy()
        
        # 保存轨迹终点信息
        self.trajectory_end = self.y[-1][0:3].copy()
        # 保存轨迹终点信息
        self.trajectory_end = self.y[-1][0:3].copy()
    
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
        self.target = self.y[self.current_episode + self.current_step]
    
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
        
        
        '''
            FIXME: 位置计算当前为简化模型，仅按照地球为规则球形计算。更改计算公式
        '''
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
    
        # 计算位置误差
        predicted_position = new_state[1:4]  # 经度、纬度、地心距（相对值）
        target_position = self.target[0:3] - np.array([self.longitude_start, self.latitude_start, self.r_start])  # 目标位置（相对值）
    
        lambda_error = abs(predicted_position[0] - target_position[0]) / max(abs(target_position[0]), 1e-6)
        phi_error = abs(predicted_position[1] - target_position[1]) / max(abs(target_position[1]), 1e-6)
        r_error = abs(predicted_position[2] - target_position[2]) / max(abs(target_position[2]), 1e-6)
    
        next_state_error = lambda_error + phi_error + r_error
        
        # 计算与轨迹终点的距离
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ])
        
        # 计算与终点的欧式距离（归一化）
        euclidean_distance_to_end = np.sqrt(
            (predicted_position[0] - trajectory_end_relative[0])**2 + 
            (predicted_position[1] - trajectory_end_relative[1])**2 + 
            (predicted_position[2] - trajectory_end_relative[2])**2
        )
        
        # 终点接近奖励
        # 距离越近，奖励越高（使用指数衰减函数）
        distance_reward = np.exp(-euclidean_distance_to_end * 0.1)
        
        # 使用欧式距离设置多级奖励
        end_reward = 0
        if euclidean_distance_to_end < 0.1:  # 非常接近终点
            end_reward = 20.0
        elif euclidean_distance_to_end < 0.3:  # 很接近终点
            end_reward = 10.0
        elif euclidean_distance_to_end < 0.5:  # 接近终点
            end_reward = 5.0
        elif euclidean_distance_to_end < 1.0:  # 距离终点不远
            end_reward = 2.0
        
        # 计算步骤进展（从起点到终点的相对进度）
        # 使用余弦相似度来衡量方向是否朝向终点
        start_to_end = trajectory_end_relative - np.zeros_like(trajectory_end_relative)
        current_to_end = trajectory_end_relative - predicted_position
        
        # 计算余弦相似度（方向对齐程度）
        start_to_end_norm = np.linalg.norm(start_to_end)
        current_to_end_norm = np.linalg.norm(current_to_end)
        
        # 避免除以零
        if start_to_end_norm > 0 and current_to_end_norm > 0:
            cos_similarity = np.dot(start_to_end, current_to_end) / (start_to_end_norm * current_to_end_norm)
        else:
            cos_similarity = 1.0
            
        # 方向奖励（越接近1表示方向越好）
        direction_reward = cos_similarity
        
        # 步数接近最大步数时，增加终点奖励权重
        remaining_steps_ratio = (self.max_steps - self.current_step) / self.max_steps
        end_weight = 1.0 + (1.0 - remaining_steps_ratio) * 2.0  # 权重随着步数增加而增加
        
        # 奖励函数：组合当前轨迹误差、终点方向奖励、终点距离奖励
        reward = -next_state_error * 0.3 + direction_reward * 0.3 + distance_reward * 0.4 + end_reward * end_weight
    
        # 更新步数
        self.current_step += 1
        
        # 达到最大步数或非常接近终点时结束
        done = (self.current_step >= self.max_steps) or (euclidean_distance_to_end < 0.1)
        
        # 如果已达到最大步数但未接近终点，给予惩罚
        if self.current_step >= self.max_steps and euclidean_distance_to_end > 0.3:
            reward -= 10.0  # 未能到达终点的惩罚
        
        # 计算剩余步数与最大步数的比例
        steps_remaining_ratio = (self.max_steps - self.current_step) / self.max_steps
        
        # 组合观测状态：历史状态 + 终点坐标 + 步数比例
        flat_history = np.concatenate(self.history_states)
        observation = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]])
    
        info = {
            'lambda_error': lambda_error,
            'phi_error': phi_error,
            'r_error': r_error,
            'next_state_error': next_state_error,
            'direction_reward': direction_reward,
            'distance_reward': distance_reward,
            'end_reward': end_reward,
            'reward': reward
        }
        
        return observation, reward, done, info
