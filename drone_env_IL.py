import gym 
import numpy as np
from gym import spaces
from missile import Missile

# 全局变量
omega_e = 7.292e-5 # 地球自转角速度，rad/s
r_e = 6371393 # 地球平均半径，m
g0 = 9.8 # 海平面重力加速度，m/s^2

params = {'m': 907,
          's': 0.4097,
          'missile_step': 1,
          'CL': np.array([
    [0.450, 0.425, 0.400, 0.380, 0.370, 0.360, 0.350],
    [0.740, 0.700, 0.670, 0.630, 0.600, 0.570, 0.557],
    [1.050, 1.000, 0.950, 0.900, 0.850, 0.800, 0.780]
]),
          'CD': np.array([
    [0.450, 0.425, 0.400, 0.380, 0.370, 0.360, 0.350],
    [0.740, 0.700, 0.670, 0.630, 0.600, 0.570, 0.557],
    [1.050, 1.000, 0.950, 0.900, 0.850, 0.800, 0.780]
]),
          'k': np.array([1, 1, 1])}


class DroneEnvIL(gym.Env):
    """
    模仿学习环境：用于训练无人机按照给定轨迹飞行
    与原始强化学习环境不同，这里使用专家轨迹作为目标
    """
    def __init__(self, X, y, dt, max_steps=100, test=False, n_obs=5):
        super(DroneEnvIL, self).__init__()
        
        self.dt = dt
        self.test = test
        self.X = X  # 状态序列，每一步为t时刻状态
        self.y = y  # 下一步状态，y[t]对应X[t+1]
        self.n_obs = n_obs  # 历史观测窗口大小
        
        self.trajectory_length = len(X)
        self.max_steps = max_steps
        
        self.current_step = 0
        self.current_episode = 0
        
        self.longitude_start = self.X[0][1]
        self.latitude_start = self.X[0][2]
        self.r_start = self.X[0][3]
        
        # 动作空间：速度、航向角和弹道倾角的调整
        # 每个动作维度的范围为[-1, 1]，将在step中映射到实际调整量
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        
        # 构建状态空间边界
        # 每个状态的边界：[速度, 经度, 纬度, 地心距, 航向角, 弹道倾角]
        state_low = np.array([0, -np.pi, -np.pi/2, 6370000, 0, -np.pi])
        state_high = np.array([10000, np.pi, np.pi/2, 6500000, 2*np.pi, np.pi])
        
        # 重复n_obs次以适应历史状态
        history_low = np.tile(state_low, n_obs)
        history_high = np.tile(state_high, n_obs)
        
        # 终点坐标边界
        endpoint_low = np.array([-np.pi, -np.pi/2, 6370000])
        endpoint_high = np.array([np.pi, np.pi/2, 6500000])
        
        # 步数比例边界
        steps_low = np.array([0.0])
        steps_high = np.array([1.0])
        
        # 组合所有边界
        obs_low = np.concatenate([history_low, endpoint_low, steps_low])
        obs_high = np.concatenate([history_high, endpoint_high, steps_high])

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
        self.state = None
        self.target_state = None
        self.expert_action = None
        self.trajectory_end = None
        
        # 存储历史观测状态
        self.history_states = []
        
        # 为IRL算法准备的数据
        self.current_features = None
        self.expert_features = None

    def reset(self):
        if not self.test:
            self.current_episode = np.random.randint(0, self.trajectory_length - self.max_steps - 1)
        else:
            self.current_episode = 0
            
        self.state = self.X[self.current_episode].copy()
        self.target_state = self.y[self.current_episode].copy()
        
        # 保存轨迹终点
        self.trajectory_end = self.y[-1][0:3].copy()
        
        # 转换为相对坐标系
        self.state[1] -= self.longitude_start
        self.state[2] -= self.latitude_start
        self.state[3] -= self.r_start
        
        # 初始化历史状态
        self.history_states = []
        
        # 填充历史状态
        while len(self.history_states) < self.n_obs:
            if self.current_episode - (self.n_obs - len(self.history_states)) >= 0:
                prev_idx = self.current_episode - (self.n_obs - len(self.history_states))
                prev_state = self.X[prev_idx].copy()
                # 转换为相对坐标
                prev_state[1] -= self.longitude_start
                prev_state[2] -= self.latitude_start
                prev_state[3] -= self.r_start
                self.history_states.append(prev_state)
            else:
                # 没有更早的状态，使用当前状态填充
                self.history_states.append(self.state.copy())
        
        # 获取专家动作
        self._calculate_expert_action()
        
        # 将终点转换为相对坐标
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ])
        
        # 计算剩余步数比例
        steps_remaining_ratio = (self.max_steps - self.current_step) / self.max_steps
        
        # 组合观测状态
        flat_history = np.concatenate(self.history_states)
        observation = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]])
        
        # 计算当前状态的特征表示(为IRL算法准备)
        self.current_features = self._extract_features(self.state)
        
        # 计算专家状态的特征表示
        next_state = self.X[self.current_episode + 1]
        next_state_rel = next_state.copy()
        next_state_rel[1] -= self.longitude_start
        next_state_rel[2] -= self.latitude_start
        next_state_rel[3] -= self.r_start
        self.expert_features = self._extract_features(next_state_rel)
        
        self.current_step = 0
        return observation
    
    def step(self, action):
        self.target_state = self.y[self.current_episode + self.current_step]
        
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
            由于计算要求，传入missile类的位置信息必须为地球坐标系，不能转为相对坐标系
        '''
        
        missile_pre_state = self.X[self.current_episode + self.current_step].copy()
        missile_pre_state = missile_pre_state[[0,5,4,1,2,3]]
        missile_desired_state = np.array([new_V, new_theta, new_psi])
        missile = Missile(pre_state=missile_pre_state, desired_state=missile_desired_state, params=params)
        
        new_state = missile.calculate_state()
        # 调整状态向量的顺序：[v, theta, psi, lambda, phi, r] -> [v, lambda, phi, r, psi, theta]
        new_state = new_state[[0, 3, 4, 5, 2, 1]]
        
        # 更新历史状态
        self.history_states.append(new_state.copy())
        if len(self.history_states) > self.n_obs:
            self.history_states.pop(0)
        
        self.state = new_state
        
        # 准备位置数据计算奖励
        predicted_position = new_state[1:4]  # 经度、纬度、地心距（相对值）
        target_position = self.target_state[0:3] - np.array([self.longitude_start, self.latitude_start, self.r_start])
        
        # 计算与轨迹终点的距离
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ])
        
        # 计算与终点的欧式距离
        euclidean_distance_to_end = np.sqrt(
            (predicted_position[0] - trajectory_end_relative[0])**2 + 
            (predicted_position[1] - trajectory_end_relative[1])**2 + 
            (predicted_position[2] - trajectory_end_relative[2])**2
        )
        
        # 计算当前状态的特征表示
        new_features = self._extract_features(new_state)
        self.current_features = new_features
        
        # 为下一步更新专家动作
        self.current_step += 1
        if self.current_episode + self.current_step < len(self.X) - 1:
            next_state = self.X[self.current_episode + self.current_step + 1]
            next_state_rel = next_state.copy()
            next_state_rel[1] -= self.longitude_start
            next_state_rel[2] -= self.latitude_start
            next_state_rel[3] -= self.r_start
            self.expert_features = self._extract_features(next_state_rel)
            self._calculate_expert_action()
        
        # 对于模仿学习，奖励是与专家动作的相似度
        imitation_reward = -np.linalg.norm(action - self.expert_action)
        
        # 与目标状态的误差
        position_error = np.linalg.norm(predicted_position - target_position)
        
        # 综合奖励
        reward = imitation_reward - 0.1 * position_error
        
        # 是否结束
        done = (self.current_step >= self.max_steps) or (euclidean_distance_to_end < 0.1)
        
        # 计算剩余步数比例
        steps_remaining_ratio = (self.max_steps - self.current_step) / self.max_steps
        
        # 组合观测状态
        flat_history = np.concatenate(self.history_states)
        observation = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]])
        
        # 信息字典
        info = {
            'position_error': position_error,
            'imitation_reward': imitation_reward,
            'expert_action': self.expert_action,
            'euclidean_distance_to_end': euclidean_distance_to_end,
            'features': self.current_features,
            'expert_features': self.expert_features
        }
        
        return observation, reward, done, info
    
    def _calculate_expert_action(self):
        """计算专家动作"""
        if self.current_episode + self.current_step + 1 >= len(self.X):
            # 已经到达轨迹末尾
            self.expert_action = np.zeros(3)
            return
            
        # 当前状态
        current_state = self.X[self.current_episode + self.current_step]
        # 下一状态
        next_state = self.X[self.current_episode + self.current_step + 1]
        
        # 速度差异
        dV = next_state[0] - current_state[0]
        # 航向角差异
        dpsi = next_state[4] - current_state[4]
        # 弹道倾角差异
        dtheta = next_state[5] - current_state[5]
        
        # 将差异映射到动作空间[-1, 1]
        action_V = np.clip(dV / 100, -1, 1)
        action_psi = np.clip(dpsi / 0.1, -1, 1)
        action_theta = np.clip(dtheta / 0.1, -1, 1)
        
        self.expert_action = np.array([action_V, action_psi, action_theta])
    
    def _extract_features(self, state):
        """
        提取状态特征，用于IRL算法
        这些特征将用来学习奖励函数
        """
        # 基础状态特征
        features = np.zeros(10)
        
        # 标准化的状态变量
        v_norm = state[0] / 10000.0  # 速度
        lambda_norm = state[1] / np.pi  # 经度
        phi_norm = state[2] / (np.pi/2)  # 纬度
        r_norm = (state[3] - 6370000) / 130000  # 地心距
        psi_norm = state[4] / (2 * np.pi)  # 航向角
        theta_norm = state[5] / np.pi  # 弹道倾角
        
        # 距离终点的相对位置
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ])
        
        # 计算与终点的欧氏距离
        dist_to_end = np.sqrt(
            (state[1] - trajectory_end_relative[0])**2 + 
            (state[2] - trajectory_end_relative[1])**2 + 
            (state[3] - trajectory_end_relative[2])**2
        )
        dist_norm = np.exp(-0.1 * dist_to_end)  # 距离特征
        
        # 朝向终点的方向特征
        start_to_end = trajectory_end_relative - np.zeros_like(trajectory_end_relative)
        current_to_end = trajectory_end_relative - state[1:4]
        
        # 计算余弦相似度（方向对齐程度）
        start_to_end_norm = np.linalg.norm(start_to_end)
        current_to_end_norm = np.linalg.norm(current_to_end)
        
        # 避免除以零
        if start_to_end_norm > 0 and current_to_end_norm > 0:
            cos_similarity = np.dot(start_to_end, current_to_end) / (start_to_end_norm * current_to_end_norm)
        else:
            cos_similarity = 1.0
        
        # 特征向量组成
        features[0] = v_norm  # 速度特征
        features[1] = lambda_norm  # 经度特征
        features[2] = phi_norm  # 纬度特征
        features[3] = r_norm  # 高度特征
        features[4] = psi_norm  # 航向角特征 
        features[5] = theta_norm  # 弹道倾角特征
        features[6] = dist_norm  # 距离特征
        features[7] = cos_similarity  # 方向特征
        features[8] = (self.max_steps - self.current_step) / self.max_steps  # 剩余步数特征
        features[9] = 1.0 if dist_to_end < 0.1 else 0.0  # 是否接近终点特征
        
        return features
    
    def get_expert_trajectory(self, start_idx=0, length=None):
        """
        获取专家轨迹片段，用于IRL算法
        返回格式：[(状态，动作，下一状态，特征)...]
        """
        if length is None:
            length = min(self.max_steps, len(self.X) - start_idx - 1)
        
        expert_trajectory = []
        
        for i in range(start_idx, start_idx + length):
            if i + 1 >= len(self.X):
                break
                
            # 当前状态(相对坐标)
            current_state = self.X[i].copy()
            current_state[1] -= self.longitude_start
            current_state[2] -= self.latitude_start
            current_state[3] -= self.r_start
            
            # 下一状态(相对坐标)
            next_state = self.X[i+1].copy()
            next_state[1] -= self.longitude_start
            next_state[2] -= self.latitude_start
            next_state[3] -= self.r_start
            
            # 速度差异
            dV = next_state[0] - current_state[0]
            # 航向角差异
            dpsi = next_state[4] - current_state[4]
            # 弹道倾角差异
            dtheta = next_state[5] - current_state[5]
            
            # 将差异映射到动作空间[-1, 1]
            action_V = np.clip(dV / 100, -1, 1)
            action_psi = np.clip(dpsi / 0.1, -1, 1)
            action_theta = np.clip(dtheta / 0.1, -1, 1)
            
            action = np.array([action_V, action_psi, action_theta])
            
            # 保存当前episode和step
            old_episode = self.current_episode
            old_step = self.current_step
            
            # 临时设置当前episode和step用于特征提取
            self.current_episode = i
            self.current_step = 0
            
            # 提取特征
            features = self._extract_features(current_state)
            
            # 恢复原来的episode和step
            self.current_episode = old_episode
            self.current_step = old_step
            
            expert_trajectory.append((current_state, action, next_state, features))
            
        return expert_trajectory 