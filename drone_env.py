import gym 
import numpy as np
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self, X, y, dt, max_steps=100, test=False):
        super(DroneEnv, self).__init__()
        
        self.dt = dt
        self.test = test
        self.X = X
        self.y = y
        
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
        
        # Observe Space：速度、经度、纬度、地心距、航向角、弹道倾角 + 终点坐标
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, -np.pi/2, 6370000, 0, -np.pi, -np.pi, -np.pi/2, 6370000]),
            high=np.array([10000, np.pi, np.pi/2, 6500000, 2*np.pi, np.pi, np.pi, np.pi/2, 6500000]),
            dtype=np.float32
        )
        
        self.state = None
        self.target = None
        self.trajectory_end = None  # 添加终点信息
    
    def reset(self):
        if not self.test:
            self.current_episode = np.random.randint(0, self.trajectory_length - self.max_steps)
        self.state = self.X[self.current_episode].copy()
        self.target = self.y[self.current_episode].copy()
        
        # 保存轨迹终点信息
        self.trajectory_end = self.y[-1][0:3].copy()
    
        # 转换为相对坐标系
        self.state[1] -= self.longitude_start  # 经度相对值
        self.state[2] -= self.latitude_start   # 纬度相对值
        self.state[3] -= self.r_start          # 地心距相对值
        
        # 将终点转换为相对坐标
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ])
        
        # 将终点添加到观测中
        state_with_end = np.concatenate([self.state, trajectory_end_relative])
    
        self.current_step = 0
        return state_with_end
    
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
    
        self.state = new_state
    
        # 计算位置误差
        predicted_position = new_state[1:4]  # 经度、纬度、地心距（相对值）
        target_position = self.target[0:3] - np.array([self.longitude_start, self.latitude_start, self.r_start])  # 目标位置（相对值）
    
        lambda_error = abs(predicted_position[0] - target_position[0]) / max(abs(target_position[0]), 1e-6)
        phi_error = abs(predicted_position[1] - target_position[1]) / max(abs(target_position[1]), 1e-6)
        r_error = abs(predicted_position[2] - target_position[2]) / max(abs(target_position[2]), 1e-6)
    
        total_error = lambda_error + phi_error + r_error
        
        # 计算与轨迹终点的距离
        trajectory_end_relative = np.array([
            self.trajectory_end[0] - self.longitude_start,
            self.trajectory_end[1] - self.latitude_start,
            self.trajectory_end[2] - self.r_start
        ])
        
        # 计算与终点的误差
        end_lambda_error = abs(predicted_position[0] - trajectory_end_relative[0]) / max(abs(trajectory_end_relative[0]), 1e-6)
        end_phi_error = abs(predicted_position[1] - trajectory_end_relative[1]) / max(abs(trajectory_end_relative[1]), 1e-6)
        end_r_error = abs(predicted_position[2] - trajectory_end_relative[2]) / max(abs(trajectory_end_relative[2]), 1e-6)
        
        end_total_error = end_lambda_error + end_phi_error + end_r_error
        
        # 到达终点的奖励
        end_reward = -min(end_total_error, 10.0)  # 限制范围，避免过大的惩罚
        
        # 如果非常接近终点，给予额外奖励
        if end_total_error < 0.1:
            end_reward += 10.0  # 接近终点的额外奖励
    
        # 奖励函数: 组合当前位置误差和终点误差的奖励
        reward = -total_error + 0.5 * end_reward
    
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # 将轨迹终点添加到观测中
        state_with_end = np.concatenate([self.state, trajectory_end_relative])
    
        info = {
            'lambda_error': lambda_error,
            'phi_error': phi_error,
            'r_error': r_error,
            'total_error': total_error,
            'end_error': end_total_error,
            'end_reward': end_reward,
            'reward': reward
        }
        
        return state_with_end, reward, done, info
