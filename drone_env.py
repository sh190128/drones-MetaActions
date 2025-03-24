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
        
        # Observe Space：速度、经度、纬度、地心距、航向角、弹道倾角
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, -np.pi/2, 6370000, 0, -np.pi]),
            high=np.array([10000, np.pi, np.pi/2, 6500000, 2*np.pi, np.pi]),
            dtype=np.float32
        )
        
        self.state = None
        self.target = None
    
    def reset(self):
        if not self.test:
            self.current_episode = np.random.randint(0, self.trajectory_length - self.max_steps)
        self.state = self.X[self.current_episode].copy()
        self.target = self.y[self.current_episode].copy()
        
    
        # 转换为相对坐标系
        self.state[1] -= self.longitude_start  # 经度相对值
        self.state[2] -= self.latitude_start   # 纬度相对值
        self.state[3] -= self.r_start          # 地心距相对值
    
        self.current_step = 0
        return self.state
    
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
    
        # 奖励函数
        reward = -total_error
    
        self.current_step += 1
        done = self.current_step >= self.max_steps
    
        info = {
            'lambda_error': lambda_error,
            'phi_error': phi_error,
            'r_error': r_error,
            'total_error': total_error,
            'reward': reward
        }
        
        return self.state, reward, done, info
