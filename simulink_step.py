import numpy as np
from stable_baselines3 import PPO

'''
    % 在仿真开始前初始化
    py.simulink_step.initialize()

    obs = numpy array([v, lambda, phi, r, psi, theta])         (速度、经度、纬度、地心距、航向角、弹道倾角)
    
    % 在模型中调用step函数
    action = py.simulink_step.step(observation)
'''

class DroneController:
    def __init__(self, model_path="./ckpt/model_ppo"):
        self.model = PPO.load(model_path)
    
    def predict_step(self, observation):
        
        obs = np.array(observation).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        
        return action.flatten()


controller = None

def initialize():
    """
    初始化函数，在Simulink仿真开始时调用
    """
    global controller
    controller = DroneController()
    return 1

def step(observation):
    """
    Simulink调用的主要接口函数
    
    Args:
        observation: 从Simulink传入的观测值
        
    Returns:
        action: 预测的动作值
    """
    global controller
    if controller is None:
        initialize()
    
    return controller.predict_step(observation) 