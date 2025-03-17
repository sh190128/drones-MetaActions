"""
在Simulink中的使用方法：

1. 添加 'Python Function' 模块:
   - 在Simulink库浏览器中搜索 'Python Function'
   - 将模块拖入到你的模型中

2. 配置输入输出:
   输入 obs: 6维向量 [v, lambda, phi, r, psi, theta]
   - v: 速度
   - lambda: 经度
   - phi: 纬度
   - r: 地心距
   - psi: 航向角
   - theta: 弹道倾角

   输出 action: 控制动作

3. 在Python Function模块中配置:
   function out = fcn(in)
   % 添加Python路径
   if count(py.sys.path,'path/to/simulink_step.py') == 0
       insert(py.sys.path,int32(0),'path/to/simulink_step.py');
   end
   
   % 初始化（仅在仿真开始时调用一次）
   persistent initialized
   if isempty(initialized)
       py.simulink_step.initialize();
       initialized = 1;
   end
   
   % 调用Python函数
   out = py.simulink_step.step(in);
"""


import numpy as np
from stable_baselines3 import PPO

class DroneController:
    def __init__(self, model_path="./model_ppo"):
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