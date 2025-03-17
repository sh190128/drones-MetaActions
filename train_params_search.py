import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from drone_env import DroneEnv
import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
matplotlib.use('Agg')

class ParameterSearchCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.reset_stats()
    
    def reset_stats(self):
        """重置所有统计数据"""
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
                self.temp_lambda_errors.append(info['lambda_error'])
                self.temp_phi_errors.append(info['phi_error'])
                self.temp_r_errors.append(info['r_error'])
                if 'reward' in info:
                    self.temp_rewards.append(info['reward'])

        if len(self.temp_lambda_errors) >= self.model.n_steps:
            avg_lambda = np.mean(self.temp_lambda_errors)
            avg_phi = np.mean(self.temp_phi_errors)
            avg_r = np.mean(self.temp_r_errors)
            avg_reward = np.mean(self.temp_rewards) if self.temp_rewards else 0

            self.lambda_errors.append(avg_lambda)
            self.phi_errors.append(avg_phi)
            self.r_errors.append(avg_r)
            self.rewards.append(avg_reward)

            # 记录损失
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logs = self.model.logger.name_to_value
                if 'train/policy_gradient_loss' in logs:
                    self.policy_losses.append(logs['train/policy_gradient_loss'])
                if 'train/value_loss' in logs:
                    self.value_losses.append(logs['train/value_loss'])

            self.temp_lambda_errors = []
            self.temp_phi_errors = []
            self.temp_r_errors = []
            self.temp_rewards = []
            
        return True

def create_env():
    # 加载数据
    data_dir = './data/processed_data'
    trajectory_files = sorted([f for f in os.listdir(data_dir) 
                     if f.startswith('output') and f.endswith('-ode1_X.npy')])
    
    # 加载所有轨迹数据
    X_trains = []
    y_trains = []
    for file in trajectory_files:
        X = np.load(os.path.join(data_dir, file))
        y = np.load(os.path.join(data_dir, file.replace('_X.npy', '_y.npy')))
        X_trains.append(X)
        y_trains.append(y)
    
    # 合并所有轨迹数据
    X_train = np.concatenate(X_trains, axis=0)
    y_train = np.concatenate(y_trains, axis=0)
    
    print(f"加载了 {len(trajectory_files)} 条轨迹，总训练样本数: {len(X_train)}")
    
    return DummyVecEnv([lambda: DroneEnv(X_train, y_train, dt=1)])

def plot_training_curves(callback, save_dir, trial_number):
    """绘制训练过程中的各项指标曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建一个2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Training Curves - Trial {trial_number}')
    
    # 绘制奖励
    axes[0, 0].plot(callback.rewards)
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Reward')
    
    # 绘制经度误差
    axes[0, 1].plot(callback.lambda_errors)
    axes[0, 1].set_title('Lambda Error')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Error')
    
    # 绘制纬度误差
    axes[0, 2].plot(callback.phi_errors)
    axes[0, 2].set_title('Phi Error')
    axes[0, 2].set_xlabel('Steps')
    axes[0, 2].set_ylabel('Error')
    
    # 绘制半径误差
    axes[1, 0].plot(callback.r_errors)
    axes[1, 0].set_title('Radius Error')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Error')
    
    # 绘制策略损失
    if callback.policy_losses:
        axes[1, 1].plot(callback.policy_losses)
        axes[1, 1].set_title('Policy Loss')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
    
    # 绘制价值损失
    if callback.value_losses:
        axes[1, 2].plot(callback.value_losses)
        axes[1, 2].set_title('Value Loss')
        axes[1, 2].set_xlabel('Steps')
        axes[1, 2].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_trial_{trial_number}.png'))
    plt.close()

def objective(trial):
    # 定义参数搜索空间
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "n_steps": trial.suggest_int("n_steps", 1024, 4096, step=1024),
        "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
        "n_epochs": trial.suggest_int("n_epochs", 5, 20),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.999),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.5),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.9)
    }
    
    env = create_env()
    
    # 创建回调
    callback = ParameterSearchCallback()
    
    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=f"./drone_ppo_tensorboard/optuna_{trial.number}",
            **params
        )
        
        # 训练模型
        model.learn(
            total_timesteps=100000,
            callback=callback
        )
        
        # 绘制训练曲线
        plot_training_curves(callback, 
                           save_dir='./optimization_results/training_curves',
                           trial_number=trial.number)
        
        # 使用最后10次评估的平均奖励作为优化目标
        mean_reward = np.mean(callback.rewards[-10:])
        
        # 保存本次试验的参数和结果
        trial_results = {
            'params': params,
            'mean_reward': mean_reward,
            'final_errors': {
                'lambda': np.mean(callback.lambda_errors[-10:]),
                'phi': np.mean(callback.phi_errors[-10:]),
                'r': np.mean(callback.r_errors[-10:])
            }
        }
        
        # 保存试验结果
        os.makedirs('./optimization_results/trial_results', exist_ok=True)
        np.save(f'./optimization_results/trial_results/trial_{trial.number}.npy', 
                trial_results)
        
        return mean_reward
    
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('-inf')

def main():
    # 创建study对象
    study = optuna.create_study(
        study_name="drone_ppo_optimization",
        direction="maximize",
        storage="sqlite:///params_optimization.db",
        load_if_exists=True
    )
    
    # 运行优化
    n_trials = 10  # 可以根据需要调整试验次数
    study.optimize(objective, n_trials=n_trials)
    
    # 打印最佳参数
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    
    # 保存优化结果
    results_dir = "./optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存最佳参数和最佳值
    with open(f"{results_dir}/best_parameters.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    
    # 保存参数重要性分析
    try:
        importance = optuna.importance.get_param_importances(study)
        with open(f"{results_dir}/parameter_importance.txt", "w") as f:
            for param, score in importance.items():
                f.write(f"{param}: {score}\n")
    except:
        print("无法计算参数重要性")
    
    # 使用最佳参数训练最终模型
    print("使用最佳参数训练最终模型...")
    env = create_env()
    final_model = PPO("MlpPolicy", env, verbose=1, **study.best_params)
    final_model.learn(total_timesteps=200000)  # 增加训练步数
    final_model.save(f"{results_dir}/best_model")

    # 在完成所有试验后，绘制参数重要性图
    if study.trials_dataframe is not None:
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame(
            list(importance.items()), 
            columns=['Parameter', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        plt.barh(importance_df['Parameter'], importance_df['Importance'])
        plt.title('Parameter Importance')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/parameter_importance.png")
        plt.close()

if __name__ == "__main__":
    main() 