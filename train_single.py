import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_env import DroneEnv
import matplotlib
from utils import plot_training_curves

matplotlib.use('Agg')
np.random.seed(42)

X = np.load('./data/processed_data/processed_X_standard.npy')
y = np.load('./data/processed_data/processed_y_standard.npy')

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print("train set size:", X_train.shape[0])

def make_env():
    return DroneEnv(X_train, y_train, dt=1)

env = DummyVecEnv([make_env])


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
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
                # 每个step数据存储到临时列表
                self.temp_lambda_errors.append(info['lambda_error'])
                self.temp_phi_errors.append(info['phi_error'])
                self.temp_r_errors.append(info['r_error'])
                if 'reward' in info:
                    self.temp_rewards.append(info['reward'])

        # 每n_steps记录一次平均值
        if len(self.temp_lambda_errors) >= self.model.n_steps:
            avg_lambda = np.mean(self.temp_lambda_errors)
            avg_phi = np.mean(self.temp_phi_errors)
            avg_r = np.mean(self.temp_r_errors)
            avg_reward = np.mean(self.temp_rewards) if self.temp_rewards else 0

            self.lambda_errors.append(avg_lambda)
            self.phi_errors.append(avg_phi)
            self.r_errors.append(avg_r)
            self.rewards.append(avg_reward)

            self.logger.record("custom/lambda_error", avg_lambda)
            self.logger.record("custom/phi_error", avg_phi)
            self.logger.record("custom/r_error", avg_r)
            self.logger.record("custom/reward", avg_reward)

            self.temp_lambda_errors = []
            self.temp_phi_errors = []
            self.temp_r_errors = []
            self.temp_rewards = []
            
            # 获取SB3记录的 policy / value loss
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logs = self.model.logger.name_to_value
                if 'train/policy_gradient_loss' in logs:
                    self.policy_losses.append(logs['train/policy_gradient_loss'])
                if 'train/value_loss' in logs:
                    self.value_losses.append(logs['train/value_loss'])

            self.logger.dump(step=self.num_timesteps)
        
        return True


callback = TrainingCallback()
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./drone_ppo_tensorboard/"
)


'''
    / ****************************************** /
    / **************** training **************** /
    / ****************************************** /
'''

total_timesteps = 1000000
model.learn(total_timesteps=total_timesteps, callback=callback)

model.save("./models/drone_ppo_model")

plot_training_curves(callback, './visualize/training_plots', trace_num=None) 