import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_env import DroneEnv
import matplotlib

matplotlib.use('Agg')
np.random.seed(42)

X = np.load('./data/processed_X_standard.npy')
y = np.load('./data/processed_y_standard.npy')

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




plt.figure(num=4, figsize=(20, 12))


def exponential_moving_average(data, smoothing=0.6):
    alpha = 1 - smoothing
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

plt.subplot(2, 3, 1)
plt.plot(callback.rewards, alpha=0.3, label='Raw')
if len(callback.rewards) > 0:
    ema_rewards = exponential_moving_average(callback.rewards)
    plt.plot(ema_rewards, 'r', label='Smoothed')
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(callback.lambda_errors, alpha=0.3, label='Raw')
if len(callback.lambda_errors) > 0:
    ema_lambda = exponential_moving_average(callback.lambda_errors)
    plt.plot(ema_lambda, 'r', label='Smoothed')
plt.title(f'Longitude ($\lambda$) Error')
plt.xlabel('Episode')
plt.ylabel('Error')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(callback.phi_errors, alpha=0.3, label='Raw')
if len(callback.phi_errors) > 0:
    ema_phi = exponential_moving_average(callback.phi_errors)
    plt.plot(ema_phi, 'r', label='Smoothed')
plt.title(f'Latitude ($\phi$) Error')
plt.xlabel('Episode')
plt.ylabel('Error')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(callback.r_errors, alpha=0.3, label='Raw')
if len(callback.r_errors) > 0:
    ema_r = exponential_moving_average(callback.r_errors)
    plt.plot(ema_r, 'r', label='Smoothed')
plt.title('Earth Center Distance (r) Error')
plt.xlabel('Episode')
plt.ylabel('Error')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(callback.policy_losses, alpha=0.3, label='Raw')
if len(callback.policy_losses) > 0:
    ema_policy = exponential_moving_average(callback.policy_losses)
    plt.plot(ema_policy, 'r', label='Smoothed')
plt.title('Policy Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(callback.value_losses, alpha=0.3, label='Raw')
if len(callback.value_losses) > 0:
    ema_value = exponential_moving_average(callback.value_losses)
    plt.plot(ema_value, 'r', label='Smoothed')
plt.title('Value Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('./visualize/learning_curves_test.png')
plt.show() 