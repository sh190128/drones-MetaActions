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
    
    def _on_step(self):
        # 记录误差和奖励
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if 'lambda_error' in info:
                self.lambda_errors.append(info['lambda_error'])
                self.phi_errors.append(info['phi_error'])
                self.r_errors.append(info['r_error'])
                if 'reward' in info:
                    self.rewards.append(info['reward'])
        
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
    / ************************** /
    / ******** training ******** /
    / ************************** /
'''

total_timesteps = 1000000
model.learn(total_timesteps=total_timesteps, callback=callback)

model.save("./models/drone_ppo_model")




plt.figure(num=4, figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(callback.rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(2, 2, 2)
plt.plot(callback.lambda_errors)
plt.title('Longitude Error')
plt.xlabel('Step')
plt.ylabel('Error')

plt.subplot(2, 2, 3)
plt.plot(callback.phi_errors)
plt.title('Latitude Error')
plt.xlabel('Step')
plt.ylabel('Error')

plt.subplot(2, 2, 4)
plt.plot(callback.r_errors)
plt.title('Earth Center Distance Error')
plt.xlabel('Step')
plt.ylabel('Error')

plt.tight_layout()
plt.savefig('./visualize/learning_curves_standard.png')
plt.show() 