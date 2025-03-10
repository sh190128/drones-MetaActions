import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from drone_env import DroneEnv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')

np.random.seed(42)

X_test = np.load('./data/processed_data/processed_X_standard.npy')
y_test = np.load('./data/processed_data/processed_y_standard.npy')
# X_test = np.load('./data/processed_X_short.npy')
# y_test = np.load('./data/processed_y_short.npy')


print(X_test.shape)
print(y_test.shape)


model = PPO.load("./models/drone_ppo_model_trace5")


test_env = DroneEnv(X_test, y_test, dt=1, test=True)

def evaluate_model(model, env, num_steps=None):
    if num_steps is None:
        num_steps = len(env.X)
    
    predictions = []
    targets = []
    
    for step in range(num_steps):
        # 设置当前step
        env.current_episode = step
        obs = env.reset()
        target = env.target

        # 采样一步，预测动作，计算next_obs
        action, _ = model.predict(obs, deterministic=True)
        next_obs, _, _, _ = env.step(action)
        
        
        predicted = next_obs[1:4]
        
        predictions.append(predicted)
        targets.append(target)
    
    return np.array(predictions), np.array(targets)


def plot_trajectories(predictions, targets):

    plt.figure(figsize=(15, 18))
    
    # 经度
    plt.subplot(3, 1, 1)
    plt.plot(predictions[:, 0], 'b-', label='Predicted')
    plt.plot(targets[:, 0], 'r--', label='Target')
    plt.title('Longitude')
    plt.xlabel('Time Step')
    plt.ylabel('Longitude')
    plt.legend()
    
    # 纬度
    plt.subplot(3, 1, 2)
    plt.plot(predictions[:, 1], 'b-', label='Predicted')
    plt.plot(targets[:, 1], 'r--', label='Target')
    plt.title('Latitude')
    plt.xlabel('Time Step')
    plt.ylabel('Latitude')
    plt.legend()
    
    # 地心距
    plt.subplot(3, 1, 3)
    plt.plot(predictions[:, 2], 'b-', label='Predicted')
    plt.plot(targets[:, 2], 'r--', label='Target')
    plt.title('Earth Center Distance')
    plt.xlabel('Time Step')
    plt.ylabel('Earth Center Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./visualize/prediction_results_trace5.png')
    
    # 3D轨迹图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], 'b-', label='Predicted')
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], 'r--', label='Target')
    ax.set_title('3D Trajectory')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Radius')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('./visualize/trajectory_3d_comparison_trace5.png')
    plt.show()
    

predictions, targets = evaluate_model(model, test_env)

plot_trajectories(predictions, targets)