import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from drone_env import DroneEnv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import argparse
matplotlib.use('Agg')

np.random.seed(42)

def evaluate_model(model, env, simulate=False, num_steps=None):

    if num_steps is None:
        num_steps = len(env.X)
    assert len(env.X) == len(env.y)

    predictions = []
    targets = []
    
    if simulate:
        for step in range(num_steps):
            env.current_episode = step
            
            if step < 5:
                # 前n步，obs从env中获取（测试集已有数据）
                obs = env.reset()
                action, _ = model.predict(obs, deterministic=True)
                next_obs, _, _, _ = env.step(action)
            else:
                # obs由计算迭代更新
                env.reset()
                env.state = next_obs
                obs = next_obs
                action, _ = model.predict(obs, deterministic=True)
                next_obs, _, _, _ = env.step(action)
                
            target = env.target
            predictions.append(next_obs[1:4]+np.array([env.longitude_start, env.latitude_start, env.r_start]))
            targets.append(target)

    else:
        for step in range(num_steps):
            env.current_episode = step
            obs = env.reset()
            target = env.target
            
            action, _ = model.predict(obs, deterministic=True)
            next_obs, _, _, _ = env.step(action)
            
            predictions.append(next_obs[1:4]+np.array([env.longitude_start, env.latitude_start, env.r_start]))
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
    plt.savefig('./visualize/testing_plots/prediction_results.png')
    
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
    plt.savefig('./visualize/testing_plots/trajectory_3d_comparison.png')
    plt.show()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true', help='whether to simulate the complete trace')
    args = parser.parse_args()

    X_test = np.load('./data/processed_data/output8-ode1_X.npy')
    y_test = np.load('./data/processed_data/output8-ode1_y.npy')
    # X_test = np.load('./data/processed_data/processed_X_standard.npy')
    # y_test = np.load('./data/processed_data/processed_y_standard.npy')
    test_env = DroneEnv(X_test, y_test, dt=1, test=True)
    model = PPO.load("./ckpt/model_ppo_trace8")

    predictions, targets = evaluate_model(model, test_env, simulate=args.simulate)

    plot_trajectories(predictions, targets)