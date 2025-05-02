import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from drone_env import DroneEnv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import argparse
matplotlib.use('Agg')

np.random.seed(42)

def evaluate_model(model, env, simulate=False, num_steps=None, n_obs=5, start_obs=347):

    if num_steps is None:
        num_steps = len(env.X)
    assert len(env.X) == len(env.y)

    predictions = []
    targets = []
    
    if simulate:
        # 存储模拟轨迹的历史状态
        history_states = []
        
        for step in range(num_steps):
            env.current_episode = step
            
            if step < start_obs:
                # 修改逻辑：直接从env.X读取历史状态，而不是通过环境迭代
                if step == 0:
                    # 第一步时初始化环境
                    obs = env.reset()
                
                # 直接从数据中获取当前状态
                current_state = env.X[step].copy()
                # 转换为相对坐标
                current_state[1] -= env.longitude_start  # 经度相对值
                current_state[2] -= env.latitude_start   # 纬度相对值
                current_state[3] -= env.r_start          # 地心距相对值
                
                # 将状态添加到历史记录
                history_states.append(current_state)
                if len(history_states) > n_obs:
                    history_states.pop(0)
                
                # 对于当前step，记录预测和目标
                target = env.y[step]
                predictions.append(env.X[step][1:4])  # 使用原始数据作为"预测"
                targets.append(target)
                
                # 如果是最后一个start_obs之前的步骤，准备好环境状态
                if step == start_obs - 1:
                    # 设置环境的当前状态和历史状态
                    env.reset()
                    env.state = current_state
                    env.history_states = history_states.copy()
                    print(f'self.target = self.y[self.current_episode + self.current_step] = {env.y[env.current_episode + env.current_step]}')

            else:
                # 使用历史状态构建观测
                # 获取观测
                trajectory_end_relative = np.array([
                    env.trajectory_end[0] - env.longitude_start,
                    env.trajectory_end[1] - env.latitude_start,
                    env.trajectory_end[2] - env.r_start
                ])
                steps_remaining_ratio = (env.max_steps - env.current_step) / env.max_steps
                flat_history = np.concatenate(env.history_states)
                obs = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]])
                
                # 预测动作并执行
                action, _ = model.predict(obs, deterministic=True)
                next_obs, _, _, _ = env.step(action)
                
                # 更新历史记录
                current_state = env.state.copy()
                history_states.append(current_state)
                if len(history_states) > n_obs:
                    history_states.pop(0)
                
                target = env.target
                # 转换为绝对坐标进行评估
                position = env.state[1:4] + np.array([env.longitude_start, env.latitude_start, env.r_start])
                predictions.append(position)
                targets.append(target)

    else:
        for step in range(num_steps):
            env.current_episode = step
            obs = env.reset()
            target = env.target
            
            action, _ = model.predict(obs, deterministic=True)
            next_obs, _, _, _ = env.step(action)
            
            # 转换为绝对坐标进行评估
            position = env.state[1:4] + np.array([env.longitude_start, env.latitude_start, env.r_start])
            predictions.append(position)
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
    parser.add_argument('--simulate', action='store_true', help='是否仿真预测完整轨迹')
    parser.add_argument('--cuda', type=int, default=0, help='GPU设备编号 (默认: 0)')
    parser.add_argument('--n_obs', type=int, default=100, help='历史观测窗口大小')
    parser.add_argument('--start_obs', type=int, default=400, help='仿真完整轨迹的起始步')
    args = parser.parse_args()

    # 设置CUDA设备
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
        print(f"使用 CUDA 设备 {args.cuda}")
    else:
        print("CUDA不可用，使用CPU")

    X_test = np.load('./data/processed_data/output8-ode1_X.npy')
    y_test = np.load('./data/processed_data/output8-ode1_y.npy')
    # X_test = np.load('./data/processed_data/processed_X_standard.npy')
    # y_test = np.load('./data/processed_data/processed_y_standard.npy')
    test_env = DroneEnv(X_test, y_test, dt=1, test=True, n_obs=args.n_obs)
    model = PPO.load("./ckpt/test_missile/model_ppo_trace8.zip")

    predictions, targets = evaluate_model(model, test_env, simulate=args.simulate, 
                                          num_steps=X_test.shape[0], n_obs=args.n_obs,
                                          start_obs=args.start_obs)

    plot_trajectories(predictions, targets)