import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from drone_env_IL import DroneEnvIL
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import argparse
import os
matplotlib.use('Agg')

np.random.seed(42)

def evaluate_il_model(model, env, simulate=False, num_steps=None, n_obs=5, start_obs=100):
    """
    评估模仿学习模型性能
    
    参数:
    model: 训练好的模型
    env: 仿真环境
    simulate: 是否进行完整轨迹仿真
    num_steps: 仿真步数
    n_obs: 历史观测窗口大小
    start_obs: 开始自主预测的步数
    
    返回:
    预测轨迹和实际轨迹
    """
    if num_steps is None:
        num_steps = len(env.X)
    
    predictions = []
    targets = []
    
    if simulate:
        # 存储模拟轨迹的历史状态
        history_states = []
        current_state = None
        
        # 第一阶段：前start_obs步，直接记录原始轨迹
        for step in range(start_obs):
            if step < len(env.X):
                # 记录原始轨迹状态
                current_state = env.X[step].copy()
                history_states.append(current_state)
                if len(history_states) > n_obs:
                    history_states.pop(0)
                
                # 记录原始轨迹和目标轨迹（相同）
                target_state = env.X[step].copy()
                position = env.X[step,1:4]
                target_position = target_state[1:4]
                predictions.append(position)
                targets.append(target_position)
        
        # 第二阶段：从start_obs步开始，进行自主预测
        if start_obs < num_steps:
            # 只需要重置一次环境
            env.reset()
            # 设置当前状态为最后一个已知状态
            env.state = current_state.copy()
            # 设置历史状态
            env.history_states = history_states.copy()
            
            # 从start_obs开始，一直预测到num_steps
            for step in range(start_obs, num_steps):
                # 获取观测
                trajectory_end_relative = np.array([
                    env.trajectory_end[0] - env.longitude_start,
                    env.trajectory_end[1] - env.latitude_start,
                    env.trajectory_end[2] - env.r_start
                ])
                steps_remaining_ratio = (env.max_steps - (step % env.max_steps)) / env.max_steps
                flat_history = np.concatenate(env.history_states)
                obs = np.concatenate([flat_history, trajectory_end_relative, [steps_remaining_ratio]])
                
                # 预测动作并执行
                action, _ = model.predict(obs, deterministic=True)
                next_obs, _, _, _ = env.step(action)
                
                # 记录预测轨迹和原始轨迹
                if step < len(env.X):
                    target_state = env.X[step].copy()
                    # 转换为绝对坐标进行评估
                    position = env.state[1:4]
                    # position += np.array([env.longitude_start, env.latitude_start, env.r_start])
                    target_position = target_state[1:4]
                    predictions.append(position)
                    targets.append(target_position)
    
    else:
        # 不进行仿真，只是依次使用环境重置并获取动作
        for step in range(num_steps):
            env.current_episode = step
            obs = env.reset()
            
            if step < len(env.X):
                target_state = env.X[step].copy()
                
                action, _ = model.predict(obs, deterministic=True)
                next_obs, _, _, _ = env.step(action)
                
                # 转换为绝对坐标进行评估
                position = env.state[1:4] + np.array([env.longitude_start, env.latitude_start, env.r_start])
                target_position = target_state[1:4]
                predictions.append(position)
                targets.append(target_position)
    
    return np.array(predictions), np.array(targets)

def compute_metrics(predictions, targets):
    """计算评估指标"""
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return {
        "MSE": mse,
        "MAE": mae,
        "R^2": r2
    }

def plot_trajectories(predictions, targets, output_dir="./visualize/il_results/"):
    """绘制轨迹比较图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 18))
    
    # 经度
    plt.subplot(3, 1, 1)
    plt.plot(predictions[:, 0], 'b-', label='prediction')
    plt.plot(targets[:, 0], 'r--', label='target')
    plt.title('longitude')
    plt.xlabel('time step')
    plt.ylabel('longitude value')
    plt.legend()
    
    # 纬度
    plt.subplot(3, 1, 2)
    plt.plot(predictions[:, 1], 'b-', label='prediction')
    plt.plot(targets[:, 1], 'r--', label='target')
    plt.title('latitude')
    plt.xlabel('time step')
    plt.ylabel('latitude value')
    plt.legend()
    
    # 地心距
    plt.subplot(3, 1, 3)
    plt.plot(predictions[:, 2], 'b-', label='prediction')
    plt.plot(targets[:, 2], 'r--', label='target')
    plt.title('geocentric distance')
    plt.xlabel('time step')
    plt.ylabel('geocentric distance value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_results.png')
    
    # 3D轨迹图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], 'b-', label='prediction')
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2], 'r--', label='target')
    ax.set_title('3D trajectory')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_zlabel('r')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trajectory_3d_comparison.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模仿学习模型评估脚本")
    parser.add_argument('--simulate', action='store_true', help='是否进行完整轨迹仿真预测')
    parser.add_argument('--cuda', type=int, default=0, help='GPU设备编号 (默认: 0)')
    parser.add_argument('--n_obs', type=int, default=5, help='历史观测窗口大小')
    parser.add_argument('--start_obs', type=int, default=5, help='仿真完整轨迹的起始步')
    parser.add_argument('--model_path', type=str, default="./ckpt/il_models/irl_policy.zip", help='模型路径')
    parser.add_argument('--test_data', type=str, default="./data/processed_data/output8-ode1_X.npy", help='测试数据路径')
    args = parser.parse_args()

    # 设置CUDA设备
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
        print(f"使用 CUDA 设备 {args.cuda}")
    else:
        print("CUDA不可用，使用CPU")

    # 加载测试数据
    X_test = np.load(args.test_data)
    print(f"加载测试数据: {args.test_data}, 形状: {X_test.shape}")
    
    # 创建环境
    dt = 1.0  # 设置固定的时间步长
    test_env = DroneEnvIL(X_test, dt=dt, max_steps=len(X_test), test=True, n_obs=args.n_obs)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = PPO.load(args.model_path)
    
    # 评估模型
    print("开始评估模型...")
    print(f"模拟模式: {'开启' if args.simulate else '关闭'}")
    print(f"起始自主预测步数: {args.start_obs}")
    
    predictions, targets = evaluate_il_model(
        model, test_env, 
        simulate=args.simulate, 
        num_steps=X_test.shape[0], 
        n_obs=args.n_obs,
        start_obs=args.start_obs
    )
    
    # 计算评估指标
    metrics = compute_metrics(predictions, targets)
    print("\n评估指标:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    
    # 绘制轨迹
    print("\n绘制轨迹图表...")
    plot_trajectories(predictions, targets)
    print("评估完成！结果保存在 './visualize/il_results/' 目录中") 