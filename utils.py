import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
np.random.seed(42)

def exponential_moving_average(data, smoothing=0.6):
    alpha = 1 - smoothing
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def plot_training_curves(callback, save_path, trace_num=None):
    """
    绘制训练过程中的学习曲线
    
    Args:
        callback: 包含训练数据的回调对象
        save_path: 图片保存路径
        trace_num: 轨迹编号
    """
    # 创建3x2的子图布局以容纳更多指标
    plt.figure(num=4, figsize=(20, 15))
    
    # 第一行：基本训练指标
    plt.subplot(3, 2, 1)
    plt.plot(callback.rewards, alpha=0.3, label='Raw')
    if len(callback.rewards) > 0:
        ema_rewards = exponential_moving_average(callback.rewards)
        plt.plot(ema_rewards, 'r', label='Smoothed')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(callback.lambda_errors, alpha=0.3, label='Raw')
    if len(callback.lambda_errors) > 0:
        ema_lambda = exponential_moving_average(callback.lambda_errors)
        plt.plot(ema_lambda, 'r', label='Smoothed')
    plt.title(f'Longitude ($\lambda$) Error')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    plt.legend()

    # 第二行：更多训练指标
    plt.subplot(3, 2, 3)
    plt.plot(callback.phi_errors, alpha=0.3, label='Raw')
    if len(callback.phi_errors) > 0:
        ema_phi = exponential_moving_average(callback.phi_errors)
        plt.plot(ema_phi, 'r', label='Smoothed')
    plt.title(f'Latitude ($\phi$) Error')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(callback.r_errors, alpha=0.3, label='Raw')
    if len(callback.r_errors) > 0:
        ema_r = exponential_moving_average(callback.r_errors)
        plt.plot(ema_r, 'r', label='Smoothed')
    plt.title('Earth Center Distance (r) Error')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    plt.legend()
    
    # 第三行：终点距离和方向指标
    # 添加终点距离指标
    plt.subplot(3, 2, 5)
    if hasattr(callback, 'distance_rewards') and len(callback.distance_rewards) > 0:
        plt.plot(callback.distance_rewards, alpha=0.3, label='Raw')
        ema_distance = exponential_moving_average(callback.distance_rewards)
        plt.plot(ema_distance, 'r', label='Smoothed')
        plt.title('Distance Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
    else:
        plt.title('Distance Reward')
    
    # 添加方向奖励指标
    plt.subplot(3, 2, 6)
    if hasattr(callback, 'direction_rewards') and len(callback.direction_rewards) > 0:
        plt.plot(callback.direction_rewards, alpha=0.3, label='Raw')
        ema_direction = exponential_moving_average(callback.direction_rewards)
        plt.plot(ema_direction, 'r', label='Smoothed')
        plt.title('Direction Alignment Reward')
        plt.xlabel('Episode')
        plt.ylabel('Cosine Similarity')
        plt.legend()
    else:
        plt.title('Direction Alignment (No Data)')

    plt.tight_layout()
    if trace_num is None:
        plt.savefig(f'{save_path}/learning_curves_trace_single.png')
    else:
        plt.savefig(f'{save_path}/learning_curves_trace{trace_num}.png')
    plt.close() 