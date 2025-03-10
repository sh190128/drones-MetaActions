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
    plt.figure(num=4, figsize=(20, 12))
    
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
    if trace_num is None:
        plt.savefig(f'{save_path}/learning_curves_trace_single.png')
    else:
        plt.savefig(f'{save_path}/learning_curves_trace{trace_num}.png')
    plt.close() 