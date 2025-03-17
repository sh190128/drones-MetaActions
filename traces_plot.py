import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')

# 获取所有轨迹文件
data_dir = './data/processed_data'
trajectory_files = [f for f in os.listdir(data_dir) if f.startswith('output') and f.endswith('-ode1_y.npy')]
trajectories = []

# 读取所有轨迹数据
for file_name in trajectory_files:
    y = np.load(os.path.join(data_dir, file_name))
    trajectories.append(y)

# 添加标准轨迹
standard_trajectory = np.load(os.path.join(data_dir, 'processed_y_standard.npy'))
trajectories.append(standard_trajectory)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 生成足够的颜色和线型
n_trajectories = len(trajectories)
colors = plt.cm.rainbow(np.linspace(0, 1, n_trajectories-1))  # 为普通轨迹使用彩虹色
styles = ['-', '--', '-.', ':'] * (n_trajectories // 4 + 1)  # 确保有足够的线型

# 绘制轨迹
for i in range(n_trajectories):
    if i < n_trajectories - 1:
        label = f'Trajectory {i+1}'
        color = colors[i]
    else:
        label = 'Standard Trajectory'
        color = 'black'
    
    ax.plot(trajectories[i][:, 0],  # 经度
            trajectories[i][:, 1],   # 纬度
            trajectories[i][:, 2],   # 地心距
            color=color,
            linestyle=styles[i],
            label=label)

ax.set_title('Trajectories')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Radius')
ax.legend()

plt.tight_layout()
plt.savefig('./visualize/multiple_trajectories.png')
plt.show()