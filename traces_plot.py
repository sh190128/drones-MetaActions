import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

trajectories = []
for i in range(1, 6):
    y = np.load(f'./data/processed_data/output{i}-ode1_y.npy')
    trajectories.append(y)

y = np.load('./data/processed_data/processed_y_standard.npy')
trajectories.append(y)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['b', 'r', 'g', 'c', 'm', 'black']
styles = ['-', '--', '-.', ':', '-', '--']

for i in range(6):
    if i < 5:
        lable = f'Trajectory {i+1}'
    else:
        lable = 'Standard Trajectory'
    ax.plot(trajectories[i][:, 0],  # 经度
            trajectories[i][:, 1],   # 纬度
            trajectories[i][:, 2],   # 地心距
            color=colors[i],
            linestyle=styles[i],
            label=lable)


ax.set_title('Trajectories')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Radius')
ax.legend()

plt.tight_layout()
plt.savefig('./visualize/multiple_trajectories.png')
plt.show()