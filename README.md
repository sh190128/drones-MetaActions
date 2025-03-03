# 无人机轨迹规划-元动作

基于强化学习控制无人机元动作，实现无人机在大气层内的轨迹规划和姿态控制。


## 数据说明

项目包含两类主要的飞行数据:

- standard_data.csv: 标准飞行数据,包含时间、速度、经纬度、高度等基本参数
- typical_data.csv: 典型飞行轨迹数据,包含更多飞行状态参数

## 项目结构

```bash
drone_ppo/
├── data/ # 数据文件夹
├── models/ # 模型文件夹
├── visualize/ # 可视化文件夹
└── README.md # 项目说明