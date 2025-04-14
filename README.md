# 无人机轨迹规划-元动作

基于强化学习控制无人机元动作，实现无人机在大气层内的轨迹规划和姿态控制。

## 数据说明

项目包含两类主要的飞行数据:

- standard_data.csv: 标准飞行数据,包含时间、速度、经纬度、高度等基本参数，轨迹较长
- typical_data.csv: 典型飞行轨迹数据模板, 轨迹较短

## 项目特性

- **历史观测窗口**: 模型使用前n_obs步的历史状态作为输入，提供更长的视野 (history horizon)
- **终点约束**: 通过改进的奖励函数，鼓励靠近终点
- **方向引导**: 使用余弦相似度计算轨迹方向与目标方向的一致性，引导无人机朝向终点
- **多级奖励**: 根据与终点的距离设置多级奖励，鼓励无人机接近终点
- **早停机制**: 当无人机非常接近终点时提前结束训练，优化训练效率

## 项目结构

```bash
drones-MetaActions/
├── data/
│   ├── raw_data/ # 原始csv飞行轨迹数据
│   └── processed_data/ # 处理后的训练数据, npy形式存储
│
├── ckpt/
│
├── visualize/
│   ├── training_plots/
│   └── testing_plots/
│
├── train_multiple.py # 多条轨迹训练脚本
├── train_single.py   # 单条轨迹训练脚本
├── train_params_search.py  # 搜参
├── traces_plot.py
├── evaluate.py
├── drone_env.py
├── preprocess.py
├── utils.py
├── requirements.txt
└── README.md
```

## 环境安装
```bash
pip install -r requirements.txt
```

## 运行
```bash
# 预处理原始数据，生成训练所需的 npy 文件
python preprocess.py

# 进行超参数搜索，找到最优的训练参数
python train_params_search.py

# 使用最优参数进行多轨迹训练
python train_multiple.py --n_obs 5 --max_steps 100

# 模型评估
# 逐步预测
python evaluate.py  

# 给定起始点，预测完整轨迹
python evaluate.py --simulate
```

## 参数说明

- `--n_obs`: 历史观测窗口大小，默认为5
- `--max_steps`: 每个episode的最大步数，默认为100

## Todo:
- 本质是Sparse Obs, Sparse Reward, Long Horizon问题，应对抵达终点添加更严格的约束
- drone env 步进函数 step() 与真实仿真模型相差较大，存在较大误差