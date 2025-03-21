# 无人机轨迹规划-元动作

基于强化学习控制无人机元动作，实现无人机在大气层内的轨迹规划和姿态控制。

## 数据说明

项目包含两类主要的飞行数据:

- standard_data.csv: 标准飞行数据,包含时间、速度、经纬度、高度等基本参数，轨迹较长
- typical_data.csv: 典型飞行轨迹数据模板, 轨迹较短

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
python multiple.py

# 模型评估
python evaluate.py
```