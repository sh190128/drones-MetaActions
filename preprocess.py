import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DATADIR = './data/raw_data'
PROCESSED_DIR = './data/processed_data'

# 获取所有符合模式的CSV文件
csv_files = [f for f in os.listdir(RAW_DATADIR) if f.startswith('output') and f.endswith('-ode1.csv')]

for input_file_name in csv_files:
    # 构建完整的输入输出路径
    input_file = os.path.join(RAW_DATADIR, input_file_name)
    # 从文件名中提取序号，用于构建输出文件名
    base_name = os.path.splitext(input_file_name)[0]
    output_X = os.path.join(PROCESSED_DIR, f'{base_name}_X.npy')
    output_y = os.path.join(PROCESSED_DIR, f'{base_name}_y.npy')
    
    data = pd.read_csv(input_file)
    print(f"\n处理文件: {input_file}")
    # print(data.head())
    # print(data.info())


    print("原始数据行数:", len(data))
    data = data.drop_duplicates(subset=['t'])
    print("删除t重复值后的行数:", len(data))

    X = []
    y = []

    for j in range(len(data)-1):
        X.append([
            data.iloc[j]['v'],
            data.iloc[j]['lambda'],
            data.iloc[j]['phi'], 
            data.iloc[j]['r'],
            data.iloc[j]['psi'],
            data.iloc[j]['theta']
        ])
        
        y.append([
            data.iloc[j+1]['lambda'],
            data.iloc[j+1]['phi'], 
            data.iloc[j+1]['r']
        ])

    X = np.array(X)
    y = np.array(y)
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)


    np.save(output_X, X)
    np.save(output_y, y)
    print(f"已保存处理后的数据到: {output_X} 和 {output_y}")
    print(f"共{len(X)}个训练样本")


