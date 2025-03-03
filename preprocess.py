import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# data = pd.read_csv('./data/typical_data.csv')

data = pd.read_csv('./data/standard_data.csv')
print(data.head())
print(data.info())

# 处理重复值
print("原始数据行数:", len(data))
data = data.drop_duplicates(subset=['t'])
print("删除重复值后的行数:", len(data))


X = []
y = []

for i in range(len(data)-1):

    # X.append([
    #     data.iloc[i]['V速度'],
    #     data.iloc[i]['lambda经度'], 
    #     data.iloc[i]['phi纬度'],
    #     data.iloc[i]['r地心距'],
    #     data.iloc[i]['psi航向角'],
    #     data.iloc[i]['theta弹道倾角']
    # ])
    
    # y.append([
    #     data.iloc[i+1]['lambda经度'],
    #     data.iloc[i+1]['phi纬度'], 
    #     data.iloc[i+1]['r地心距']
    # ])
    
    X.append([
        data.iloc[i]['v'],
        data.iloc[i]['lambda'],
        data.iloc[i]['phi'], 
        data.iloc[i]['r'],
        data.iloc[i]['psi'],
        data.iloc[i]['theta']
    ])
    
    y.append([
        data.iloc[i+1]['lambda'],
        data.iloc[i+1]['phi'], 
        data.iloc[i+1]['r']
    ])

# 前901条数据作为样本（t），后901条数据作为标签（t+1），共901条训练样本
X = np.array(X)
y = np.array(y)
print("X.shape:", X.shape)
print("y.shape:", y.shape)

np.save('./data/processed_X_standard.npy', X)
np.save('./data/processed_y_standard.npy', y)

print(f"共{len(X)}个训练样本")


