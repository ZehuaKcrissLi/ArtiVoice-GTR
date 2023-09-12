import os
import plotly.express as px
import re
from collections import Counter

# 获取所有文件名
file_path = "/home/kcriss/artivoice/picked_sliced"
files = [f for f in os.listdir(file_path) if f.endswith('.wav')]

# 初始化G, T, R列表
G = []
T = []
R = []

for file in files:
    try:
        basename = os.path.basename(file)  # 获取文件名
        basename = basename.split('_')[0]  # 获取下划线前的部分
        digits = re.findall(r'\d', basename)  # 提取所有数字
        
        # 获取最后三个数字并确保它们都是整数
        g = int(digits[-3])
        t = int(digits[-2])
        r = int(digits[-1])
        
        G.append(g)
        T.append(t)
        R.append(r)
        
    except (ValueError, IndexError) as e:
        print(f"文件名 {file} 不符合预期的格式: {str(e)}")

# 计算每个GTR组合的频率
combinations = list(zip(G, T, R))
frequency = Counter(combinations)

# 获取每个数据点的频率
counts = [frequency[combo] for combo in combinations]

# 创建一个交互式的3D散点图
fig = px.scatter_3d(x=G, y=T, z=R, color=counts, labels={'x': 'Glottalization', 'y': 'Tenseness', 'z': 'Resonance', 'color': 'Frequency'})
# fig = px.scatter_3d(x=G, y=T, z=R, color=counts, labels={'x': 'G Axis', 'y': 'T Axis', 'z': 'R Axis', 'color': 'Frequency'})

# 显示图形
fig.show()

combinations = list(zip(G, T, R))
unique_combinations = set(combinations)

# 打印出每一个唯一的GTR组合
for combo in unique_combinations:
    print(combo)
