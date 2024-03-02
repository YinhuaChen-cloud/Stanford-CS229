import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 步骤1: 创建预测概率的网格
# np.linspace(start, stop, num)函数返回num个在闭区间[start, stop]内均匀分布的样本值。
# start：序列的起始值，在这个例子中是 0.01。
# stop：序列的结束值，在这个例子中是 0.99。
# num：生成的样本数，在这个例子中是 100。
p1 = np.linspace(-100, 100, 200)  # 避免log(0)的情况
# print(p1)
p2 = 0 - p1
# print(p2)
P1, P2 = np.meshgrid(p1, p2)

# 真实标签（这里假设真实标签为类别1）
labels = torch.tensor([1])

# 步骤2: 计算交叉熵损失
# 由于PyTorch的交叉熵函数需要输入和标签来计算损失，
# 我们将预测概率转换成需要的格式并计算每个点的损失
losses = np.zeros(P1.shape)
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        preds = torch.tensor([[P1[i, j], P2[i, j]]], dtype=torch.float)
        loss = F.cross_entropy(preds, labels)
        losses[i, j] = loss.item()

# 步骤3: 绘制曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P1, P2, losses, cmap='viridis')

# 标签和标题
ax.set_xlabel('P(Class=1)')
ax.set_ylabel('P(Class=0)')
ax.set_zlabel('Loss')
ax.set_title('Cross Entropy Loss Surface')

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

