import torch
import matplotlib.pyplot as plt

# 步骤 2: 创建一个张量，包含从-10到10的一系列点，步长为0.1
x = torch.arange(-10., 10., 0.1)

# 步骤 3: 计算这些点的Sigmoid值
y = torch.sigmoid(x)

# 步骤 4: 使用Matplotlib绘制Sigmoid函数的曲线图
plt.plot(x.numpy(), y.numpy())  # 将张量转换为NumPy数组以用于绘图
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid(True)
plt.show()




