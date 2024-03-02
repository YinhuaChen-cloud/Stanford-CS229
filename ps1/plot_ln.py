import torch
import matplotlib.pyplot as plt

# 创建一个张量，包含从1到10的一系列点，步长为0.1
x = torch.arange(1, 10, 0.1)

# 计算这些点的自然对数
y = torch.log(x)

# # 使用exp(1)来获取自然常数e的值
# e = torch.exp(torch.tensor(1.0))

# 使用Matplotlib绘制torch.log(x)的曲线图
plt.plot(x.numpy(), y.numpy())  # 将张量转换为NumPy数组以用于绘图
plt.title("Natural Logarithm Function")
plt.xlabel("x")
plt.ylabel("log(x)")
plt.grid(True)
plt.show()


