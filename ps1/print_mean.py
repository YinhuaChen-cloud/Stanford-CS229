import torch

# 创建一个张量
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 计算均值
mean_val = torch.mean(x)

# 打印均值
print(f"Mean value: {mean_val.item()}")

# ==========================================================

# 创建一个二维张量
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 计算每列的均值
mean_col = torch.mean(x, dim=0)

# 计算每行的均值
mean_row = torch.mean(x, dim=1)

print(f"Mean of each column: {mean_col}")
print(f"Mean of each row: {mean_row}")



