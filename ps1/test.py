import torch

# 创建一个一维向量
vector = torch.arange(5)  # tensor([0, 1, 2, 3, 4])
print(vector)

# 转换为行向量
row_vector = vector.view(1, -1)  # tensor([[0, 1, 2, 3, 4]])
print(row_vector)

# 转换为列向量
col_vector = vector.view(-1, 1)
print(col_vector)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4]])

