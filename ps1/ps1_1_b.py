import torch

# 这是 J loss function 函数的定义
# def cross_entr

# pytorch 不需要我们自己定义 sigmoid，它本身有 sigmoid 函数方法
# y = torch.sigmoid(x)

# 1. 加载样本数据和标签数据 (要记得对 x 进行扩充 “1”)
data_X_path = 'logistic_x.txt'  # X样本文件路径
data_Y_path = 'logistic_y.txt'  # Y标签文件路径
data_X_list = []  # 创建空样本列表存储样本数据
data_Y_list = []  # 创建空标签列表存储样本数据

with open(data_X_path, 'r') as file:
    for line in file.readlines():
        # 假设数据是用空格分隔的
        x1, x2 = line.strip().split()  # 解析每一行的数据
        data_X_list.append([float(x1), float(x2)])  # 转换为浮点数并添加到列表中

with open(data_Y_path, 'r') as file:
    for line in file.readlines():
        y = line.strip()  # 解析每一行的数据
        data_Y_list.append([float(y)])  # 转换为浮点数并添加到列表中

# 将列表转换为Tensor
data_X_tensor = torch.tensor(data_X_list, dtype=torch.double)  # 转换列表为Tensor，类型为 double
data_Y_tensor = torch.tensor(data_Y_list, dtype=torch.double)  # 转换列表为Tensor，类型为 double

# 设置打印选项，precision指定小数点后的位数
torch.set_printoptions(precision=10)

# 获取这个 tensor 的行数
m = data_X_tensor.size(0)

# 创建一个大小为m x 1的全1的列Tensor
ones_column = torch.ones(m, 1)

# 将全1的列Tensor添加到原始Tensor的右侧，产生经过处理的 result_X
result_X = torch.cat((data_X_tensor, ones_column), dim=1)
# 产生经过处理的 result_Y
result_Y = data_Y_tensor

# 打印经过处理的 X矩阵
print(result_X)
print(result_Y)

# 2. 初始化 theta = [0 0 0] 向量    (可以看到这是一个三维向量，这意味着我们可以绘图，观察曲线)
theta = [[0, 0, 0]]
theta = torch.tensor(theta, dtype=torch.double)  # 转换列表为Tensor，类型为 double

# -. 绘制 sigmoid 函数的曲线
# TODO: here

# 3. 根据 theta 绘制 J(theta) 的三维曲面，观察它的样子
print(theta)


print(theta @ result_X[0])

# print(sigmoid(theta, result_X[1]))


zero_tensor = torch.tensor([0], dtype=torch.double)
print(sigmoid(zero_tensor))











