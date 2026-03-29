# 导入用于张量操作的PyTorch库。
import torch

# 从PyTorch导入神经网络模块。
import torch.nn as nn

# 导入用于数值运算的'numpy'库。
import numpy as np

# 定义一个用于编码时间步长的高斯随机特征模块
class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        """
        参数：
        - embed_dim：嵌入的维度（输出维度）
        - scale：随机权重（频率）的缩放因子
        """
        super().__init__()

        # 在初始化期间随机采样权重（频率）。这些权重（频率）在优化过程中是固定的，不可训练。
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        """
        参数：
        - x：表示时间步的输入张量
        """
        # 计算余弦和正弦投影：Cosine(2 pi freq x), Sine(2 pi freq x)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi

        # 在最后一个维度上连接正弦和余弦投影
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# 定义一个用于将输出重塑为特征图的全连接层模块
class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        参数：
        - input_dim：输入特征的维度
        - output_dim：输出特征的维度
        """
        super().__init__()

        # 定义一个全连接层
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        参数：
        - x：输入张量

        返回：
        - 经过全连接层并重塑为4D张量（特征图）后的输出张量
        """

        # 应用全连接层并将输出重塑为4D张量
        return self.dense(x)[..., None, None]
        # 这将2D张量广播到4D张量，在空间上添加相同的值。
