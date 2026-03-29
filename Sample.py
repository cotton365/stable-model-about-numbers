# 导入用于张量操作的PyTorch库。
import torch

# 从PyTorch导入神经网络模块。
import torch.nn as nn

# 从PyTorch导入功能操作。
import torch.nn.functional as F

# 导入用于数值运算的'numpy'库。
import numpy as np

# 导入用于高阶函数的'functools'模块。
import functools

# 从PyTorch导入Adam优化器。
from torch.optim import Adam

# 从PyTorch导入DataLoader类以处理数据集。
from torch.utils.data import DataLoader

# 从torchvision导入数据变换函数。
import torchvision.transforms as transforms

# 从torchvision导入MNIST数据集。
from torchvision.datasets import MNIST

# 导入用于在训练过程中创建进度条的'tqdm'库。
import tqdm

# 特别为笔记本兼容性导入'trange'和'tqdm'。（已修改为终端兼容版）
from tqdm import trange, tqdm

# 从PyTorch导入学习率调度器。
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR

# 导入用于绘制图形的'matplotlib.pyplot'库。
import matplotlib.pyplot as plt

# 从torchvision.utils导入'make_grid'函数以可视化图像网格。
from torchvision.utils import make_grid

# 从'einops'库导入'rearrange'函数。
from einops import rearrange

# 导入'math'模块以进行数学运算。
import math


# 采样步骤数
num_steps = 500


def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           x_shape=(1, 28, 28),
                           num_steps=num_steps,
                           device='cuda',
                           eps=1e-3, y=None):
    """
    使用Euler-Maruyama求解器从基于得分的模型生成样本。

    参数：
    - score_model: 表示时间相关的基于得分的模型的PyTorch模型。
    - marginal_prob_std: 提供扰动核的标准差的函数。
    - diffusion_coeff: 提供SDE的扩散系数的函数。
    - batch_size: 每次调用该函数生成的采样数。
    - x_shape: 样本的形状。
    - num_steps: 采样步骤数，相当于离散化的时间步数。
    - device: 'cuda'表示在GPU上运行，'cpu'表示在CPU上运行。
    - eps: 数值稳定性的最小时间步。
    - y: 目标张量（在此函数中未使用）。

    返回：
    - 样本。
    """

    # 初始化时间和初始样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]

    # 生成时间步
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    # 使用Euler-Maruyama方法采样
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

    # 最后的采样步骤中不包含任何噪声。
    return mean_x
