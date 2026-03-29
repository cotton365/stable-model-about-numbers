# 导入用于张量操作的PyTorch库。
import torch

# 导入用于数值运算的'numpy'库。
import numpy as np

# 导入用于高阶函数的'functools'模块。
import functools

# 使用GPU
device = "cuda"


# 边际概率标准差函数
def marginal_prob_std(t, sigma):
    """
    计算 $p_{0t}(x(t) | x(0))$ 的均值和标准差。

    参数：
    - t: 时间步向量。
    - sigma: SDE 中的 $\sigma$。

    返回：
    - 标准差。
    """
    # 将时间步转换为PyTorch张量
    t = torch.tensor(t, device=device)

    # 根据给定公式计算并返回标准差
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """
    计算SDE的扩散系数。

    参数：
    - t: 时间步向量。
    - sigma: SDE 中的 $\sigma$。

    返回：
    - 扩散系数向量。
    """
    # 根据给定公式计算并返回扩散系数
    return torch.tensor(sigma**t, device=device)


# Sigma值
sigma = 25.0

# 边际概率标准差
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

# 扩散系数
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """
    用于训练基于得分的生成模型的损失函数。

    参数：
    - model: 表示时间相关的基于得分的模型的PyTorch模型实例。
    - x: 训练数据的小批量。
    - marginal_prob_std: 提供扰动核的标准差的函数。
    - eps: 数值稳定性的容差值。
    """
    # 在范围(eps, 1-eps)内均匀采样时间
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - 2 * eps) + eps
    # 在采样时间`t`找到噪声标准差
    std = marginal_prob_std(random_t)

    # 生成正态分布的噪声
    z = torch.randn_like(x)

    # 使用生成的噪声扰动输入数据
    perturbed_x = x + z * std[:, None, None, None]

    # 使用扰动数据和时间从模型获取得分
    score = model(perturbed_x, random_t)

    # 基于得分和噪声计算损失
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))

    return loss

def loss_fn_cond(model, x, y, marginal_prob_std, eps=1e-5):
    """使用条件信息训练得分生成模型的损失函数。

    参数:
    - model: 表示时间依赖得分模型的 PyTorch 模型实例。
    - x: 一小批训练数据。
    - y: 条件信息（目标张量）。
    - marginal_prob_std: 一个函数，返回扰动核的标准差。
    - eps: 数值稳定性的容差值。

    返回:
    - loss: 计算出的损失。
    """
    # 在范围 [eps, 1-eps] 内均匀采样时间
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    # 生成与输入形状相同的随机噪声
    z = torch.randn_like(x)
    # 计算采样时间下扰动核的标准差
    std = marginal_prob_std(random_t)
    # 用生成的噪声和标准差扰动输入数据
    perturbed_x = x + z * std[:, None, None, None]
    # 获取模型对扰动输入的得分，考虑条件信息
    score = model(perturbed_x, random_t, y=y)
    # 使用得分和扰动计算损失
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    return loss
