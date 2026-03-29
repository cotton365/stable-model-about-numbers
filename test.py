# 从PyTorch导入Adam优化器。
from torch.optim import Adam

# 从PyTorch导入DataLoader类以处理数据集。
from torch.utils.data import DataLoader

# 从torchvision导入数据变换函数。
import torchvision.transforms as transforms

# 从torchvision导入MNIST数据集。
from torchvision.datasets import MNIST

# 特别为笔记本兼容性导入'trange'和'tqdm'。
from tqdm.notebook import trange, tqdm

# 从PyTorch导入学习率调度器。
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR

# 导入用于绘制图形的'matplotlib.pyplot'库。
import matplotlib.pyplot as plt

# 从torchvision.utils导入'make_grid'函数以可视化图像网格。
from torchvision.utils import make_grid

from Unet import *
from loss import *

device = "cuda"
# 定义基于得分的模型并将其移动到指定设备
score_model = torch.nn.DataParallel(UNet_res(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

# 指定是否继续训练或初始化新模型
continue_training = False # 设置为 True 或 False

if not continue_training:
    # 初始化一个新的带 Transformer 的 UNet 模型
    score_model = torch.nn.DataParallel(UNet_Tranformer(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

# 设置训练超参数
n_epochs =   100   # {'type':'integer'}
batch_size =  1024 # {'type':'integer'}
lr = 10e-4         # {'type':'number'}

# 加载 MNIST 数据集并创建数据加载器
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 定义优化器和学习率调度器
optimizer = Adam(score_model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

# 使用 tqdm 显示 epoch 的进度条
tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0

    # 遍历数据加载器中的批次
    for x, y in tqdm(data_loader):
        x = x.to(device)

        # 使用条件得分模型计算损失
        loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    # 使用调度器调整学习率
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]

    # 打印 epoch 信息，包括平均损失和当前学习率
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

    # 在每个 epoch 结束后保存模型检查点
    torch.save(score_model.state_dict(), 'ckpt_transformer.pth')




from Sample import Euler_Maruyama_sampler

# 加载预训练的模型检查点
ckpt = torch.load('ckpt_transformer.pth', map_location=device)
score_model.load_state_dict(ckpt)

# 设置采样批量大小和步骤数
sample_batch_size = 64
num_steps = 500

# 选择Euler-Maruyama采样器
sampler = Euler_Maruyama_sampler

# 生成从 0 到 9 的条件标签 y，以便可视化每个类的生成情况
y_cond = torch.arange(sample_batch_size, device=device) % 10

# 使用指定的采样器生成样本
samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  num_steps=num_steps,
                  device=device,
                  y=y_cond)

# 将样本裁剪到范围[0, 1]
samples = samples.clamp(0.0, 1.0)

# 可视化生成的样本

sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

# 绘制样本网格
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu().detach().numpy(), vmin=0., vmax=1.)
plt.show()