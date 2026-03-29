# 导入用于张量操作的PyTorch库。
import torch

# 从PyTorch导入神经网络模块。
import torch.nn as nn

# 从PyTorch导入功能操作。
import torch.nn.functional as F


# 从'einops'库导入'rearrange'函数。
from einops import rearrange

# 导入'math'模块以进行数学运算。
import math

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1):
        """
        初始化 CrossAttention 模块。

        参数:
        - embed_dim: 输出嵌入的维度。
        - hidden_dim: 隐藏表示的维度。
        - context_dim: 上下文表示的维度（如果不是自注意力）。
        - num_heads: 注意力头的数量（目前支持1个头）。

        注意: 为了简化实现，假设使用1头注意力。
        可以通过复杂的张量操作实现多头注意力。
        """
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim

        # 查询投影的线性层
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)

        # 判断是自注意力还是交叉注意力
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)

    def forward(self, tokens, context=None):
        """
        CrossAttention 模块的前向传播。

        参数:
        - tokens: 输入的 tokens，形状为 [batch, sequence_len, hidden_dim]。
        - context: 上下文信息，形状为 [batch, context_seq_len, context_dim]。
                   如果 self_attn 为 True，则忽略 context。

        返回:
        - ctx_vecs: 注意力后的上下文向量，形状为 [batch, sequence_len, embed_dim]。
        """
        if self.self_attn:
            # 自注意力情况
            Q = self.query(tokens)
            K = self.key(tokens)
            V = self.value(tokens)
        else:
            # 交叉注意力情况
            Q = self.query(tokens)
            K = self.key(context)
            V = self.value(context)

        # 计算分数矩阵、注意力矩阵和上下文向量
        scoremats = torch.einsum("BTH,BSH->BTS", Q, K)  # Q 和 K 的内积
        attnmats = F.softmax(scoremats / math.sqrt(self.embed_dim), dim=-1)  # scoremats 的 softmax
        ctx_vecs = torch.einsum("BTS,BSH->BTH", attnmats, V)  # 使用 attnmats 加权平均 V 向量

        return ctx_vecs

class TransformerBlock(nn.Module):
    """结合自注意力、交叉注意力和前馈神经网络的 Transformer 块"""
    def __init__(self, hidden_dim, context_dim):
        """
        初始化 TransformerBlock。

        参数:
        - hidden_dim: 隐藏状态的维度。
        - context_dim: 上下文张量的维度。

        注意: 为了简化，自注意力和交叉注意力使用相同的 hidden_dim。
        """
        super(TransformerBlock, self).__init__()

        # 自注意力模块
        self.attn_self = CrossAttention(hidden_dim, hidden_dim)

        # 交叉注意力模块
        self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)

        # 层归一化模块
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # 实现一个具有 3 * hidden_dim 隐藏单元的 2 层 MLP，使用 nn.GELU 激活函数
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.GELU(),
            nn.Linear(3 * hidden_dim, hidden_dim)
        )

    def forward(self, x, context=None):
        """
        TransformerBlock 的前向传播。

        参数:
        - x: 输入张量，形状为 [batch, sequence_len, hidden_dim]。
        - context: 上下文张量，形状为 [batch, context_seq_len, context_dim]。

        返回:
        - x: 经过 TransformerBlock 后的输出张量。
        """
        # 使用层归一化和残差连接应用自注意力
        x = self.attn_self(self.norm1(x)) + x

        # 使用层归一化和残余连接应用交叉注意力
        x = self.attn_cross(self.norm2(x), context=context) + x

        # 使用层归一化和残余连接应用前馈神经网络
        x = self.ffn(self.norm3(x)) + x

        return x


class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        """
        初始化 SpatialTransformer。

        参数:
        - hidden_dim: 隐藏状态的维度。
        - context_dim: 上下文张量的维度。
        """
        super(SpatialTransformer, self).__init__()

        # 用于空间变换的 TransformerBlock
        self.transformer = TransformerBlock(hidden_dim, context_dim)

    def forward(self, x, context=None):
        """
        SpatialTransformer 的前向传播。

        参数:
        - x: 输入张量，形状为 [batch, channels, height, width]。
        - context: 上下文张量，形状为 [batch, context_seq_len, context_dim]。

        返回:
        - x: 经过空间变换后的输出张量。
        """
        b, c, h, w = x.shape
        x_in = x

        # 合并空间维度并将通道维度移动到最后
        x = rearrange(x, "b c h w -> b (h w) c")

        # 应用序列 transformer
        x = self.transformer(x, context)

        # 逆向过程
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        # 残差连接
        return x + x_in
