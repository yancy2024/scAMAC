import copy
import logging
import math
from os.path import join as pjoin

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(self.all_head_size, hidden_size)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        #weights = attention_probs if self.vis else None
        #attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        #attention_output = self.proj_dropout(attention_output)
        return attention_output #, weights


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, hidden_size//2)
        self.fc2 = Linear(hidden_size//2, hidden_size)
        self.act_fn = nn.LeakyReLU()
        self.dropout = Dropout(p=0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        # 继承父类初始化
        super(eca_block, self).__init__()

        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size

        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            #                m.bias.data.zero_()

    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]          意为[batch_size, channels, height, width]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        x = x.view([b, 1, c])
        # 1D卷积 [b,1,c]==>[b,1,c]
        x = self.conv(x)
        # 权值归一化
        x = self.sigmoid(x)
        # 维度调整 [b,1,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = x * inputs
        return outputs

# class EMPA(nn.Module):
#     def __init__(self,  num_heads, hidden_size, dropout=0.1):
#         super(EMPA, self).__init__()
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size
#         self.head_size = hidden_size // num_heads
#         self.all_head_size = self.head_size * self.num_heads
#         self.qk_linear = nn.Linear( hidden_size, self.all_head_size, bias=True)
#         self.v_linear = nn.Linear( hidden_size, self.all_head_size, bias=True)
#         self.out_linear = nn.Linear(self.all_head_size, hidden_size, bias=True)
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.dropout = nn.Dropout(dropout)
#         self.attn_drop = nn.Dropout(dropout)
#         self.out_proj = nn.Linear(self.all_head_size, int(hidden_size // 2))
#         self.out_proj2 = nn.Linear(self.all_head_size, int(hidden_size // 2))
#         self.E = self.F = nn.Linear(self.head_size, int(self.head_size // 3))
#
#     def forward(self, x):
#         B, N, C = x.size()
#
#         # shared k and q
#         k = self.qk_linear(x)#.chunk(2, dim=-1)
#         q = self.qk_linear(x)
#         k = k.view(B, N, self.num_heads, self.head_size).transpose(1, 2)
#         q = q.view(B, N, self.num_heads, self.head_size).transpose(1, 2)
#
#         # shared v
#         v = self.v_linear(x).view(B, N, self.num_heads, self.head_size).transpose(1, 2)
#
#         # attention
#         attn_weights = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
#         attn_weights = attn_weights.softmax(dim=-1)
#         attn_weights = self.dropout(attn_weights)
#
#         k_shared_projected = self.E(k)
#         q_shared_projected = self.F(q)
#         # v_SA_projected = self.F(v)
#
#         # attn_CA = (q @ k.transpose(-2, -1)) * self.temperature
#         #
#         # attn_CA = attn_CA.softmax(dim=-1)
#         # attn_CA = self.attn_drop(attn_CA)
#
#         attn_SA = (q_shared_projected @ k_shared_projected.permute(0, 1, 3, 2)) * self.temperature
#
#         attn_SA = attn_SA.softmax(dim=-1)
#         attn_SA = self.attn_drop(attn_SA)
#
#         # output
#         out = attn_weights @ v
#         out = out.transpose(1, 2).contiguous() #.view(B, N, self.hidden_size)
#         new_out = out.size()[:-2] + (self.head_size * self.num_heads,)
#         out = out.view(*new_out)
#         out = self.out_linear(out)
#
#         x_CA = (attn_SA @ v).permute(0, 2, 1, 3).contiguous()
#
#         # x_CA = (attn_CA @ v).permute(0, 2, 1, 3).contiguous()
#         new_x = x_CA.size()[:-2] + (self.head_size * self.num_heads,)
#         x_CA = x_CA.view(*new_x)
#         x_CA  = self.out_linear(x_CA)
#         out = 0.5 * (out + x_CA)
#         return out


class EPA(nn.Module):
    def __init__(self,  num_heads, hidden_size, dropout=0.1):
        super(EPA, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.head_size * self.num_heads
        self.qk_linear = nn.Linear( hidden_size, self.all_head_size, bias=True)
        self.v_linear = nn.Linear( hidden_size, self.all_head_size, bias=True)
        self.out_linear = nn.Linear(self.all_head_size, hidden_size, bias=True)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.all_head_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(self.all_head_size, int(hidden_size // 2))
        self.E = self.F = nn.Linear(self.head_size, int(self.head_size // 3))  #

    def forward(self, x):
        B, N, C = x.size()

        # shared k and q
        k = self.qk_linear(x)#.chunk(2, dim=-1)
        q = self.qk_linear(x)
        k = k.view(B, N, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(B, N, self.num_heads, self.head_size).transpose(1, 2)

        # shared v
        v = self.v_linear(x).view(B, N, self.num_heads, self.head_size).transpose(1, 2)

        # attention
        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        k_shared_projected = self.E(k.transpose(1, 2))
        v_shared_projected = self.F(v.transpose(1, 2))
        # v_SA_projected = self.F(v)

        # attn_CA = (q @ k.transpose(-2, -1)) * self.temperature
        #
        # attn_CA = attn_CA.softmax(dim=-1)
        # attn_CA = self.attn_drop(attn_CA)

        attn_SA = (k_shared_projected.permute(0, 1, 3, 2) @ (q.transpose(1, 2)))#* self.temperature

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop(attn_SA)

        # output
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous() #.view(B, N, self.hidden_size)
        new_out = out.size()[:-2] + (self.head_size * self.num_heads,)
        out = out.view(*new_out)
        out = self.out_linear(out)

        x_CA = (v_shared_projected @ attn_SA).contiguous()

        # x_CA = (attn_CA @ v).permute(0, 2, 1, 3).contiguous()
        new_x = x_CA.size()[:-2] + (self.head_size * self.num_heads,)
        x_CA = x_CA.view(*new_x)
        x_CA  = self.out_linear(x_CA)
        out = 0.5 * (out + x_CA)
        # out = self.out_proj(out)
        # x_CA = self.out_proj2(x_CA)
        # out = torch.cat((out, x_SA), dim=-1)
        return out


class Block(nn.Module):
    def __init__(self, num_attention_heads,hidden_size): # , input_size
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.fc_layer = nn.Linear(hidden_size, 1)
        self.attn = Attention(num_attention_heads, hidden_size)
        self.ffn = eca_block(3)
        self.epa = EPA(num_attention_heads, hidden_size)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.epa(x)
        #
        #x = self.attn(x)
        x = (x+ h).unsqueeze(2).permute([1, 0, 2, 3])    #

        # x = h+x
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = (x + h).permute([1, 0, 2, 3]).squeeze(2)

        return (x[0]+ x[1]+x[2])/3 #, weights +x[3]+ x[4]+x[5]
