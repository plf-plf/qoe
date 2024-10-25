import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math
import numpy as np
import time

# from model.optimizations import VariationalHidDropout

from IPython import embed

class BTCN(nn.Module):
    def __init__(self, nodes, groups, dropout):
        super(BTCN, self).__init__()

        kernel_size = 2
        in_channels = nodes
        out_channels = nodes
        stride = 1
        layer1 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=1, padding= 1, dropout=dropout) # padding=(kernel_size - 1) * 1
        layer2 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=2, padding= 2, dropout=dropout)
        layer3 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=4, padding= 4, dropout=dropout)
        layer4 = TemporalBlock(in_channels, out_channels, kernel_size, stride, groups, dilation=4, padding= 4, dropout=dropout)

        self.network_p = nn.Sequential(layer1, layer2, layer3, layer4)
        self.network_b = nn.Sequential(layer1, layer2, layer3, layer4)

    def forward(self, x): # (B, T, N, D)
        B, T, N, D = x.shape
        x = x.reshape(B, T, N*D)

        x = x.transpose(1,2) # (B, N*D, T)
        x_p = self.network_p(x)

        x_re = torch.flip(x,dims=[2]) #  (B, N*D, T)
        x_b = self.network_b(x_re)
        x_b = torch.flip(x_b,dims=[2])
        x = x_p + x_b
        x = x.transpose(1,2) # (B, T, N*D)
        x = x.reshape(B, T, N, D)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class AttentionBlock(nn.Module):
    # key_size600=in_channels = emsize = 600   nhid = 600
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, in_channels)
        self.linear_keys = nn.Linear(in_channels, in_channels)
        self.linear_values = nn.Linear(in_channels, in_channels)
        self.sqrt_key_size = math.sqrt(in_channels) #键大小的平方根，用于缩放点积注意力得分

    def forward(self, input):
        # input is dim (N, in_channels, T) where N is the batch_size, and T is the sequence length
        # 生成一个上三角矩阵掩码，用于屏蔽未来的时间步
        mask = np.array([[1 if i>j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        if input.is_cuda:
            mask = torch.ByteTensor(mask).cuda(input.get_device())
        else:
            mask = torch.ByteTensor(mask)
        # mask = mask.bool()
        
        input = input.permute(0,2,1) # input: [N, T, inchannels]
        keys = self.linear_keys(input) # keys: (N, T, key_size)
        query = self.linear_query(input) # query: (N, T, key_size)
        values = self.linear_values(input) # values: (N, T, value_size)
        # torch.bmm批量矩阵乘法，将 query 和 keys 的转置进行矩阵乘法，得到形状为 (N, T, T) 的 temp 矩阵
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        # masked_fill_ 方法将掩码位置的值填充为负无穷大（-float('inf')），这样在后续的 softmax 操作中，这些位置的权重会变得非常小，接近于零
        mask = mask.bool()
        temp.data.masked_fill_(mask, -float('inf'))
        #  temp 矩阵除以 self.sqrt_key_size 进行缩放，防止数值过大导致梯度消失,得到注意力权重矩阵 weight_temp
        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_vert = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_hori = F.softmax(temp / self.sqrt_key_size, dim=2) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp = (weight_temp_hori + weight_temp_vert)/2
        # 注意力权重矩阵与 values 矩阵进行矩阵乘法，得到注意力加权后的 values 矩阵
        value_attentioned = torch.bmm(weight_temp, values).permute(0,2,1) # shape: (N, T, value_size)
        #   注意力加权后的 values 矩阵，注意力权重矩阵
        return value_attentioned, weight_temp # value_attentioned: [N, in_channels, T], weight_temp: [N, T, T]
    
# TCN(input_size,output_size,features_size)
# 输入通道数，输出通道数，卷积核大小，卷积步长，扩张率，填充，dropout
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, key_size, num_sub_blocks, temp_attn, nheads, en_res, 
                conv, stride, dilation, padding, vhdrop_layer, visual, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # multi head
        self.nheads = nheads
        self.visual = visual
        self.en_res = en_res
        self.conv = conv
        self.temp_attn = temp_attn
        if self.temp_attn:
            if self.nheads > 1:
                self.attentions = [AttentionBlock(n_inputs, key_size, n_inputs) for _ in range(self.nheads)]
                for i, attention in enumerate(self.attentions):
                    self.add_module('attention_{}'.format(i), attention)
                # self.cat_attentions = AttentionBlock(n_inputs * self.nheads, n_inputs, n_inputs)
                self.linear_cat = nn.Linear(n_inputs * self.nheads, n_inputs)
            else:
                self.attention = AttentionBlock(n_inputs, key_size, n_inputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        if self.conv:
            self.net = self._make_layers(num_sub_blocks, n_inputs, n_outputs, kernel_size, stride, dilation, 
                                        padding, vhdrop_layer, dropout)
            self.init_weights()


    def _make_layers(self, num_sub_blocks, n_inputs, n_outputs, kernel_size, stride, dilation, 
                    padding, vhdrop_layer, dropout=0.2):
        layers_list = []

        if vhdrop_layer is not None:
            layers_list.append(vhdrop_layer)
        layers_list.append(
            weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
        layers_list.append(Chomp1d(padding)) 
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(dropout))
        # num_sub_blocks = 2
        for _ in range(num_sub_blocks-1):
            layers_list.append(
                weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
            layers_list.append(Chomp1d(padding)) 
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout))

        return nn.Sequential(*layers_list)

    def init_weights(self):
        layer_idx_list = []
        for name, _ in self.net.named_parameters():
            inlayer_param_list = name.split('.')
            layer_idx_list.append(int(inlayer_param_list[0]))
        layer_idxes = list(set(layer_idx_list))
        for idx in layer_idxes:
            getattr(self.net[idx], 'weight').data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: [N, emb_size, T]
        if self.temp_attn == True:
            en_res_x = None
            if self.nheads > 1:
                # will create some bugs when nheads>1
                # 多头注意力机制：将每个注意力头的输出沿着通道维度（dim=1）拼接起来，形成 x_out
                x_out = torch.cat([att(x) for att in self.attentions], dim=1)
                out = self.net(self.linear_cat(x_out.transpose(1,2)).transpose(1,2))
            else:
                # 单头注意力机制
                # x = x if self.downsample is None else self.downsample(x)
                # attention输出【out_attn=注意力加权后的 values 矩阵，attn_weight=注意力权重矩阵】
                # attention+conv--残差连接
                out_attn, attn_weight = self.attention(x)
                if self.conv:
                    out = self.net(out_attn)
                else:
                    out = out_attn
                # 这一部分是论文中的残差增强连接部分，将注意力权重矩阵与输入进行矩阵乘法，得到残差连接后的输入
                # 对注意力权重在最后一个维度上求和
                weight_x = F.softmax(attn_weight.sum(dim=2),dim=1)
                en_res_x = weight_x.unsqueeze(2).repeat(1,1,x.size(1)).transpose(1,2) * x   #计算加权输入 weight_x*x
                en_res_x = en_res_x if self.downsample is None else self.downsample(en_res_x)
                
            res = x if self.downsample is None else self.downsample(x)

            if self.visual:
                attn_weight_cpu = attn_weight.detach().cpu().numpy()
            else:
                attn_weight_cpu = [0]*10
            del attn_weight
            
            if self.en_res:
                return self.relu(out + res + en_res_x), attn_weight_cpu
            else:
                return self.relu(out + res), attn_weight_cpu

        else:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res) # return: [N, emb_size, T]

# TCN(input_size,output_size,features_size)
# 输入通道数，输出通道数，卷积核大小，卷积步长，扩张率，填充，dropout   kernel_size=2,padding=1,2,4,8;dilation=1,2,4,8
class TemporalConvNet(nn.Module):
    def __init__(self, emb_size,output_size, num_channels, num_sub_blocks, temp_attn, nheads, en_res,
                conv, key_size, kernel_size, visual, vhdropout=0.0, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.vhdrop_layer = None
        # layers.append(nn.Conv1d(emb_size*2, emb_size, 1))
        self.temp_attn = temp_attn
        # self.temp_attn_share = AttentionBlock(emb_size, key_size, emb_size)
        if vhdropout != 0.0:
            print("no vhdropout")
            # self.vhdrop_layer = VariationalHidDropout(vhdropout)
            # self.vhdrop_layer.reset_mask()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = emb_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, key_size, num_sub_blocks, \
                temp_attn, nheads, en_res, conv, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, \
                vhdrop_layer=self.vhdrop_layer, visual=visual, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batchsize, seq_len, emb_size]
        attn_weight_list = []
        if self.temp_attn:
            out = x
            for i in range(len(self.network)):
                out, attn_weight = self.network[i](out)
                # print("the len of attn_weight", len(attn_weight))
                # if len(attn_weight) == 64:
                #     attn_weight_list.append([attn_weight[18], attn_weight[19]])
                attn_weight_list.append([attn_weight[0], attn_weight[-1]])
            return out, attn_weight_list
        else:
            return self.network(x)
