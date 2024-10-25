import torch
import logging
from torch import nn
import torch.nn.functional as F
from TCAN.tcn_block import TemporalConvNet
# from TCAN.BiTCN_block import TemporalConvNet

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
    )

class TCANet(nn.Module):
    # num_channels=[128, 64, 32, 16] [128, 128, 128, 128]
    # emb_size相当于输入的词向量维度，output_size相当于输出的维度，num_channels是TCANet的卷积核数量，seq_len是输入序列的长度，
    # num_sub_blocks是TCANet的子块数量，temp_attn是是否使用时序注意力，nheads是注意力头的数量，en_res是是否使用残差连接，conv是是否使用卷积核，
    # key_size是卷积核的大小，kernel_size是卷积核的大小，dropout是dropout的概率，wdrop是残差连接的概率，emb_dropout是词向量的dropout的概率，
    # tied_weights是是否使用共享权重，dataset_name是数据集的名称，visual是是否使用可视化

    # key_size=300, lr=1e-4, epochs=200, gpu_id=2, num_subblocks=0, levels=4, en_res=False,
    # temp_attn=True, seq_len=80, valid_len=40, conv=False, visual=False, log="tcanet_num_subblocks-1_levels-4_conv-False
    def __init__(self, emb_size, output_size, features_size,num_channels=[128, 128, 128, 128], num_sub_blocks=2, temp_attn=True, nheads=1, en_res=False,
    conv=True, key_size=300, kernel_size=2, dropout=0.3, emb_dropout=0.1, visual=True):
        super(TCANet, self).__init__()
        self.temp_attn = temp_attn
        self.num_levels = len(num_channels)

        # emb_size,output_size, num_channels, num_sub_blocks, temp_attn, nheads, en_res,conv, key_size, kernel_size, visual, vhdropout=0.0, dropout=0.2
        self.tcanet = TemporalConvNet(emb_size,output_size, num_channels, 
            num_sub_blocks, temp_attn, nheads, en_res, conv, key_size, kernel_size, visual=visual, dropout=dropout)
        self.drop = nn.Dropout(emb_dropout)
        # nn.Linear(num_channels[-1]*features_size, output_size)
        self.decoder = nn.Linear(num_channels[-1], output_size)  # 修改输出维度为1，用于预测单个值
        self.emb_dropout = emb_dropout
        self.final_conv = nn.Conv1d(num_channels[-1], output_size, 1)  # 最后一层卷积将输出通道数调整为10 卷积核大小为1意味着卷积操作不会跨越时间步，只会在每个时间步上独立地应用
        self.init_weights()

    def init_weights(self):

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):

        if self.temp_attn:
            y, attn_weight_list = self.tcanet(input)
            y = self.final_conv(y)
            return y
        # 没有调用，不用管
        else:
            y = self.tcanet(input.transpose(1, 2))
            y = self.decoder(y.transpose(1, 2))
            return y.contiguous()