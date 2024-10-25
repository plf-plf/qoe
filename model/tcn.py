import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# 对输入张量进行裁剪（chomp），即去掉最后几个时间步长的数据;这里是因果卷积的特点，即输入序列长度要大于输出序列长度
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    # 将权重初始化为服从均值为0、标准差为0.01的正态分布
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# #这里才是关键，被后续利用
#````````````````````````````````````````````````````````````
class TemporalConvNet(nn.Module):
    # TCN(input_size,output_size,features_size)
    # 输入通道数，输出通道数，卷积核大小，卷积步长，扩张率，填充，dropout
    def __init__(self, num_inputs, output_size,features_size,num_channels=[128, 128, 128, 128], kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = [] #初始化一个空列表，用于存储网络的各个层
        num_levels = len(num_channels) #计算网络深度，即有多少层 4层

        # 假设输入序列长度为10，输出序列长度为2  output_size是你想要预测的未来时间点的数量
        # 我们需要定义一个合适的num_channels列表来满足这个要求
        # 例如，我们可以使用逐渐减少的通道数 [128, 64, 32, 16] 4层
        # in_channels=10 ; len(num_channels)是通道层数， 这里是定义一个n层卷积
        for i in range(num_levels):
            dilation_size = 2 ** i #扩张卷积（Dilated Convolutions） dilation_size=1,2,4,8
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(num_channels[-1], output_size, 1)  # 最后一层卷积将输出通道数调整为10 卷积核大小为1意味着卷积操作不会跨越时间步，只会在每个时间步上独立地应用
        self.linear = nn.Linear(num_channels[-1]*features_size, output_size) #num_channels[-1]是最后一个卷积层的输出通道数,第一个值应该是进行view之后的output(第二个值));output_size是线性层的输出

    def forward(self, x):#x-->[256,60,51]
        x = self.network(x) #[256,16,51]
        x = self.final_conv(x)  # 应用最后一层卷积 [256,10,51]
        # x=x[:, -10:, -1:]#如果是多-->单，则f_dim为-1；如果是多-->多f_dim为0
        # x = x.view(x.size(0), -1)  # 展平特征以适应线性层
        # x = self.linear(x)  # 应用线性层 [256,10]
        return x

# 当输入数据x传递给TemporalConvNet实例时，它直接通过self.network，
# 即之前定义的顺序模型。
# 这意味着输入数据会依次通过所有的TemporalBlock层s
# x是data
# 定义模型，输入通道数为input_size，输出通道数逐渐减少;可以从128开始，逐渐减半，直到输出长度为10
