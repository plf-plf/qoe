import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBloss_xgb:
    #类的构造函数，用于初始化类的属性。
    def __init__(self, per_class_num, beta=0.9999, ):
        """
        :param per_class_num: 每个类别的样本个数;
        per_class_num是一个列表，它表示每个类别的样本个数，如[100, 10]表示有两个类别，第一个类别有100个样本，第二个类别有10个样本。
        :param beta: 超参数，与有效样本数N有关:beta=(N-1)/N
        """
        self.beta = beta
        self.per_class_num = per_class_num

    #是类的主要方法，用于计算每个类别的权重，用于加权交叉熵损失函数。它没有参数，返回一个数值，表示权重的比值
    def get_alpha(self):
        #如[1.0, 0.1]，表示第一个类别的有效样本数为1.0，第二个类别的有效样本数为0.1
        effective_num = 1.0 - np.power(self.beta, self.per_class_num) #计算每个类别的有效样本数
        weights = (1.0 - self.beta) / np.array(effective_num) #计算每个类别的权重
        weights = weights / np.sum(weights) * 2  # 对weights进行标准化
        alpha = weights[1] / weights[0]  # 计算权重，用于加权交叉熵损失函数
        return alpha

    # 待实现
    def CBLOSS_cross_entropy(self):
        return

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # 确保 alpha 在与 inputs 相同的设备上
        # self.alpha = self.alpha.to(inputs.device)

        pt = torch.exp(-BCE_loss)  # 预测为正的概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # Focal loss公式

        return F_loss.mean()  # 返回平均损失
