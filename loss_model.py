import torch
from torch import nn


def log_rmse(output, target):
    """
    用于相对数的均方误差
    :param output: 神经网络的输出结果
    :param target: 目标数据
    :return: 返回对数的均方误差
    """
    cliped_preds = torch.clamp(output, 1, float('inf'))
    loss = nn.MSELoss()
    rmse_loss = torch.sqrt(loss(torch.log(cliped_preds), torch.log(target)))
    return rmse_loss.item()
