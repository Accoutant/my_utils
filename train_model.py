import torch
from my_utils.loss_model import log_rmse
from d2l import torch as d2l
from torch.utils.data import TensorDataset, DataLoader
from my_utils.process_data import get_k_fold_data
from torch import nn


class TrainerMSE(nn.Module):
    def __init__(self, model, loss, optimizer, lr, k_fold=False):
        """
        损失函数为均方误差的训练器
        :param model: net
        :param loss: nn.MESLoss
        :param optimizer: optimizer
        :param lr: lr
        :param k_fold: 是否为k折交叉验证
        """
        super(TrainerMSE, self).__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), self.lr)
        self.k_fold = k_fold

    def fit(self, train_iter, test_features, test_labels, max_epochs, device):
        """
        :param train_iter: 迭代器
        :param test_features: 验证数据的特征
        :param test_labels: 验证数据的标签
        :return:
        """
        self.model = self.model.to(device)
        if not self.k_fold:
            animator = d2l.Animator(xlabel='epoch', ylabel='loss', legend=['train, test'],
                                    xlim=[0, max_epochs], ylim=[0, 1], figsize=(6, 3))

        for epoch in range(max_epochs):
            loss_list = []
            for x, y in train_iter:
                x = x.to(device)
                y = y.to(device)
                output = self.model(x).squeeze(-1)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # 梯度裁减
                self.optimizer.step()
                loss_rmse = log_rmse(output, y)
                loss_list.append(loss_rmse)

            if test_labels is not None:
                test_features, test_labels = test_features.to(device), test_labels.to(device)
                test_loss = log_rmse(self.model(test_features).squeeze(-1), test_labels)
            else:
                test_loss = 0

            print('| epoch %d | train_loss %.3f | test_loss %.3f |' % (epoch+1, loss_rmse, test_loss))

            if not self.k_fold:
                animator.add(epoch, [loss_rmse, test_loss])


def k_fold_train(k, x, y, trainer, max_epochs, batch_size, device):
    """
    k折交叉验证
    :param k: 折数
    :param x: 打乱后的特征数据
    :param y: 打乱后的标签数据
    :param trainer: 训练器，trainer.fit(self, train_iter, test_features, test_labels, max_epochs, device)
    :param max_epochs: epochs
    :param batch_size: batch_size
    :param device: device
    :return: none
    """
    trainer = trainer
    for i in range(k):
        print('-'*25, 'k_fold: %d' % (i+1), '-'*25)
        # 得到训练数据
        (x_train, y_train), (x_test, y_test) = get_k_fold_data(k, i+1, x, y)
        # 将numpy转为tensor
        x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        x_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
        train_dataset = TensorDataset(x_train, y_train)
        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        trainer.fit(train_iter, x_test, y_test, max_epochs, device)


class TrainKFold(nn.Module):
    def __init__(self, trainer, batch_size):
        """
        k折交叉验证
        :param trainer: 实例化训练器
        :param batch_size: 批次
        """
        super(TrainKFold, self).__init__()
        self.trainer = trainer
        self.batch_size = batch_size

    def fit(self, k, x, y, *args):
        """
        训练
        :param k: 折数
        :param x: 打乱后的特征数据
        :param y: 打乱后的标签数据
        :param args: 训练器trainer.fit所需要的参数，除了train_iter, test_features, test_labels以外
        :return:
        """
        for i in range(k):
            print('-' * 25, 'k_fold: %d' % (i + 1), '-' * 25)
            # 得到训练数据
            (x_train, y_train), (x_test, y_test) = get_k_fold_data(k, i + 1, x, y)
            # 将numpy转为tensor
            x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
            x_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
            train_dataset = TensorDataset(x_train, y_train)
            train_iter = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.trainer.fit(train_iter, x_test, y_test, *args)