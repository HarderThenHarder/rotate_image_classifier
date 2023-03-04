# !/usr/bin/env python3
"""
旋转图片角度计算器。

Author: pankeyu
Date: 2022/05/17
"""
import os

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim

from utils import get_filenames
from ImageDataset import RotateImageDataset
from iTrainingLogger import iSummaryWriter


batch_size = 32
n_epoch = 50
input_shape = (3, 244, 244)
log_interval = 5000
eval_interval = 10000

writer = iSummaryWriter(log_path='.', log_name='Rotate Net Training Log')

data_path = os.path.join('/Volumes/Samsung_T5/datasets/训练数据集/', 'street_view')
train_filenames, test_filenames = get_filenames(data_path)

train_dataset = RotateImageDataset(input=train_filenames, input_shape=input_shape, normalize=True)
test_dataset = RotateImageDataset(input=test_filenames, input_shape=input_shape, normalize=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


class RotateNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(2048, 360)                  # 360度，一度一个分类

    def forward(self, x):
        """
        前向传播, 使用resnet 50作为backbone, 后面接一个线性层。

        Args:
            x (_type_): (batch, 3, 224, 224)

        Returns:
            _type_: 360维的一个tensor, 表征属于每一个角度类别的概率
        """
        x = self.model.conv1(x)             # (batch, 64, 122, 122)
        x = self.model.bn1(x)               # (batch, 64, 122, 122)
        x = self.model.relu(x)              # (batch, 64, 122, 122)
        x = self.model.maxpool(x)           # (batch, 64, 61, 61)
        x = self.model.layer1(x)            # (batch, 256, 61, 61)
        x = self.model.layer2(x)            # (batch, 512, 31, 31)
        x = self.model.layer3(x)            # (batch, 1024, 16, 16)
        x = self.model.layer4(x)            # (batch, 2048, 8, 8)
        x = self.model.avgpool(x)           # (batch, 2048, 1, 1)
        x = x.view(x.size(0), x.size(1))    # (batch, 2048)
        x = self.output(x)                  # (batch, 360)

        return x


model = RotateNet()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()


def train():
    """
    训练分类器。
    """
    for i in range(n_epoch):
        for batch_idx, (imgs, targets) in enumerate(train_dataloader):
            imgs, targets = torch.tensor(imgs, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)    # imgs (batch, 244, 3, 3), targets (64, )
            logits = model(imgs)
            logits = F.softmax(logits, dim=-1)
            optimizer.zero_grad()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            current_steps = i * len(train_dataloader) + batch_idx * batch_size
            if current_steps % log_interval == 0:
                writer.add_scalar('train_loss', loss.item(), current_steps)
                writer.record()
            
            if current_steps % eval_interval == 0:
                evaluate(current_steps)


def evaluate(current_steps: int):
    """
    测试训练器的效果。

    Args:
        current_steps (int): _description_
    """
    with torch.no_grad():
        test_loss, correct = 0, 0
        for imgs, targets in test_dataloader:
            imgs, targets = torch.tensor(imgs, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)
            logits = model(imgs)
            test_loss += criterion(logits, targets)
            pred = logits.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_dataloader.dataset)
        writer.add_scalar('eval_loss', test_loss.cpu().item(), current_steps)
        writer.add_scalar('eval_acc', 100. * correct / len(test_dataloader.dataset), current_steps)
        writer.record()
        print('Eval Acc: {:.2f}%'.format(100. * correct / len(test_dataloader.dataset)))
        torch.save(model, 'models/model_{:.2f}.pth'.format(correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    train()