# !/usr/bin/env python3
"""
验证训练模型。

Author: pankeyu
Date: 2022/05/19
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from utils import RotNetDataset


input_shape = (3, 244, 244)


class RotateNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 360)

    def forward(self, x):
        """
        前向传播，使用resnet 50作为backbone，后面接一个线性层。

        Args:
            x (_type_): (batch, 3, 224, 224)

        Returns:
            _type_: 360维的一个tensor，表征属于每一个角度类别的概率
        """
        x = self.model(x)

        return x


if __name__ == '__main__':
    with torch.no_grad():
        model = torch.load('models/model_13.pth', map_location=torch.device('cpu')).eval()
        data_path = './img_examples'
        files = os.listdir(data_path)
        labels = [file.split('.')[0].split('_')[1] for file in files]
        files = [os.path.join(data_path, f) for f in files]
        val_dataset = RotNetDataset(files, input_shape=input_shape, rotate=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
        
        for (img, rotate_angle), label in zip(val_dataloader, labels):
            img = img.float()
            logits = model(img)
            logits = F.softmax(logits, dim=-1)
            pred = np.argmax(logits.numpy(), axis=1)
        
        avg_diff = 0
        for p, l in zip(pred, labels):
            print('Infer / Label: {} / {}'.format(p, l))
            avg_diff += abs(int(p) - int(l))

        print('Avg diff: {:.2f}'.format(avg_diff / len(pred)))