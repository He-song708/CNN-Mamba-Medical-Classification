# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
from mamba_ssm.modules.mamba_simple import Mamba


class CNN_Mamba(nn.Module):
    def __init__(self, n_class=10, d_model=512, n_mamba=2, pretrained=True):
        super().__init__()
        # CNN骨干网络
        try:
            backbone = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception as e:
            print(f"⚠️  预训练权重下载失败，改用随机初始化：{e}")
            backbone = resnet18(weights=None)
        self.cnn = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool,
            backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4
        )
        # 特征展平为序列
        self.project = nn.Linear(d_model, d_model)
        # Mamba块
        self.mambas = nn.Sequential(*[Mamba(d_model=d_model) for _ in range(n_mamba)])
        # 分类头
        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, n_class)

    def forward(self, x):
        x = self.cnn(x)  # B, C, h, w
        B, C, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, (h*w), C -> token序列
        x = self.project(x)
        x = self.mambas(x)  # 交给Mamba
        x = self.norm(x.mean(1))  # token均值池化
        return self.cls_head(x)