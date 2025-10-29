import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HCTNet(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.resnet = timm.create_model('resnet50d', pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        resnet_feat = self.resnet.get_classifier().in_features
        self.resnet.fc = nn.Identity()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        vit_feat = self.vit.head.in_features
        self.vit.head = nn.Identity()

        fused_dim = resnet_feat + vit_feat
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        resnet_feat = self.resnet(x)
        vit_feat = self.vit(x)
        fused = torch.cat([resnet_feat, vit_feat], dim=1)
        return self.classifier(fused)