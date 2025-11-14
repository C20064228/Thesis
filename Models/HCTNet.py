import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HCTNet(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.resnet = timm.create_model('resnet50d', pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = True
        resnet_feat = self.resnet.get_classifier().in_features
        self.resnet.fc = nn.Identity()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = True
        vit_feat = self.vit.head.in_features
        self.vit.head = nn.Identity()

        self.resnet_proj = nn.Sequential(
            nn.Linear(resnet_feat, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_feat, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        resnet_feat = self.resnet_proj(self.resnet(x))
        vit_feat = self.vit_proj(self.vit(x))
        fused = torch.cat([resnet_feat, vit_feat], dim=1)
        return self.classifier(fused)

class HCTNet_F(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.resnets = nn.ModuleList([self._create_resnet() for _ in range(2)])
        self.vits = nn.ModuleList([self._create_vit() for _ in range(2)])

        resnet_feat = self.resnets[0].num_features
        vit_feat = self.vits[0].num_features

        self.resnet_proj = nn.Sequential(
            nn.Linear(resnet_feat, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_feat, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

        fused_dim = 256 * 4
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )

    def _create_resnet(self):
        model = timm.create_model('resnet50d', pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        model.fc = nn.Identity()
        return model

    def _create_vit(self):
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        model.head = nn.Identity()
        return model

    def forward(self, top, side):
        resnet_feat = [self.resnet_proj(model(x)) for model, x in zip(self.resnets, [top, side])]
        vit_feat = [self.vit_proj(model(x)) for model, x in zip(self.vits, [top, side])]
        fused_feat = [torch.cat([r, v], dim=1) for r, v in zip(resnet_feat, vit_feat)]
        fused = torch.cat(fused_feat, dim=1)
        return self.classifier(fused)