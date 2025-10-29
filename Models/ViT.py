import torch
import torch.nn as nn
import timm

class ViT(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.model(x)

class ViT_F(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.backbones = nn.ModuleList([self._create_backbone() for _ in range(2)])
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_classes)
        )

    def _create_backbone(self):
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Identity()
        return model

    def forward(self, top, side):
        feats = [model(x) for model, x in zip(self.backbones, [top, side])]
        fused = torch.cat(feats, dim=1)
        return self.classifier(fused)