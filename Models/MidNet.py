import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MidNet(nn.Module):
    def __init__(self, n_classes, dropout=0.3, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

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
    
        self.resnet_proj = nn.Linear(resnet_feat, 256)
        self.vit_proj = nn.Linear(vit_feat, 256)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x, kd=False):
        resnet_feat = self.resnet(x)
        vit_feat = self.vit(x)

        resnet_feat = self.resnet_proj(resnet_feat)
        vit_feat = self.vit_proj(vit_feat)

        resnet_feat = F.normalize(resnet_feat, dim=1)
        vit_feat = F.normalize(vit_feat, dim=1)

        fused = torch.cat([resnet_feat, vit_feat], dim=1)
        output = self.classifier(fused)

        if kd:
            T = self.temperature
            resnet_logits = resnet_feat / T
            vit_logits = vit_feat / T
            kl_loss = (
                F.kl_div(F.log_softmax(resnet_logits, dim=1),
                         F.softmax(vit_logits, dim=1),
                         reduction='batchmean') +
                F.kl_div(F.log_softmax(vit_logits, dim=1),
                         F.softmax(resnet_logits, dim=1),
                         reduction='batchmean')
            ) / 2.0
            return output, kl_loss * (T ** 2) * self.alpha

        return output