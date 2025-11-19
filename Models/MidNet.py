import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MidNet(nn.Module):
    def __init__(self, n_classes, dropout=0.3, temperature=4.0):
        super().__init__()
        self.temperature = temperature

        self.resnet = timm.create_model('resnet18', pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = True
        resnet_feat = self.resnet.get_classifier().in_features
        self.resnet.fc = nn.Identity()
        self.resnet_proj = nn.Sequential(
            nn.Linear(resnet_feat, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256))

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = True
        vit_feat = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_feat, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256))

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes))

        self.resnet_kd_head = nn.Linear(256, n_classes)
        self.vit_kd_head = nn.Linear(256, n_classes)

    def forward(self, x, kd=False):
        resnet_feat = self.resnet_proj(self.resnet(x))  # 256次元
        vit_feat = self.vit_proj(self.vit(x))           # 256次元

        fused = torch.cat([resnet_feat, vit_feat], dim=1)
        output = self.classifier(fused)

        if kd:
            resnet_feat = F.normalize(resnet_feat, dim=1)
            vit_feat = F.normalize(vit_feat, dim=1)
            T = self.temperature
            resnet_logits = self.resnet_kd_head(resnet_feat) / T
            vit_logits = self.vit_kd_head(vit_feat) / T
            kl_loss = (
                F.kl_div(F.log_softmax(resnet_logits, dim=1),
                        F.softmax(vit_logits, dim=1),
                        reduction='batchmean') +
                F.kl_div(F.log_softmax(vit_logits, dim=1),
                        F.softmax(resnet_logits, dim=1),
                        reduction='batchmean')
            ) / 2.0
            return output, kl_loss * (T ** 2)

        return output

class MidNet_F(nn.Module):
    def __init__(self, n_classes, dropout=0.3, temperature=4.0):
        super().__init__()
        self.temperature = temperature
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

        self.resnet_kd_head = nn.ModuleList([
            nn.Linear(256, n_classes),
            nn.Linear(256, n_classes)
        ])
        self.vit_kd_head = nn.ModuleList([
            nn.Linear(256, n_classes),
            nn.Linear(256, n_classes)
        ])

        self.fused_kd = nn.ModuleList([
            nn.Linear(512, n_classes),
            nn.Linear(512, n_classes)
        ])

        fused_dim = 256 * 4
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )

    def _create_resnet(self):
        model = timm.create_model('resnet18', pretrained=True)
        for p in model.parameters():
            p.requires_grad = True
        model.fc = nn.Identity()
        return model

    def _create_vit(self):
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        for p in model.parameters():
            p.requires_grad = True
        model.head = nn.Identity()
        return model

    def forward(self, top, side, kd=False):
        resnet_feat = [
            self.resnet_proj(model(x)) for model, x in zip(self.resnets, [top, side])
        ]
        vit_feat = [
            self.vit_proj(model(x)) for model, x in zip(self.vits, [top, side])
        ]

        fused_feat = [
            torch.cat([r, v], dim=1) for r, v in zip(resnet_feat, vit_feat)
        ]

        fused = torch.cat(fused_feat, dim=1)
        logits = self.classifier(fused)

        if not kd:
            return logits

        T = self.temperature
        kd_resvit = 0.0

        for i in range(2):
            r = F.normalize(resnet_feat[i], dim=1)
            v = F.normalize(vit_feat[i], dim=1)
            r_logits = self.resnet_kd_head[i](r) / T
            v_logits = self.vit_kd_head[i](v) / T

            kd_resvit += (
                F.kl_div(F.log_softmax(r_logits, dim=1),
                        F.softmax(v_logits, dim=1),
                        reduction='batchmean')
                +
                F.kl_div(F.log_softmax(v_logits, dim=1),
                        F.softmax(r_logits, dim=1),
                        reduction='batchmean')
            ) / 2.0
        kd_resvit = kd_resvit * (T ** 2)

        kd_topside = 0.0
        for i in range(2):
            f = F.normalize(fused_feat[i], dim=1)
            logits_f = self.fused_kd[i](f) / T

            other = F.normalize(fused_feat[1 - i], dim=1)
            logits_other = self.fused_kd[1 - i](other) / T

            kd_topside += (
                F.kl_div(F.log_softmax(logits_f, dim=1),
                        F.softmax(logits_other, dim=1),
                        reduction='batchmean')
                +
                F.kl_div(F.log_softmax(logits_other, dim=1),
                        F.softmax(logits_f, dim=1),
                        reduction='batchmean')
            ) / 2.0

        kd_topside = kd_topside * (T ** 2)

        kd_total = kd_resvit + kd_topside

        return logits, kd_total