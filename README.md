## 修正点
- `前処理の修正`
    - RandomResizedCrop
    - ColorJitter
    - RandomAffine
    - RandomErasing（元からあったが、引数を修正）
    - Normalize
- `クラス重み付け損失関数の導入`：
    $$\frac{1}{\sqrt{\log{x}}}$$
- `ResNet18 → ResNet50dへ変更`：
- `macro F1 & Kappa係数の導入`：
- `Schedulerの変更`： CosineAnnealingLR