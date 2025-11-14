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
- `Schedulerの変更＆エポック数`： CosineAnnealingLR & 10
- `unfreeze`
- `知識蒸留の扱い方`
- `Batchsize`: 16

 ## Data Summary
<!-- CSV_TABLE_START-->

<table>
<thead><tr><th>View</th><th>Model</th><th>Loss</th><th>Acc</th><th>macro F1</th></tr></thead>
<tbody>
<tr style="background-color:#ffe5e5;"><td rowspan="4"><b>Top</b></td><td>ResNet50</td><td>0.0292 ± 0.0007</td><td>0.9423 ± 0.0085</td><td>0.8617 ± 0.0586</td></tr>
<tr style="background-color:#ffe5e5;"><td>ViT</td><td>0.0290 ± 0.0006</td><td>0.9423 ± 0.0055</td><td>0.8711 ± 0.0524</td></tr>
<tr style="background-color:#ffe5e5;"><td>HCTNet</td><td><u>0.0285 ± 0.0006</u></td><td><u>0.9479 ± 0.0085</u></td><td>0.8775 ± 0.0480</td></tr>
<tr style="background-color:#ffe5e5;"><td>MidNet</td><td><u>0.0285 ± 0.0008</u></td><td>0.9444 ± 0.0077</td><td><u>0.8805 ± 0.0367</u></td></tr>
<tr style="background-color:#e5f0ff;"><td rowspan="4"><b>Side</b></td><td>ResNet50</td><td>0.0291 ± 0.0007</td><td>0.9409 ± 0.0103</td><td><u>0.8732 ± 0.0496</u></td></tr>
<tr style="background-color:#e5f0ff;"><td>ViT</td><td>0.0322 ± 0.0007</td><td>0.9301 ± 0.0083</td><td>0.8557 ± 0.0532</td></tr>
<tr style="background-color:#e5f0ff;"><td>HCTNet</td><td><u>0.0287 ± 0.0008</u></td><td>0.9453 ± 0.0071</td><td>0.8684 ± 0.0413</td></tr>
<tr style="background-color:#e5f0ff;"><td>MidNet</td><td>0.0290 ± 0.0007</td><td><u>0.9455 ± 0.0042</u></td><td>0.8674 ± 0.0442</td></tr>
<tr style="background-color:#e5ffe5;"><td rowspan="2"><b>Fusion</b></td><td>ResNet50</td><td>0.0285 ± 0.0006</td><td><u>0.9479 ± 0.0090</u></td><td>0.8715 ± 0.0343</td></tr>
<tr style="background-color:#e5ffe5;"><td>ViT</td><td><u>0.0284 ± 0.0007</u></td><td>0.9467 ± 0.0091</td><td><u>0.8873 ± 0.0367</u></td></tr>
</tbody>
</table>

<!-- CSV_TABLE_END-->