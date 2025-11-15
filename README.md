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
  <thead>
    <tr>
      <th>View</th>
      <th>Model</th>
      <th>Loss</th>
      <th>Acc</th>
      <th>macro F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">Top</td>
      <td>ResNet50</td>
      <td>0.0568 ± 0.0018</td>
      <td>0.9348 ± 0.0125</td>
      <td>0.8579 ± 0.0399</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>0.0577 ± 0.0013</td>
      <td>0.9368 ± 0.0050</td>
      <td>0.8667 ± 0.0259</td>
    </tr>
    <tr>
      <td>HCTNet</td>
      <td>0.0562 ± 0.0016</td>
      <td>0.9476 ± 0.0068</td>
      <td>0.8872 ± 0.0264</td>
    </tr>
    <tr>
      <td>MidNet</td>
      <td>0.0565 ± 0.0013</td>
      <td>0.9423 ± 0.0088</td>
      <td>0.8704 ± 0.0626</td>
    </tr>
    <tr>
      <td rowspan="4">Side</td>
      <td>ResNet50</td>
      <td>0.0571 ± 0.0013</td>
      <td>0.9388 ± 0.0088</td>
      <td>0.8829 ± 0.0425</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>0.0578 ± 0.0014</td>
      <td>0.9319 ± 0.0072</td>
      <td>0.8702 ± 0.0561</td>
    </tr>
    <tr>
      <td>HCTNet</td>
      <td>0.0566 ± 0.0018</td>
      <td>0.9391 ± 0.0101</td>
      <td>0.8628 ± 0.0440</td>
    </tr>
    <tr>
      <td>MidNet</td>
      <td>0.0290 ± 0.0007</td>
      <td>0.9455 ± 0.0042</td>
      <td>0.8674 ± 0.0442</td>
    </tr>
    <tr>
      <td rowspan="3">Fusion</td>
      <td>ResNet50</td>
      <td>0.0285 ± 0.0006</td>
      <td>0.9479 ± 0.0090</td>
      <td>0.8715 ± 0.0343</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>0.0284 ± 0.0007</td>
      <td>0.9467 ± 0.0091</td>
      <td>0.8873 ± 0.0367</td>
    </tr>
    <tr>
      <td>HCTNet</td>
      <td>0.0554 ± 0.0009</td>
      <td>0.9464 ± 0.0066</td>
      <td>0.8827 ± 0.0184</td>
    </tr>
  </tbody>
</table>

<!-- CSV_TABLE_END-->