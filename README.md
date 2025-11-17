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
      <td>0.0561 ± 0.0016</td>
      <td>0.9426 ± 0.0063</td>
      <td>0.8633 ± 0.0535</td>
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
      <td>0.0558 ± 0.0013</td>
      <td>0.9470 ± 0.0089</td>
      <td>0.8967 ± 0.0340</td>
    </tr>
    <tr>
      <td rowspan="3">Fusion</td>
      <td>ResNet50</td>
      <td>0.0553 ± 0.0015</td>
      <td>0.9496 ± 0.0065</td>
      <td>0.8996 ± 0.0272</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>0.0560 ± 0.0014</td>
      <td>0.9438 ± 0.0088</td>
      <td>0.8758 ± 0.0506</td>
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