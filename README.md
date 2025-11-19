## 修正点
- `前処理の修正`
    - RandomResizedCrop
    - ColorJitter
    - RandomAffine
    - RandomErasing（元からあったが、引数を修正）
    - Normalize
- `クラス重み付け損失関数の導入`：
    $$\frac{1}{\sqrt{\log{x}}}$$
- `macro F1 & Kappa係数の導入`：
- `Schedulerの変更＆エポック数`： CosineAnnealingLR & 10
- `知識蒸留の扱い方`

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
      <td>ResNet18</td>
      <td>0.0296 ± 0.0008</td>
      <td>0.9313 ± 0.0094</td>
      <td>0.8611 ± 0.0488</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>0.0290 ± 0.0006</td>
      <td>0.9377 ± 0.0031</td>
      <td>0.8840 ± 0.0346</td>
    </tr>
    <tr>
      <td>HCTNet</td>
      <td>0.0296 ± 0.0008</td>
      <td>0.9394 ± 0.0073</td>
      <td>0.8769 ± 0.0494</td>
    </tr>
    <tr>
      <td>MidNet</td>
      <td>0.0295 ± 0.0011</td>
      <td>0.9394 ± 0.0114</td>
      <td>0.8644 ± 0.0524</td>
    </tr>
    <tr>
      <td rowspan="4">Side</td>
      <td>ResNet18</td>
      <td>0.0335 ± 0.0007</td>
      <td>0.9045 ± 0.0116</td>
      <td>0.6118 ± 0.0407</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>0.0293 ± 0.0010</td>
      <td>0.9386 ± 0.0091</td>
      <td>0.8733 ± 0.0534</td>
    </tr>
    <tr>
      <td>HCTNet</td>
      <td>0.0297 ± 0.0011</td>
      <td>0.9362 ± 0.0117</td>
      <td>0.8678 ± 0.0404</td>
    </tr>
    <tr>
      <td>MidNet</td>
      <td>0.0295 ± 0.0009</td>
      <td>0.9409 ± 0.0068</td>
      <td>0.8591 ± 0.0394</td>
    </tr>
    <tr>
      <td rowspan="4">Fusion</td>
      <td>ResNet18</td>
      <td>0.0311 ± 0.0009</td>
      <td>0.9255 ± 0.0116</td>
      <td>0.7397 ± 0.0272</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>0.0285 ± 0.0006</td>
      <td>0.9479 ± 0.0105</td>
      <td>0.8941 ± 0.0365</td>
    </tr>
    <tr>
      <td>HCTNet</td>
      <td>0.0285 ± 0.0009</td>
      <td>0.9482 ± 0.0081</td>
      <td>0.8840 ± 0.0491</td>
    </tr>
    <tr>
      <td>MidNet</td>
      <td>0.0552 ± 0.0008</td>
      <td>0.9473 ± 0.0037</td>
      <td>0.8834 ± 0.0169</td>
    </tr>
    <tr>
      <td rowspan="3">nan</td>
      <td>ResNet50</td>
      <td>0.0568 ± 0.0018</td>
      <td>0.9348 ± 0.0125</td>
      <td>0.8579 ± 0.0399</td>
    </tr>
    <tr>
      <td>ResNet50</td>
      <td>0.0571 ± 0.0013</td>
      <td>0.9388 ± 0.0088</td>
      <td>0.8829 ± 0.0425</td>
    </tr>
    <tr>
      <td>ResNet50</td>
      <td>0.0553 ± 0.0015</td>
      <td>0.9496 ± 0.0065</td>
      <td>0.8996 ± 0.0272</td>
    </tr>
  </tbody>
</table>

<!-- CSV_TABLE_END-->