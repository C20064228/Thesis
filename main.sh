python train.py \
--view 'Top' \
--model 'ResNet18'

python train.py \
--view 'Top' \
--model 'ViT'

python train.py \
--view 'Top' \
--model 'HCTNet'

python train.py \
--view 'Side' \
--model 'ResNet18'

python train.py \
--view 'Side' \
--model 'ViT'

python train.py \
--view 'Side' \
--model 'HCTNet'

python train.py \
--view 'Fusion' \
--model 'ResNet18'

python train.py \
--view 'Fusion' \
--model 'ViT'

python train.py \
--view 'Fusion' \
--model 'HCTNet'

git add .
git commit -m 'Auto commit'
git push origin main