python test_alpha.py \
--view 'Top' \
--model 'MidNet'

python train.py \
--view 'Fusion' \
--model 'ResNet50'

python train.py \
--view 'Fusion' \
--model 'ViT'

git add .
git commit -m 'Auto commit'
git push origin main