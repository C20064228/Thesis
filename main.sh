python train.py \
--view 'Top' \
--model 'ResNet50'

python train.py \
--view 'Side' \
--model 'ResNet50'

python train.py \
--view 'Top' \
--model 'ViT'

python train.py \
--view 'Side' \
--model 'ViT'

git add .
git commit -m 'Auto commit'
git push origin main