python train.py \
--view 'Top' \
--model 'MidNet'

python train.py \
--view 'Side' \
--model 'MidNet'

python train.py \
--view 'Fusion' \
--model 'MidNet'

git add .
git commit -m 'Auto commit'
git push origin main