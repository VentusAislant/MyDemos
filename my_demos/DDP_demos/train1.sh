# training original model with single gpu
python DDP1_original_training.py \
  --batch_size 128 \
  --lr 1e-3 \
  --device cuda:1 \
  --save_every 10 \
  --total_epochs 10