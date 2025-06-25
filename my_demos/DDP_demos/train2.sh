export CUDA_VISIBLE_DEVICES=5,6,7
export NCCL_P2P_DISABLE=1
# training original model with multi gpu
python DDP2_ddp_training.py \
  --batch_size 128 \
  --lr 1e-3 \
  --save_every 10 \
  --total_epochs 10