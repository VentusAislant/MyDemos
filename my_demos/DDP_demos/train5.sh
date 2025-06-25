export CUDA_VISIBLE_DEVICES=0

# nnodes means 2 machine
# node_rank means current machine's rank
# nproc_per_node means use how many gpu, if value='gpu' then use all
# rdzv_id means convergence point, this argument should be identical for all nodes
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=gpu \
  --rdzv_id=1235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=183.169.124.233:12355 \
DDP5_multi_node_torchrun_training.py \
  --batch_size 128 \
  --lr 1e-3 \
  --save_every 10 \
  --total_epochs 10
