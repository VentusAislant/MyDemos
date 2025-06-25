export CUDA_VISIBLE_DEVICES=1,4
export NCCL_P2P_DISABLE=1

# standalone means single machine with multi gpu
# nproc_per_node means use how many gpu, if value='gpu' then use all
torchrun --standalone --nproc_per_node=gpu DDP4_fsdp_torchrun_training.py \
  --batch_size 128 \
  --lr 1e-3 \
  --save_every 10 \
  --num_epochs 10 \
  --log_every 100 \
  --output_dir ./checkpoints \
  --sharding_strategy zero3 \
  --mixed_precision fp32 \
  --log_strategy epoch
