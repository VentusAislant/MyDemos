#model_path="pretrained_models/openai/clip-vit-large-patch14"
model_path="pretrained_models/dino/dinov2-large"
image_path="data/img_1.png"
output_dir="output"
threshold=0.6

python my_demos/attn_map_demos/visualize_attention.py \
    --model_path $model_path \
    --image_path $image_path \
    --image_size 518 518 \
    --output_dir $output_dir \
    --threshold $threshold