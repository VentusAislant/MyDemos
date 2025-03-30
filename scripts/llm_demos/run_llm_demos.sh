export CUDA_VISIBLE_DEVICES=0

#python my_demos/llm_demos/run.py \
#    --llm_type qwen \
#    --llm_name qwen2.5_7b_instruct \
#    --mode multi \
#    --device 'cuda'

python my_demos/llm_demos/run.py \
    --llm_type tinyllama \
    --llm_name tinyllama_1.1b_chat \
    --mode multi \
    --device 'cuda'