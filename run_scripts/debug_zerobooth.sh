export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="output/pretrain-dbg"

cd ..

TORCH_DISTRIBUTED_DEBUG=INFO accelerate launch train_zerobooth.py \
--config_path "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_dbg.yaml" \
--debug

