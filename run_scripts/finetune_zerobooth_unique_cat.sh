cd ..

TORCH_DISTRIBUTED_DEBUG=INFO accelerate launch finetune_zerobooth.py \
--config_path "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_finetune_unique_cat.yaml"
# --config_path "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_openimage.yaml"
