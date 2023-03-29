cd ..

TORCH_DISTRIBUTED_DEBUG=INFO accelerate launch train_zerobooth.py \
--config_path "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_unet_only.yaml"
