cd ..

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_LAUNCH_BLOCKING=1 accelerate launch train_zerobooth_openimage.py \
--config_path "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_openimage_capfilt6b7.yaml"
# --config_path "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_openimage.yaml"
# --config_path "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_openimage_nq=32.yaml"
