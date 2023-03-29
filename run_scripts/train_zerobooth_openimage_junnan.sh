cd ..

TORCH_DISTRIBUTED_DEBUG=INFO accelerate launch train_zerobooth_openimage_junnan.py \
--config_path "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_openimage_capfilt6b7_junnan.yaml"   
