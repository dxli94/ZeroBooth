cd ..

# python -m torch.distributed.run --nproc_per_node=16 capfilt_openimage.py
python -m torch.distributed.run --nproc_per_node=16 capfilt_openimage.py
