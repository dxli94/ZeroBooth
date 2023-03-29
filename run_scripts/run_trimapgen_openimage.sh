cd ..

# python -m torch.distributed.run --nproc_per_node=16 trimapgen_openimage.py
python -m torch.distributed.run --nproc_per_node=16 trimapgen_openimage.py
# python -m torch.distributed.run --nproc_per_node=1 maskgen_openimage.py
