import os
import torch
import yaml
import torch.distributed as dist
import pandas as pd
from types import SimpleNamespace
import torch.backends.cudnn as cudnn
from lavis.processors.blip_processors import BlipCaptionProcessor
import numpy as np
import random

debug = False
seed = 1234

def get_timestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d%H%M")

now = get_timestamp()

cudnn.benchmark = False
cudnn.deterministic = True

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

default_checkpoint = "/export/home/workspace/dreambooth/diffusers/output/pretrain-202302315-unet-textenc-v1.5-capfilt6b7-synbbox-matting-rr0-drop15-500k/500000"

txt_tsfm = BlipCaptionProcessor()
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

live_prompts = [
    'in the jungle',
    'in the snow',
    'on the beach',
    'on a cobblestone street',
    'on top of pink fabric',
    'on top of a wooden floor',
    'with a city in the background',
    'with a mountain in the background',
    'with a blue house in the background',
    'on top of a purple rug in a forest',
    'wearing a red hat',
    'wearing a santa hat',
    'wearing a rainbow scarf',
    'wearing a black top hat and a monocle',
    'in a chef outfit',
    'in a firefighter outfit',
    'in a police outfit',
    'wearing pink glasses',
    'wearing a yellow shirt',
    'in a purple wizard outfit',
    'coloured red',
    'coloured purple',
    'shiny',
    'wet',
    'cube shaped'
]

object_prompts = [
    'in the jungle',
    'in the snow',
    'on the beach',
    'on a cobblestone street',
    'on top of pink fabric',
    'on top of a wooden floor',
    'with a city in the background',
    'with a mountain in the background',
    'with a blue house in the background',
    'on top of a purple rug in a forest',
    'with a wheat field in the background',
    'with a tree and autumn leaves in the background',
    'with the Eiffel Tower in the background',
    'floating on top of water',
    'floating in an ocean of milk',
    'on top of green grass with sunflowers around it',
    'on top of a mirror',
    'on top of the sidewalk in a crowded street',
    'on top of a dirt road',
    'on top of a white rug',
    'coloured red',
    'coloured purple',
    'shiny',
    'wet',
    'cube shaped'
]


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def create_model():
    from modeling_zerobooth import ZeroBooth
    # load config
    print("Creating model...")
    config_path = "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_openimage.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)

    model = ZeroBooth(config=config.model)
    model = model.to(device)

    print("Finished creating model.")
    return model

def load_checkpoint(checkpoint_path):
    if checkpoint_path:
        print("Loading checkpoint from {} on rank {}...".format(checkpoint_path, get_rank()))
        model.load_checkpoint(checkpoint_path)
        print("Finished loading checkpoint.")

def generate_images(
        class_token,
        prompt,
        seed,
        num_inference_steps=100,
        guidance_scale=7.5,
        num_repeat=15
    ):
    prompt = ", ".join([prompt] * int(num_repeat))
    prompt = "a {} {}".format(class_token, prompt)

    class_names = [txt_tsfm(class_token)]
    prompt = [txt_tsfm(prompt)]
    ctx_begin_pos = [2]

    samples = {
        "class_names": class_names,
        "input_images": None,
        "prompt": prompt,
        "ctx_begin_pos": ctx_begin_pos,
    }

    print(f"Generating image with seed {seed} on rank {get_rank()}...")
    print(f"Prompt: {prompt}")
    output = model.generate(
        samples,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
    )

    return output[0]


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', default=True, type=bool)
args = parser.parse_args()
    
init_distributed_mode(args)

device = torch.device(args.device)
world_size = get_world_size()
rank = get_rank()

num_images = 4

checkpoint_base = "/export/home/workspace/dreambooth/diffusers/output/benchmark/checkpoints"
image_out_base = "/export/home/workspace/dreambooth/diffusers/output/benchmark/images"

# csv file looks like this:
# model_id, path_to_best_checkpoint, class_token, is_live_object (bool)
best_ckpt_file = "/export/home/workspace/dreambooth/diffusers/evaluations/best-checkpoints.csv"
df = pd.read_csv(best_ckpt_file)
df = df if not debug else df[:4]

num_models = len(df)
print(f"Found {num_models} models.")

records = []

item_per_rank = num_models // world_size
start_id = item_per_rank * rank
end_id = min(item_per_rank*(rank+1), num_models)

model = create_model()
df_rank = df[start_id:end_id+1]
print("Rank {} will process {} to {}.".format(rank, start_id, end_id))

# iterate over each model
for index, row in df_rank.iterrows():
    model_id, best_ckpt, class_token, is_live_object = row

    load_checkpoint(best_ckpt)

    image_outdir = os.path.join(image_out_base, now, model_id)
    os.makedirs(image_outdir, exist_ok=True)

    if is_live_object:
        prompts = live_prompts
    else:
        prompts = object_prompts
    
    prompts = prompts if not debug else prompts[:2]

    for prompt in prompts:
        for i in range(num_images):
            this_seed = seed + i
            image = generate_images(class_token, prompt, this_seed)

            image_name = f"""{"-".join(prompt.split()) + str(i)}.jpg"""[:128]
            image_path = os.path.join(image_outdir, image_name)
            image.save(image_path)

            record = {
                "image_name": image_name,
                "prompt": prompt,
                "model_id": model_id,
                "seed": this_seed,
            }
            records.append(record)
            # image.save("dog-bucket.png")

import json

with open(os.path.join(image_out_base, now, "records_%d.json"%rank), "w") as f:
    json.dump(records, f)

dist.barrier()

# merge json files
if rank == 0:
    all_records = []
    for i in range(world_size):
        with open(os.path.join(image_out_base, now, "records_%d.json"%i), "r") as f:
            records = json.load(f)
            all_records += records

    with open(os.path.join(image_out_base, now, "records.json"), "w") as f:
        json.dump(all_records, f)

# python -m torch.distributed.launch --nproc_per_node=2 predict_benchmark.py --world_size 4 --distributed True
