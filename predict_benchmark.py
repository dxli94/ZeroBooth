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
from PIL import Image
import clip

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


def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    return model, preprocess


def get_clip_image_features_dir(dirpath, clip_model, preprocess):
    filenames = sorted(os.listdir(dirpath))
    # keep only images
    filenames = [f for f in filenames if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
    # concatenate with dirpath
    filenames = [os.path.join(dirpath, f) for f in filenames]

    features = []

    for filename in filenames:
        image = Image.open(filename)
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

        features.append(image_features)
    
    return torch.cat(features, dim=0)


def get_clip_text_features(text, clip_model):
    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features


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

def compute_clip_score(clip_model, preprocess, image, ref_features):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    sims = image_features @ ref_features.T

    return sims.mean().item()


def get_scores_by_class_tokens(total_records):
    # records = [{"model_id": "sth", "clip_i_score": 0.3, "clip_t_score": 0.3}]
    records = total_records
    scores_by_class_tokens = {}

    for record in records:
        class_token = record["model_id"]
        clip_i_score = record["clip_i_score"]
        clip_t_score = record["clip_t_score"]

        if class_token not in scores_by_class_tokens:
            scores_by_class_tokens[class_token] = {
                "clip_i_scores": [],
                "clip_t_scores": [],
            }
        
        scores_by_class_tokens[class_token]["clip_i_scores"].append(clip_i_score)
        scores_by_class_tokens[class_token]["clip_t_scores"].append(clip_t_score)

    # take mean
    for class_token in scores_by_class_tokens:
        scores_by_class_tokens[class_token]["clip_i_scores"] = sum(scores_by_class_tokens[class_token]["clip_i_scores"]) / len(scores_by_class_tokens[class_token]["clip_i_scores"])
        scores_by_class_tokens[class_token]["clip_t_scores"] = sum(scores_by_class_tokens[class_token]["clip_t_scores"]) / len(scores_by_class_tokens[class_token]["clip_t_scores"])
    
    return scores_by_class_tokens


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

image_in_base = "/export/home/workspace/dreambooth/diffusers/official_benchmark/dreambooth/dataset"
image_out_base = "/export/home/workspace/dreambooth/diffusers/output/benchmark/images"
if debug:
    image_out_base = "/export/home/workspace/dreambooth/diffusers/output/debug/images"

# csv file looks like this:
# model_id, path_to_best_checkpoint, class_token, is_live_object (bool)
best_ckpt_file = "/export/home/workspace/dreambooth/diffusers/evaluations/best-checkpoints.csv"
df = pd.read_csv(best_ckpt_file, header=None)
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

clip_model, clip_preprocess = load_clip()

# iterate over each model
for index, row in df_rank.iterrows():
    model_id, best_ckpt, class_token, is_live_object = row
    load_checkpoint(best_ckpt)

    image_outdir = os.path.join(image_out_base, now, model_id)
    os.makedirs(image_outdir, exist_ok=True)

    image_indir = os.path.join(image_in_base, model_id)
    ref_image_features = get_clip_image_features_dir(image_indir, clip_model, clip_preprocess)

    if is_live_object:
        prompts = live_prompts
    else:
        prompts = object_prompts
    
    prompts = prompts if not debug else prompts[:2]

    for prompt in prompts:
        prompt_with_class = f"a {class_token} {prompt}"
        ref_text_features = get_clip_text_features(prompt_with_class, clip_model)

        for i in range(num_images):
            this_seed = seed + i
            image = generate_images(class_token, prompt, this_seed)

            image_name = f"""{"-".join(prompt.split()) + str(i)}.jpg"""[:128]
            image_path = os.path.join(image_outdir, image_name)
            image.save(image_path)

            clip_i_scores = compute_clip_score(clip_model, clip_preprocess, image, ref_image_features)
            clip_t_scores = compute_clip_score(clip_model, clip_preprocess, image, ref_text_features)
            print(clip_i_scores, clip_t_scores)

            record = {
                "image_name": image_name,
                "prompt": prompt,
                "model_id": model_id,
                "seed": this_seed,
                "clip_i_score": clip_i_scores,
                "clip_t_score": clip_t_scores,
            }
            records.append(record)

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

    # compute average scores
    total_clip_i_score = 0
    total_clip_t_score = 0

    for record in all_records:
        total_clip_i_score += record["clip_i_score"]
        total_clip_t_score += record["clip_t_score"]

    avg_clip_i_score = total_clip_i_score / len(all_records)
    avg_clip_t_score = total_clip_t_score / len(all_records)

    metrics = {
        "avg_clip_i_score": avg_clip_i_score,
        "avg_clip_t_score": avg_clip_t_score,
    }

    scores_by_class_tokens = get_scores_by_class_tokens(all_records)
    metrics["scores_by_class_tokens"] = scores_by_class_tokens

    with open(os.path.join(image_out_base, now, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

