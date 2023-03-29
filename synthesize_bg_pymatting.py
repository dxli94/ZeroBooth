#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import os
import torch
from dataset import OpenImageDataset
from transformers import CLIPTokenizer

import matplotlib.pyplot as plt
import numpy as np
import cv2

import random

from pymatting import *


bg_image_dir = "/export/share/dongxuli/BG59K/images"

split = "train"

from PIL import Image

def sample_bg_image():
    return Image.open(random.choice(image_filenames)).convert("RGB")


def rand_crop_image(image, width, height):
    iw, ih = image.size
    
    if iw < width or ih < height:
        raise ValueError("bg size smaller than bbox size")
        
    left = random.randint(0, iw - width)
    top = random.randint(0, ih - height)
    
    return image.crop((left, top, left+width, top+height))


def synthesize(fg_image, labels, trimap_path, device="cpu", bg_image=None, min_threshold=60, max_threshold=120):
    import random
    rank = random.randint(0, 10000000)
    
    save_fg_to = "/export/home/workspace/dreambooth/diffusers/data/tmp/input_{}.jpg".format(rank)
    save_bg_to = "/export/home/workspace/dreambooth/diffusers/data/tmp/bg_{}.jpg".format(rank)
    save_blend_to = "/export/home/workspace/dreambooth/diffusers/data/tmp/blend_{}.jpg".format(rank)
    
    fg_image_orig_size = fg_image.size
    if fg_image_orig_size[0] > fg_image_orig_size[1]:
        factor = 256 / fg_image_orig_size[1] 
    else:
        factor = 256 / fg_image_orig_size[0]
    fg_image_orig_size = int(fg_image_orig_size[0] * factor), int(fg_image_orig_size[1] * factor)
    
    fg_image = fg_image.resize((256, 256))
    fg_image.save(save_fg_to)
    
    # estimate alpha
    image = load_image(save_fg_to, "RGB", 1, "box")
    trimap = load_image(trimap_path, "GRAY", 256/352, "nearest")
    alpha = estimate_alpha_cf(image, trimap)
    # estimate foreground from image and alpha
    foreground = estimate_foreground_ml(image, alpha)
    
    # sample background crop
    background = sample_bg_image()
    background = rand_crop_image(background, 256, 256).resize((256, 256))
#     print(background.size)
    background.save(save_bg_to)
    background = load_image(save_bg_to, "RGB", 1, "box")
    
#     print(background.shape)
#     print(foreground.shape)
#     print(alpha.shape)
    
    # blend foreground and background
    new_image = blend(foreground, background, alpha)
    save_image(save_blend_to, new_image)
    new_image = Image.open(save_blend_to).convert("RGB").resize(fg_image_orig_size)

    # cleanup
    os.remove(save_fg_to)
    os.remove(save_bg_to)
    os.remove(save_blend_to)

    return new_image


def safe_synthesize(example):
    while True:
        try:
            trimap_dir = "/export/home/workspace/dreambooth/diffusers/data/openimage-trimap/train"

            image_id = example["image_id"]
            label = example["class_name"]

            trimap_path = str(Path(trimap_dir) / f"{image_id}_{'_'.join(label.split(' '))}.png")

            syn_image = synthesize(
                fg_image=example["image"].convert("RGB"),
    #                 bg_image=sample_bg_image(),
                labels=example["class_name"],
                trimap_path=trimap_path,
                # processor=processor,
                # model=model,
            )
        except ValueError as e:
            print(e)
            syn_image = None

        if syn_image is not None:
            break
            
    return syn_image


def syn_and_save(example):
    # print(example["caption"])
    syn_image = safe_synthesize(example)
    
    filename = os.path.basename(example["image_path"])
    # filename = filename.replace(" ", "_")
    outdir = "/export/home/workspace/dreambooth/diffusers/data/openimage-image-syn-matting/train"
    outpath = os.path.join(outdir, filename)

    # print(outpath)
    syn_image.save(outpath)

    return syn_image


if __name__ == "__main__":
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="tokenizer",
    )

    image_filenames = os.listdir(bg_image_dir)
    image_filenames = [os.path.join(bg_image_dir, img_fn) for img_fn in image_filenames]

    dataset = OpenImageDataset(
        split=f"{split}",
        load_cache=True,
        clip_tokenizer=clip_tokenizer,
    )


    from multiprocessing import Pool
    from tqdm import tqdm

    # multiprocessing.set_start_method('spawn')

    from multiprocessing import get_context

    # with get_context("spawn").Pool(2) as pool:
    # # with Pool(96) as pool:
    #     results = list(tqdm(pool.starmap(syn_and_save, dataset), total=len(dataset)))
        # results = pool.starmap(syn_and_save, zip(dataset, [processor] * len(dataset), [model] * len(dataset)))
        
    # for i in tqdm(range(len(dataset))):
    # for i in tqdm(range(100000, 200000)):
    for i in tqdm(range(200000, len(dataset))):
        syn_and_save(dataset[i])


    # In[ ]:




