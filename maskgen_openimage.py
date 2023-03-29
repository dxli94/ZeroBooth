import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torch
import torch.backends.cudnn as cudnn
from dataset import OpenImageDataset
from lavis.common.dist_utils import (get_rank, get_world_size,
                                     init_distributed_mode)
from lavis.common.logger import MetricLogger
from torch.utils.data import DataLoader, Dataset

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import CLIPTokenizer
import cv2


class capfilt_dataset(Dataset):
    def __init__(self):

        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="tokenizer",
        )

        self.inner_dataset = OpenImageDataset(
            split=f"{split}",
            # split="train",
            load_cache=True,
            # debug=True,
            # text_transform=lambda x: x,
            clip_tokenizer=clip_tokenizer,
            # simple_prompt=True
        )

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):

        example = self.inner_dataset[index]

        box_image = example["bbox_image"]
        label = example["class_name"]
        image_id = example["image_id"]

        bbox = example["bbox"]
        orig_image = example["image"]

        return box_image, label, image_id, bbox, orig_image
    
    def collate_fn(self, batch):
        images = []
        labels = []
        image_ids = []

        bbox_list = []
        orig_image_list = []

        for image, label, image_id, bbox, orig_image in batch:
            images.append(image)
            labels.append(label)
            image_ids.append(image_id)

            bbox_list.append(bbox)
            orig_image_list.append(orig_image)

        return images, labels, image_ids, bbox_list, orig_image_list


def pad_bbox_to_full_image(bbox_image, image_size, bbox):
    # first generate a 0 image with the same size as the original image
    full_image = Image.new("RGB", image_size, (0, 0, 0))

    # get width and height of the full image (PIL Image)
    full_width, full_height = full_image.size

    # bbox is by ratio, so we need to convert it to pixel
    # paste the bbox image to the full image
    upper_left_corner = int(bbox[0] * full_width), int(bbox[1] * full_height)
    full_image.paste(bbox_image, upper_left_corner)

    return full_image


@torch.no_grad()
def main(args):
    init_distributed_mode(args)

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    dataset = capfilt_dataset()
    print("dataset size: ", len(dataset))
    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=5,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
        collate_fn=dataset.collate_fn,
        # shuffle=False,
        # shuffle=True,
    )

    metric_logger = MetricLogger(delimiter="  ")
    header = "ClipSeg:"
    print_freq = 10

    with torch.no_grad():
        for batch_id, (images, labels, image_ids, bbox, orig_image) in enumerate(
            metric_logger.log_every(dataloader, print_freq, header)
        ):
            inputs = processor(
                text=labels,
                images=images,
                padding="max_length",
                return_tensors="pt",
            )

            # move inputs to device
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            # inputs = inputs.to(device)

            logits = model(**inputs).logits
            heatmap = torch.sigmoid(logits).cpu()

            # save the heatmap
            for i in range(len(heatmap)):
                this_bbox = bbox[i]
                this_image_size = orig_image[i].size

                save_dir = args.output_dir

                image_id = image_ids[i]
                label = labels[i]

                save_path = str(Path(save_dir) / f"{image_id}_{label}.jpg")
                plt.imsave(save_path, heatmap[i])

                img2 = cv2.imread(save_path)
                gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                (thresh, bw_image) = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)

                # fix color format
                cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
                # resize to the original bbox image size
                bw_image = Image.fromarray(bw_image).resize(images[i].size)
                # pad the bbox image to the full image size
                bw_image = pad_bbox_to_full_image(bw_image, this_image_size, this_bbox)
                # save the mask
                bw_image.save(save_path)


if __name__ == "__main__":
    split = "train"
    # split = "validation"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        # default="/export/home/workspace/dreambooth/diffusers/data/openimage-annotations/{}".format(split)
        default="/export/home/workspace/dreambooth/diffusers/data/openimage-mask-full/{}".format(split)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", default=True, type=bool)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
