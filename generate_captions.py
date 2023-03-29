import os
import cv2
import yaml
import argparse
import numpy as np
import random
import torch
from PIL import Image
from dataset import ImageDirDataset
import matplotlib.pyplot as plt

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class ImageSynthesizer:
    def __init__(
            self,
            threshold=100,
            bg_image_dir="/export/home/workspace/dreambooth/diffusers/data/BG1",
            # bg_image_dir="/export/home/workspace/dreambooth/diffusers/data/BG118",
            device="cuda"
        ):
        self.threshold = threshold

        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

        self.bg_image_paths = [os.path.join(bg_image_dir, image_name) for image_name in os.listdir(bg_image_dir)]

        self.device = device

    def synthesize_and_save(self, image_inp_paths, image_out_paths, labels):
        images = [Image.open(image_path) for image_path in image_inp_paths]

        # 1. detect the object mask with clipseg
        heatmaps = self._get_clipseg_heatmap(images, labels)
        # 2. get the foreground mask
        masks = self._get_foreground_mask(heatmaps, images)
        # 3. paste the object to a random background
        syn_images = self._paste_to_background(images, masks)
        # 4. save the synthetic images
        for syn_image, image_out_path in zip(syn_images, image_out_paths):
            syn_image.save(image_out_path)

    def _get_clipseg_heatmap(self, images, prompts):
        inputs = self.clipseg_processor(
            text=prompts,
            images=images,
            padding="max_length",
            return_tensors="pt"
        )

        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        logits = self.clipseg_model(**inputs).logits
        heatmap = torch.sigmoid(logits).detach().cpu()

        # unsqueeze to make it a batch of 1
        if len(heatmap.shape) == 2:
            heatmap = heatmap.unsqueeze(0)

        return heatmap
    
    def _get_foreground_mask(self, heatmap, orig_images):
        masks = []

        for i in range(len(heatmap)):
            save_path = "tmp-bw.jpg"

            plt.imsave(save_path, heatmap[i])

            img2 = cv2.imread(save_path)
            gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            (thresh, bw_image) = cv2.threshold(
                gray_image, self.threshold, 255, cv2.THRESH_BINARY
            )

            # fix color format
            cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
            bw_image = Image.fromarray(bw_image).resize(orig_images[i].size)
            masks.append(bw_image)
        
        return masks
    
    def _get_foreground_bbox(self, masks):
        bboxes = []

        for mask in masks:
            mask = np.array(mask)
            mask = mask[:, :, 0]
            mask = mask.astype(np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])

            bboxes.append((x, y, w, h))
        
        return bboxes
    
    def _draw_bg_image(self):
        bg_image_path = random.choice(self.bg_image_paths)
        bg_image = Image.open(bg_image_path)

        return bg_image

    def _resize_image_with_aspect_ratio(self, image, max_side=368):
        # Determine the new size based on the aspect ratio
        if isinstance(image, np.ndarray):
            width, height = image.shape[1], image.shape[0]
        else:
            width, height = image.size

        long_side = width if width > height else height
        ratio = max_side / long_side
        
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        return image.resize((new_width, new_height)), new_width, new_height

    def _composite(self, fg_image, mask, bg_image=None):
        mask = mask.convert("1")
        if bg_image is None:
            bg_image = Image.new('RGB', fg_image.size)
        return Image.composite(fg_image, bg_image, mask)

    def _rand_crop_image(self, image, width, height):
        iw, ih = image.size
        
        if iw < width or ih < height:
            raise ValueError("bg size smaller than bbox size")
            
        left = random.randint(0, iw - width)
        top = random.randint(0, ih - height)
        
        return image.crop((left, top, left+width, top+height))

    def _paste_to_background(self, images, masks):
        composite_images = []

        for fg_image, mask in zip(images, masks):
            bg_image = self._draw_bg_image()

            fg_image, r_width, r_height = self._resize_image_with_aspect_ratio(fg_image)
            mask, _, _ = self._resize_image_with_aspect_ratio(mask)
            bg_image = self._rand_crop_image(bg_image, r_width, r_height)

            composite_image = self._composite(
                fg_image=fg_image,
                mask=mask,
                bg_image=bg_image
            )

            composite_images.append(composite_image)
        
        return composite_images


def is_image(path):
    img_postfix = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG", ".bmp", ".BMP", ".tif", ".tiff"]

    return any([path.endswith(postfix) for postfix in img_postfix])

def add_suffix(filename_w_ext, suffix="_syn"):
    filename, ext = os.path.splitext(filename_w_ext)
    return filename + suffix + ext

# parse arguments
parser = argparse.ArgumentParser()

# the first argument is the path to the image dir
# parser.add_argument("-c", '--image_dir', type=str, help='path to the image directory')
# parser.add_argument("-s", "--subject", type=str, help="subject of the image directory")

parser.add_argument("-c", '--config_path', type=str, help='path to the config yaml file')

args = parser.parse_args()

config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)

image_dir = config.get("image_dir")
subject = config.get("subject")

annotation_path = os.path.join(image_dir, "annotations.json")

if not os.path.exists(annotation_path):
    ImageDirDataset.generate_annotations(
        subject=subject,
        image_dir=image_dir,
        annotation_path=annotation_path,
    )
else:
    print("annotations.json already exists")

# syn_image_dir = os.path.join(image_dir + "_synthetic_input")

# if not os.path.exists(syn_image_dir):

image_names = [img_name for img_name in os.listdir(image_dir) if is_image(img_name)]

inp_image_paths = [os.path.join(image_dir, image_name) for image_name in image_names]
out_image_paths = [os.path.join(image_dir, add_suffix(image_name)) for image_name in image_names]

labels = [subject] * len(image_names)

if not os.path.exists(out_image_paths[0]):
    print("synthesizing images...")
    synthesizer = ImageSynthesizer(threshold=110)
    synthesizer.synthesize_and_save(
        image_inp_paths=inp_image_paths,
        image_out_paths=out_image_paths,
        labels=labels
    )
else:
    print("synthetic_input already exists")


