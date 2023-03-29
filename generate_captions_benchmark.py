import os
import yaml
import argparse
from dataset import ImageDirDataset

# # parse arguments
# parser = argparse.ArgumentParser()

# # the first argument is the path to the image dir
# # parser.add_argument("-c", '--image_dir', type=str, help='path to the image directory')
# # parser.add_argument("-s", "--subject", type=str, help="subject of the image directory")

# parser.add_argument("-c", '--config_path', type=str, help='path to the config yaml file')

# args = parser.parse_args()

# config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)

# image_dir = config.get("image_dir")
# subject = config.get("subject")

# root = "/export/home/workspace/dreambooth/diffusers/data/benchmark/few-shot"
root = "/export/home/workspace/dreambooth/diffusers/data/benchmark/one-shot"

dirnames = os.listdir(root)

for dirname in dirnames:
    image_dir = os.path.join(root, dirname)
    subject = dirname.split("-")[-1]

    annotation_path = os.path.join(image_dir, "annotations.json")

    if not os.path.exists(annotation_path):
        ImageDirDataset.generate_annotations(
            subject=subject,
            image_dir=image_dir,
            annotation_path=annotation_path,
        )
    else:
        print("annotations.json already exists")