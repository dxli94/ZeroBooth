import json
import os
import random
import time

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def load_dataset(
    dataset_name,
    inp_image_transform,
    tgt_image_transform,
    text_transform,
    clip_tokenizer,
    inp_bbox_transform,
    tgt_bbox_transform,
    msk_bbox_transform,
    **kwargs,
):
    if dataset_name == "imagenet":
        return ImageNetDataset(
            inp_image_transform,
            tgt_image_transform,
            text_transform,
            clip_tokenizer,
            **kwargs,
        )
    elif dataset_name == "coco":
        return COCODataset(
            inp_image_transform,
            tgt_image_transform,
            text_transform,
            clip_tokenizer,
            min_ratio=0.1,
            max_ratio=0.9,
            split="train2017",
            exclude_categories=set(["person", "bed", "dining table", "couch", "truck"]),
            **kwargs,
        )
    elif dataset_name == "openimage":
        return OpenImageDataset(
            inp_image_transform=inp_image_transform,
            inp_bbox_transform=inp_bbox_transform,
            tgt_image_transform=tgt_image_transform,
            tgt_bbox_transform=tgt_bbox_transform,
            msk_bbox_transform=msk_bbox_transform,
            text_transform=text_transform,
            clip_tokenizer=clip_tokenizer,
            split="validation" if "debug" in kwargs and kwargs["debug"] else "train",
        )
    elif dataset_name == "imagedir":
        return ImageDirDataset(
            inp_image_transform=inp_image_transform,
            tgt_image_transform=tgt_image_transform,
            text_transform=text_transform,
            clip_tokenizer=clip_tokenizer,
            subject=kwargs["subject"],
            image_dir=kwargs["image_dir"],
            shuffle_input=kwargs["shuffle_input"],
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


class OpenImageDataset(Dataset):
    def __init__(
        self,
        inp_image_transform=None,
        tgt_image_transform=None,
        inp_bbox_transform=None,
        tgt_bbox_transform=None,
        msk_bbox_transform=None,
        text_transform=None,
        clip_tokenizer=None,
        root_dir="/export/share/dongxuli/fiftyone/open-images-v6",
        split="validation",
        imagedir_path="data",
        maskdir="/export/home/workspace/dreambooth/diffusers/data/openimage-mask-full/",
        bbox_imagedir="/export/home/workspace/dreambooth/diffusers/data/openimage-image-syn-matting/",
        annotation_path="labels/detections.csv",
        filtered_annotation_path="labels/detections_filtered.csv",
        cls_mapping_path="metadata/classes.csv",
        # capfilt_caption_path="labels/capfilt2.json",
        capfilt_caption_path="labels/capfilt_opt6b7.json",
        load_cache=True,
        save_after_filter=False,
        min_size=0.30,
        max_size=0.80,
        **kwargs,
    ):
        self.inp_image_transform = inp_image_transform
        self.inp_bbox_transform = inp_bbox_transform

        self.tgt_image_transform = tgt_image_transform
        self.tgt_bbox_transform = tgt_bbox_transform

        self.msk_bbox_transform = msk_bbox_transform

        self.text_transform = text_transform
        self.clip_tokenizer = clip_tokenizer

        self.root_dir = root_dir
        self.split = split

        self.imagedir_path = os.path.join(root_dir, split, imagedir_path)
        self.maskdir_path = os.path.join(maskdir, split)
        self.bbox_syndir_path = os.path.join(bbox_imagedir, split)

        self.cls_id2name = self._load_cls_id2name(cls_mapping_path)
        self.forbid_labels = self._create_forbid_labels()

        capfilt_caption_path = os.path.join(root_dir, split, capfilt_caption_path)
        self.img2captions = self._load_captions(capfilt_caption_path)

        annotation_path = os.path.join(root_dir, split, annotation_path)
        filtered_annotation_path = os.path.join(
            root_dir, split, filtered_annotation_path
        )

        if load_cache and os.path.exists(filtered_annotation_path):
            self.annotations = pd.read_csv(filtered_annotation_path)
            print("All labels:", len(self.annotations))
        else:
            self.annotations = self._load_annotations(
                annotation_path, min_size, max_size
            )

            if save_after_filter:
                print("Saving filtered labels to", filtered_annotation_path)
                self.annotations.to_csv(filtered_annotation_path, index=False)

    def _load_captions(self, capfilt_caption_path):
        content = json.load(open(capfilt_caption_path))

        img2captions = {}

        for item in content:
            image_id = item["image_id"]
            label = item["label"]

            captions = [f"{cap}, the {label} is" for cap in item["caption"]]

            if image_id not in img2captions:
                img2captions[image_id] = dict()

            img2captions[image_id][label] = captions

        return img2captions

    def _load_annotations(self, annotation_path, min_size, max_size):
        def get_size(row):
            xmin, xmax, ymin, ymax = row["XMin"], row["XMax"], row["YMin"], row["YMax"]

            return (xmax - xmin) * (ymax - ymin)

        def is_extreme_aspect_ratio(row, max_ratio=2):
            xmin, xmax, ymin, ymax = row["XMin"], row["XMax"], row["YMin"], row["YMax"]

            width = xmax - xmin
            height = ymax - ymin

            return width / height > max_ratio or height / width > max_ratio

        # read labels from pandas csv
        annotations = pd.read_csv(annotation_path)

        print("All labels:", len(annotations))
        annotations = annotations[annotations["IsGroupOf"] == 0]
        # labels = labels[labels["IsOccluded"] == 0]
        annotations = annotations[annotations["IsInside"] == 0]
        print("After drop group of, inside:", len(annotations))

        # discard an image if it contains more than 1 object of the same class
        annotations = annotations.drop_duplicates(subset=["ImageID", "LabelName"])
        print("After drop duplicates:", len(annotations))

        # add class name to labels
        annotations["LabelString"] = annotations["LabelName"].apply(
            lambda x: self.cls_id2name[x]
        )
        # filter out forbidden labels
        annotations = annotations[
            ~annotations["LabelName"].apply(self._isin_forbid_labels)
        ]
        print("After filter out forbidden labels:", len(annotations))

        # filter out small objects
        # filter out full-frame objects
        annotations = annotations[annotations.apply(get_size, axis=1) < max_size]
        annotations = annotations[annotations.apply(get_size, axis=1) > min_size]
        print("After filter out small and full-frame objects:", len(annotations))

        # filter out object with extreme aspect ratio
        annotations = annotations[~annotations.apply(is_extreme_aspect_ratio, axis=1)]
        print("After filter out extreme aspect ratio:", len(annotations))

        # filter out images if it contains more than 1 object
        # max_obj = 1
        # labels = labels.groupby("ImageID").filter(lambda x: len(x) <= max_obj)
        # labels = labels.groupby("ImageID").filter(lambda x: len(x) <= 2)
        # print(f"After filter out images with more than {max_obj} objects:", len(labels))

        total_unique_images = len(annotations["ImageID"].unique())
        print("Total unique images:", total_unique_images)

        return annotations

    def _isin_forbid_labels(self, label):
        return label in self.forbid_labels

    def _load_cls_id2name(self, cls_mapping_path):
        cls_mapping_path = os.path.join(self.root_dir, self.split, cls_mapping_path)

        # csv file
        cls_mapping = pd.read_csv(cls_mapping_path, header=None)

        cls_id2name = {}
        for _, row in cls_mapping.iterrows():
            cls_id2name[row[0]] = row[1]

        return cls_id2name

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        image_path = os.path.join(self.imagedir_path, row["ImageID"] + ".jpg")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        label = self.cls_id2name[row["LabelName"]].lower()
        caption = random.choice(self.img2captions[row["ImageID"]][label])
        input_ids = self.clip_tokenizer(
            caption,
            padding="do_not_pad",
            truncation=True,
            max_length=self.clip_tokenizer.model_max_length - 16,
        ).input_ids
        input_ids_label = self.clip_tokenizer(
            f"a {label}",
            padding="do_not_pad",
            truncation=True,
            max_length=self.clip_tokenizer.model_max_length,
        ).input_ids
        ctx_begin_pos = len(input_ids) - 1  # exclude eos token
        ctx_begin_pos_label = 2  # exclude eos token

        # xxyy format
        bbox = (row["XMin"], row["YMin"], row["XMax"], row["YMax"])
        bbox_image = crop_bbox(image, bbox, width, height)

        bbox_mask_filename = row["ImageID"] + "_" + label + ".jpg"
        bbox_mask_path = os.path.join(self.maskdir_path, bbox_mask_filename)
        bbox_mask = Image.open(bbox_mask_path).convert("L")

        bbox_image_syn = os.path.join(self.bbox_syndir_path, bbox_mask_filename)
        bbox_image_syn = Image.open(bbox_image_syn).convert("RGB")
        bbox_image_syn = crop_bbox(bbox_image_syn, bbox, bbox_image_syn.size[0], bbox_image_syn.size[1])

        # transform
        if self.inp_image_transform is not None:
            inp_image = self.inp_image_transform(image)
        else:
            inp_image = None

        if self.inp_bbox_transform is not None:
            # bbox_inp_image = self.inp_bbox_transform(bbox_image)
            bbox_inp_image = self.inp_bbox_transform(bbox_image_syn)
        else:
            bbox_inp_image = None

        if self.tgt_image_transform is not None:
            tgt_image = self.tgt_image_transform(image)
        else:
            tgt_image = None

        if self.tgt_bbox_transform is not None:
            bbox_tgt_image = self.tgt_bbox_transform(bbox_image)
            bbox_mask = self.msk_bbox_transform(bbox_mask)
        else:
            bbox_tgt_image = None

        sample = {
            "image_path": bbox_mask_filename,
            "image": image,
            "input_image": inp_image,
            "target_image": tgt_image,
            #
            "bbox_image": bbox_image,
            "bbox_image_syn": bbox_image_syn,
            "bbox_input_image": bbox_inp_image,
            "bbox_target_image": bbox_tgt_image,
            "bbox_mask": bbox_mask,
            # used by model
            "caption": caption,
            "class_name": label,
            #
            "input_ids": input_ids,
            "ctx_begin_pos": ctx_begin_pos,
            #
            "input_ids_label": input_ids_label,
            "ctx_begin_pos_label": ctx_begin_pos_label,
            # metainfo
            "image_id": row["ImageID"],
            "bbox": bbox,
            "row": row,
        }

        return sample

    def get_label_distribution(self):
        label_counts = self.annotations["LabelName"].value_counts()
        label_counts = label_counts.rename(self.cls_id2name)

        return label_counts

    def _isin_forbid_labels(self, label):
        return self.cls_id2name[label] in self.forbid_labels

    def _create_forbid_labels(self):
        flabels = set(
            [
                # remove all people related labels
                "Boy",
                "Girl",
                "Person",
                "Man",
                "Mammal",
                "Woman",
                "Human body",
                "Human head",
                "Human hair",
                "Human arm",
                "Human face",
                "Human leg",
                "Human hand",
                "Human foot",
                "Human eye",
                "Human mouth",
                "Human nose",
                "Human ear",
                "Clothing",  # oftentimes contain a human
                "Suit",  # oftentimes contain a human
                # "Coat",  # same reason as suit
                # "Dress",  # same reason as suit
                "Tree",  # usually hardly contain visual details, e.g. silhouette-like
                "Plant",  # same reason as tree
                "Houseplant",  # same reason as tree
                "Desk",  # can be too crowded with other objects
                "Table",  # same reason as desk
                "Poster",  # quite text-heavy, and contain very different visual concepts
                "Billboard",  # same reason as poster
            ]
        )

        return flabels


class ImageNetDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(
        self,
        inp_image_transform,
        tgt_image_transform,
        text_transform,
        clip_tokenizer,
        ctx_special_token="sks",
        superclass_filename="data/imagenet_superclasses.txt",
        **kwargs,
    ):
        self.debug = "debug" in kwargs and kwargs["debug"]

        self.inner_dataset, self.classnames = self.load_imagenet_train()

        self.inp_image_transform = inp_image_transform
        self.tgt_image_transform = tgt_image_transform

        self.text_transform = text_transform

        self.clip_tokenizer = clip_tokenizer
        self.ctx_special_token = ctx_special_token

        self.prompt = "A {}."

        self.superclass_filename = superclass_filename
        self.label2sclassnames = self.load_superclass_names()

    def __len__(self):
        return len(self.inner_dataset)

    def get_classname(self, label_id):
        return self.classnames[label_id]

    def get_superclassname(self, label_id):
        return self.label2sclassnames[label_id]

    def __getitem__(self, index):
        example = self.inner_dataset[index]

        example["input_image"] = self.inp_image_transform(example["image"])
        example["target_image"] = self.tgt_image_transform(example["image"])

        class_name = self.text_transform(self.get_classname(example["label"]))
        superclass_name = self.text_transform(self.get_superclassname(example["label"]))

        example["class_name"] = ", ".join([class_name, superclass_name])

        prompt = self.text_transform(self.prompt_from_label(example["label"]))

        # ctx begin position
        example["ctx_begin_pos"] = 2

        example["instance_prompt_ids"] = self.clip_tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            # max_length=self.clip_tokenizer.model_max_length,
            max_length=25,
        ).input_ids

        return example

    def prompt_from_label(self, label):
        classname = self.get_superclassname(label)
        prompt = self.prompt.format(classname)

        return prompt

    def load_imagenet(self):
        from lavis.datasets.builders import load_dataset

        if self.debug:
            dataset = load_dataset(
                name="imagenet",
                # cfg_path="/export/home/workspace/LAVIS/lavis/configs/datasets/imagenet/defaults_val.yaml",
            )["val"]
        else:
            dataset = load_dataset("imagenet")["train"]

        return dataset

    def load_imagenet_train(self):
        dataset = self.load_imagenet()
        classnames = dataset.classnames

        return dataset, classnames

    def load_superclass_names(self):
        with open(self.superclass_filename, "r") as f:
            classnames = f.readlines()

        classnames = [c.strip() for c in classnames]

        label2sclassnames = {}
        for i, classname in enumerate(classnames):
            label2sclassnames[i] = classname

        return label2sclassnames


class COCODataset(Dataset):
    def __init__(
        self,
        inp_image_transform=None,
        tgt_image_transform=None,
        text_transform=None,
        clip_tokenizer=None,
        ctx_special_token="sks",
        annotation_path="data/coco2017-annotations",
        image_dir="/export/share/datasets/vision/coco",
        split="train2017",
        iscrowd=False,
        min_area=32768,
        min_ratio=0.10,
        max_ratio=0.90,
        crop_pad=32,
        exclude_categories=set(["person"]),
        **kwargs,
    ):
        self.inp_image_transform = inp_image_transform
        self.tgt_image_transform = tgt_image_transform

        self.text_transform = text_transform

        self.clip_tokenizer = clip_tokenizer
        self.ctx_special_token = ctx_special_token

        self.iscrowd = iscrowd
        self.min_area = min_area
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.crop_pad = crop_pad
        self.exclude_categories = exclude_categories

        self.annotation_path = os.path.join(annotation_path, f"instances_{split}.json")
        self.caption_path = os.path.join(
            annotation_path, f"capfilt_captions_{split}.json"
        )
        self.image_dir = os.path.join(image_dir, split)

        (
            self.annotations,
            self.category_id2name,
            self.image_id2info,
        ) = self.load_annotations()

    def load_annotations(self):
        import json

        with open(self.annotation_path, "r") as f:
            content = json.load(f)

        images = content["images"]
        annotations = content["annotations"]
        categories = content["categories"]

        category_id2name = {}
        for category in categories:
            category_id2name[category["id"]] = category["name"]

        image_id2info = {}
        for img in images:
            image_id2info[img["id"]] = img

        filtered_annotations = []
        for annotation in annotations:
            if bool(annotation["iscrowd"]) != self.iscrowd:
                continue

            if category_id2name[annotation["category_id"]] in self.exclude_categories:
                continue

            if annotation["area"] < self.min_area:
                image_width = image_id2info[annotation["image_id"]]["width"]
                image_height = image_id2info[annotation["image_id"]]["height"]
                ratio = annotation["area"] / (image_width * image_height)

                if ratio < self.min_ratio:
                    continue

            if annotation["area"] / (image_width * image_height) > self.max_ratio:
                continue

            filtered_annotations.append(annotation)

        capfilt_captions = json.load(open(self.caption_path, "r"))

        for annotation in capfilt_captions:
            image_id = annotation["image_id"]

            image_info = image_id2info[image_id]

            label = annotation["label"]
            caption = annotation["caption"]

            label2caption = {label: caption}
            if "labelled_captions" not in image_info:
                image_info["labelled_captions"] = label2caption
            else:
                if label in image_info["labelled_captions"]:
                    image_info["labelled_captions"][label].extend(caption)
                else:
                    image_info["labelled_captions"].update(label2caption)

        return filtered_annotations, category_id2name, image_id2info

    def __len__(self):
        return len(self.annotations)

    def crop_bbox(self, image, bbox):
        xmin, ymin, width, height = bbox

        # random offset
        max_w_offset = width / 3
        max_h_offset = height / 3

        w_offset = random.uniform(-max_w_offset, max_w_offset)
        h_offset = random.uniform(-max_h_offset, max_h_offset)

        xmin += w_offset
        ymin += h_offset

        xmax = xmin + width
        ymax = ymin + height

        # crop a region that contains the bbox
        # add padding
        xmin -= self.crop_pad
        ymin -= self.crop_pad
        xmax += self.crop_pad
        ymax += self.crop_pad

        # clip
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.width, xmax)
        ymax = min(image.height, ymax)

        return image.crop((xmin, ymin, xmax, ymax))

    def __getitem__(self, index):
        annotation = self.annotations[index]

        image_id = annotation["image_id"]
        image_info = self.image_id2info[image_id]

        image_path = os.path.join(self.image_dir, image_info["file_name"])

        image = Image.open(image_path).convert("RGB")

        example = {
            "image": image,
            "image_id": image_id,
            "image_path": image_path,
            "label": annotation["category_id"],
            "bbox": annotation["bbox"],
        }

        # image tensors
        if self.inp_image_transform is not None:
            example["input_image"] = self.inp_image_transform(
                self.crop_bbox(example["image"], example["bbox"])
            )

        if self.tgt_image_transform is not None:
            example["target_image"] = self.tgt_image_transform(example["image"])

        # text: class name and prompt
        # class name used by BLIP referring expression
        # prompt used by stable diffusion
        raw_class_name = self.get_classname(example["label"])

        if self.text_transform is not None:
            class_name = self.text_transform(raw_class_name)

        class_name = raw_class_name

        example["class_name"] = class_name
        example["captions"] = image_info["labelled_captions"][raw_class_name]
        if len(example["captions"]) == 0:
            example["captions"] = ["A {}".format(raw_class_name)]

        # sample a caption, sample first caption with largest weight, last with least
        prompts = example["captions"][::-1]

        total_shares = sum(range(len(prompts))) + len(prompts)
        weights = [i / total_shares for i in range(len(prompts))]
        prompt = random.choices(prompts, weights=weights)[0]

        if self.clip_tokenizer is not None:
            example["instance_prompt_ids"] = self.clip_tokenizer(
                prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.clip_tokenizer.model_max_length,
            ).input_ids

        # ctx begin position
        overlap_word_start = prompt.index(raw_class_name)
        # trace back to the first space before the class name
        while overlap_word_start > 0 and prompt[overlap_word_start - 1] != " ":
            overlap_word_start -= 1
        # end at the first space after the class name or the end of the string
        overlap_word_end = (
            prompt.index(" ", overlap_word_start)
            if " " in prompt[overlap_word_start:]
            else len(prompt)
        )

        overlap_word = prompt[overlap_word_start:overlap_word_end]
        label_token_id = self.clip_tokenizer.encode(overlap_word)[1:-1]

        ctx_begin_pos = example["instance_prompt_ids"].index(label_token_id[0])
        example["ctx_begin_pos"] = ctx_begin_pos

        return example

    def get_classname(self, label_id):
        return self.category_id2name[label_id]


class ImageDirDataset(Dataset):
    def __init__(
        self,
        image_dir,
        subject,
        annotation_filename="annotations.json",
        shuffle_input=False,
        inp_image_transform=None,
        tgt_image_transform=None,
        # inp_bbox_transform=None,
        # tgt_bbox_transform=None,
        text_transform=None,
        clip_tokenizer=None,
    ):
        self.image_dir = image_dir
        self.subject = subject

        self.shuffle_input = shuffle_input

        image_paths = os.listdir(self.image_dir)
        # image paths are jpg png webp
        self.image_paths = [
            os.path.join(self.image_dir, imp)
            for imp in image_paths
            if os.path.splitext(imp)[1][1:]
            in ["jpg", "png", "webp", "jpeg", "JPG", "PNG", "WEBP", "JPEG"]
        ]

        # annotation_filepath = os.path.join(image_dir, annotation_filename)

        # if not os.path.exists(annotation_filepath) or force_init_annotations:
        #     print("Generating annotations...")
        #     ImageDirDataset.generate_annotations(
        #         subject, image_dir, annotation_filepath
        #     )

        self.annotations = json.load(open(os.path.join(image_dir, annotation_filename)))
        self.actual_len = len(self.annotations)
        # a hacky way to create infinite long dataset
        self.annotations = self.annotations * 100000

        self.inp_image_transform = inp_image_transform
        self.tgt_image_transform = tgt_image_transform
        # self.inp_bbox_transform = inp_bbox_transform
        # self.tgt_bbox_transform = tgt_bbox_transform

        self.text_transform = text_transform
        self.clip_tokenizer = clip_tokenizer

    def __len__(self):
        return len(self.annotations)

    def get_shuffle_input_image(self):
        # a random index
        # while True:
        index = random.randint(0, len(self) - 1)
        #     if index != forbid_index:
        #         break

        image_basename = self.annotations[index]["image_path"]
        image_path = os.path.join(self.image_dir, image_basename)
        image = Image.open(image_path).convert("RGB")

        print("Getting shuffle input image from {}".format(image_path))

        return image

    def __getitem__(self, index):
        image_basename = self.annotations[index]["image_path"]
        image_path = os.path.join(self.image_dir, image_basename)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        label = self.subject.lower()
        caption = random.choice(self.annotations[index]["captions"])
        input_ids = self.clip_tokenizer(
            caption,
            padding="do_not_pad",
            truncation=True,
            max_length=self.clip_tokenizer.model_max_length,
        ).input_ids
        input_ids_label = self.clip_tokenizer(
            f"a {label}",
            padding="do_not_pad",
            truncation=True,
            max_length=self.clip_tokenizer.model_max_length,
        ).input_ids
        ctx_begin_pos = len(input_ids) - 1  # exclude eos token
        ctx_begin_pos_label = 2  # exclude eos token

        # bbox = tuple(self.annotations[index]["bbox"])
        # bbox_image = crop_bbox(image, bbox, width, height)

        if self.shuffle_input and len(self) > 1:
            input_image = self.get_shuffle_input_image()
        else:
            input_image = image

        # transform
        if self.inp_image_transform is not None:
            inp_image = self.inp_image_transform(input_image)
        else:
            inp_image = input_image

        # if self.inp_bbox_transform is not None:
        #     bbox_inp_image = self.inp_bbox_transform(bbox_image)
        # else:
        #     bbox_inp_image = bbox_image

        if self.tgt_image_transform is not None:
            tgt_image = self.tgt_image_transform(image)
        else:
            tgt_image = image

        # if self.tgt_bbox_transform is not None:
        #     bbox_tgt_image = self.tgt_bbox_transform(bbox_image)
        # else:
        #     bbox_tgt_image = bbox_image

        sample = {
            #
            "image": image,
            "input_image": inp_image,
            "target_image": tgt_image,
            #
            # "bbox_image": bbox_image,
            # "bbox_input_image": bbox_inp_image,
            # "bbox_target_image": bbox_tgt_image,
            # used by model
            "caption": caption,
            "class_name": label,
            #
            "input_ids": input_ids,
            "ctx_begin_pos": ctx_begin_pos,
            #
            "input_ids_label": input_ids_label,
            "ctx_begin_pos_label": ctx_begin_pos_label,
            # metainfo
            # "bbox": bbox,
        }

        return sample

    @classmethod
    def generate_annotations(
        cls, subject, image_dir, annotation_path, n_caps=20, skip_if_exists=True
    ):
        if skip_if_exists and os.path.exists(annotation_path):
            return

        annotations = []

        image_paths = os.listdir(image_dir)
        # image paths are jpg png webp
        image_paths = [
            os.path.join(image_dir, imp)
            for imp in image_paths
            if os.path.splitext(imp)[1][1:]
            in ["jpg", "png", "webp", "jpeg", "JPG", "PNG", "WEBP", "JPEG"]
        ]

        captions = cls._generate_captions(image_paths, subject, n_caps=n_caps)
        # bbox = cls._generate_bbox(image_paths, subject)

        for image_path, caption in zip(image_paths, captions):
            ann = {
                "image_path": os.path.basename(image_path),
                "captions": caption,
            }

            annotations.append(ann)

        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

    @classmethod
    def _generate_captions(cls, image_paths, subject, n_caps):
        import torch.nn.functional as F
        from lavis.models import load_model
        from lavis.models.clip_models.tokenizer import \
            tokenize as clip_tokenize
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_captioner = load_model(
                name="blip_caption",
                model_type="large_coco",
                device=device,
                is_eval=True,
            )
            model_filter = load_model(
                name="clip", model_type="ViT-L-14", device=device, is_eval=True
            )

            normalize = transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
            transform_blip = transforms.Compose(
                [
                    transforms.Resize(
                        (384, 384), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            transform_clip = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            print("Generating captions for {} images...".format(len(image_paths)))
            all_caps = []
            for image_path in tqdm(image_paths):
                image = Image.open(image_path).convert("RGB")

                image_blip = (
                    transform_blip(image).to(device, non_blocking=True).unsqueeze(0)
                )
                image_clip = (
                    transform_clip(image).to(device, non_blocking=True).unsqueeze(0)
                )

                captions_blip = model_captioner.generate(
                    samples={"image": image_blip},
                    use_nucleus_sampling=True,
                    top_p=0.95,
                    min_length=5,
                    num_captions=n_caps,
                )

                tokens_blip = clip_tokenize(captions_blip).to(device)
                image_features = F.normalize(
                    model_filter.encode_image(image_clip), dim=-1
                )
                text_features_blip = F.normalize(
                    model_filter.encode_text(tokens_blip), dim=-1
                )

                sim_blip = torch.bmm(
                    image_features.unsqueeze(1),
                    text_features_blip.view(image_blip.size(0), n_caps, -1).permute(
                        0, 2, 1
                    ),
                )

                _, indices = torch.topk(sim_blip, k=n_caps, dim=-1)
                caps = [
                    "{}, the {} is".format(captions_blip[n], subject)
                    for n in indices[0][0]
                ][:10]

                all_caps.append(caps)

        return all_caps

class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def crop_bbox(image, bbox, width, height):
    xmin, ymin, xmax, ymax = bbox

    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image.width, xmax)
    ymax = min(image.height, ymax)

    return image.crop((xmin, ymin, xmax, ymax))


if __name__ == "__main__":
    # dataset = COCODataset(
    #     inp_image_transform=lambda x: x,
    #     tgt_image_transform=lambda x: x,
    #     text_transform=lambda x: x,
    #     clip_tokenizer=lambda x: x,
    # )

    # print(len(dataset))

    imagedir_dataset = ImageDirDataset(
        image_dir="/export/home/workspace/dreambooth/diffusers/data/alvan-nee",
        subject="dog",
    )

    # ImageDirDataset.generate_annotations(
    #     image_dir="/export/home/workspace/dreambooth/diffusers/data/alvan-nee",
    #     subject="dog",
    # )

    import pdb

    pdb.set_trace()
