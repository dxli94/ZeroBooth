import time
import os
import random
from torch.utils.data import Dataset
from PIL import Image


def load_dataset(
    dataset_name,
    inp_image_transform,
    tgt_image_transform,
    text_transform,
    clip_tokenizer,
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
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


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
                cfg_path="/export/home/workspace/LAVIS/lavis/configs/datasets/imagenet/defaults_val.yaml",
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


if __name__ == "__main__":
    dataset = COCODataset(
        inp_image_transform=lambda x: x,
        tgt_image_transform=lambda x: x,
        text_transform=lambda x: x,
        clip_tokenizer=lambda x: x,
    )

    print(len(dataset))
