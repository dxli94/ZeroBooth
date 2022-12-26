from torch.utils.data import Dataset


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
        superclass_filename="data/imagenet_superclasses.txt",
        **kwargs,
    ):
        self.debug = "debug" in kwargs and kwargs["debug"]

        self.inner_dataset, self.classnames = self.load_imagenet_train()

        self.inp_image_transform = inp_image_transform
        self.tgt_image_transform = tgt_image_transform

        self.text_transform = text_transform

        self.clip_tokenizer = clip_tokenizer

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
