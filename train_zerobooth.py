import argparse
import itertools
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import torch
import torch.utils.checkpoint
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from BLIP2.constant import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from dataset import IterLoader, load_dataset
from lavis.processors.blip_processors import BlipCaptionProcessor
from modeling_zerobooth import ZeroBooth
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

from torch.utils.data import DistributedSampler

from diffusers.optimization import get_scheduler

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def create_transforms(config):
    # preprocess
    # blip image transform
    inp_image_transform = transforms.Compose(
        [
            transforms.Resize(
                config.image_size, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )

    inp_image_transform_coco = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.image_size,
                scale=(0.7, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )

    # stable diffusion image transform
    tgt_image_transform = transforms.Compose(
        [
            transforms.Resize(
                config.tgt_image_size, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(config.tgt_image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    text_transform = BlipCaptionProcessor()

    return {
        "inp_image_transform": inp_image_transform,
        "inp_image_transform_coco": inp_image_transform_coco,
        "tgt_image_transform": tgt_image_transform,
        "text_transform": text_transform,
    }


def unwrap_dist_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    # only process rank 0 should log
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if accelerator.is_main_process or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    print(args)
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # if (
    #     args.train_text_encoder
    #     and args.gradient_accumulation_steps > 1
    #     and accelerator.num_processes > 1
    # ):
    #     raise ValueError(
    #         "Gradient accumulation is not supported when training the text encoder in distributed training. "
    #         "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
    #     )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # =============== model ==================
    processors = create_transforms(config)
    model = ZeroBooth(config=config.model)

    optimizer_class = torch.optim.AdamW
    model_params = model.parameters()

    optimizer = optimizer_class(
        model_params,
        lr=float(args.learning_rate),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        weight_decay=float(args.adam_weight_decay),
        eps=float(args.adam_epsilon),
    )

    # ====== Dataset ======
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["target_image"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = (
            unwrap_dist_model(model)
            .tokenizer.pad(
                {"input_ids": input_ids},
                padding="max_length",
                # max_length=tokenizer.model_max_length,
                max_length=25,
                return_tensors="pt",
            )
            .input_ids
        )

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,  # used by stable diffusion
            "input_images": torch.stack(  # used by blip
                [example["input_image"] for example in examples]
            ),
            "class_names": [example["class_name"] for example in examples],
            "ctx_begin_pos": [example["ctx_begin_pos"] for example in examples],
        }

        return batch

    print("Loading dataset")
    # train_dataset = ImageNetDataset(**processors, clip_tokenizer=tokenizer)
    train_dataset = load_dataset(
        dataset_name="imagenet",
        inp_image_transform=processors["inp_image_transform"],
        tgt_image_transform=processors["tgt_image_transform"],
        text_transform=processors["text_transform"],
        clip_tokenizer=model.tokenizer,
        debug=input_args.debug,
    )
    print(f"Loaded {len(train_dataset)} training examples")

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=sampler,
    )

    train_dataset_coco = load_dataset(
        dataset_name="coco",
        inp_image_transform=processors["inp_image_transform_coco"],
        tgt_image_transform=processors["tgt_image_transform"],
        text_transform=processors["text_transform"],
        clip_tokenizer=model.tokenizer,
        debug=input_args.debug,
    )

    sampler = DistributedSampler(
        train_dataset_coco,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )

    train_dataloader_coco = torch.utils.data.DataLoader(
        train_dataset_coco,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=sampler,
    )
    print("Loaded data {} samples.".format(len(train_dataset_coco)))

    train_dataloader = IterLoader(train_dataloader)
    train_dataloader_coco = IterLoader(train_dataloader_coco)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    (
        model,
        optimizer,
        # train_dataloader,
        # train_dataloader_coco,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        # train_dataloader,
        # train_dataloader_coco,
        lr_scheduler,
    )

    # weight_dtype = torch.float32
    # if args.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif args.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # vae.to(accelerator.device, dtype=weight_dtype)
    # if not args.train_text_encoder:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)
    # if not args.train_unet:
    #     unet.to(accelerator.device, dtype=weight_dtype)

    # blip_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers("dreambooth", config=vars(args))
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    # for epoch in range(args.num_train_epochs):
    model.train()

    # for step, batch in enumerate(train_dataloader):
    while True:
        if random.random() < 0.5:
            batch = next(train_dataloader)
        else:
            batch = next(train_dataloader_coco)

        loss = model(batch)

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model_params, args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

        if global_step % args.logging_steps == 0:
            print(logs)

        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        # Create the pipeline using using the trained modules and save it.
        if global_step % args.save_steps == 0:
            print(f"Saving model at step {global_step}.")
            save_to = os.path.join(args.output_dir, f"{global_step}")

            if accelerator.is_main_process:
                unwrap_dist_model(model).save_checkpoint(save_to, accelerator)

            validate(
                model=unwrap_dist_model(model),
                transforms=processors,
                out_dir=os.path.join(save_to, "out_images"),
                rank=accelerator.process_index,
            )

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    accelerator.end_training()


def get_val_dataset():
    img_paths = [
        "/export/home/workspace/dreambooth/diffusers/data/NationalGeographic_2731043_4x3.webp",
        "/export/home/workspace/dreambooth/diffusers/data/NationalGeographic_2731043_4x3.webp",
        "/export/home/workspace/dreambooth/diffusers/data/NationalGeographic_2731043_4x3.webp",
        #
        "/export/home/workspace/dreambooth/diffusers/data/hat-dog.png",
        "/export/home/workspace/dreambooth/diffusers/data/hat-dog.png",
        "/export/home/workspace/dreambooth/diffusers/data/hat-dog.png",
        #
        "/export/home/workspace/dreambooth/diffusers/data/istockphoto-1311993425-170667a.jpeg",
        "/export/home/workspace/dreambooth/diffusers/data/istockphoto-1311993425-170667a.jpeg",
        "/export/home/workspace/dreambooth/diffusers/data/istockphoto-1311993425-170667a.jpeg",
        #
        "/export/home/workspace/dreambooth/diffusers/data/purple-flower.jpg",
    ]

    subj_names = [
        "dog",
        "dog",
        "dog",
        "dog",
        "dog",
        "dog",
        "cat",
        "cat",
        "cat",
        "flower",
    ]

    prompts = [
        "a dog swimming in the ocean",
        "a dog at the grand canyon, photo by National Geographic",
        "a dog wearing a space suit",
        "a dog swimming in the ocean",
        "a dog at the grand canyon, photo by National Geographic",
        "a dog wearing a space suit",
        "a cat swimming in the ocean",
        "a cat at the grand canyon, photo by National Geographic",
        "a cat wearing a space suit",
        "a flower wreath",
    ]

    return img_paths, subj_names, prompts


def validate(model, transforms, out_dir, rank):
    from PIL import Image

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    subj_image_paths, subj_names, prompts = get_val_dataset()
    model.eval()

    inp_tsfm = transforms["inp_image_transform"]
    txt_tsfm = transforms["text_transform"]

    for i, (img_path, subject, prompt) in enumerate(
        zip(subj_image_paths, subj_names, prompts)
    ):
        image = Image.open(img_path).convert("RGB")

        samples = {
            "input_images": inp_tsfm(image).unsqueeze(0).to(model.device),
            "class_names": [txt_tsfm(subject)],
            "prompt": [txt_tsfm(prompt)],
        }

        for gs, theta in [
            (7.5, -1),
            (7.5, 1),
            # (7.5, 2),
            (7.5, 4),
            (7.5, 7),
            # (7.5, 10),
        ]:
            output = model.generate(
                samples,
                seed=3876998111 + int(rank),
                guidance_scale=gs,
                num_inference_steps=250,
                theta=theta,
                disable_bg_model=theta < 0,
            )

            prompt = prompt.replace(" ", "_")
            out_filename = f"{i}_{prompt}_gs={gs}_theta={theta}_rank{rank}.png"
            out_filepath = os.path.join(out_dir, out_filename)

            output[0].save(out_filepath)

    model.train()


if __name__ == "__main__":
    input_args = parse_args()

    config = yaml.load(open(input_args.config_path), Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)

    main(config)
