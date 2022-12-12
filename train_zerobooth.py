import argparse
import itertools
import math
import os
from pathlib import Path

import torch
import torch.utils.checkpoint
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from lavis.processors.blip_processors import BlipCaptionProcessor
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm
from transformers.activations import QuickGELUActivation as QuickGELU

from BLIP2.constant import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from dataset import load_dataset
from diffusers.optimization import get_scheduler
from modeling_zerobooth import ZeroBooth

from types import SimpleNamespace


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
    )
    print(f"Loaded {len(train_dataset)} training examples")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

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

    (model, optimizer, train_dataloader, lr_scheduler,) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
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

    for epoch in range(args.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
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
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                print(f"Saving model at step {global_step}.")

                save_to = os.path.join(args.output_dir, f"{global_step}")
                unwrap_dist_model(model).save_checkpoint(save_to, accelerator)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    input_args = parse_args()

    config = yaml.load(open(input_args.config_path), Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)

    main(config)
