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
from dataset import IterLoader, load_dataset, ImageDirDataset
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

    inp_bbox_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.image_size,
                scale=(0.9, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            # transforms.Resize(
            #     config.tgt_image_size, interpolation=InterpolationMode.BICUBIC
            # ),
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

    tgt_bbox_transform = transforms.Compose(
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
        "inp_bbox_transform": inp_bbox_transform,
        "tgt_bbox_transform": tgt_bbox_transform,
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

    # load checkpoint
    print("loading checkpoint: ", config.checkpoint)
    model.load_checkpoint(config.checkpoint)

    # optimization
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
        # random choice from a "referring" batch and a "completion" batch
        # is_referring = random.choice([True, False])
        is_two_stage = True

        if is_two_stage:
            is_referring = random.uniform(0, 1) < 1.0
            # is_referring = False

            if is_referring:
                # in the referring batch:
                #   - the input image is the full image
                #   - the target image is the bbox image
                #   - input caption is "a [v] label"
                # [option 1]
                input_images_key = "input_image"
                # pixel_values_key = "bbox_target_image"
                pixel_values_key = "target_image"
                input_id_key = "input_ids_label"

                ctx_begin_pos_key = "ctx_begin_pos_label"

                # [option 2]
                # input_images_key = "input_image"
                # pixel_values_key = "target_image"
                # input_id_key = "input_ids_label"

                # ctx_begin_pos_key = "ctx_begin_pos_label"
                batch_type = "referring"

            else:  # completion
                # in the completion batch:
                #   - the input image is the bbox image
                #   - the target image is the full image
                #   - input caption is the full caption + "the [v] is"
                input_images_key = "bbox_input_image"
                pixel_values_key = "target_image"
                input_id_key = "input_ids"

                ctx_begin_pos_key = "ctx_begin_pos"

                batch_type = "completion"
        else:
            input_images_key = "input_image"
            pixel_values_key = "target_image"
            input_id_key = "input_ids_label"

            ctx_begin_pos_key = "ctx_begin_pos_label"

            batch_type = "one-stage"

        input_ids = [example[input_id_key] for example in examples]
        ctx_begin_pos = [example[ctx_begin_pos_key] for example in examples]
        pixel_values = (
            torch.stack([example[pixel_values_key] for example in examples])
            .to(memory_format=torch.contiguous_format)
            .float()
        )
        input_images = (
            torch.stack([example[input_images_key] for example in examples])
            .to(memory_format=torch.contiguous_format)
            .float()
        )

        input_ids = (
            unwrap_dist_model(model)
            .tokenizer.pad(
                {"input_ids": input_ids},
                padding="longest",
                # max_length=tokenizer.model_max_length,
                max_length=35,
                return_tensors="pt",
            )
            .input_ids
        )
        class_names = [example["class_name"] for example in examples]

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "input_images": input_images,
            "class_names": class_names,
            "ctx_begin_pos": ctx_begin_pos,
            "batch_type": batch_type,
        }

        return batch

    print("Loading dataset")
    # train_dataset = ImageNetDataset(**processors, clip_tokenizer=tokenizer)
    subject = config.subject
    image_dir = config.image_dir
    annotation_path = os.path.join(image_dir, "annotations.json")
    force_init_annotations = config.force_init_annotations

    if accelerator.is_main_process:
        if force_init_annotations or not os.path.exists(annotation_path):
            ImageDirDataset.generate_annotations(
                subject=subject,
                image_dir=image_dir,
                annotation_path=annotation_path,
            )
    accelerator.wait_for_everyone()

    train_dataset = load_dataset(
        dataset_name="imagedir",
        inp_image_transform=processors["inp_image_transform"],
        inp_bbox_transform=processors["inp_bbox_transform"],
        tgt_image_transform=processors["tgt_image_transform"],
        tgt_bbox_transform=processors["tgt_bbox_transform"],
        text_transform=processors["text_transform"],
        clip_tokenizer=model.tokenizer,
        subject=config.subject,
        image_dir=config.image_dir,
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

    train_dataloader = IterLoader(train_dataloader)

    # Scheduler and math around the number of training steps.
    # overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(
    #     len(train_dataloader) / args.gradient_accumulation_steps
    # )
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True

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
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
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

    save_to = os.path.join(args.output_dir, f"{global_step}")

    validate(
        model=unwrap_dist_model(model),
        transforms=processors,
        out_dir=os.path.join(save_to, "out_images"),
        rank=accelerator.process_index,
    )

    # for epoch in range(args.num_train_epochs):
    model.train()

    # for step, batch in enumerate(train_dataloader):
    while True:
        batch = next(train_dataloader)

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
            # print(f"Saving model at step {global_step}.")
            save_to = os.path.join(args.output_dir, f"{global_step}")

            # if accelerator.is_main_process:
            #     unwrap_dist_model(model).save_checkpoint(save_to, accelerator)

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


# def get_val_dataset():
#     img_paths = [
#         "/export/home/workspace/dreambooth/diffusers/data/cybertrucks/1.png",
#         "/export/home/workspace/dreambooth/diffusers/data/cybertrucks/1.png",
#         "/export/home/workspace/dreambooth/diffusers/data/cybertrucks/1.png",
#     ]

#     subj_names = [
#         "truck",
#         "truck",
#         "truck",
#         # "dog",
#     ]

#     prompts = [
#     #     # "a dog swimming in the ocean, the dog is",
#         "a truck " + ", ".join(["in the mountain"] * 20),
#     #     # "a dog at the grand canyon, photo by National Geographic, the dog is",
#         "a truck "
#         +", ".join(["at the grand canyon, photo by National Geographic"] * 20),
#         "a truck "
#         +", ".join(["in an ocean"] * 20),
#     #     # "a dog wearing a superman suit, the dog is",
#     #     # "a dog wearing sunglasses, the dog is",
#     #     # "a dog " + ", ".join(["wearing a superman suit"] * 20),
#     #     # "a dog " + ", ".join(["wearing a sunglasses"] * 20),
#     ]

#     return img_paths, subj_names, prompts

def get_val_dataset():
    img_paths = [
        "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
        "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
        "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
        "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
        "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
    ]

    subj_names = [
        "dog",
        "dog",
        "dog",
        "dog",
        "dog",
    ]

    prompts = [
        # "a dog swimming in the ocean, the dog is",
        "a dog " + ", ".join(["swimming in the ocean"] * 20),
        # "a dog at the grand canyon, photo by National Geographic, the dog is",
        "a dog "
        +", ".join(["at the grand canyon, photo by National Geographic"] * 20),
        # "a dog wearing a superman suit, the dog is",
        # "a dog wearing sunglasses, the dog is",
        "a dog " + ", ".join(["wearing a superman suit"] * 20),
        "a dog " + ", ".join(["wearing a sunglasses"] * 20),
        "a dog " + ", ".join(["at a wood doghouse"] * 20),
    ]

    return img_paths, subj_names, prompts


def validate(model, transforms, out_dir, rank, debug=False):
    negative_prompt = "over-exposed, saturated, blur, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

    from PIL import Image

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    subj_image_paths, subj_names, prompts = get_val_dataset()
    # ctx_begin_pos = [
    #     len(
    #         model.tokenizer(
    #             prompt,
    #             padding="do_not_pad",
    #         ).input_ids
    #     )
    #     - 1 for prompt in prompts
    # ]
    ctx_begin_pos = [2 for prompt in prompts]

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
            "ctx_begin_pos": [ctx_begin_pos[i]],
        }

        for gs, theta in [
            (7.5, -1),
            # (7.5, 1),
            # (7.5, 2),
            # (7.5, 4),
            # (7.5, 7),
            # (7.5, 10),
        ]:
            output = model.generate(
                samples,
                seed=3876998111 + int(rank),
                guidance_scale=gs,
                num_inference_steps=50,
                theta=theta,
                disable_bg_model=theta < 0,
                neg_prompt=negative_prompt,
            )

            prompt = prompt.replace(" ", "_")
            out_filename = f"{i}_{prompt[:20]}_gs={gs}_theta={theta}_rank{rank}.png"
            out_filepath = os.path.join(out_dir, out_filename)

            output[0].save(out_filepath)
        
    model.train()


if __name__ == "__main__":
    input_args = parse_args()

    config = yaml.load(open(input_args.config_path), Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)

    main(config)
