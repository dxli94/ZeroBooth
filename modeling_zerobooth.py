import os
import inspect
import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from transformers import CLIPTokenizer
from transformers.activations import QuickGELUActivation as QuickGELU

from BLIP2.models.blipv2_feature_extractor import blip
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from modeling_clip import CtxCLIPTextModel


class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x):
        x_in = x

        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x


class ZeroBooth(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # BLIP-2
        print("Creating model")
        self.blip = blip(config=config)
        state_dict = torch.load(config["finetuned"], map_location="cpu")["model"]
        msg = self.blip.load_state_dict(state_dict, strict=False)
        print("Loaded BLIP checkpoint from {}, {}".format(config["finetuned"], msg))

        # projection layer
        proj_in_dim, proj_out_dim = 768, 768
        proj_rate = 4
        self.proj_layer = ProjLayer(
            in_dim=proj_in_dim,
            out_dim=proj_out_dim,
            hidden_dim=proj_in_dim * proj_rate,
            drop_p=0.1,
            eps=1e-12,
        )
        self.num_query_token = (
            config["num_proj_query_token"] if "num_proj_query_token" in config else 32
        )
        # assert 32 % self.num_query_token == 0
        # pool_kernel_size = pool_stride = 32 // self.num_query_token
        # self.pool_layer = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_stride)

        # stable diffusion
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config["pretrained_model_name_or_path"],
            subfolder="tokenizer",
            revision=config["revision"],
        )

        # Load models and create wrapper for stable diffusion
        self.text_encoder = CtxCLIPTextModel.from_pretrained(
            config["pretrained_model_name_or_path"],
            subfolder="text_encoder",
            revision=config["revision"],
        )

        self.vae = AutoencoderKL.from_pretrained(
            config["pretrained_model_name_or_path"],
            subfolder="vae",
            revision=config["revision"],
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            config["pretrained_model_name_or_path"],
            subfolder="unet",
            revision=config["revision"],
        )

        # if config.gradient_checkpointing:
        #     self.unet.enable_gradient_checkpointing()
        #     if config.train_text_encoder:
        #         self.text_encoder.gradient_checkpointing_enable()

        self.noise_scheduler = DDPMScheduler.from_config(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )

        self.max_txt_len = 25

        self.freeze_modules()

        self._background_model = None

    @property
    def background_model(self):
        from types import SimpleNamespace

        if self._background_model is None:
            # Load models and create wrapper for stable diffusion
            text_encoder = CtxCLIPTextModel.from_pretrained(
                self.config["pretrained_model_name_or_path"],
                subfolder="text_encoder",
                revision=self.config["revision"],
            ).to(self.device)

            unet = UNet2DConditionModel.from_pretrained(
                self.config["pretrained_model_name_or_path"],
                subfolder="unet",
                revision=self.config["revision"],
            ).to(self.device)

            background_model = SimpleNamespace(text_encoder=text_encoder, unet=unet)

            self._background_model = background_model

        return self._background_model

    def freeze_modules(self):
        self.vae.eval()
        self.vae.train = self.disabled_train
        self.vae.requires_grad_(False)

        if not self.config["train_unet"]:
            print("Freezing UNet")
            self.unet.eval()
            self.unet.train = self.disabled_train
            self.unet.requires_grad_(False)

        if not self.config["train_text_encoder"]:
            print("Freezing text encoder")
            self.text_encoder.eval()
            self.text_encoder.train = self.disabled_train
            self.text_encoder.requires_grad_(False)

    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, batch):
        """
        batch contains:
            - input_images: [B, 3, 392, 392], for BLIP to extract features
            - class_names: string list of class names
            - pixel_values: [B, 3, 512, 512], GT for stable diffusion to compute loss
            - input_ids
        """
        # ctx_embeddings = self.pool_layer(ctx_embeddings.transpose(1, 2)).transpose(1, 2)

        # TODO update CLIP embedding layer with projected blip embeddings
        # Convert images to latent space
        latents = self.vae.encode(
            # batch["pixel_values"].to(dtype=weight_dtype)
            batch["pixel_values"]
        ).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        blip_embeddings = self.blip(
            image=batch["input_images"],
            text=batch["class_names"],
        )

        # projected as clip text embeddings
        blip_embeddings = blip_embeddings[:, : self.num_query_token, :]
        ctx_embeddings = self.proj_layer(blip_embeddings)

        # Get the text embedding for conditioning
        # TODO make it configurable rather than hardcoding 2 (2 = len(["[pad]", "a"])
        encoder_hidden_states = self.text_encoder(
            input_ids=batch["input_ids"],
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=batch["ctx_begin_pos"],
        )[0]

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss

    @torch.no_grad()
    def generate(
        self,
        samples,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=250,
        eta=1,
        k=125,
        theta=10,
        disable_bg_model=False,
        v_condition=False,
    ):

        input_image = samples["input_images"]  # reference image
        text_input = samples["class_names"]  # category
        prompt = samples["prompt"]  # prompt for stable diffusion

        scheduler = self.eval_noise_scheduler

        # 1. extract BLIP query features and proj to text space -> (bs, 32, 768)
        query_embeds = self.blip(image=input_image, text=text_input)
        query_embeds = query_embeds[:, : self.num_query_token, :]
        query_embeds = self.proj_layer(query_embeds)
        # query_embeds = self.pool_layer(query_embeds.transpose(1, 2)).transpose(1, 2)

        # 2. embeddings for prompt, with query_embeds as context
        # ctx_begin_pos = torch.LongTensor(
        #     [p.index(self.special_token_inference) for p in prompt]
        # )

        # prompt = [p.replace(self.special_token_inference, "") for p in prompt]
        tokenized_prompt = self.tokenize_text(prompt).to(self.device)

        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            # ctx_begin_pos=[2],
            ctx_begin_pos=samples["ctx_begin_pos"],
        )[0]

        if not disable_bg_model:
            text_embeddings_bg = self.background_model.text_encoder(
                input_ids=tokenized_prompt.input_ids,
                ctx_embeddings=None,
            )[0]

        # 3. unconditional embedding
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            max_length = tokenized_prompt.input_ids.shape[-1]

            if not v_condition:
                max_length += self.num_query_token

            uncond_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # FIXME use context embedding for uncond_input or not?
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device),
                ctx_embeddings=None if not v_condition else query_embeds,
                ctx_begin_pos=[1],
            )[0]

            if not disable_bg_model:
                max_length = tokenized_prompt.input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    [""],
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                # FIXME use context embedding for uncond_input or not?
                uncond_embeddings_bg = self.background_model.text_encoder(
                    input_ids=uncond_input.input_ids.to(self.device),
                    ctx_embeddings=None,
                    # ctx_begin_pos=torch.ones_like(ctx_begin_pos),
                    # ctx_begin_pos=2,
                )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            if not disable_bg_model:
                text_embeddings_bg = torch.cat(
                    [uncond_embeddings_bg, text_embeddings_bg]
                )

        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)

        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
        )

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(scheduler, LMSDiscreteScheduler):
            latents = latents * scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        iterator = tqdm.tqdm(scheduler.timesteps)

        for i, t in enumerate(iterator):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            # predict the noise residual
            noise_pred_fg = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            if not disable_bg_model:
                noise_pred_bg = self.background_model.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings_bg
                )["sample"]

                # compute value of a in an exponential decay wrt to i
                a = np.exp(-theta * i / num_inference_steps)

                noise_pred = a * noise_pred_bg + (1 - a) * noise_pred_fg
            else:
                noise_pred = noise_pred_fg

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
                "prev_sample"
            ]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = numpy_to_pil(image)

        return image

    @torch.no_grad()
    def generate_dual(
        self,
        samples,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=250,
        eta=1,
        k=125,
        theta=10,
    ):

        input_image = samples["input_images"]  # reference image
        text_input = samples["class_names"]  # category
        prompt = samples["prompt"]  # prompt for stable diffusion

        scheduler = self.eval_noise_scheduler

        # 1. extract BLIP query features and proj to text space -> (bs, 32, 768)
        query_embeds = self.blip(image=input_image, text=text_input)
        query_embeds = query_embeds[:, : self.num_query_token, :]
        query_embeds = self.proj_layer(query_embeds)
        # query_embeds = self.pool_layer(query_embeds.transpose(1, 2)).transpose(1, 2)

        # 2. embeddings for prompt, with query_embeds as context
        # ctx_begin_pos = torch.LongTensor(
        #     [p.index(self.special_token_inference) for p in prompt]
        # )

        # prompt = [p.replace(self.special_token_inference, "") for p in prompt]
        tokenized_prompt = self.tokenize_text(prompt).to(self.device)

        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            # ctx_begin_pos=[2],
            ctx_begin_pos=samples["ctx_begin_pos"],
        )[0]

        text_embeddings_bg = text_embeddings.clone()

        # 3. unconditional embedding
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            max_length = tokenized_prompt.input_ids.shape[-1]

            # Foreground model
            max_length += self.num_query_token

            uncond_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # FIXME use context embedding for uncond_input or not?
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device), ctx_embeddings=None
            )[0]

            # Background model
            max_length = tokenized_prompt.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            # FIXME use context embedding for uncond_input or not?
            uncond_embeddings_bg = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device),
                ctx_embeddings=query_embeds,
                ctx_begin_pos=[1],
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            text_embeddings_bg = torch.cat([uncond_embeddings_bg, text_embeddings_bg])

        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)

        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
        )

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(scheduler, LMSDiscreteScheduler):
            latents = latents * scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        iterator = tqdm.tqdm(scheduler.timesteps)

        for i, t in enumerate(iterator):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            # predict the noise residual
            noise_pred_fg = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            noise_pred_bg = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings_bg
            )["sample"]

            # compute value of a in an exponential decay wrt to i
            a = np.exp(-theta * i / num_inference_steps)

            noise_pred = a * noise_pred_bg + (1 - a) * noise_pred_fg
            # noise_pred = noise_pred_bg

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
                "prev_sample"
            ]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = numpy_to_pil(image)

        return image

    def tokenize_text(self, text_input):
        tokenized_text = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )

        return tokenized_text

    @torch.no_grad()
    def save_checkpoint(self, path, accelerator=None):
        assert (
            accelerator is not None
        ), "only support distributed training with accelerator for now."

        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config["pretrained_model_name_or_path"],
            unet=accelerator.unwrap_model(self.unet),
            text_encoder=accelerator.unwrap_model(self.text_encoder),
            revision=self.config["revision"],
        )
        pipeline.save_pretrained(path)

        # save blip model and proj weights
        blip_without_ddp = accelerator.unwrap_model(self.blip)
        proj_without_ddp = accelerator.unwrap_model(self.proj_layer)

        blip_save_to = os.path.join(path, "blip_model")
        proj_save_to = os.path.join(path, "proj_layer")

        os.makedirs(blip_save_to)
        os.makedirs(proj_save_to)

        torch.save(blip_without_ddp.state_dict(), blip_save_to + "/blip_weight.pt")
        torch.save(proj_without_ddp.state_dict(), proj_save_to + "/proj_weight.pt")

    @torch.no_grad()
    def load_checkpoint(self, checkpoint_dir):
        import os

        self.proj_layer.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "proj_layer/proj_weight.pt"),
                map_location="cpu",
            )
        )
        self.blip.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "blip_model/blip_weight.pt"),
                map_location="cpu",
            )
        )

        self.unet.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "unet/diffusion_pytorch_model.bin"),
                map_location="cpu",
            )
        )
        self.vae.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "vae/diffusion_pytorch_model.bin"),
                map_location="cpu",
            )
        )

        self.text_encoder.load_state_dict(
            torch.load(
                os.path.join(checkpoint_dir, "text_encoder/pytorch_model.bin"),
                map_location="cpu",
            )
        )

    @property
    def eval_noise_scheduler(self):
        if not hasattr(self, "_eval_noise_scheduler"):
            self._eval_noise_scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        return self._eval_noise_scheduler


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images
