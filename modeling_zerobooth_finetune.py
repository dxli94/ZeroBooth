import os
import inspect
import tqdm

import torch
import torch.nn.functional as F
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

        if "finetuned" in config and config["finetuned"] is not None:
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
        self.num_query_token = config["num_query_token"]
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

        self.noise_scheduler = DDPMScheduler.from_config(
            config["pretrained_model_name_or_path"], subfolder="scheduler"
        )

        self.freeze_modules()

    def freeze_modules(self):
        self.vae.eval()
        self.vae.train = self.disabled_train
        self.vae.requires_grad_(False)

        if not self.config["train_unet"]:
            print("Freezing UNet")
            # self.unet.eval()
            # self.unet.train = self.disabled_train
            # self.unet.requires_grad_(False)
            def freeze_module(module):
                module.eval()
                module.train = self.disabled_train
                module.requires_grad_(False)

            modules = [
                # self.unet.conv_out,
                # self.unet.conv_act,
                # self.unet.conv_norm_out,
                self.unet.down_blocks,
                self.unet.up_blocks[:1],
                self.unet.mid_block,
                ]
            
            for module in modules:
                freeze_module(module)

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
    
    def apply_weights(self, text_embeddings, embedding_weights, ctx_begin_pos):
        subj_weight = embedding_weights[0]
        prpt_weight = embedding_weights[1]

        ctx_begin_pos = ctx_begin_pos[0]

        prev_mean = text_embeddings.mean(axis=[1, 2])

        cur_text_embeddings = text_embeddings.clone()

        cur_text_embeddings[0, :ctx_begin_pos] *= prpt_weight
        cur_text_embeddings[0, ctx_begin_pos + self.num_query_token:] *= prpt_weight
        cur_text_embeddings[0, ctx_begin_pos: ctx_begin_pos + self.num_query_token] *= subj_weight

        cur_mean = cur_text_embeddings.mean(axis=[1, 2])

        # ensure the mean is the same
        cur_text_embeddings *= prev_mean / cur_mean

        return cur_text_embeddings

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
        neg_prompt="",
        embedding_weights=None
    ):

        input_image = samples["input_images"]  # reference image
        text_input = samples["class_names"]  # category
        prompt = samples["prompt"]  # prompt for stable diffusion

        scheduler = self.eval_noise_scheduler

        # 1. extract BLIP query features and proj to text space -> (bs, 32, 768)
        query_embeds = self.blip(image=input_image, text=text_input)
        query_embeds = query_embeds[:, : self.num_query_token, :]
        query_embeds = self.proj_layer(query_embeds)

        # 2. embeddings for prompt, with query_embeds as context
        tokenized_prompt = self.tokenize_text(prompt).to(self.device)

        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            # ctx_begin_pos=[2],
            ctx_begin_pos=samples["ctx_begin_pos"],
        )[0]

        if embedding_weights is not None:
            text_embeddings = self.apply_weights(text_embeddings, embedding_weights, samples["ctx_begin_pos"])

        # 3. unconditional embedding
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings 

            uncond_input = self.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # FIXME use context embedding for uncond_input or not?
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device),
                ctx_embeddings=None,
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

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
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

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
    def generate_attn_ctrl(
        self,
        samples,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=250,
        eta=1,
        neg_prompt="",
        prompt_edit_token_weights=[],
    ):

        input_image = samples["input_images"]  # reference image
        text_input = samples["class_names"]  # category
        prompt = samples["prompt"]  # prompt for stable diffusion

        scheduler = self.eval_noise_scheduler

        # 1. extract BLIP query features and proj to text space -> (bs, 32, 768)
        query_embeds = self.blip(image=input_image, text=text_input)
        query_embeds = query_embeds[:, : self.num_query_token, :]
        query_embeds = self.proj_layer(query_embeds)

        # 2. embeddings for prompt, with query_embeds as context
        tokenized_prompt = self.tokenize_text(prompt).to(self.device)

        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            # ctx_begin_pos=[2],
            ctx_begin_pos=samples["ctx_begin_pos"],
        )[0]

        # 3. unconditional embedding
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings 

            uncond_input = self.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # FIXME use context embedding for uncond_input or not?
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device),
                ctx_embeddings=None,
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        self.unet = init_attention_func(self.unet)
        self.unet = init_attention_weights(self.unet, prompt_edit_token_weights)

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
            # latent_model_input = (
            #     torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # )
            latent_model_input = (
                latents if do_classifier_free_guidance else latents
            )

            # predict the noise residual
            noise_pred_uncond = self.unet(
                latent_model_input, t, encoder_hidden_states=uncond_embeddings
            )["sample"]

            self.unet = use_last_tokens_attention_weights(self.unet)

            noise_pred_text = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
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
            # max_length=self.text_encoder,
            max_length=self.text_encoder.text_model.config.max_position_embeddings - self.num_query_token,
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


def init_attention_func(unet):
    #ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276
    def new_attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn_slice = attention_scores.softmax(dim=-1)
        # compute attention output
        
        # if self.use_last_attn_slice:
        #     if self.last_attn_slice_mask is not None:
        #         new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
        #         attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
        #     else:
        #         attn_slice = self.last_attn_slice

        #     self.use_last_attn_slice = False

        # if self.save_last_attn_slice:
        #     self.last_attn_slice = attn_slice
        #     self.save_last_attn_slice = False

        if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
            attn_slice = attn_slice * self.last_attn_slice_weights
            self.use_last_attn_weights = False
        
        hidden_states = torch.matmul(attn_slice, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
    
    def new_sliced_attention(self, query, key, value, sequence_length, dim):
        
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            
            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                    attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice
                
                self.use_last_attn_slice = False
                    
            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False
                
            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False
            
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.save_last_attn_slice = False
            module._sliced_attention = new_sliced_attention.__get__(module, type(module))
            module._attention = new_attention.__get__(module, type(module))
        
    return unet


def init_attention_weights(unet, weight_tuples, tokens_length=77):
    weights = torch.ones(tokens_length)
    
    for i, w in weight_tuples:
        if i < tokens_length and i >= 0:
            weights[i] = w
    
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_weights = weights.to(unet.device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_weights = None

    return unet


def use_last_tokens_attention_weights(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use
    
    return unet