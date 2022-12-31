"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""
from BLIP2.models.multimodal_bert import BertConfig, BertLMHeadModel
import transformers

transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from BLIP2.models.create_vision_model import create_clip_vit
from BLIP2.CLIP.clip.model import LayerNorm

from transformers import BertTokenizerFast as BertTokenizer

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from torch.cuda.amp import autocast as autocast


class BLIP(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.visual_encoder, vision_width = create_clip_vit(
            config["vision_model"],
            config["image_size"],
            # use_grad_checkpointing=config["use_grad_checkpointing"],
            precision="fp32",
        )
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
            print("freeze %s" % name)

        self.ln_vision = LayerNorm(vision_width)

        self.tokenizer = BertTokenizer.from_pretrained(config["text_model"])
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        encoder_config = BertConfig.from_pretrained(config["text_model"])
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        if "cross_attention_freq" in config:
            print("cross_attention_freq: %d" % config["cross_attention_freq"])
            encoder_config.cross_attention_freq = config["cross_attention_freq"]
        # else:
        #     encoder_config.cross_attention_freq = 1
        encoder_config.query_length = config["num_query_token"]

        self.text_model = BertLMHeadModel.from_pretrained(
            config["text_model"], config=encoder_config
        )
        self.text_model.resize_token_embeddings(len(self.tokenizer))

        # [TODO] is this the right way to do it?
        if config["vision_model"] == "evaclip":
            state_dict = self.text_model.state_dict()
            for name, param in self.text_model.named_parameters():
                if "_query" in name:
                    key_orig = name.replace("_query", "")
                    param.data.copy_(state_dict[key_orig])
                    print("copy from %s to %s" % (key_orig, name))

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config["num_query_token"], encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        # self.vision_proj = nn.Linear(
        #     self.text_model.config.hidden_size, config["embed_dim"]
        # )
        # self.text_proj = nn.Linear(
        #     self.text_model.config.hidden_size, config["embed_dim"]
        # )

        # self.itm_head = nn.Linear(encoder_config.hidden_size, 2)

        # self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_text_length = config["max_text_length"]

        if "pretrained" in config:
            checkpoint = torch.load(config["pretrained"], map_location="cpu")
            state_dict = checkpoint["model"]
            for k in list(state_dict.keys()):
                if "visual_encoder.positional_embedding" in k:
                    del state_dict[k]
            msg = self.load_state_dict(state_dict, strict=False)
            print("load checkpoint from %s" % config["pretrained"])
            print(msg)

    def forward(self, image, text):

        image_embeds = self.visual_encoder(image)
        image_embeds = self.ln_vision(image_embeds)

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        ).to(image.device)

        text_ids = text_tokens.input_ids
        text_atts = text_tokens.attention_mask

        query_tokens = self.query_tokens.expand(text_ids.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output = self.text_model.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        vl_embeddings = output.last_hidden_state[:, : query_tokens.size(1), :]
        # mean over query tokens
        # vl_embeddings = vl_embeddings.mean(dim=1)

        return vl_embeddings


def blip(**kwargs):
    model = BLIP(**kwargs)
    return model


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
