"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""
from models.multimodal_bert import BertConfig, BertLMHeadModel
import transformers

transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from models.create_vision_model import create_clip_vit
from CLIP.clip.model import LayerNorm

from transformers import BertTokenizerFast as BertTokenizer

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from torch.cuda.amp import autocast as autocast


class BLIP_Pretrain(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.visual_encoder, vision_width = create_clip_vit(
            config["vision_model"], config["image_size"]
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
        else:
            encoder_config.cross_attention_freq = 1
        encoder_config.query_length = config["num_query_token"]

        self.text_model = BertLMHeadModel.from_pretrained(
            config["text_model"], config=encoder_config
        )
        self.text_model.resize_token_embeddings(len(self.tokenizer))

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

        self.vision_proj = nn.Linear(
            self.text_model.config.hidden_size, config["embed_dim"]
        )
        self.text_proj = nn.Linear(
            self.text_model.config.hidden_size, config["embed_dim"]
        )

        self.itm_head = nn.Linear(encoder_config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_text_length = config["max_text_length"]

    @autocast()
    def compute_itc(self, image, text):
        image_embeds = self.visual_encoder(image)

        image_embeds = self.ln_vision(image_embeds)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.text_model.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.text_model.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        return image_feats, text_feat, image_embeds, text_tokens

    @autocast()
    def compute_itm(self, image_embeds, text):
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )

        text_tokens = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        ).to(image_embeds.device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        output_itm = self.text_model.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        vl_output = self.itm_head(vl_embeddings)

        vl_output = vl_output[:, :, 1].mean(dim=1)

        return vl_output

    def forward(self, image, text):

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        image_embeds = self.visual_encoder(image)

        image_embeds = self.ln_vision(image_embeds)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.text_model.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.text_model.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        image_feats_all = all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        smooth = 0.1
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=smooth)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=smooth)
        ) / 2

        ###============== Image-text Matching ===================###
        #         with torch.no_grad():
        #             weights_t2i = F.softmax(sim_t2i[:,rank*bs:rank*bs+bs],dim=1)+1e-4
        #             weights_t2i.fill_diagonal_(0)
        #             weights_i2t = F.softmax(sim_i2t[:,rank*bs:rank*bs+bs],dim=1)+1e-4
        #             weights_i2t.fill_diagonal_(0)

        #         # select a negative image for each text
        #         image_embeds_neg = []
        #         for b in range(bs):
        #             neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #             image_embeds_neg.append(image_embeds[neg_idx])
        #         image_embeds_neg = torch.stack(image_embeds_neg,dim=0)

        #         # select a negative text for each image
        #         text_ids_neg = []
        #         text_atts_neg = []
        #         for b in range(bs):
        #             neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #             text_ids_neg.append(text_tokens.input_ids[neg_idx])
        #             text_atts_neg.append(text_tokens.attention_mask[neg_idx])

        text_input_ids_world = all_gather(text_tokens.input_ids)
        text_attention_mask_world = all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output_itm = self.text_model.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)

        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.text_model(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )
        loss_lm = lm_output.loss

        return loss_itc, loss_itm, loss_lm

    @autocast()
    def generate(
        self, image, sample=False, num_beams=3, max_length=25, min_length=5, top_p=0.9
    ):
        image_embeds = self.visual_encoder(image)
        image_embeds = self.ln_vision(image_embeds)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if sample:
            # nucleus sampling
            outputs = self.text_model.generate(
                input_ids=input_ids,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            # beam search
            outputs = self.text_model.generate(
                input_ids=input_ids,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


def blip_pretrain(**kwargs):
    model = BLIP_Pretrain(**kwargs)
    return model


@torch.no_grad()
def all_gather(tensor):
    world_size = dist.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
