try:
    from BLIP2.CLIP import clip
except ModuleNotFoundError:
    from CLIP import clip
import torch
from scipy import interpolate
import numpy as np

def create_clip_vit(vit, image_size, use_grad_checkpointing=False, precision='fp16'):

    if vit=='openclip':
        import open_clip
        clip_model = open_clip.create_model('ViT-H-14', pretrained='laion2b_s32b_b79k', 
                                            precision=precision, 
                                            image_size=image_size,
                                            device=torch.device('cuda'),
                                           )
        if use_grad_checkpointing:
            clip_model.visual.set_grad_checkpointing() 

    elif vit=='evaclip':
        from eva_clip import build_eva_model_and_transforms
        eva_clip_path = "eva_clip/eva_clip_psz14.pt" 
        clip_model, _ = build_eva_model_and_transforms("EVA_CLIP_g_14", precision=precision, pretrained=eva_clip_path, device=torch.device('cuda'))

    else:
        device = 'cuda' if precision=='fp16' else 'cpu'
        clip_model, _ = clip.load('ViT-L/14', device=device, image_resolution=image_size, 
                                  use_grad_checkpointing=use_grad_checkpointing)        
    visual_encoder = clip_model.visual       
    vision_width = visual_encoder.num_features
        
    return visual_encoder, vision_width

# from beit.model import beit_large_patch16_224
# def create_beit_vit(image_size):
#     model = beit_large_patch16_224(img_size=image_size)
#     checkpoint = torch.load('beit/beitv2_large_patch16_224_pt1k_ft21k.pth')
#     checkpoint_model = checkpoint['module']
    
#     all_keys = list(checkpoint_model.keys())
#     for key in all_keys:
#         if "relative_position_index" in key:
#             checkpoint_model.pop(key)

#         if "relative_position_bias_table" in key and key in model.state_dict():
#             rel_pos_bias = checkpoint_model[key]
#             src_num_pos, num_attn_heads = rel_pos_bias.size()
#             dst_num_pos, _ = model.state_dict()[key].size()
#             dst_patch_shape = model.patch_embed.patch_shape
#             if dst_patch_shape[0] != dst_patch_shape[1]:
#                 raise NotImplementedError()
#             num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
#             src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
#             dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
#             if src_size != dst_size:
#                 print("Position interpolate for %s from %dx%d to %dx%d" % (
#                     key, src_size, src_size, dst_size, dst_size))
#                 extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
#                 rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

#                 def geometric_progression(a, r, n):
#                     return a * (1.0 - r ** n) / (1.0 - r)

#                 left, right = 1.01, 1.5
#                 while right - left > 1e-6:
#                     q = (left + right) / 2.0
#                     gp = geometric_progression(1, q, src_size // 2)
#                     if gp > dst_size // 2:
#                         right = q
#                     else:
#                         left = q

#                 # if q > 1.090307:
#                 #     q = 1.090307

#                 dis = []
#                 cur = 1
#                 for i in range(src_size // 2):
#                     dis.append(cur)
#                     cur += q ** (i + 1)

#                 r_ids = [-_ for _ in reversed(dis)]

#                 x = r_ids + [0] + dis
#                 y = r_ids + [0] + dis

#                 t = dst_size // 2.0
#                 dx = np.arange(-t, t + 0.1, 1.0)
#                 dy = np.arange(-t, t + 0.1, 1.0)

#                 print("Original positions = %s" % str(x))
#                 print("Target positions = %s" % str(dx))

#                 all_rel_pos_bias = []

#                 for i in range(num_attn_heads):
#                     z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
#                     f = interpolate.interp2d(x, y, z, kind='cubic')
#                     all_rel_pos_bias.append(
#                         torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

#                 rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

#                 new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
#                 checkpoint_model[key] = new_rel_pos_bias

#     # interpolate position embedding
#     if 'pos_embed' in checkpoint_model:
#         pos_embed_checkpoint = checkpoint_model['pos_embed']
#         embedding_size = pos_embed_checkpoint.shape[-1]
#         num_patches = model.patch_embed.num_patches
#         num_extra_tokens = model.pos_embed.shape[-2] - num_patches
#         # height (== width) for the checkpoint position embedding
#         orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
#         # height (== width) for the new position embedding
#         new_size = int(num_patches ** 0.5)
#         # class_token and dist_token are kept unchanged
#         if orig_size != new_size:
#             print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
#             extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
#             # only the position tokens are interpolated
#             pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
#             pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
#             pos_tokens = torch.nn.functional.interpolate(
#                 pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
#             pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
#             new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
#             checkpoint_model['pos_embed'] = new_pos_embed       
    
#     msg = model.load_state_dict(checkpoint['module'],strict=False)
#     print(msg)
#     vision_width = model.num_features
#     return model, vision_width
    

# from convnext.convnext import convnext_large, convnext_xlarge
# def create_convnext():
    
#     visual_encoder = convnext_xlarge(pretrained=True, in_22k=True)
#     vision_width = visual_encoder.feature_dim
    
#     return visual_encoder, vision_width
    
    