import gradio as gr
import yaml
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
from annotator.util import HWC3, resize_image
from types import SimpleNamespace
from modeling_zerobooth import ZeroBooth
from train_zerobooth import create_transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from lavis.processors.blip_processors import BlipCaptionProcessor
import PIL
from ptp_utils import LocalBlend, AttentionRefine


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

device = "cuda" if torch.cuda.is_available() else "cpu"

default_checkpoint = "/export/home/workspace/dreambooth/diffusers/output/pretrain-202302315-unet-textenc-v1.5-capfilt6b7-synbbox-matting-rr0-drop15-500k/500000"

def create_model():
    # load config
    print("Creating model...")
    config_path = "/export/home/workspace/dreambooth/diffusers/configs/zerobooth_openimage.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)

    model = ZeroBooth(config=config.model)
    model = model.to(device)

    print("Finished creating model.")
    return model

def prepare_sample(
        src_prompt,
        tgt_prompt,
        src_subject,
        tgt_subject,
        image_input,
    ):
    src_prompt = "a {} {}".format(src_subject, src_prompt)
    tgt_prompt = "a {} {}".format(tgt_subject, tgt_prompt)

    input_images = inp_tsfm(image_input).unsqueeze(0).to(device)
    class_names = [txt_tsfm(src_subject)]
    src_prompt = [txt_tsfm(src_prompt)]
    ctx_begin_pos = [2]

    samples = {
        "input_images": input_images,
        "class_names": class_names,
        "prompt": src_prompt,
        "ctx_begin_pos": ctx_begin_pos,
    }

    prompts = []
    prompts.append(samples["prompt"][0])

    tokens = tgt_prompt.split(" ")
    tokens = [tokens[0]] + ["sks"] * 16 + tokens[1:]
    new_text = " ".join(tokens)
    tgt_prompt = new_text

    prompts.append(tgt_prompt)
    samples["prompt"] = prompts

    return samples

def generate_images(
        image_input,
        src_subject,
        tgt_subject,
        src_replace_token,
        tgt_replace_token,
        prompt,
        negative_prompt,
        seed,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_out=4
    ):
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    num_out = int(num_out)

    cudnn.benchmark = False
    cudnn.deterministic = True

    samples = prepare_sample(
        prompt,
        prompt,
        src_subject,
        tgt_subject,
        image_input,
    )

    prompts = samples["prompt"]
    lb = LocalBlend(
        prompts,
        (src_replace_token, tgt_replace_token),
        device=device, tokenizer=model.tokenizer
    )

    output_images = []
    attn_maps = []

    for i in range(num_out):
        controller = AttentionRefine(
            prompts,
            num_inference_steps,
            cross_replace_steps=.8,
            self_replace_steps=.4,
            tokenizer=model.tokenizer,
            device=device,
            local_blend=lb
        )

        iter_seed = seed + random.randint(0, 1000000)
        print(f"Generating image {i+1}/{num_out} with seed {iter_seed}...")
        output = model.generate_p2p_refine(
            samples,
            # seed=99839198429 + random.randint(0, 1000000),
            seed=iter_seed,
            guidance_scale=7.5,
            controller=controller,
            num_inference_steps=num_inference_steps,
            neg_prompt=negative_prompt,
            height=512,
            width=512,
        )

        # output_images.append(output[0])
        output_images.extend(output)

        attn_map = model.show_cross_attention(prompts, controller, res=16, disable_subject=True, from_where=("up", "down"), select=0) # run only after inference once. the controller must be update to get the cross-attn map.
        attn_maps.append(attn_map)

    return output_images, attn_maps


def load_checkpoint(checkpoint_path):
    if checkpoint_path:
        print("Loading checkpoint...")
        model.load_checkpoint(checkpoint_path)
        print("Finished loading checkpoint.")

def load_default():
    print("Loading default checkpoint...")
    load_checkpoint(default_checkpoint)
    print("Finished loading checkpoint.")

def create_transforms(image_size=224):
    # preprocess
    # blip image transform
    inp_image_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )

    text_transform = BlipCaptionProcessor()

    return {
        "inp_image_transform": inp_image_transform,
        "text_transform": text_transform,
    }


with gr.Blocks(
    css="""
    .message.svelte-w6rprc.svelte-w6rprc.svelte-w6rprc {font-size: 20px; margin-top: 20px}
    #component-21 > div.wrap.svelte-w6rprc {height: 600px;}
    """
) as iface:
    title = "BLIP-Diffusion Prompt-to-Prompt Editing"

    gr.Markdown(title)
    # gr.Markdown(description)
    # gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=1):

            with gr.Row():
                image_input = gr.Image(type="pil", interactive=True, label="subject image")
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_path = gr.Textbox(label="Model checkpoint", value=default_checkpoint)
                
                with gr.Column(scale=0.25):
                    # load button
                    load_btn = gr.Button(value="Load Checkpoint", interactive=True, variant="primary")
                    load_btn.click(
                        load_checkpoint,
                        inputs=[model_path],
                    )

                    reset_btn = gr.Button(value="Reset Checkpoint", interactive=True, variant="primary")
                    reset_btn.click(load_default)

            # other options
            with gr.Row():
                src_subject = gr.Textbox(lines=1, label="source subject")
                tgt_subject = gr.Textbox(lines=1, label="target subject")

            with gr.Row():
                src_replace_token = gr.Textbox(lines=1, label="token to replace")
                tgt_replace_token = gr.Textbox(lines=1, label="token to replace with")

            prompt = gr.Textbox(lines=1, label="Prompt")
            negative_prompt = gr.Textbox(lines=1, label="Negative prompt", value="over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate")

        with gr.Column(scale=1):
            with gr.Column():
                gallery = gr.Gallery(label='Output',
                                    show_label=False,
                                    elem_id='gallery',
                                    ).style(grid=4, height=600)

                attn_gallery = gr.Gallery(label='Attention Maps',
                                          show_label=False,
                                          elem_id='gallery',
                                         ).style(grid=1, height=600)

                seed = gr.Textbox(lines=1, label="seed", value=42)
                num_out = gr.Slider(maximum=16, minimum=2, value=2, step=2, label="num_output")
                num_inference_steps = gr.Textbox(lines=1, label="num_inference_steps", value=50)
                guidance_scale = gr.Slider(maximum=20, minimum=0, value=7.5, step=0.5, label="guidance_scale")

                run_btn = gr.Button(
                    value="Run", interactive=True, variant="primary"
                )
                run_btn.click(
                    generate_images,
                    inputs=[
                        image_input,
                        src_subject,
                        tgt_subject,
                        src_replace_token,
                        tgt_replace_token,
                        prompt,
                        negative_prompt,
                        seed,
                        num_inference_steps,
                        guidance_scale,
                        num_out,
                    ],
                    outputs=[gallery, attn_gallery],
                )


if __name__ == "__main__":
    model = create_model()
    load_checkpoint(default_checkpoint)

    # create transforms
    t = create_transforms()

    inp_tsfm = t["inp_image_transform"]
    txt_tsfm = t["text_transform"]

    iface.queue(concurrency_count=1, api_open=False, max_size=10)
    iface.launch(enable_queue=True, server_port=7864)
   