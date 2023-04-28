import gradio as gr
import yaml
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
from annotator.midas import MidasDetector
from annotator.util import HWC3, resize_image
from types import SimpleNamespace
from modeling_zerobooth_controlnet import ZeroBooth
from train_zerobooth import create_transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from lavis.processors.blip_processors import BlipCaptionProcessor
import PIL
import cv2

apply_midas = MidasDetector()

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

    model = ZeroBooth(config=config.model, control_type="depth")
    model = model.to(device)

    print("Finished creating model.")
    return model

def preprocess_depth(
    input_image: np.ndarray,
    image_resolution: int,
):
    # image = resize_image(HWC3(input_image), image_resolution)
    # control_image = apply_canny(image, low_threshold, high_threshold)
    # control_image = HWC3(control_image)
    # vis_control_image = 255 - control_image
    # return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
    #     vis_control_image)
    input_image = HWC3(input_image)
    detected_map, _ = apply_midas(resize_image(input_image, image_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    # return PIL.Image.fromarray(control_image)
    return detected_map

def generate_depth(cond_image_input):
    # convert cond_image_input to numpy array
    cond_image_input = np.array(cond_image_input).astype(np.uint8)

    # canny_input, vis_control_image = preprocess_canny(cond_image_input, 512, low_threshold=100, high_threshold=200)
    vis_control_image = preprocess_depth(cond_image_input, 512)

    return vis_control_image 

def generate_images(
        image_input,
        cond_image_input,
        src_subject,
        tgt_subject,
        prompt,
        negative_prompt,
        prompt_strength,
        seed,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_out=4
    ):
    seed = int(seed)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    num_out = int(num_out)

    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    assert prompt_strength >= 0 and prompt_strength <= 1
    num_repeat = 20 * prompt_strength
    prompt = " ".join([prompt] * int(num_repeat))
    prompt = "a {} {}".format(tgt_subject, prompt)
    print(prompt)

    input_images = inp_tsfm(image_input).unsqueeze(0).to(device)
    class_names = [txt_tsfm(src_subject)]
    prompt = [txt_tsfm(prompt)]
    ctx_begin_pos = [2]

    samples = {
        "input_images": input_images,
        "class_names": class_names,
        "prompt": prompt,
        "ctx_begin_pos": ctx_begin_pos,
        "cond_image": cond_image_input.convert("RGB"),
    }

    output_images = []

    for i in range(num_out):
        iter_seed = seed + random.randint(0, 1000000)
        print(f"Generating image {i+1}/{num_out} with seed {iter_seed}...")
        output = model.generate(
            samples,
            seed=iter_seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            neg_prompt=negative_prompt,
            height=512,
            width=512,
        )

        output_images.append(output[0])

    return output_images


def load_checkpoint(checkpoint_path):
    if checkpoint_path:
        print("Loading checkpoint...")
        model.load_checkpoint(checkpoint_path)
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
    title = "BLIP-Diffusion Inference"

    gr.Markdown(title)
    # gr.Markdown(description)
    # gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=1):

            with gr.Row():
                image_input = gr.Image(type="pil", interactive=True, label="subject image")
                cond_image_input = gr.Image(type="pil", interactive=True, label="conditional image")
                # canny_image = gr.Image(type="pil", interactive=False, label="canny image", shape=(128, 128))
                depth_image = gr.Image(type="pil", interactive=True, label="depth image")

            cond_image_input.change(
                generate_depth,
                inputs=[cond_image_input],
                outputs=[depth_image],
            )

            with gr.Row():
                with gr.Column(scale=1):
                    model_path = gr.Textbox(label="Model checkpoint", value=default_checkpoint)
                
                with gr.Column(scale=0.25):
                    # load button
                    load_btn = gr.Button(value="Load Pretrained Checkpoint", interactive=True, variant="primary")
                    load_btn.click(
                        load_checkpoint,
                        inputs=[model_path],
                    )

            # other options
            with gr.Row():
                src_subject = gr.Textbox(lines=1, label="source subject")
                tgt_subject = gr.Textbox(lines=1, label="target subject")

            prompt = gr.Textbox(lines=1, label="Prompt")
            negative_prompt = gr.Textbox(lines=1, label="Negative prompt", value="over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate")
            prompt_strength = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="prompt strength (increase for more prompt influence)",
            )

        with gr.Column(scale=1):

            with gr.Column():
                gallery = gr.Gallery(label='Output',
                                    show_label=False,
                                    elem_id='gallery',
                                    ).style(grid=4, height=600)

                seed = gr.Textbox(lines=1, label="seed", value=42)
                num_out = gr.Slider(maximum=16, minimum=2, value=4, step=2, label="num_output")
                num_inference_steps = gr.Textbox(lines=1, label="num_inference_steps", value=50)
                guidance_scale = gr.Slider(maximum=20, minimum=0, value=7.5, step=0.5, label="guidance_scale")

                run_btn = gr.Button(
                    value="Run", interactive=True, variant="primary"
                )
                run_btn.click(
                    generate_images,
                    inputs=[
                        image_input,
                        depth_image,
                        src_subject,
                        tgt_subject,
                        prompt,
                        negative_prompt,
                        prompt_strength,
                        seed,
                        num_inference_steps,
                        guidance_scale,
                        num_out,
                    ],
                    outputs=[gallery],
                )


if __name__ == "__main__":
    model = create_model()
    load_checkpoint(default_checkpoint)

    # create transforms
    t = create_transforms()

    inp_tsfm = t["inp_image_transform"]
    txt_tsfm = t["text_transform"]

    # iface.queue(concurrency_count=1, api_open=False, max_size=10)
    # iface.launch(enable_queue=True, server_port=7861)
    iface.launch()
   