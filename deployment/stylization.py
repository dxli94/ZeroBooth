import streamlit as st
from PIL import Image
import yaml
import torch
import numpy as np
import random
from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image
from types import SimpleNamespace
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from lavis.processors.blip_processors import BlipCaptionProcessor
from streamlit_image_select import image_select


# add parent directory to path
import sys
sys.path.append("..")

from modeling_zerobooth_controlnet import ZeroBooth
import PIL

apply_canny = CannyDetector()

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

device = "cuda" if torch.cuda.is_available() else "cpu"
default_checkpoint = "/export/share/dongxuli/zerobooth/500000/"

negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
prompt_strength = 1
num_inference_steps = 50
guidance_scale = 7.5

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

def preprocess_canny(
    input_image: np.ndarray,
    image_resolution: int,
    low_threshold: int,
    high_threshold: int,
):
    image = resize_image(HWC3(input_image), image_resolution)
    control_image = apply_canny(image, low_threshold, high_threshold)
    control_image = HWC3(control_image)
    vis_control_image = 255 - control_image
    return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
        vis_control_image)

def generate_canny(cond_image_input, low_threshold, high_threshold):
    # convert cond_image_input to numpy array
    cond_image_input = np.array(cond_image_input).astype(np.uint8)

    canny_input, vis_control_image = preprocess_canny(cond_image_input, 512, low_threshold=low_threshold, high_threshold=high_threshold)
    # vis_control_image = preprocess_canny(cond_image_input, 512, low_threshold=low_threshold, high_threshold=high_threshold)

    return canny_input, vis_control_image

t = create_transforms()

inp_tsfm = t["inp_image_transform"]
txt_tsfm = t["text_transform"]


def app():
    st.title("BLIP-Diffusion Zero-shot Stylization")

    # make the following into st.sidebar
    # guidance_scale = 7.5
    # num_out = 2
    # seed = 42
    default_seed = 88889
    num_out = st.sidebar.slider("num_out", 2, 8, 2, step=2)
    seed = st.sidebar.text_input("seed", default_seed)

    subj_img = st.file_uploader("Upload Subject Image", type=["jpg", "jpeg", "png"])

    if subj_img is not None:
        subj_img = Image.open(subj_img).convert("RGB")
    else:
        subj_img = load_demo_image()

    style_img = image_select(
        label="Select a stlying image",
        images=[
            "images/flower.jpg",
            "images/vase-2.jpeg",
            "images/wood.jpg",
            "images/yarn.png",
        ],
        captions=["floral", "glass", "wood", "yarn"],
    )

    # style_img = load_demo_style_image()
    if style_img == "images/flower.jpg":
        src_subj_str = "flower"
    elif style_img == "images/vase-2.jpeg":
        src_subj_str = "vase"
    elif style_img == "images/wood.jpg":
        src_subj_str = "vase"
    elif style_img == "images/yarn.png":
        src_subj_str = "ball"
    
    style_img = Image.open(style_img).convert("RGB")

    canny_low_threshold = st.sidebar.slider(
        "canny_low_threshold",
        0, 255, 70,
    )
    canny_high_threshold = st.sidebar.slider(
        "canny_high_threshold",
        0, 255, 140,
    )

    try:
        seed = int(seed)
    except ValueError:
        st.warning("Seed must be an integer, found {}. Using default seed {}.".format(seed, default_seed))
        seed = default_seed

    col_m, col_r = st.columns(2)

    # with col_l:
    #     style_img = st.image(style_img, width=210, caption="Style Image")

    with col_m:
        st.image(subj_img, width=210, caption="Subject Image")
    
    col1, col2 = st.columns(2)

    src_subj = col1.text_input("style subject", src_subj_str)
    tgt_subj = col2.text_input("target subject", "pot")

    prompt = st.text_input("Prompt", "on the table")
    
    cap_button = st.button("Generate")
    if cap_button:
        print(canny_low_threshold, canny_high_threshold)
        canny_image, edge_img_vis = generate_canny(subj_img, canny_low_threshold, canny_high_threshold)
        with col_r:
            st.image(edge_img_vis, width=210, caption="Edge Maps")

        with st.spinner('Loading model...'):
            model = load_model_cache()
            print('Loading model done!')
        
        output = generate_images(
            model,
            style_img,
            canny_image,
            src_subj,
            tgt_subj,
            prompt,
            negative_prompt,
            prompt_strength,
            seed,
            num_inference_steps,
            guidance_scale,
            num_out,
        )

        # create grid of images, each row contains 2 images
        for i in range(0, num_out, 2):
            img_col1, img_col2 = st.columns(2)

            img_col1.image(output[i], width=318)
            img_col2.image(output[i + 1], width=318)


def load_image(img_url):
    raw_image = Image.open(img_url).convert("RGB")
    return raw_image

def load_demo_image():
    img_url = "images/pot.jpeg"

    return load_image(img_url)

def load_demo_style_image():
    img_url = "images/vase-2.jpeg"

    return load_image(img_url)

def load_checkpoint(model, checkpoint_path):
    if checkpoint_path:
        print("Loading checkpoint...")
        model.load_checkpoint(checkpoint_path)
        print("Finished loading checkpoint.")

@st.cache_resource
def load_model_cache():
    # load config
    print("Creating model...")
    config_path = "../configs/zerobooth_openimage.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)

    model = ZeroBooth(config=config.model)
    model = model.to(device)

    print("Finished creating model.")

    print("Loading checkpoint...")
    load_checkpoint(model, default_checkpoint)
    print("Finished loading checkpoint.")

    return model


def generate_images(
        model,
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
        iter_seed = seed + i
        print(f"Generating image {i+1}/{num_out} with seed {iter_seed}...")
        with st.spinner(f"Generating image {i+1}/{num_out} with seed {iter_seed}..."):
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
