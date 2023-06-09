import streamlit as st
from PIL import Image
import yaml
import torch
import random
from types import SimpleNamespace
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from lavis.processors.blip_processors import BlipCaptionProcessor

# add parent directory to path
import sys
sys.path.append("..")

from modeling_zerobooth import ZeroBooth

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

device = "cuda" if torch.cuda.is_available() else "cpu"
# default_checkpoint = "/export/home/workspace/dreambooth/diffusers/output/pretrain-202302315-unet-textenc-v1.5-capfilt6b7-synbbox-matting-rr0-drop15-500k/500000"
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

t = create_transforms()

inp_tsfm = t["inp_image_transform"]
txt_tsfm = t["text_transform"]


def app():
    st.title("BLIP-Diffusion Zero-shot Generation")

    # make the following into st.sidebar
    # guidance_scale = 7.5
    # num_out = 2
    # seed = 42
    default_seed = 88888
    num_out = st.sidebar.slider("num_out", 2, 8, 4, step=2)
    seed = st.sidebar.text_input("seed", default_seed)

    try:
        seed = int(seed)
    except ValueError:
        st.warning("Seed must be an integer, found {}. Using default seed {}.".format(seed, default_seed))
        seed = default_seed

    image_input = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if image_input is not None:
        raw_img = Image.open(image_input).convert("RGB")
    else:
        raw_img = load_demo_image()

    col_l, col_r = st.columns(2)

    with col_l:
        st.image(raw_img, width=318)

    with col_r:
        src_subj = st.text_input("source subject", "dog")
        tgt_subj = st.text_input("target subject", src_subj)

        prompt = st.text_area("Prompt", "wearing sunglasses, at the grand canyon, painting by van gogh", height=100)
    
    cap_button = st.button("Generate")
    if cap_button:
        with st.spinner('Loading model...'):
            model = load_model_cache()
            print('Loading model done!')
        
        output = generate_images(
            model,
            raw_img,
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


def load_demo_image():
    img_url = "images/dog-square.png"

    raw_image = Image.open(img_url).convert("RGB")
    return raw_image

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
    prompt = prompt.strip()

    if tgt_subject != src_subject:
        prompt = "a {} {}".format(tgt_subject, prompt)
        prompt = ", ".join([prompt] * int(num_repeat))
    else:
        prompt = ", ".join([prompt] * int(num_repeat))
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

