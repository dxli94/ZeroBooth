from diffusers import DiffusionPipeline, UNet2DConditionModel
import torch

model_id = "runwayml/stable-diffusion-v1-5"
# checkpoint = "/export/home/workspace/dreambooth/diffusers/output/original-dreambooth/dog-backpack/checkpoint-300/unet"
checkpoint = "/export/home/workspace/dreambooth/diffusers/output/original-dreambooth/dog-backpack/checkpoint-800/unet"

unet = UNet2DConditionModel.from_pretrained(checkpoint)
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    unet=unet).to("cuda")

# prompt = "A photo of sks backpack at the grand canyon"
# prompt = "A photo of sks backpack at the grand canyon"
prompts = [
    "A photo of sks backpack with a city in the background",
    "A photo of sks backpack in the snow",
    "A photo of sks backpack on a cobblestone street",
    "A photo of sks backpack with a wheat field in the background",
    "A photo of sks backpack with a mountain in the background",
    "A photo of sks backpack on top of a white rug",
]

for prompt in prompts:
    for i in range(5):
        image = pipe(
            prompt,
            num_inference_steps=100,
            guidance_scale=7.5
        ).images[0]

        image.save(f"""{"-".join(prompt.split()) + str(i)}.png"""[:128])
        # image.save("dog-bucket.png")