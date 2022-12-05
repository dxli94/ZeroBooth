# from diffusers import StableDiffusionPipeline
# from diffusers.schedulers import DDIMScheduler
# import torch

# # model_id = "path-to-your-trained-model"
# model_id = "output/alvan-nee/"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
#     "cuda"
# )

# # prompt = "A photo of sks dog in a bucket"
# # prompt = "A photo of an sks dog in a bucket"
# # prompt = "a sks dog in front of eifle tower"
# # prompt = "a sks dog in water"
# prompt = "a transparent sks dog cup"
# # prompt = "A photo of sks dog on mount fuji"
# # prompt = "A photo of sks dog in the sky"
# # prompt = "a photo of sks dog pokemon."
# # prompt = "A photo of sks dog in New York"

# scheduler = DDIMScheduler(
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     clip_sample=False,
#     set_alpha_to_one=False,
# )

# # image = pipe(
# #     prompt, num_inference_steps=50, scheduler=scheduler, guidance_scale=7.5
# # ).images[0]

# image = pipe(
#     prompt, num_inference_steps=250, scheduler=scheduler, guidance_scale=7.5, seed=1337
# ).images[0]

# image.save(f"""{"-".join(prompt.split())}.png""")

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
import torch

# model_id = "path-to-your-trained-model"
model_id = "output/yb-cat"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None
).to("cuda")

# prompt = "A photo of sks dog in a bucket"
# prompt = "A photo of an sks dog in a bucket"
# prompt = "A photo of sks cat in front of eifle tower, high-definition"
# prompt = "A photo of sks cat in a bucket"
# prompt = "A photo of sks cat on mount fuji, high-definition"
# prompt = "A sks cat crying, high-definition"
# prompt = "A sks cat drinking coffee, high-definition"
# prompt = "A sks cat in the grand canyon, high-definition"
# prompt = "A sks cat in space, high-definition"
# prompt = "A sks cat on moon, high-definition"
# prompt = "Sks cat as astronaut on the moon sigma 1 4 mm f / 1, 8 the earth is seen in the background surrounded by an asteroid belt made of cat toys"
# prompt = "Photo of sks cat floating inside the international space station, realistic award-winning"
prompt = "A portrait of sks cat behind the space suit helmet in outer space, realistic award-winning"
# prompt = "a sks dog in water"
# prompt = "A photo of sks dog on mount fuji"
# prompt = "A photo of sks dog in the sky"
# prompt = "a photo of sks dog pokemon."
# prompt = "A photo of sks dog in New York"

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)

# image = pipe(
#     prompt, num_inference_steps=50, scheduler=scheduler, guidance_scale=7.5
# ).images[0]

for i in range(30):
    image = pipe(
        prompt,
        num_inference_steps=250,
        scheduler=scheduler,
        guidance_scale=7.5,
        seed=1337,
    ).images[0]

    image.save(f"""{"-".join(prompt.split()) + str(i)}.png"""[:128])
