# https://huggingface.co/stabilityai/sd-turbo


# text to image
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

# load lora file
#adapter_id = "lora_test.safetensors"

#pipe.load_lora_weights(adapter_id)

prompt = "a person giving thumbs up in an office setting"
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

image.show()

# image to image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForImage2Image.from_pipe(pipe)
pipe.to("cuda")

init_image = image
prompt = "a hairy ape with a banana"

for i in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    gen = torch.manual_seed(0)
    image = pipe(prompt, image=init_image, num_inference_steps=2, strength=i, guidance_scale=0.0, generator=gen).images[0]
    image.show()