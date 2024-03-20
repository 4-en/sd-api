
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image

import torch
import torchvision.transforms as transforms
import random
import base64
import PIL
from io import BytesIO
import sys
import argparse


# fastapi
from fastapi import FastAPI

# uvicorn
import uvicorn

# types for fastapi requests
from pydantic import BaseModel

# request and response types
class Image2ImageRequest(BaseModel):
    image_data: str | list[str] # base64 encoded image or list of base64 encoded images
    prompt: str = ""
    seed: int = -1
    num_inference_steps: int = 2
    guidance_scale: float = 0.0
    strength: float = 0.5 # noise strength
    size: tuple[int, int] = (512, 512)
    timesteps: list[int] | None = None

class Image2ImageResponse(BaseModel):
    image_data: str | list[str] # base64 encoded image or list of base64 encoded images


# simple fastapi app for sd-turbo image to image
class SdTurboApi:
    def __init__(self, model_name="stabilityai/sdxl-turbo"):
        self.app = FastAPI()
        self.image2image = AutoPipelineForImage2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16")
        self.image2image.to("cuda")
        self.image2image.enable_xformers_memory_efficient_attention() # enable memory efficient attention

        # if linux, compile model
        if "linux" in sys.platform and False: # disable compiling for now since it causes issues
            print("Compiling model")
            self.image2image.unet = torch.compile(self.image2image.unet, mode="reduce-overhead", fullgraph=True)

        self.image2image.upcast_vae()

        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])

        self.transform_size = 512

        self.to_PIL = transforms.ToPILImage()

        @self.app.post("/image2image")
        async def image2image(request: Image2ImageRequest):
            image_data = request.image_data
            prompt = request.prompt
            seed = request.seed
            if seed == -1:
                seed = random.randint(0, 1000000)
            num_inference_steps = request.num_inference_steps
            guidance_scale = request.guidance_scale
            strength = request.strength
            size = request.size
            timesteps = request.timesteps
            _ = timesteps # TODO: implement timesteps usage

            # change transform if necessary
            if self.transform_size != size:
                self.transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor()
                ])
                self.transform_size = size

            # print(f"seed: {seed}, strength: {strength}")

            if isinstance(image_data, str):
                image_data = [image_data]


            # convert images from base64 to PIL
            converted_images = []
            for img in image_data:
                img = base64.b64decode(img)
                img = PIL.Image.open(BytesIO(img))
                converted_images.append(img)

            # convert PIL to torch tensor
            image_data = []
            for img in converted_images:
                img = self.transform(img)
                image_data.append(img)

            images = []

            for img in image_data:
                rand_gen = torch.Generator(device="cuda").manual_seed(seed)

                neg_prompt = "low quality, deformed, distorted, corrupted, glitch, error, noise, artifact, low resolution, low res"
                image = self.image2image(prompt, image=img, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, neg_prompt=neg_prompt, strength=strength, generator=rand_gen)[0]
                if type(image) == list or type(image) == tuple:
                    for i in image:
                        images.append(i)
                else:
                    images.append(image)

            # convert images from torch tensor to base64
            converted_images = []
            for img in images:
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                img = base64.b64encode(buffer.getvalue()).decode()
                converted_images.append(img)

            print(len(converted_images))

            return Image2ImageResponse(image_data=converted_images)
        
    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

_mapping = {
        "xl_turbo": "stabilityai/sdxl-turbo",
        "turbo": "stabilityai/sd-turbo",
        "xl": "stabilityai/sdxl",
        "base": "stabilityai/sd-base",
    }

def get_model_name(short_name: str) -> str:
    short_name = short_name.lower()
    return _mapping.get(short_name, "stabilityai/sdxl-turbo")

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run sd-turbo api")
    parser.add_argument("--model_name", type=str, default="xl_turbo", help="model name", choices=[short_name for short_name in _mapping.keys()])
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    model_name = get_model_name(args.model_name)
    api = SdTurboApi(model_name=model_name)
    api.run()
