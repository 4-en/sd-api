
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image

import torch
import random
import base64

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

class Image2ImageResponse(BaseModel):
    image_data: str | list[str] # base64 encoded image or list of base64 encoded images


# simple fastapi app for sd-turbo image to image
class SdTurboApi:
    def __init__(self):
        self.app = FastAPI()
        self.image2image = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
        self.image2image.to("cuda")

        @self.app.post("/image2image")
        async def image2image(request: Image2ImageRequest):
            image_data = request.image_data
            prompt = request.prompt
            seed = request.seed
            if seed == -1:
                seed = random.randint(0, 1000000)
            num_inference_steps = request.num_inference_steps
            guidance_scale = request.guidance_scale

            if isinstance(image_data, str):
                image_data = [image_data]

            # convert images from base64 to PIL
            converted_images = []
            for img in image_data:
                img = base64.b64decode(img)
                converted_images.append(img)

            # convert PIL to torch tensor
            image_data = []
            for img in converted_images:
                image = self.image2image.load_image(img).resize((512, 512))
                image_data.append(image)            

            images = []
            for img in image_data:
                image = self.image2image(prompt, image=img, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, seed=seed).images[0]
                images.append(image)

            # convert images from torch tensor to base64
            converted_images = []
            for img in images:
                img = self.image2image.to_pil_image(img)
                img = self.image2image.to_base64(img)
                converted_images.append(img)

            return Image2ImageResponse(image_data=converted_images)
        
    def run(self):
        uvicorn.run(self.app, host="localhost", port=8000)

if __name__ == "__main__":
    api = SdTurboApi()
    api.run()
