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
import numpy as np

from worker_api import worker


class SDWorker:
    def __init__(self, model_name="stabilityai/sdxl-turbo"):
        self.worker = worker.Worker(receive_port=5556, task_callback=self._task_callback)
        self.worker.start()

        self.image2image = AutoPipelineForImage2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16")
        self.image2image.to("cuda")
        self.image2image.enable_xformers_memory_efficient_attention()

        if "linux" in sys.platform and False:
            print("Compiling model")
            self.image2image.unet = torch.compile(self.image2image.unet, mode="reduce-overhead", fullgraph=True)

        if "xl" in model_name:
            self.image2image.upcast_vae()

        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])

    def _task_callback(self, task):
        params = task.get('params', {"prompt": "a cute cat"}) # default prompt
        shape = params.get('shape', (512, 512, 3)) # default shape
        image_bytes = base64.b64decode(params['image_data'])
        np_image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = np_image.reshape(shape)

        # turn image into black and white with numpy
        image = np.mean(image, axis=2)
        image = np.stack([image, image, image], axis=2)
        

        # send back the image
        b64image = base64.b64encode(image.tobytes()).decode()
        return {
            'image_data': b64image,
            'shape': image.shape
        }

    def start(self):
        self.worker.start()

    def stop(self):
        self.worker.stop()

def get_params(args=None):
    parser = argparse.ArgumentParser(description='SD Worker')
    parser.add_argument('--model_name', type=str, default="stabilityai/sdxl-turbo", help='model name')
    return parser.parse_args(args)

def main():
    args = get_params()
    worker = SDWorker(model_name=args.model_name)
    worker.start()

if __name__ == "__main__":
    main()