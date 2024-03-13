# test performance of huggingface diffusers

from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image

import torch

import argparse

import time


def create_random_input(size, batch_size=1):
    return torch.rand(batch_size, 3, size, size).cuda()

def main():
    parser = argparse.ArgumentParser(description='Benchmark diffusers')
    parser.add_argument('--images', type=int, default=20, help='Number of images to benchmark')
    parser.add_argument('--model', type=str, default="stabilityai/sd-turbo", help='Model to benchmark')
    parser.add_argument('--steps', type=int, default=1, help='Number of inference steps')
    parser.add_argument('--size', type=int, default=512, help='Size of images')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')


    args = parser.parse_args()

    strength = 1.0

    print(f"Running benchmark for {args.images} images with model {args.model} and {args.steps} steps")

    image2image = AutoPipelineForImage2Image.from_pretrained(args.model, torch_dtype=torch.float16, variant="fp16")
    image2image.to("cuda")
    #image2image.enable_xformers_memory_efficient_attention()

    # compile unet (windows not supported)
    #image2image.unet = torch.compile(image2image.unet, mode="reduce-overhead", fullgraph=True)

    inputs = create_random_input(args.size, args.batch_size)

    # warmup
    outputs = image2image(["a picture of a cute cat"]*args.batch_size, inputs, num_inference_steps=args.steps, strength=strength)[0]
    inputs = outputs

    print("Starting benchmark")
    start = time.time()
    processed_count = 0
    for _ in range(0, args.images, args.batch_size):
        outputs = image2image(["a picture of a cute cat"]*args.batch_size, inputs, num_inference_steps=args.steps, strength=strength)[0]
        # use outputs as new inputs
        inputs = outputs
        processed_count += args.batch_size
    end = time.time()
    fps = processed_count / (end - start)
    print("Benchmark completed")
    print(f"Time for {args.images} images: {end - start}")
    print(f"FPS: {fps}")

if __name__ == "__main__":
    main()
