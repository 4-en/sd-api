import PIL
import base64
import requests

# test image2image endpoint
import os
import sys

# get image file from args
if len(sys.argv) < 2:
    print("Please provide an image file to test.")
    sys.exit(1)

image_file = sys.argv[1]
if not os.path.exists(image_file):
    print(f"Image file {image_file} not found.")
    sys.exit(1)

# read image file
with open(image_file, "rb") as f:
    image_data = f.read()

# convert image to base64
image_data = base64.b64encode(image_data).decode("utf-8")

# prompt
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

# send request
url = "http://localhost:8000/image2image"

data = {
    "image_data": image_data,
    "prompt": prompt,
    "num_inference_steps": 2,
    "guidance_scale": 0.0
}

response = requests.post(url, json=data)

# get response
response_data = response.json()
image_data = response_data["image_data"]

# convert base64 to image
image = base64.b64decode(image_data[0])

import io
from PIL import Image
# open image in window
img = Image.open(io.BytesIO(image))
img.show()

