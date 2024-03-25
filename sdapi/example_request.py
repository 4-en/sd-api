import numpy as np
import cv2
import base64
import requests
from typing import List

def image2image(frame: np.ndarray | List[np.ndarray],
                prompt: str = "a cute cat",
                num_inference_steps: int = 2,
                guidance_scale: float = 0.0,
                seed: int = 123,
                strength: float = 0.8,
                size: int = (512, 512),
                sd_ip: str = "localhost"
                ) -> np.ndarray | List[np.ndarray]:

        # get url with default port and endpoint
        url = f"http://{sd_ip}:8000/image2image"

        return_list = True
        # put frame in list if not already
        if not isinstance(frame, list):
            frame = [frame]
            return_list = False

        # encode frame to base64
        frame = [base64.b64encode(cv2.imencode('.jpg', f)[1]).decode("utf-8") for f in frame]

        # setting data for request
        data = {
            "image_data": frame,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "strength": strength,
            "size": size
        }

        # send request, this will block until response is received
        response = requests.post(url, json=data)

        # get image data from response
        response_data = response.json()
        image_data = response_data["image_data"]

        # decode image data
        res = []
        for img in image_data:
            image = base64.b64decode(img)
            image = np.frombuffer(image, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            res.append(image)
        
        if return_list:
            return res
        return res[0]