import cv2
import requests
import base64
import numpy as np


def image2image(image_data, prompt=None):
    # send request
    url = "http://localhost:8000/image2image"

    prompt = prompt or "a hairy ape, high quality, high resolution"

    data = {
        "image_data": image_data,
        "prompt": prompt,
        "num_inference_steps": 2,
        "guidance_scale": 0.0,
        "seed": 123,
        "strength": 0.6
    }

    response = requests.post(url, json=data)

    # get response
    response_data = response.json()
    image_data = response_data["image_data"]


    res = []
    for img in image_data:
        image = base64.b64decode(img)
        res.append(image)
    return res


import time

# open webcam
cap = cv2.VideoCapture(0)

first_frame = None

while True:
    t_start = time.time()
    # read frame
    ret, frame = cap.read()

    # resize to hight = 512
    h, w, _ = frame.shape
    new_w = int(w * 512 / h)
    frame = cv2.resize(frame, (new_w, 512))

    # crop center
    h, w, _ = frame.shape
    frame = frame[:, (w - 512) // 2:(w + 512) // 2]



    # convert image to base64
    img_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
    image_data = base64.b64encode(img_bytes).decode("utf-8")


    # send frame to sd turbo
    first_frame = first_frame or image_data
    image_data = image2image(image_data)

    # convert bytes to image
    image = cv2.imdecode(np.frombuffer(image_data[0], np.uint8), -1)

    d_time = time.time() - t_start
    # show fps top left
    cv2.putText(image, f"FPS: {1/d_time:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # show image
    cv2.imshow('frame', image)

    # wait for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

