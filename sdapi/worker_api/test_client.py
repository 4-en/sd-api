import numpy as np
import cv2
import base64

from worker_api import client

sd_client = client.Client(5555, ["localhost:5556"], "localhost")

def send_frame(frame):
    # frame to bytes
    shape = frame.shape
    frame = frame.tobytes()
    frame = base64.b64encode(frame).decode('utf-8')

    message = {
        "shape": shape,
        "image_data": frame
    }
    sd_client.send_task(message)

    # receive result
    result = sd_client.receive(blocking=False)
    res_frame = None
    if result:
        res_frame = base64.b64decode(result["image_data"])
        res_frame = np.frombuffer(res_frame, dtype=np.uint8)
        res_frame = res_frame.reshape(result["shape"])
    return res_frame

# open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # scale to 512 height
    h, w = frame.shape[:2]
    scale = 512 / h
    frame = cv2.resize(frame, (int(w * scale), 512))

    # crop to 512x512 from center
    h, w = frame.shape[:2]
    x = (w - 512) // 2
    y = (h - 512) // 2
    frame = frame[y:y+512, x:x+512]

    # send frame to worker
    result = client.send_frame(frame)

    # display result
    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break
