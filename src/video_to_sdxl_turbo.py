import cv2
import requests
import base64
import numpy as np


def image2image(image_data, prompt=None):
    # send request
    url = "http://localhost:8000/image2image"

    prompt = prompt or "pennywise, stephen king, IT, horror, clown, high quality, high resolution"

    data = {
        "image_data": image_data,
        "prompt": prompt,
        "num_inference_steps": 2,
        "guidance_scale": 0.0,
        "seed": 123,
        "strength": 0.7
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

import threading
from queue import Queue

running = True

def process_and_display(frame_queue, processed_frame_queue, target_fps):
    global running
    last_frame_time = time.time()
    last_frame = None
    while running:
        if not processed_frame_queue.empty():
            frame, t_start = processed_frame_queue.get()
            d_time = time.time() - last_frame_time
            last_frame_time = time.time()

            # fps top left
            cv2.putText(frame, f"FPS: {1 / d_time:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            last_frame = frame
            cv2.imshow('Processed Frame', frame)
            
            # Calculate display time to match target FPS
            time_to_wait = max(0, 1 / target_fps - (time.time() - t_start))
            time.sleep(time_to_wait)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Processed Frame', cv2.WND_PROP_VISIBLE) < 1:
                running = False
                break
        else:
            if last_frame is not None:
                cv2.imshow('Processed Frame', last_frame)
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Processed Frame', cv2.WND_PROP_VISIBLE) < 1:
                    running = False
                    break
            time.sleep(target_fps / 2000)

def capture_and_send(frame_queue, processed_frame_queue, batch_size, target_fps):
    cap = cv2.VideoCapture(0)
    batch = []
    next_frame_time = time.time() + 1 / target_fps

    global running
    while running:
        t_start = time.time()

        if t_start >= next_frame_time:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # scale so that hight is 512
            scale = 512 / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), 512))

            # crop 512x512
            frame = frame[:, frame.shape[1] // 2 - 256:frame.shape[1] // 2 + 256]

            # Convert image to base64 for processing
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = base64.b64encode(buffer).decode("utf-8")
            batch.append(image_data)

            if len(batch) == batch_size:
                # Send batch for processing
                processed_images = image2image(batch)  # Assuming this function processes the batch
                
                # Assuming processed_images is a list of base64 images
                for img_bytes in processed_images:
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    processed_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    processed_frame_queue.put((processed_frame, time.time()))
                
                batch.clear()

            next_frame_time += 1 / target_fps

    cap.release()

if __name__ == "__main__":
    batch_size = 1
    target_fps = 10

    frame_queue = Queue()
    processed_frame_queue = Queue()

    capture_thread = threading.Thread(target=capture_and_send, args=(frame_queue, processed_frame_queue, batch_size, target_fps))
    capture_thread.daemon = True
    display_thread = threading.Thread(target=process_and_display, args=(frame_queue, processed_frame_queue, target_fps))
    display_thread.daemon = True

    capture_thread.start()
    display_thread.start()
    try:
        # wait for keyboard interrupt
        capture_thread.join()
        display_thread.join()
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()