import cv2
import requests
import base64
import numpy as np
import random

SEED = 123
SIZE = 512

PROMPTS = [
    "a picture of a cute cat",
    "an expressionist painting by Picasso",
    "a picture of a robot in the rain, sci-fi style, futuristic, neon lights, light reflections",
    "a hairy ape with a banana",
    "picture of a lego world with lego people, lego, lego bricks, blocks, colorful"
]

def randomize_seed():
    global SEED
    print("Randomizing seed")
    SEED = random.randint(0, 1000000)

def image2image(image_data, prompt=None):
    global SEED
    # send request
    # 10.35.2.162 4090
    url = "http://10.35.2.135:8000/image2image"

    prompt = prompt or "a picture of a cute cat"
    #prompt = "a picture of a fairy with bright clothes, wings, and a magic wand, sitting in front of an open flame"
    #prompt = "a marble statue of a woman, ancient, roman"

    data = {
        "image_data": image_data,
        "prompt": prompt,
        "num_inference_steps": 2,
        "guidance_scale": 0.0,
        "seed": SEED,
        "strength": 0.8,
        "size": (SIZE, SIZE)
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
            # scale to 1024x1024
            frame = cv2.resize(frame, (1024, 1024))
            d_time = time.time() - last_frame_time

            # mirror image
            frame = cv2.flip(frame, 1)

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

            # change seed if s is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                randomize_seed()
        else:
            if last_frame is not None:
                cv2.imshow('Processed Frame', last_frame)
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Processed Frame', cv2.WND_PROP_VISIBLE) < 1:
                    running = False
                    break

                # change seed if s is pressed
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    randomize_seed()
            
            time.sleep(target_fps / 2000)

    running = False

def capture_and_send(frame_queue, processed_frame_queue, batch_size, target_fps):
    cap = cv2.VideoCapture(0)
    batch = []
    next_frame_time = time.time() + 1 / target_fps

    prompt_idx = 0
    last_prompt_change = time.time()
    time_per_prompt = 10

    global running
    while running:
        t_start = time.time()

        if t_start - last_prompt_change > time_per_prompt:
            prompt_idx = (prompt_idx + 1) % len(PROMPTS)
            last_prompt_change = t_start
            


        if t_start >= next_frame_time:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # scale so that hight is 512
            scale = SIZE / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), SIZE))

            

            # crop SIZExSIZE from the center
            frame = frame[:, (frame.shape[1] - SIZE) // 2:(frame.shape[1] + SIZE) // 2]


            # Convert image to base64 for processing
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = base64.b64encode(buffer).decode("utf-8")
            batch.append(image_data)

            if len(batch) == batch_size:

                prompt = PROMPTS[prompt_idx]

                # Send batch for processing
                processed_images = image2image(batch, prompt=prompt) # Assuming this function processes the batch
                
                # Assuming processed_images is a list of base64 images
                for img_bytes in processed_images:
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    processed_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    processed_frame_queue.put((processed_frame, time.time()))
                
                batch.clear()

            next_frame_time += 1 / target_fps

    cap.release()

    running = False

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
