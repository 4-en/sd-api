import cv2
import requests
import base64
import numpy as np
import random
import argparse

# threading
from threading import Thread

class FrameBuffer:
    def __init__(self, size=3):
        self.size = size
        self.frames = [None] * size
        self.idx = 0

    def add_frame(self, frame):
        self.frames[self.idx] = frame
        self.idx = (self.idx + 1) % self.size

    def get_frame(self):
        return self.frames[self.idx]

    def get_frame_at(self, idx):
        return self.frames[idx % self.size]

    def get_frame_count(self):
        return self.size

    def get_frames(self):
        return self.frames



import time

class VideoDisplay:
    def __init__(self, frame_source: callable, target_fps: int = 30, window_name: str = "Video Display"):
        self.frame_source = frame_source
        self.target_fps = target_fps
        self.running = False
        self.window_name = window_name

    def start(self):
        if self.running:
            print("Already running")
            return
        self.running = True
        self._run()

    def stop(self):
        self.running = False

    def _run(self):

        last_frame_time = time.time()
        d_time = 0

        last_frame = None

        while self.running:
            frame = self.frame_source()
            if frame is None:
                if last_frame is not None:
                    frame = last_frame
                else:
                    time.sleep(1 / self.target_fps)
                    continue
            else:
                last_frame = frame

            cv2.imshow(self.window_name, frame)

            d_time = time.time() - last_frame_time
            last_frame_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                break
            time.sleep(1 / self.target_fps)

        # destroy named window if not already destroyed
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(self.window_name)



class HeadSizeCalculator:
    """
    Class to calculate the size of the head in pixels in a frame
    """
    def __init__(self):
        self.buffer = FrameBuffer(3)


        self.last_sizes = []
        self.recalculate = True
        self.running = False

    def add_frame(self, frame):
        self.buffer.add_frame(frame)
        self.recalculate = True

    def get_head_size(self):
        # remove -1 values
        sizes = [s for s in self.last_sizes if s != -1]
        if len(sizes) == 0:
            return -1
        # average
        return sum(sizes) / len(sizes)
    
    def calculate_head_size(self):
        # calculate head size
        latest_frame = self.buffer.get_frame()
        if latest_frame is None:
            return
        
        # convert to grayscale
        gray = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # calculate head sizes and save biggest
        max_size = -1
        for (x, y, w, h) in faces:
            max_size = max(max_size, max(w, h))

        self.last_sizes.append(max_size)
        if len(self.last_sizes) > 10:
            self.last_sizes = self.last_sizes[1:]
        self.recalculate = False

    def _run(self):
        while self.running:
            if self.recalculate:
                self.calculate_head_size()
            time.sleep(1 / 30)

    def start(self):
        if self.running:
            print("Already running")
            return
        self.running = True
        self._run()


class HeadBasedDisplay(VideoDisplay):

    def __init__(self, frame_source: callable, target_fps: int = 30, window_name: str = "Video Display"):
        super().__init__(frame_source, target_fps, window_name)
        self.head_size_calculator = HeadSizeCalculator()
        self.head_size_calculator_thread = Thread(target=self.head_size_calculator.start, daemon=True, name="HeadSizeCalculatorThread")
        self.head_threshold = 50

        self.original_frame_source = self.frame_source

        def frame_source_wrapper():
            frame = self.original_frame_source()
            self.head_size_calculator.add_frame(frame)
            size = self.head_size_calculator.get_head_size()
            if size is not None and size > self.head_threshold:
                # color frame red
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame[:, :, 0] = frame[:, :, 0] + 100
                
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        
        self.frame_source = frame_source_wrapper

    def start(self):
        self.head_size_calculator_thread.start()
        super().start()

    def stop(self):
        super().stop()
        self.head_size_calculator.running = False
        self.head_size_calculator_thread.join()    



class VideoCapture:
    """
    Class to capture video from webcam or other source and send it to a processing pipeline
    """
    def __init__(self):
        self.resolution = (512, 512)
        self.fps = -1
        self.running = False

        self.processed_buffer = FrameBuffer()
        self.raw_buffer = FrameBuffer()

        self.init_source()

        self.last_frame_times = [time.time()] * 10

    def get_fps(self):
        mean = np.mean(np.diff(self.last_frame_times))
        if mean == 0:
            return self.fps
        self.fps = 1 / mean
        return self.fps


    def get_next_frame(self) -> np.ndarray:
        succ, frame = self.cap.read()
        if not succ:
            raise Exception("Failed to capture frame")
        return frame
    
    def init_source(self):
        self.cap = cv2.VideoCapture(0)

    def release(self):
        self.cap.release()

    def process_frame(self, frame) -> np.ndarray:
        # scale so that width and height are at least self.resolution
        scale = max(self.resolution[0] / frame.shape[0], self.resolution[1] / frame.shape[1])
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        # crop self.resolution from the center
        frame = frame[(frame.shape[0] - self.resolution[0]) // 2:(frame.shape[0] + self.resolution[0]) // 2,
                      (frame.shape[1] - self.resolution[1]) // 2:(frame.shape[1] + self.resolution[1]) // 2]

        return frame

    def send_frame(self, frame):
        self.receive_frame(frame) # for testing, remove later
        return
        res = self.image2image_rest(frame)
        for img in res:
            self.receive_frame(img)

    def receive_frame(self, frame):
        # store frame in buffer
        self.processed_buffer.add_frame(frame)

        # update fps
        self.last_frame_times = self.last_frame_times[1:] + [time.time()]

    def _run(self):
        while self.running:
            try:
                frame = self.get_next_frame()
                frame = self.process_frame(frame)
                self.raw_buffer.add_frame(frame)
                self.send_frame(frame)
            except Exception as e:
                print(e)
                self.running = False

        self.release()

    def start(self):
        if self.running:
            print("Already running")
            return
        self.running = True
        self._run()

    def stop(self):
        self.running = False

    def image2image_rest(self, frame, prompt=None):
        # send request
        # 10.35.2.162 4090
        url = "http://10.35.2.135:8000/image2image"

        prompt = prompt or "a picture of a cute cat"
        #prompt = "a picture of a fairy with bright clothes, wings, and a magic wand, sitting in front of an open flame"
        #prompt = "a marble statue of a woman, ancient, roman"

        # put frame in list if not already
        if not isinstance(frame, list):
            frame = [frame]

        # encode frame to base64
        frame = [base64.b64encode(cv2.imencode('.jpg', f)[1]).decode("utf-8") for f in frame]

        data = {
            "image_data": frame,
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
            image = np.frombuffer(image, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            res.append(image)
        return res

import zmq
class VideoCaptureZMQ(VideoCapture):
    def __init__(self, zmq_port: int = 5555, zmq_ip: str = "localhost"):
        super().__init__()
        self.zmq_port = zmq_port
        self.zmq_ip = zmq_ip

    def init_source(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(f"tcp://{self.zmq_ip}:{self.zmq_port}")

    def get_next_frame(self) -> np.ndarray:
        encoded_image = self.socket.recv()
        image_data = base64.b64decode(encoded_image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def release(self):
        self.socket.close()
        self.context.term()



def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Capture and process video from webcam")

    # enable zmq
    parser.add_argument("--zmq", action="store_true", help="Enable ZMQ communication")

    # zmq args
    parser.add_argument("--zmq-port", type=int, default=5555, help="Port for ZMQ communication")
    parser.add_argument("--zmq-ip", type=str, default="localhost", help="IP for ZMQ communication")
    return parser.parse_args(args)

if __name__ == "__main__":

    args = parse_args()


    # create video capture object
    vc = VideoCapture() if not args.zmq else VideoCaptureZMQ(args.zmq_port, args.zmq_ip)

    # create video display object
    #vd = VideoDisplay(vc.processed_buffer.get_frame, 30, "SDXL-Turbo")
    vd = HeadBasedDisplay(vc.processed_buffer.get_frame, 30, "SDXL-Turbo")

    # start video capture in separate thread
    capture_thread = Thread(target=vc.start, daemon=True, name="CaptureThread")
    capture_thread.start()

    # start video display in this process
    vd.start()

    # stop video capture
    vc.stop()
    capture_thread.join()

    print("Done")