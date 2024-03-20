from abc import ABC, abstractmethod
import cv2
import zmq
import base64
import numpy as np

class VideoSource(ABC):
    """
    Abstract class for video sources
    """
    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def init_source(self):
        pass

    @abstractmethod
    def release(self):
        pass


class CVSource(VideoSource):
    """
    Video source using OpenCV
    Can be a file or a camera or a stream
    """
    def __init__(self, source):
        self.source = source
        self.cap = None

    def init_source(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", self.source)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # try to loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("Unable to read frame from video source", self.source)
            
        return frame

    def release(self):
        self.cap.release()

class ZMQSource(VideoSource):
    """
    Video source using ZMQ
    """
    def __init__(self, port):
        self.context = zmq.Context()
        self.port = port
        self.socket = None

    def init_source(self):
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(f"tcp://*:{self.port}")

    def get_frame(self):
        encoded_image = self.socket.recv()
        image_data = base64.b64decode(encoded_image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def release(self):
        self.socket.close()
        self.context.term()