from dataclasses import dataclass
import json


@dataclass
class WorkerRequest:
    frame_id: int
    image_data: list[str]
    prompt: str = ""
    num_inference_steps: int = 2
    guidance_scale: float = 0.0
    seed: int = 0
    strength: float = 0.8
    size: tuple = (512, 512)

    @staticmethod
    def from_json(json_str):
        return WorkerRequest(**json.loads(json_str))

@dataclass
class WorkerReply:
    frame_id: int
    image_data: list[str]
    current_queue: int = 0

    @staticmethod
    def from_json(json_str):
        return WorkerReply(**json.loads(json_str))

@dataclass
class WorkerInfo:
    ip: str
    task_port: int
    queue_size: int = 0

    @staticmethod
    def from_json(json_str):
        return WorkerInfo(**json.loads(json_str))
