import zmq
import json
import threading
import asyncio

import queue

from shared import WorkerInfo, WorkerRequest, WorkerReply


class Worker:
    def __init__(self, discovery_port=5555, task_port=5556, handle_task=None):
        self.discovery_port = discovery_port
        self.task_port = task_port
        self.context = zmq.Context()
        self.queue = queue.Queue()
        self.task_worker = threading.Thread(target=self.work_on_tasks)
        self.running = False
        self.handle_task = handle_task

    def work_on_tasks(self):
        while self.running:
            try:
                task = self.queue.get()
                if task is None:
                    break
                worker_request = WorkerRequest(**task)
                image_data = self.handle_task(worker_request)
                reply = WorkerReply(frame_id=worker_request.frame_id, image_data=image_data, current_queue=self.queue.qsize())
                self.send_reply(reply)
            except Exception as e:
                print(f"Error processing task: {e}")
    
    async def listen_for_discovery(self):
        print("Listening for discovery")
        socket = self.context.socket(zmq.DISH)
        socket.bind(f"udp://*:{self.discovery_port}")
        socket.join('discovery')
        while self.running:
            message = await socket.recv_string()
            if message == "discover":
                response_socket = self.context.socket(zmq.RADIO)
                response_socket.connect(f"udp://client_ip:{self.task_port}")
                my_info = WorkerInfo(ip="worker_ip", task_port=self.task_port, queue_size=self.queue.qsize())
                await response_socket.send_json(my_info)
    
    async def listen_for_tasks(self):
        print("Listening for tasks")
        socket = self.context.socket(zmq.DISH)
        socket.bind(f"udp://*:{self.task_port}")
        while self.running:
            task = await socket.recv_json()
            worker_request = WorkerRequest(**task)
            self.queue.put(worker_request)

    def send_reply(self, reply):
        asyncio.create_task(self._send_reply(reply))

    async def _send_reply(self, reply):
        socket = self.context.socket(zmq.DISH)
        socket.connect(f"udp://client_ip:{self.discovery_port}")
        await socket.send_json(reply)

    def start(self):
        loop = asyncio.get_event_loop()
        tasks = [self.listen_for_discovery(), self.listen_for_tasks()]
        self.running = True
        self.task_worker.start()
        loop.run_until_complete(asyncio.gather(*tasks))
        loop.close()
        self.running = False
        # add none to the queue to unblock the worker
        self.queue.put(None)
        self.task_worker.join()

if __name__ == "__main__":
    worker = Worker(discovery_port=5555, task_port=5556)
    worker.start()