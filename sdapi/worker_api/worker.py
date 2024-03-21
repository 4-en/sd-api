import asyncio
import zmq.asyncio
from concurrent.futures import ThreadPoolExecutor

# for testing
import time
import random

class Worker:
    def __init__(self, receive_port=5556):
        self.receive_port = receive_port
        self.context = zmq.asyncio.Context()
        self.task_queue = asyncio.Queue()  # Queue for incoming tasks
        self.response_sockets = {}  # Dictionary to hold dynamic PUSH sockets
        self.executor = ThreadPoolExecutor(max_workers=1)  # Executor for task processing
        self.running = False

    async def process_tasks(self):
        while self.running:
            try:
                task_json = await asyncio.wait_for(self.task_queue.get(), timeout=1)

                # Process the task in the executor to prevent blocking the event loop
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.process_task, task_json)

                # check if the result is None
                if result is None:
                    self.task_queue.task_done()
                    continue
                
                # After processing, send the response
                await self.send_response(result)
                self.task_queue.task_done()
            except asyncio.TimeoutError:
                pass


    def process_task(self, task_json):
        # Simulate task processing that would block the event loop
        # This is where you'd interact with the GPU or perform other intensive computations
        time.sleep(random.random() * 2) # Sleep for a random amount of time, 0-2 seconds
        task = task_json.get('task', None)
        if task is None:
            return None
        return task_json

    async def send_response(self, result):
        sender_ip = result['sender_ip']
        sender_port = result['sender_port']
        sender_id = f"{sender_ip}:{sender_port}"

        if sender_id not in self.response_sockets:
            self.response_sockets[sender_id] = self.context.socket(zmq.PUSH)
            self.response_sockets[sender_id].connect(f"tcp://{sender_ip}:{sender_port}")

        socket = self.response_sockets[sender_id]
        await socket.send_json(result)

    async def run(self):
        self.running = True
        pull_socket = self.context.socket(zmq.PULL)
        pull_socket.bind(f"tcp://*:{self.receive_port}")

        # Start the background task processing coroutine
        asyncio.create_task(self.process_tasks())

        while self.running:
            try:
                message = await asyncio.wait_for(pull_socket.recv_json(), timeout=1)
                await self.task_queue.put(message)
            except asyncio.TimeoutError:
                pass


    def stop(self):
        self.running = False
        # Close dynamic PUSH sockets
        for socket in self.response_sockets.values():
            socket.close()
        self.response_sockets.clear()

if __name__ == "__main__":
    worker = Worker()
    asyncio.run(worker.run())