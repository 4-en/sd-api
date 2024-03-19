import zmq
import json
import asyncio

from shared import WorkerInfo, WorkerRequest, WorkerReply



class Client:
    def __init__(self, discovery_port=5555, worker_task_port=5556, reply_handler=None):
        self.discovery_port = discovery_port
        self.worker_task_port = worker_task_port
        self.context = zmq.Context()
        self.worker_addresses = []
        self.running = False
        self.reply_handler = reply_handler
        self.running = False

    async def discover_workers(self):
        socket = self.context.socket(zmq.RADIO)
        socket.bind(f"udp://*:{self.discovery_port}")
        await socket.send_string("discover", group='discovery')

        # Listen for worker responses
        response_socket = self.context.socket(zmq.DISH)
        response_socket.bind(f"udp://*:{self.worker_task_port}")
        response_socket.join('tasks')
        while self.running:
            try:
                message = await response_socket.recv_string(zmq.NOBLOCK)  # Non-blocking
                worker_info = WorkerInfo.from_json(message)
                self.worker_addresses.append(worker_info)

            except zmq.Again:
                break  # No more messages

    def send_task(self, task):
        # create a task and send it to the worker
        asyncio.create_task(self._send_task(task))

    async def _send_task(self, task):
        # find the worker with the smallest queue
        min_queue = -1
        min_worker = None
        for worker in self.worker_addresses:
            if min_queue == -1 or worker.queue_size < min_queue:
                min_queue = worker.queue_size
                min_worker = worker

        if min_worker is None:
            print("No workers available")
            return
        
        # send the task to the worker
        socket = self.context.socket(zmq.DISH)
        socket.connect(f"udp://{min_worker.ip}:{min_worker.task_port}")
        await socket.send_json(task)
        # increment the queue size
        min_worker.queue_size += 1

    async def wait_for_reply(self):
        # listen for the reply
        socket = self.context.socket(zmq.DISH)
        socket.bind(f"udp://*:{self.worker_task_port}")
        while self.running:
            message = await socket.recv_json()
            reply = WorkerReply.from_json(message)
            print(f"Received reply: {reply}")

            if self.reply_handler is not None:
                self.reply_handler(reply)

            # set queue size
            for worker in self.worker_addresses:
                if worker.ip == reply.ip:
                    worker.queue_size = reply.queue_size
                    break

    def start(self):
        loop = asyncio.get_event_loop()
        self.running = True
        loop.run_until_complete(self.discover_workers())
        # check if we found any workers
        if len(self.worker_addresses) == 0:
            print("No workers found")
            return
        
        print(f"Starting with {len(self.worker_addresses)} workers")

        loop.run_until_complete(self.wait_for_reply())
        loop.close()
        self.running = False

if __name__ == "__main__":
    client = Client(discovery_port=5555, worker_task_port=5556)
    client.start()