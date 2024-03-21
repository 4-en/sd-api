import zmq
import json
import asyncio
import zmq.asyncio
from queue import Queue

from shared import WorkerInfo



class Client:
    def __init__(self, receive_port=5555, workers=["localhost:5556"]):
        self.receive_port = receive_port
        self.context = zmq.asyncio.Context()
        self.max_worker_queue = 3

        self._next_worker_id = 0
        self.worker_addresses = []
        self.worker_sockets = {}
        for worker in workers:
            ip, port = worker.split(":")
            self._add_worker(ip, port)

        self.running = False
        self.ready_queue = Queue() # queue of completed tasks that can be collected
        self._last_result = None
        self.loop = asyncio.get_event_loop()

    def _add_worker(self, ip, port):
        self.worker_addresses.append(WorkerInfo(ip, port, 0, self._next_worker_id))
        self._next_worker_id += 1

    async def _discover_workers(self):
        # TODO: discover workers on network, for now use fixed address
        pass

    def is_worker_available(self):
        for worker in self.worker_addresses:
            if worker.queue_size < self.max_worker_queue:
                return True
        return False

    def send_task(self, task: str) -> bool:
        if not self.running or not self.is_worker_available():
            return False
        # create a task and send it to a worker
        asyncio.run_coroutine_threadsafe(self._send_task(task), self.loop)
        return True
    
    def receive(self, blocking=True) -> str:
        if not self.running:
            return None
        if blocking:
            return self.ready_queue.get()
        
        if not self.ready_queue.empty():
            self._last_result = self.ready_queue.get()
        return self._last_result
        
    
    def _get_send_socket(self, worker: WorkerInfo):
        wid = worker.id
        if wid not in self.worker_sockets:
            self.worker_sockets[wid] = self.context.socket(zmq.PUSH)
            self.worker_sockets[wid].connect(f"tcp://{worker.ip}:{worker.port}")
        return self.worker_sockets[wid]
    
    def _close_all_sockets(self):
        for socket in self.worker_sockets.values():
            socket.close()
        self.worker_sockets = {}

    async def _send_task(self, task):
        # find the worker with the smallest queue
        min_queue = -1
        min_worker = None
        for worker in self.worker_addresses:
            if min_queue == -1 or worker.queue_size < min_queue and worker.queue_size < self.max_worker_queue:
                min_queue = worker.queue_size
                min_worker = worker

        if min_worker is None:
            print("No workers available for task")
            return
        
        # check if running again
        if not self.running:
            return
        
        socket = self._get_send_socket(min_worker)
        await socket.send_json(task)
        min_worker.queue_size += 1

    def _handle_reply(self, reply):
        # handle the reply
        # just add it to queue for now
        self.ready_queue.put(reply)


    async def _wait_for_reply(self):
        # listen for the reply
        socket = self.context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{self.receive_port}")
        while self.running:
            # timeout to check if still running
            try:
                reply = await asyncio.wait_for(socket.recv_json(), timeout=1)
                self.ready_queue.put(reply)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(e) # print, otherwise ignore and continue
        socket.close()


    def start(self):
        self.loop = asyncio.get_event_loop()
        loop = self.loop
        self.running = True
        
        if len(self.worker_addresses) == 0:
            print("No workers found")
            return
        
        print(f"Starting with {len(self.worker_addresses)} workers")

        loop.run_until_complete(self._wait_for_reply())
        loop.close()
        self.running = False
        self._close_all_sockets()
        print("Client stopped")

    def stop(self):
        self.running = False
        self.loop.stop()
        self._close_all_sockets()

if __name__ == "__main__":
    client = Client()
    client.start()