import zmq
import json
import asyncio
import zmq.asyncio
from queue import Queue
from sortedcontainers import SortedList

from shared import WorkerInfo



class Client:
    def __init__(self, receive_port=5555, workers=["localhost:5556"], my_ip="localhost"):
        self.receive_port = receive_port
        self.my_ip = my_ip
        self.context = zmq.asyncio.Context()
        self.max_worker_queue = 30

        self._max_wait_size = 20
        self._receive_in_order = True
        self._next_expected_task_id = 0
        self._next_task_id = 0
        self._next_worker_id = 0
        self.worker_addresses = []
        self.worker_sockets = {}
        for worker in workers:
            ip, port = worker.split(":")
            self._add_worker(ip, port)

        self.running = False
        self.ready_queue = Queue() # queue of completed tasks that can be collected
        self.wait_for_turn_queue = SortedList(key=lambda x: x['task_id'])
        self._last_result = None
        self.loop = None

    def _get_task_id(self):
        self._next_task_id += 1
        return self._next_task_id

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

    def send_task(self, task: dict) -> bool:
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
        min_queue = 999999
        min_worker = None
        for worker in self.worker_addresses:
            if worker.queue_size < min_queue and worker.queue_size < self.max_worker_queue:
                min_queue = worker.queue_size
                min_worker = worker

        if min_worker is None:
            print("No workers available for task")
            return
        
        # check if running again
        if not self.running:
            return
        
        message = {
            'task': task,
            'task_id': self._get_task_id(),
            'worker_id': min_worker.id,
            'sender_ip': self.my_ip,
            'sender_port': self.receive_port
        }
        
        socket = self._get_send_socket(min_worker)
        print(f"Sending task to {min_worker.ip}:{min_worker.port}")
        await socket.send_json(message)
        min_worker.queue_size += 1

    def _prepare_reply_for_receive(self, reply):
        # prepare the reply to be received
        result = reply['result']
        task_id = reply['task_id']
        # set next expected task id to the next task id
        self._next_expected_task_id = task_id + 1
        self.ready_queue.put(result)

    def _set_queue_size(self, worker_id, queue_size):
        for worker in self.worker_addresses:
            if worker.id == worker_id:
                worker.queue_size = queue_size
                break

    def _add_to_wait_queue(self, reply):
        # add the reply to the wait queue if it is not in order
        result = reply['result']
        task_id = reply['task_id']

        if task_id == self._next_expected_task_id:
            # we can immediately put this in the ready queue, since it is the next expected task
            self.ready_queue.put(result)
            self._next_expected_task_id += 1
            return
        
        # check if it is the next task_id or lower
        if task_id < self._next_expected_task_id:
            # ignore, since we can't return this in order
            # for example, a video stream would play earlier frames if we return them out of order
            return
        
        # add to the wait queue
        self.wait_for_turn_queue.add(reply)

        if len(self.wait_for_turn_queue) > self._max_wait_size:
            # set the next expected task id to the next task id
            self._next_expected_task_id = self.wait_for_turn_queue[0]['task_id']

        while len(self.wait_for_turn_queue) > 0 and self._next_expected_task_id == self.wait_for_turn_queue[0]['task_id']:
            # put the result in the ready queue
            self.ready_queue.put(self.wait_for_turn_queue.pop(0)['result'])
            self._next_expected_task_id += 1

        

    def _handle_reply(self, reply):
        # handle the reply

        # get worker id and decrease queue size
        worker_id = reply['worker_id']
        queue_size = reply['queue_size']
        self._set_queue_size(worker_id, queue_size)

        # if not in order, just put the result in the queue
        if not self._receive_in_order:
            self._prepare_reply_for_receive(reply)
            return
        
        # if in order, check if it is the next expected task
        self._add_to_wait_queue(reply)


    async def _wait_for_reply(self):
        # listen for the reply
        socket = self.context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{self.receive_port}")
        while self.running:
            # timeout to check if still running
            try:
                reply = await asyncio.wait_for(socket.recv_json(), timeout=1)
                self._handle_reply(reply)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                raise e
        socket.close()


    def start(self, loop=None):
        if self.running:
            print("Client already running")
            return
        self.loop = loop or asyncio.new_event_loop()
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