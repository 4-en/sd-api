from client import Client
from worker import Worker

POSITIONS = 11
def encrypt(c):
    return chr((ord(c) + 1) % 256)

def decrypt(c):
    return chr((ord(c) - 1) % 256)

def encrypt_string(s):
    return ''.join([encrypt(c) for c in s])

def decrypt_string(s):
    return ''.join([decrypt(c) for c in s])

def test_callback(task):
    in_c = task['c']
    out_c = decrypt_string(in_c)
    return {
        'c': out_c
    }

input_strings = [
    "This is a very secret message that I want to send to the workers",
    "They will process it and send it back to me",
    "I hope this works :)"
]


W_COUNT = 10
start_port = 5556

worker_adr = [f"localhost:{start_port + i}" for i in range(W_COUNT)]
client = Client(workers=worker_adr)
workers = [Worker(receive_port=start_port + i, task_callback=test_callback) for i in range(W_COUNT)]

import threading
import asyncio


# Function to run the asyncio event loop in a separate thread
def start_asyncio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def start_client():
    loop = asyncio.new_event_loop()
    client.start()

# Function to start workers in their own threads
def start_workers():
    for worker in workers:
        threading.Thread(target=worker.start, args=(None,)).start()

# Function to send encrypted strings to the workers
def send_encrypted_strings():
    for s in input_strings:
        print("Sending:", s)
        for c in encrypt_string(s):
            res = client.send_task({'c': c})

# Function to receive and decrypt messages
def receive_and_decrypt():
    while True:
        task = client.receive()  # Blocking call
        if task is not None:
            result = task
            print("Received:", result['c'])

import time
if __name__ == "__main__":
    # Start workers in separate threads
    threading.Thread(target=start_workers).start()

    # Start client in its own thread for sending tasks
    threading.Thread(target=start_client).start()
    time.sleep(1)
    threading.Thread(target=send_encrypted_strings).start()

    # Use the main thread to receive and print decrypted messages
    receive_and_decrypt()

    # stop client and workers
    client.stop()
    for worker in workers:
        worker.stop()