
# test streaming video by converting from numpy to bytes and back to numpy

import cv2
import numpy as np
import time
import base64
import json

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    shape = frame.shape



    # convert frame to bytes
    frame_bytes = frame.tobytes()

    # convert to base 64    
    frame_base64 = base64.b64encode(frame_bytes).decode()

    # convert to json
    data = {
        'frame': frame_base64,
        'shape': shape
    }

    json_str = json.dumps(data)

    # here we would send the json_str to the server
    # ...beep boop...

    data_dict = json.loads(json_str)

    frame_base64 = data_dict['frame']
    shape = data_dict['shape']

    # convert back to bytes
    frame_bytes = base64.b64decode(frame_base64)

    # convert back to numpy array
    frame = np.frombuffer(frame_bytes, dtype=np.uint8)

    # reshape
    frame = frame.reshape(shape)


    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break