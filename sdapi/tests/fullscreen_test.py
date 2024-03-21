import cv2
import numpy as np

SCREEN_RESOLUTION = (1920, 1080)


# Open webcam with max resolution
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initially set to windowed mode
is_fullscreen = False
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

_ws_ticks = 60
_last_ws = SCREEN_RESOLUTION
def get_window_size(window_name):
    global _ws_ticks, _last_ws
    # ensure we don't call cv2.getWindowImageRect too often
    if _ws_ticks > 0:
        _ws_ticks -= 1
        return _last_ws
    ws = cv2.getWindowImageRect(window_name)
    if not ws:
        return None
    _last_ws = (ws[2] - ws[0], ws[3] - ws[1])
    _ws_ticks = 60
    return _last_ws

def toggle_fullscreen():
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    if is_fullscreen:
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

def show_fullscreen(frame, min_window_size=(512, 512)):
    screen_resolution = get_window_size('frame') or SCREEN_RESOLUTION
    frame_height, frame_width = frame.shape[:2]

    # Ensure window size does not go below a minimum size
    screen_resolution = (
        max(min_window_size[0], screen_resolution[0]),
        max(min_window_size[1], screen_resolution[1])
    )

    # Calculate the aspect ratios
    frame_aspect_ratio = frame_width / frame_height
    screen_aspect_ratio = screen_resolution[0] / screen_resolution[1]

    # Determine padding type based on aspect ratios
    if frame_aspect_ratio > screen_aspect_ratio:
        # Pad vertically
        scale_factor = screen_resolution[0] / frame_width
        new_height = int(frame_height * scale_factor)
        padded_frame = cv2.resize(frame, (screen_resolution[0], new_height))
        top_padding = max(0, (screen_resolution[1] - new_height) // 2)
        bottom_padding = max(0, screen_resolution[1] - new_height - top_padding)
    else:
        # Pad horizontally
        scale_factor = screen_resolution[1] / frame_height
        new_width = int(frame_width * scale_factor)
        padded_frame = cv2.resize(frame, (new_width, screen_resolution[1]))
        left_padding = max(0, (screen_resolution[0] - new_width) // 2)
        right_padding = max(0, screen_resolution[0] - new_width - left_padding)

    padded_frame = cv2.copyMakeBorder(
        padded_frame,
        top_padding if frame_aspect_ratio > screen_aspect_ratio else 0,
        bottom_padding if frame_aspect_ratio > screen_aspect_ratio else 0,
        left_padding if frame_aspect_ratio <= screen_aspect_ratio else 0,
        right_padding if frame_aspect_ratio <= screen_aspect_ratio else 0,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    
    # Display the padded frame
    cv2.imshow('frame', padded_frame)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # crop square of frame
    width = frame.shape[1]
    height = frame.shape[0]
    if width > height:
        frame = frame[:, (width - height) // 2:(width - height) // 2 + height]
    elif height > width:
        frame = frame[(height - width) // 2:(height - width) // 2 + width, :]
    
    show_fullscreen(frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:  # 'q' key or ESC key
        break
    elif key == ord('f'):
        toggle_fullscreen()
    

cap.release()
cv2.destroyAllWindows()