import cv2

# Open camera
def open_cam(device):
    cam = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fourcc_value = cv2.VideoWriter_fourcc(*'MJPG')

    image_height = 1080
    image_width = 1920

    image_size = (image_height, image_width)

    frames_per_second = 30

    cam.set(cv2.CAP_PROP_FOURCC, fourcc_value)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cam.set(cv2.CAP_PROP_FPS, frames_per_second)
    
    return cam