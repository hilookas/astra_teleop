
import cv2
import time
from pathlib import Path

# Open camera
cam = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
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



calibration_directory = "./calibration_images"
Path(calibration_directory).mkdir(parents=True, exist_ok=True)

num_images_to_collect = 60
time_between_images_sec = 0.5

prev_save_time = time.time()
num_images = 0

while num_images < num_images_to_collect:
    ret, color_image = cam.read()

    cv2.imshow('image from camera', color_image)
    cv2.waitKey(1)

    curr_time = time.time()

    if (curr_time - prev_save_time) > time_between_images_sec:
        num_images = num_images + 1
        file_name = Path(calibration_directory) / (str(num_images).zfill(4) + '.png')
        print('save', file_name)
        cv2.imwrite(file_name, color_image)
        prev_save_time = curr_time

