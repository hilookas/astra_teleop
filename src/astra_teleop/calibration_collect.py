
import cv2
import time
from pathlib import Path
from .cam import open_cam
import argparse

def calibration_collect(device="/dev/video0", calibration_directory="./calibration_images", num_images_to_collect=60):
    cam = open_cam(device)

    Path(calibration_directory).mkdir(parents=True, exist_ok=True)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device name.", default="/dev/video0")
    parser.add_argument("-c", "--calibration_directory", help="Calibration directory.", default="./calibration_images")
    parser.add_argument("-n", "--num_images_to_collect", help="num_images_to_collect", default=60, type=int)
    args = parser.parse_args()

    calibration_collect(args.device, args.calibration_directory)