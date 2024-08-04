import numpy as np
import math
import argparse
from .process import calibration_load


# https://blog.csdn.net/Vpn_zc/article/details/125976495
def camera_intrinsic_transform(fov_x=45, fov_y=60, pixel_width=320, pixel_height=240):
    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = (pixel_width / 2.0) / math.tan(math.radians(fov_x / 2.0))
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = (pixel_height / 2.0) / math.tan(math.radians(fov_y / 2.0))
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics


def camera_intrinsic_fov(intrinsic):
    w, h = intrinsic[0][2] * 2, intrinsic[1][2] * 2
    fx, fy = intrinsic[0][0], intrinsic[1][1]

    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))
    # https://blog.csdn.net/qq_42957717/article/details/125508235
    fov = np.rad2deg(2 * np.arctan2(math.sqrt(w * w + h * h), fx + fy))
    return fov, fov_x, fov_y, w, h


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--calibration_directory", help="Calibration directory.", default="./calibration_images")
    args = parser.parse_args()

    # Load calibration
    camera_matrix, distortion_coefficients = calibration_load(args.calibration_directory)
    fov, fov_x, fov_y, w, h = camera_intrinsic_fov(camera_matrix)
    print(fov, fov_x, fov_y, w, h)
    print(camera_intrinsic_transform(fov_x, fov_y, w, h))
