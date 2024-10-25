import cv2
import numpy as np
from pathlib import Path
import glob
import yaml
from yaml.loader import SafeLoader
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import math
from astra_teleop.cam import open_cam
import argparse
from pprint import pprint
from astra_teleop.process import calibration_load, get_detect, get_solve
import socket

# # visualize via matplotlib
# import matplotlib.pyplot as plt
# from pytransform3d.plot_utils import make_3d_axis
# plt.ion()
# plt.cla()
# ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
# pt.plot_transform(ax=ax)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device name.", default="/dev/video0")
    parser.add_argument("-c", "--calibration_directory", help="Calibration directory.", default="./calibration_images")
    args = parser.parse_args()

    device = args.device
    calibration_directory = args.calibration_directory
    debug = True

    # Open camera
    cam = open_cam(device)

    # Load calibration
    camera_matrix, distortion_coefficients = calibration_load(calibration_directory)
    
    detect = get_detect()
    solve = get_solve()

    while True:
        ret, rgb_image = cam.read()
        
        if debug:
            # draw results
            debug_image = rgb_image.copy()
            
        aruco_corners, aruco_ids = detect(
            rgb_image,
            debug,
            debug_image,
        )
        
        debug_image2 = debug_image.copy()
            
        right_tag2cam = None
        def right_hand_cb(tag2cam):
            global right_tag2cam
            right_tag2cam = tag2cam
        solve(
            camera_matrix, distortion_coefficients,
            aruco_corners, aruco_ids,
            None, right_hand_cb,
            debug,
            debug_image,
            min_aruco_thres=4
        )
          
        right_tag2cam2 = None  
        def right_hand_cb(tag2cam):
            global right_tag2cam2
            right_tag2cam2 = tag2cam
        solve(
            camera_matrix, distortion_coefficients,
            aruco_corners, aruco_ids,
            None, right_hand_cb,
            debug,
            debug_image2,
            min_aruco_thres=1
        )

        if debug:
            debug_image = np.concatenate((debug_image[:, 480:480+960], debug_image2[:, 480:480+960]), axis=1)
            debug_image = cv2.resize(debug_image, (1280, 720))
            
            # debug_image = cv2.flip(debug_image, 1)
            cv2.imshow('Debug Image', debug_image)

            # debug_image2 = cv2.undistort(debug_image, camera_matrix, distortion_coefficients)
            # cv2.imshow('Debug Image 2', debug_image2)
            if (cv2.waitKey(1) == 27): # Must wait, otherwise imshow will show black screen
                break
            
            # pprint(right_tag2cam)
            # pt.plot_transform(ax, A2B=right_tag2cam)
            # # plt.show()
            # plt.draw()
            # plt.pause(0.001)
            
            # Plot with vofa
            if right_tag2cam is not None and right_tag2cam2 is not None:
                pq = pt.pq_from_transform(right_tag2cam)
                pq2 = pt.pq_from_transform(right_tag2cam2)
                
                sock.sendto(bytes(f"{pq[0]}, {pq[1]}, {pq[2]}, {pq[3]}, {pq[4]}, {pq[5]}, {pq[6]}, {pq2[0]}, {pq2[1]}, {pq2[2]}, {pq2[3]}, {pq2[4]}, {pq2[5]}, {pq2[6]}\n", "utf-8"), ("127.0.0.1", 10086))
