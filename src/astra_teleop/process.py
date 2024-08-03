import cv2
import numpy as np
from pathlib import Path
import glob
import yaml
from yaml.loader import SafeLoader
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import math
from .cam import open_cam
import argparse
from pprint import pprint

def calibration_load(calibration_directory="./calibration_images"):
    file_names = glob.glob(str(Path(calibration_directory) / 'calibration_results_*.yaml'))
    file_names.sort()
    if len(file_names) > 0:
        file_name = file_names[-1]
        with open(file_name) as f:
            camera_calibration = yaml.load(f, Loader=SafeLoader)
    else:
        print('Webcam: No camera calibration files found.')

    assert camera_calibration, 'Webcam: Failed to successfully load camera calibration results.'

    print('Webcam: Loaded camera calibration results from file =', file_name)
    print('Webcam: Loaded camera calibration results =', camera_calibration)
    return np.array(camera_calibration['camera_matrix']), np.array(camera_calibration['distortion_coefficients'])

def transform_from_rvec_tvec(rvec, tvec):
    return pt.transform_from(
        pr.matrix_from_compact_axis_angle(rvec), 
        tvec
    )

def rvec_tvec_from_transform(transform):
    rvec = pr.compact_axis_angle_from_matrix(transform[:3,:3])
    tvec = transform[:3,3]
    return rvec, tvec

def process(
    device="/dev/video0", calibration_directory="./calibration_images", 
    left_handle_cb=None, right_handle_cb=None,
    debug=False
):
    # Open camera
    cam = open_cam(device)

    # Load calibration
    camera_matrix, distortion_coefficients = calibration_load(calibration_directory)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_detection_parameters = cv2.aruco.DetectorParameters()
    # aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    # aruco_detection_parameters.cornerRefinementWinSize = 2
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detection_parameters)

    # set coordinate system
    marker_length_mm = 0.056
    marker_z_to_center = -0.075/2
    marker_rotation_matrix = {
        # right gripper
        239: pr.matrix_from_euler([-math.pi/2, -math.pi/2, 0], 2, 1, 0, False), # original: right_bottom, now: front
        241: pr.matrix_from_euler([-math.pi/2, 0, 0], 2, 1, 0, False), # original: right_front, now: top
        242: pr.matrix_from_euler([-math.pi/2, math.pi/2, 0], 2, 0, 1, False), # original: left_side, now: left
        243: pr.matrix_from_euler([-math.pi/2, -math.pi/2, 0], 2, 0, 1, False), # original: right_side, now: right
        229: pr.matrix_from_euler([math.pi/2, math.pi/2, 0], 2, 1, 0, False), # original: right_top, now: back
        # left gripper
        233: pr.matrix_from_euler([-math.pi/2, -math.pi/2, 0], 2, 1, 0, False), # original: right_bottom, now: front
        235: pr.matrix_from_euler([-math.pi/2, 0, 0], 2, 1, 0, False), # original: right_front, now: top
        236: pr.matrix_from_euler([-math.pi/2, math.pi/2, 0], 2, 0, 1, False), # original: left_side, now: left
        237: pr.matrix_from_euler([-math.pi/2, -math.pi/2, 0], 2, 0, 1, False), # original: right_side, now: right
        231: pr.matrix_from_euler([math.pi/2, math.pi/2, 0], 2, 1, 0, False), # original: right_top, now: back
    }
    left_hand_markers = [ 233, 235, 236, 237, 231 ]
    obj_points = np.array([
        (-marker_length_mm / 2, marker_length_mm / 2, 0), # top left
        (marker_length_mm / 2, marker_length_mm / 2, 0), # top right
        (marker_length_mm / 2, -marker_length_mm / 2, 0), # bottom right
        (-marker_length_mm / 2, -marker_length_mm / 2, 0), # bottom left
    ])

    while True:
        ret, rgb_image = cam.read()

        aruco_corners, aruco_ids, aruco_rejected_image_points = detector.detectMarkers(rgb_image)

        if debug:
            # draw results
            debug_image = rgb_image.copy()
            cv2.aruco.drawDetectedMarkers(debug_image, aruco_corners, aruco_ids)

            # # Rejected
            # cv2.aruco.drawDetectedMarkers(debug_image, aruco_rejected_image_points, None, (100, 0, 255))

            # tag_transforms = []

        max_area = 0
        tag2cam_in_use = None
        left_max_area = 0
        left_tag2cam_in_use = None

        # Calculate pose for each marker
        if aruco_ids is not None:
            for aruco_corner, aruco_id in zip(aruco_corners, aruco_ids):
                aruco_id = aruco_id.item()
                is_left_hand = aruco_id in left_hand_markers

                # rvec shape (3, 1)
                # tvec shape (3, 1)
                unknown_variable, rvec, tvec = cv2.solvePnP(
                    obj_points, # shape: (4, 3) # point coord in 3d space
                    aruco_corner, # shape: (1, 4, 2) # point coord in camera 2d space
                    camera_matrix, distortion_coefficients
                )
                
                # if debug:
                #     cv2.drawFrameAxes(
                #         debug_image,
                #         camera_matrix, distortion_coefficients,
                #         rvec, tvec,
                #         marker_length_mm * 1, 2
                #     )

                # Coordinate setting: 
                # https://stackoverflow.com/questions/53277597/fundamental-understanding-of-tvecs-rvecs-in-opencv-aruco
                # / 1000 : Convert ArUco position estimate to be in meters
                # cv2.Rodrigues(rvec.squeeze())[0] == pr.matrix_from_compact_axis_angle(rvec.squeeze())
                tag2cam = transform_from_rvec_tvec(rvec.squeeze(), tvec.squeeze())
                # tag_transforms.append((aruco_id, tag2cam))

                # move to center
                tag2cam = pt.concat(pt.transform_from_pq([0, 0, marker_z_to_center, 1, 0, 0, 0]), tag2cam)

                # rotate to right pose
                if aruco_id in marker_rotation_matrix:
                    tag2cam = pt.concat(pt.transform_from(marker_rotation_matrix[aruco_id], [0, 0, 0]), tag2cam)

                # use the marker have largest area size
                area = cv2.contourArea(aruco_corner)
                if is_left_hand:
                    if area > left_max_area:
                        left_max_area = area
                        left_tag2cam_in_use = tag2cam
                else:
                    if area > max_area:
                        max_area = area
                        tag2cam_in_use = tag2cam
            
            if tag2cam_in_use is not None:
                if right_handle_cb is not None:
                    right_handle_cb(tag2cam_in_use)

                if debug:
                    rvec2, tvec2 = rvec_tvec_from_transform(tag2cam_in_use)
                    cv2.drawFrameAxes(
                        debug_image,
                        camera_matrix, distortion_coefficients,
                        rvec2, tvec2,
                        marker_length_mm * 1, 2
                    )

            if left_tag2cam_in_use is not None:
                if left_handle_cb is not None:
                    left_handle_cb(left_tag2cam_in_use)

                if debug:
                    rvec2, tvec2 = rvec_tvec_from_transform(left_tag2cam_in_use)
                    cv2.drawFrameAxes(
                        debug_image,
                        camera_matrix, distortion_coefficients,
                        rvec2, tvec2,
                        marker_length_mm * 1, 2
                    )

        if debug:
            # # visualize via matplotlib
            # import matplotlib.pyplot as plt
            # from pytransform3d.plot_utils import make_3d_axis
            # plt.ion()
            # plt.cla()
            # ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
            # pt.plot_transform(ax=ax)
            # for tag2cam in transforms:
            #     pprint(tag2cam)
            #     pt.plot_transform(ax, A2B=tag2cam)
            # # plt.show()
            # plt.draw()
            # plt.pause(0.001)

            cv2.imshow('Debug Image', debug_image)
            if (cv2.waitKey(1) == 27): # Must wait, otherwise imshow will show black screen
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device name.", default="/dev/video0")
    parser.add_argument("-c", "--calibration_directory", help="Calibration directory.", default="./calibration_images")
    args = parser.parse_args()

    process(args.device, args.calibration_directory, pprint, pprint, debug=True)
