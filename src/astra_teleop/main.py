import cv2
import numpy as np
from pathlib import Path
import glob
import yaml
from yaml.loader import SafeLoader

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


# Load calibration
calibration_directory = "./calibration_images"

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
camera_matrix = np.array(camera_calibration['camera_matrix'])
distortion_coefficients = np.array(camera_calibration['distortion_coefficients'])



aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_detection_parameters = cv2.aruco.DetectorParameters()
# aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
# aruco_detection_parameters.cornerRefinementWinSize = 2
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detection_parameters)

# set coordinate system
marker_length_mm = 56
obj_points = np.array([
    (-marker_length_mm / 2, marker_length_mm / 2, 0),
    (marker_length_mm / 2, marker_length_mm / 2, 0),
    (marker_length_mm / 2, -marker_length_mm / 2, 0),
    (-marker_length_mm / 2, -marker_length_mm / 2, 0),
])

while True:
    ret, rgb_image = cam.read()
    debug_image = rgb_image.copy()

    aruco_corners, aruco_ids, aruco_rejected_image_points = detector.detectMarkers(rgb_image)

    rvecs = []
    tvecs = []

    # Calculate pose for each marker
    for aruco_corner in aruco_corners:
        unknown_variable, rvec, tvec = cv2.solvePnP(
            obj_points,
            aruco_corner,
            camera_matrix, distortion_coefficients
        )

        rvecs.append(rvec)
        tvecs.append(tvec)

    # draw results
    cv2.aruco.drawDetectedMarkers(debug_image, aruco_corners, aruco_ids)

    if aruco_ids is not None:
        for aruco_id, (rvec, tvec) in zip(aruco_ids, zip(rvecs, tvecs)):
            cv2.drawFrameAxes(
                debug_image,
                camera_matrix, distortion_coefficients,
                rvec, tvec,
                marker_length_mm * 1.5, 2
            )

    # # Rejected
    # cv2.aruco.drawDetectedMarkers(debug_image, aruco_rejected_image_points, None, (100, 0, 255))

    cv2.imshow('Debug Image', debug_image)
    if (cv2.waitKey(1) == 27): # Must wait, otherwise imshow will show black screen
        break
