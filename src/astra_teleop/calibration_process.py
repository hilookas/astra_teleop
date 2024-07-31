
import cv2
import time
import glob
from datetime import datetime
import yaml
from pprint import pprint
from pathlib import Path

# Aruco Board
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
aruco_board = cv2.aruco.CharucoBoard(
    size = (5,7),
    squareLength = 0.04,
    markerLength = 0.02,
    dictionary = aruco_dict
)

# Load images
calibration_directory = "./calibration_images"

file_names = glob.glob(str(Path(calibration_directory) / '*.png'))
print('found ' + str(len(file_names)) + ' calibration images')

all_object_points = []
all_image_points = []

image_size = None
number_of_images = 0
number_of_points = 0

images_used_for_calibration = []

detector_parameters = cv2.aruco.DetectorParameters()
refine_parameters = cv2.aruco.RefineParameters()
# aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_parameters, refine_parameters)
charuco_parameters = cv2.aruco.CharucoParameters()
charuco_detector = cv2.aruco.CharucoDetector(aruco_board, charuco_parameters, detector_parameters, refine_parameters)


for f in file_names:
    color_image = cv2.imread(f)
    number_of_images = number_of_images + 1
    if image_size is None:
        image_size = color_image.shape
        print('image_size =', image_size)
    elif image_size != color_image.shape:
        print('ERROR: previous image_size', image_size, ' is not equal to the current image size', color_image.shape)
        exit()

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(color_image)

    print('filename =', f)

    if (marker_ids is None) or (charuco_ids is None):
        print('marker_ids =', marker_ids)
        print('charuco_ids =', charuco_ids)
    else:
        print('len(charuco_ids) =', len(charuco_ids))

        if len(charuco_ids) > 0:
            object_points, image_points = aruco_board.matchImagePoints(charuco_corners, charuco_ids)

            print('len(object_points) =', len(object_points))
            print('len(image_points) =', len(image_points))

            # A view with fewer than eight points results in
            # cv2.calibrateCamera throwing an error like the following:
            # projection_error, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera( cv2.error: OpenCV(4.8.1) /io/opencv/modules/calib3d/src/calibration.cpp:1213: error: (-215:Assertion failed) fabs(sc) > DBL_EPSILON in function 'cvFindExtrinsicCameraParams2'

            if (len(object_points) >= 8):
                number_of_points = number_of_points + len(object_points)
                all_object_points.append(object_points)
                all_image_points.append(image_points)
                images_used_for_calibration.append(f)

            cv2.aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids)

    cv2.imshow('Detected Charuco Corners', color_image)
    cv2.waitKey(1)

# Perform Calibration
size = (image_size[1], image_size[0])

print()
print('POINTS USED FOR CALIBRATION')
print('number of images with suitable points =', len(images_used_for_calibration))
print('len(all_object_points) =', len(all_object_points))
print('len(all_image_points) =', len(all_image_points))


projection_error, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
    all_object_points,
    all_image_points,
    size,
    None,
    None
)

calibration_results = {
    'calibration_date' : datetime.now(),
    'image_size' : list(image_size),
    'number_of_images_processed' : number_of_images,
    'number_of_images_used' : len(images_used_for_calibration),
    'number_of_corresponding_points_used' : number_of_points,
    'projection_error' : projection_error,
    'camera_matrix' : camera_matrix,
    'distortion_coefficients' : distortion_coefficients
}

print()
print('calibration_results =')
pprint(calibration_results)
print()

# Convert from Numpy arrays to human-readable lists
calibration_results = { (k) : (v.tolist() if 'tolist' in dir(v) else v) for k, v in calibration_results.items()}

results_file_name = Path(calibration_directory) / ('calibration_results_' + time.strftime("%Y%m%d%H%M%S") + '.yaml')
with open(results_file_name, 'w') as file:
    yaml.dump(calibration_results, file, sort_keys=True)
print('saved calibration results to', results_file_name)
print()
