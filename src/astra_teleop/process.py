import cv2
import numpy as np
from pathlib import Path
import glob
import yaml
from yaml.loader import SafeLoader
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from .cam import open_cam
import argparse
from pprint import pprint
import time

def calibration_load(calibration_directory="./calibration_images"):
    file_names = glob.glob(str(Path(calibration_directory) / 'calibration_results_*.yaml'))
    file_names.sort()
    assert len(file_names) > 0, 'Webcam: No camera calibration files found.'

    file_name = file_names[-1]
    with open(file_name) as f:
        camera_calibration = yaml.load(f, Loader=SafeLoader)
    assert camera_calibration, 'Webcam: Failed to successfully load camera calibration results.'

    print('Webcam: Loaded camera calibration results from file =', file_name)
    print('Webcam: Loaded camera calibration results =', camera_calibration)
    return np.array(camera_calibration['camera_matrix']), np.array(camera_calibration['distortion_coefficients'])

def transform_from_rvec_tvec(rvec, tvec):
    # cv2.Rodrigues(rvec.squeeze())[0] == pr.matrix_from_compact_axis_angle(rvec.squeeze())
    return pt.transform_from(
        pr.matrix_from_compact_axis_angle(rvec),
        tvec
    )

def rvec_tvec_from_transform(transform):
    rvec = pr.compact_axis_angle_from_matrix(transform[:3,:3])
    tvec = transform[:3,3]
    return rvec, tvec

def get_detect():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_detection_parameters = cv2.aruco.DetectorParameters()
    # aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX # ~30ms
    # aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG # Most accurate but also slowest with ~200-300ms
    # aruco_detection_parameters.aprilTagQuadDecimate = 2
    # aruco_detection_parameters.cornerRefinementWinSize = 2
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detection_parameters)

    def detect(
        rgb_image,
        debug,
        debug_image
    ):
        aruco_corners, aruco_ids, aruco_rejected_image_points = detector.detectMarkers(rgb_image)

        if debug:
            cv2.aruco.drawDetectedMarkers(debug_image, aruco_corners, aruco_ids)

            # # Rejected
            # cv2.aruco.drawDetectedMarkers(debug_image, aruco_rejected_image_points, None, (100, 0, 255))

            # tag_transforms = []

        return aruco_corners, aruco_ids
    return detect

def get_solve(scale=1):
    # set coordinate system
    # Coordinate setting:
    # https://stackoverflow.com/questions/53277597/fundamental-understanding-of-tvecs-rvecs-in-opencv-aruco
    obj_points_map = { # right part
        # top_left, top_right, bottom_right, bottom_left
        0:  [ (-15.00, -15.00,  48.28), (-15.00,  15.00,  48.28), ( 15.00,  15.00,  48.28), ( 15.00, -15.00,  48.28) ],

        1:  [ ( 23.54, -15.00,  44.75), ( 23.54,  15.00,  44.75), ( 44.75,  15.00,  23.54), ( 44.75, -15.00,  23.54) ],
        2:  [ (-15.00, -23.54,  44.75), ( 15.00, -23.54,  44.75), ( 15.00, -44.75,  23.54), (-15.00, -44.75,  23.54) ],
        3:  [ (-23.54,  15.00,  44.75), (-23.54, -15.00,  44.75), (-44.75, -15.00,  23.54), (-44.75,  15.00,  23.54) ],
        4:  [ ( 15.00,  23.54,  44.75), (-15.00,  23.54,  44.75), (-15.00,  44.75,  23.54), ( 15.00,  44.75,  23.54) ],

        5:  [ ( 48.28, -15.00,  15.00), ( 48.28,  15.00,  15.00), ( 48.28,  15.00, -15.00), ( 48.28, -15.00, -15.00) ],
        6:  [ ( 23.54, -44.75,  15.00), ( 44.75, -23.54,  15.00), ( 44.75, -23.54, -15.00), ( 23.54, -44.75, -15.00) ],
        7:  [ (-15.00, -48.28,  15.00), ( 15.00, -48.28,  15.00), ( 15.00, -48.28, -15.00), (-15.00, -48.28, -15.00) ],
        8:  [ (-44.75, -23.54,  15.00), (-23.54, -44.75,  15.00), (-23.54, -44.75, -15.00), (-44.75, -23.54, -15.00) ],
        9:  [ (-48.28,  15.00,  15.00), (-48.28, -15.00,  15.00), (-48.28, -15.00, -15.00), (-48.28,  15.00, -15.00) ],
        10: [ (-23.54,  44.75,  15.00), (-44.75,  23.54,  15.00), (-44.75,  23.54, -15.00), (-23.54,  44.75, -15.00) ],
        11: [ ( 15.00,  48.28,  15.00), (-15.00,  48.28,  15.00), (-15.00,  48.28, -15.00), ( 15.00,  48.28, -15.00) ],
        12: [ ( 44.75,  23.54,  15.00), ( 23.54,  44.75,  15.00), ( 23.54,  44.75, -15.00), ( 44.75,  23.54, -15.00) ],

        13: [ ( 44.75, -15.00, -23.54), ( 44.75,  15.00, -23.54), ( 23.54,  15.00, -44.75), ( 23.54, -15.00, -44.75) ],
        14: [ (-15.00, -44.75, -23.54), ( 15.00, -44.75, -23.54), ( 15.00, -23.54, -44.75), (-15.00, -23.54, -44.75) ],
        15: [ (-44.75,  15.00, -23.54), (-44.75, -15.00, -23.54), (-23.54, -15.00, -44.75), (-23.54,  15.00, -44.75) ],
        16: [ ( 15.00,  44.75, -23.54), (-15.00,  44.75, -23.54), (-15.00,  23.54, -44.75), ( 15.00,  23.54, -44.75) ],
    }
    for i in range(18 - 1): # left part
        obj_points_map[i + 18] = obj_points_map[i]

    def solve(
        camera_matrix, distortion_coefficients,
        aruco_corners, aruco_ids,
        debug=False,
        debug_image=None,
        min_aruco_thres=4
    ):
        camera_matrix = np.array(camera_matrix, dtype=np.float32)
        distortion_coefficients = np.array(distortion_coefficients, dtype=np.float32)

        if aruco_ids is None:
            aruco_ids = np.array([])
            aruco_corners = np.array([])
        aruco_ids = np.array(aruco_ids)
        aruco_corners = np.array(aruco_corners, dtype=np.float32)

        tags = {
            "left": [],
            "right": [],
        }

        for aruco_id, aruco_corner in zip(aruco_ids, aruco_corners):
            aruco_id = aruco_id.item()
            aruco_corner = aruco_corner.squeeze() # shape: (4, 2)
            is_left_hand = aruco_id >= 18 # 18~34 is left hand

            if is_left_hand:
                tags["left"].append((aruco_id, aruco_corner))
            else:
                tags["right"].append((aruco_id, aruco_corner))

        tag2cam = {
            "left": None,
            "right": None,
        }
        for side in ["left", "right"]:
            tags[side].sort(key=lambda tag: cv2.contourArea(tag[1]), reverse=True)
            if len(tags[side]) < min_aruco_thres:
                # print(f"detected less than {min_aruco_thres} aruco tags")
                continue
            tags[side] = tags[side][:min_aruco_thres] # pick biggest four tag
            # print(tags[side][0][0])

            obj_points = []
            img_points = []

            for aruco_id, aruco_corner in tags[side]:
                if aruco_id in obj_points_map:
                    obj_points.extend(obj_points_map[aruco_id])
                    img_points.extend(aruco_corner)

            # rvec shape (3, 1)
            # tvec shape (3, 1)
            unknown_variable, rvec, tvec = cv2.solvePnP(
                np.array(obj_points) / 1000 * scale, # shape: (4 * n, 3) # point coord in 3d space
                np.array(img_points), # shape: (4 * n, 2) # point coord in camera 2d space
                camera_matrix, distortion_coefficients
            )

            tag2cam[side] = transform_from_rvec_tvec(rvec.squeeze(), tvec.squeeze())

        if tag2cam["left"] is not None:
            if debug:
                rvec, tvec = rvec_tvec_from_transform(tag2cam["left"])
                cv2.drawFrameAxes(
                    debug_image,
                    camera_matrix, distortion_coefficients,
                    rvec, tvec,
                    50/2, 2
                )

        if tag2cam["right"] is not None:
            if debug:
                print(1111)
                rvec, tvec = rvec_tvec_from_transform(tag2cam["right"])
                cv2.drawFrameAxes(
                    debug_image,
                    camera_matrix, distortion_coefficients,
                    rvec, tvec,
                    50/2, 2
                )

        return tag2cam["left"], tag2cam["right"]
    return solve

def get_process(
    device="/dev/video0", calibration_directory="./calibration_images",
    debug=False,
):
    # Open camera
    cam = open_cam(device)

    # Load calibration
    camera_matrix, distortion_coefficients = calibration_load(calibration_directory)

    detect = get_detect()
    solve = get_solve()

    def process():
        ret, rgb_image = cam.read()

        if debug:
            # draw results
            debug_image = rgb_image.copy()
        else:
            debug_image = None

        t0 = time.perf_counter()
        aruco_corners, aruco_ids = detect(
            rgb_image,
            debug,
            debug_image,
        ) # 10ms@1080p
        t1 = time.perf_counter()

        tag2cam_left, tag2cam_right = solve(
            camera_matrix, distortion_coefficients,
            aruco_corners, aruco_ids,
            debug,
            debug_image,
        ) # 1ms@1080p
        t2 = time.perf_counter()

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

            # debug_image = cv2.flip(debug_image, 1)
            cv2.imshow('Debug Image', debug_image)
            # debug_image2 = cv2.undistort(debug_image, camera_matrix, distortion_coefficients)
            # cv2.imshow('Debug Image 2', debug_image2)
            if (cv2.waitKey(1) == 27): # Must wait, otherwise imshow will show black screen
                raise Exception("Stop")

        print(f"detect time: {t1 - t0}")
        print(f"solve time: {t2 - t1}")
        print(f"total time: {t2 - t0}")

        return tag2cam_left, tag2cam_right

    return process

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device name.", default="/dev/video0")
    parser.add_argument("-c", "--calibration_directory", help="Calibration directory.", default="./calibration_images")
    args = parser.parse_args()

    import open3d as o3d

    # lookAt function implementation
    # https://github.com/hilookas/Helper3D/blob/master/trimesh_render/src/camera.py
    def lookAt(eye, target, up, yz_flip=False):
        # Normalize the up vector
        up /= np.linalg.norm(up)
        forward = eye - target
        forward /= np.linalg.norm(forward)
        if np.dot(forward, up) == 1 or np.dot(forward, up) == -1:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        new_up = np.cross(forward, right)
        new_up /= np.linalg.norm(new_up)

        # Construct a rotation matrix from the right, new_up, and forward vectors
        rotation = np.eye(4)
        rotation[:3, :3] = np.row_stack((right, new_up, forward))

        # Apply a translation to the camera position
        translation = np.eye(4)
        translation[:3, 3] = [
            np.dot(right, eye),
            np.dot(new_up, eye),
            -np.dot(forward, eye),
        ]

        if yz_flip:
            # This is for different camera setting, like Open3D
            rotation[1, :] *= -1
            rotation[2, :] *= -1
            translation[1, 3] *= -1
            translation[2, 3] *= -1

        camera_pose = np.linalg.inv(np.matmul(translation, rotation))

        return camera_pose

    class PointCloudViewer:
        def __init__(self):
            self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.eef = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.eef_T_inv = np.eye(4)
            self.eef_right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.eef_right_T_inv = np.eye(4)

            # Initialize the pointcloud viewer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Point Cloud")

            self.vis.add_geometry(self.origin)
            self.vis.add_geometry(self.eef)
            self.vis.add_geometry(self.eef_right)

            self.vis.get_render_option().point_size = 1
            # self.vis.get_render_option().background_color = np.asarray([0, 0, 0])

            view_control = self.vis.get_view_control()
            view_control.set_constant_z_far(1000)

            # Retrieve the camera parameters
            camera_params = view_control.convert_to_pinhole_camera_parameters()
            # Set the extrinsic parameters, yz_flip is for Open3D camera configuration
            camera_pose = lookAt(eye=np.array([0., 0., -1.]), target=np.array([0. ,0., 0.]), up=np.array([0.0, -1.0, 0.0]), yz_flip=True)
            camera_params.extrinsic = np.linalg.inv(camera_pose)
            # Set the camera parameters
            view_control.convert_from_pinhole_camera_parameters(camera_params)

        def update(self, Teef2cam, Teef2cam_right):
            if Teef2cam is not None:
                self.eef.transform(self.eef_T_inv)
                self.eef.transform(Teef2cam)
                self.eef_T_inv = pt.invert_transform(Teef2cam)
                self.vis.update_geometry(self.eef)

            if Teef2cam_right is not None:
                self.eef_right.transform(self.eef_right_T_inv)
                self.eef_right.transform(Teef2cam_right)
                self.eef_right_T_inv = pt.invert_transform(Teef2cam_right)
                self.vis.update_geometry(self.eef_right)

            # Update the visualizer
            self.vis.poll_events()
            self.vis.update_renderer()

        def close(self):
            self.vis.destroy_window()

    viewer = PointCloudViewer()

    process = get_process(args.device, args.calibration_directory, debug=True)
    while True:
        tag2cam_left, tag2cam_right = process()
        pprint(tag2cam_left)
        pprint(tag2cam_right)
        viewer.update(tag2cam_left, tag2cam_right)
