import cv2
import numpy as np
from pathlib import Path
import glob
import yaml
from yaml.loader import SafeLoader
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from src.astra_teleop.cam import open_cam
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


# Copy from: https://github.com/NVlabs/FoundationPose/blob/main/Utils.py
def draw_xyz_axis(color, ob_in_cam, K=np.eye(3), scale=0.1, thickness=3, transparency=0, is_input_rgb=True, save_path=None, color_preset="rgb"):
    """
    Draw XYZ coordinate axes on an image.

    Args:
        color: Input image (RGB or BGR)
        ob_in_cam: Object pose in camera frame (4x4 transformation matrix)
        scale: Scale factor for axis length
        K: Camera intrinsic matrix (3x3)
        thickness: Line thickness for drawing
        transparency: Transparency factor (0-1)
        is_input_rgb: Whether input is RGB (True) or BGR (False)
        save_path: Optional path to save the result image

    Returns:
        Image with XYZ axes drawn
    """
    def project_3d_to_2d(pt, K, ob_in_cam):
        """Project 3D point to 2D image coordinates."""
        pt = pt.reshape(4, 1)
        projected = K @ ((ob_in_cam@pt)[:3,:])
        projected = projected.reshape(-1)
        projected = projected / projected[2]
        return projected.reshape(-1)[:2].round().astype(int)

    # Convert RGB to BGR if needed (OpenCV uses BGR)
    if is_input_rgb:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    xx = np.array([1,0,0,1]).astype(float)
    yy = np.array([0,1,0,1]).astype(float)
    zz = np.array([0,0,1,1]).astype(float)
    xx[:3] = xx[:3]*scale
    yy[:3] = yy[:3]*scale
    zz[:3] = zz[:3]*scale
    origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = color.copy()
    if transparency == 0:
        if color_preset == "rgb":
            tmp = cv2.arrowedLine(tmp, origin, xx, color=(0,0,255), thickness=thickness, line_type=line_type, tipLength=arrow_len)
            tmp = cv2.arrowedLine(tmp, origin, yy, color=(0,255,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
            tmp = cv2.arrowedLine(tmp, origin, zz, color=(255,0,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
        elif color_preset == "cmy":
            tmp = cv2.arrowedLine(tmp, origin, xx, color=(255,255,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
            tmp = cv2.arrowedLine(tmp, origin, yy, color=(255,0,255), thickness=thickness, line_type=line_type, tipLength=arrow_len)
            tmp = cv2.arrowedLine(tmp, origin, zz, color=(0,255,255), thickness=thickness, line_type=line_type, tipLength=arrow_len)
        else:
            assert False, "Unknown color preset"
    else:
        tmp1 = tmp.copy()
        if color_preset == "rgb":
            tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness, line_type=line_type, tipLength=arrow_len)
            tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
            tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
        elif color_preset == "cmy":
            tmp = cv2.arrowedLine(tmp, origin, xx, color=(255,255,0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
            tmp = cv2.arrowedLine(tmp, origin, yy, color=(255,0,255), thickness=thickness, line_type=line_type, tipLength=arrow_len)
            tmp = cv2.arrowedLine(tmp, origin, zz, color=(0,255,255), thickness=thickness, line_type=line_type, tipLength=arrow_len)
        else:
            assert False, "Unknown color preset"
        mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
        tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
        tmp = tmp.astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, tmp)

    # Convert back to RGB if input was RGB
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    return tmp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device name.", default="/dev/video0")
    parser.add_argument("-c", "--calibration_directory", help="Calibration directory.", default="./calibration_images")
    args = parser.parse_args()

    device = args.device
    calibration_directory = args.calibration_directory

    # Open camera
    cam = open_cam(device)

    # Load calibration
    camera_matrix, distortion_coefficients = calibration_load(calibration_directory)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_detection_parameters = cv2.aruco.DetectorParameters()
    # aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX # ~30ms
    # aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG # Most accurate but also slowest with ~200-300ms
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detection_parameters)

    # 3D object points for a single 30mm ArUco marker (z=0 plane)
    # Corner order: top-left, top-right, bottom-right, bottom-left
    marker_half = 30.0 / 2 / 1000  # 15mm -> metres
    single_marker_obj_points = np.array([
        [-marker_half, -marker_half, 0],
        [ marker_half, -marker_half, 0],
        [ marker_half,  marker_half, 0],
        [-marker_half,  marker_half, 0],
    ], dtype=np.float32)

    tag2cam_left = None
    tag2cam_right = None

    while True:
        ret, bgr_image = cam.read()

        debug_image = bgr_image.copy()

        aruco_corners, aruco_ids, _ = detector.detectMarkers(bgr_image)
        cv2.aruco.drawDetectedMarkers(debug_image, aruco_corners, aruco_ids)

        tag2cam_left = None
        tag2cam_right = None

        if aruco_ids is not None:
            for aruco_id, aruco_corner in zip(aruco_ids, aruco_corners):
                aruco_id = aruco_id.item()
                img_points = aruco_corner.squeeze().astype(np.float32)  # shape: (4, 2)

                # Solve pose for this single marker
                _, rvec, tvec = cv2.solvePnP(
                    single_marker_obj_points,
                    img_points,
                    camera_matrix, distortion_coefficients
                )

                tag2cam = transform_from_rvec_tvec(rvec.squeeze(), tvec.squeeze())

                # Draw coordinate axes on the debug image (BGR input)
                debug_image = draw_xyz_axis(
                    debug_image, tag2cam,
                    K=camera_matrix, scale=0.05, thickness=3, transparency=0,
                    is_input_rgb=False
                )

                # Keep the last detected transform per side for the 3D viewer
                if aruco_id >= 18:
                    tag2cam_left = tag2cam
                else:
                    tag2cam_right = tag2cam

        cv2.imshow('Debug Image', debug_image)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

        viewer.update(tag2cam_left, tag2cam_right)

    cv2.destroyAllWindows()
    viewer.close()
