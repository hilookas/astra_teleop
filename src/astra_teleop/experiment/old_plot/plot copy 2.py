import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from pytransform3d.plot_utils import Trajectory, Frame
from pytransform3d.rotations import passive_matrix_from_angle, R_id
from pytransform3d.transformations import transform_from, concat
import json
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from pytorch3d.transforms import se3_exp_map
import torch
import torch.nn as nn

from pytorch3d.transforms import matrix_to_axis_angle

def compute_rotation_angle(transform1, transform2):
    """
    计算两个变换矩阵之间的旋转角度差（弧度）
    
    参数:
        transform1 (torch.Tensor): 第一个变换矩阵，形状为 (..., 4, 4)
        transform2 (torch.Tensor): 第二个变换矩阵，形状为 (..., 4, 4)
    
    返回:
        angle (torch.Tensor): 旋转角度差，形状为 (...)
    """
    # 提取旋转部分 R1 和 R2（假设左上角3x3为旋转矩阵）
    R1 = transform1[..., :3, :3]
    R2 = transform2[..., :3, :3]
    
    # 计算相对旋转矩阵 R_rel = R2 * R1^T
    R_rel = torch.matmul(R2, R1.transpose(-1, -2))
    
    # 将相对旋转矩阵转换为轴角表示
    axis_angle = matrix_to_axis_angle(R_rel)
    
    # 计算角度（轴角的模长）
    angle = torch.norm(axis_angle, dim=-1)
    
    return angle


tag2cams = []
timestamps = []

with open("data.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        
        timestamp = data["timestamp"]
        tag2cam_left = data["tag2cam_left"]
        tag2cam_right = data["tag2cam_right"]
        
        if tag2cam_right is not None:
            tag2cam_right = np.array(tag2cam_right, dtype=np.float32)
            tag2cams.append(tag2cam_right)
            timestamps.append(timestamp)

tag2cams = np.asarray(tag2cams, dtype=np.float32)

timestamps = np.array(timestamps, dtype=np.float32)
timestamps -= timestamps[0]

# see: https://github.com/NVlabs/FoundationPose/blob/bf2518348eb2ef8f1fdf786b79ed698db02e8703/bundlesdf/nerf_helpers.py#L44

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wv_plate = torch.zeros([1, 6], dtype=torch.float32, requires_grad=True, device=device)

x = torch.zeros([], dtype=torch.float32, requires_grad=True, device=device)
y = torch.zeros([], dtype=torch.float32, requires_grad=True, device=device)
theta = torch.zeros([], dtype=torch.float32, requires_grad=True, device=device)

omega = torch.zeros([], dtype=torch.float32, requires_grad=True, device=device)

optimizer = torch.optim.Adam([wv_plate, x, y, theta, omega], lr=1)

for i in range(100):
    print(1)
    for timestamp, tag2cam in zip(timestamps, tag2cams):
        tag2cam = torch.tensor(tag2cam, dtype=torch.float32, device=device)

        T_plate = se3_exp_map(wv_plate)

        T_tag_rot = torch.eye(4, dtype=torch.float32, device=device)
        T_tag_rot[0][0] = torch.cos(theta)
        T_tag_rot[0][1] = -torch.sin(theta)
        T_tag_rot[1][0] = torch.sin(theta)
        T_tag_rot[1][1] = torch.cos(theta)
        T_tag_rot[0][3] = x
        T_tag_rot[1][3] = y

        T_rot_plate = torch.eye(4, dtype=torch.float32, device=device)
        T_rot_plate[0][0] = torch.cos(omega * timestamp)
        T_rot_plate[0][1] = -torch.sin(omega * timestamp)
        T_rot_plate[1][0] = torch.sin(omega * timestamp)
        T_rot_plate[1][1] = torch.cos(omega * timestamp)

        T = T_tag_rot @ T_rot_plate @ T_plate[0]
        
        rot_loss = compute_rotation_angle(T, tag2cam)
        trans_loss = (T[0:3, 3] - tag2cam[0:3, 3]).norm(dim=-1)
        
        loss = rot_loss + trans_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    print(i, loss.item())
    
    








fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# # scale the plot
ax.set_xlim(-0.2, 0)
ax.set_ylim(0, 0.2)
ax.set_zlim(0.7, 0.9)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


H = np.array(tag2cams, dtype=np.float32)
trajectory = Trajectory(H, show_direction=False, s=0.2, c="k")
trajectory.add_trajectory(ax)

frame = Frame(np.eye(4), label="camera frame", s=0.5)
frame.add_frame(ax)

plt.show()

pr.matrix_from_quaternion