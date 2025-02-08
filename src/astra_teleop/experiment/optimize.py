import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import Trajectory, Frame
import json
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from pytorch3d.transforms import se3_exp_map
import torch

from pytorch3d.transforms import matrix_to_axis_angle
import math

import argparse
import json

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_directory", help="Experiment directory.", default="./experiment_results")
    args = parser.parse_args()

    tag2cams = []
    timestamps = []

    with open(f"{args.experiment_directory}/data.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            
            timestamp = data["timestamp"]
            tag2cam = data["tag2cam_left"]
            
            if tag2cam is not None:
                tag2cam = np.array(tag2cam, dtype=np.float32)
                tag2cams.append(tag2cam)
                timestamps.append(timestamp)

    tag2cams = np.asarray(tag2cams, dtype=np.float32)

    timestamps = np.array(timestamps, dtype=np.float64)
    timestamps -= timestamps[0]

    # see: https://github.com/NVlabs/FoundationPose/blob/bf2518348eb2ef8f1fdf786b79ed698db02e8703/bundlesdf/nerf_helpers.py#L44

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vw_plate = torch.zeros([1, 6], dtype=torch.float32, requires_grad=True, device=device)

    x = torch.zeros([], dtype=torch.float32, requires_grad=True, device=device)
    y = torch.zeros([], dtype=torch.float32, requires_grad=True, device=device)
    theta = torch.zeros([], dtype=torch.float32, requires_grad=True, device=device)

    omega = torch.full([], dtype=torch.float32, requires_grad=True, device=device, fill_value=-2 * math.pi / 7) # 这个参数是需要调的，如果发现rot_loss有周期性大波动的话，考虑符号问题

    phi = torch.zeros([], dtype=torch.float32, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([vw_plate, x, y, theta, omega, phi], lr=0.01)

    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32, device=device)
    tag2cams_tensor = torch.tensor(tag2cams, dtype=torch.float32, device=device)

    for i in range(2000):
        T_plate = se3_exp_map(vw_plate).transpose(-1, -2)

        T_rot_plate = torch.eye(4, dtype=torch.float32, device=device)
        T_rot_plate = T_rot_plate.repeat(timestamps_tensor.shape[0], 1, 1)
        T_rot_plate[:,0,0] = torch.cos(omega * timestamps_tensor + phi)
        T_rot_plate[:,0,1] = -torch.sin(omega * timestamps_tensor + phi)
        T_rot_plate[:,1,0] = torch.sin(omega * timestamps_tensor + phi)
        T_rot_plate[:,1,1] = torch.cos(omega * timestamps_tensor + phi)

        T_tag_rot = torch.eye(4, dtype=torch.float32, device=device)
        T_tag_rot.unsqueeze_(0) # [1, 4, 4]
        T_tag_rot[:,0,0] = torch.cos(theta)
        T_tag_rot[:,0,1] = -torch.sin(theta)
        T_tag_rot[:,1,0] = torch.sin(theta)
        T_tag_rot[:,1,1] = torch.cos(theta)
        T_tag_rot[:,0,3] = x
        T_tag_rot[:,1,3] = y

        # T = T_tag_rot @ T_rot_plate @ T_plate
        T = T_plate @ T_rot_plate @ T_tag_rot
        
        rot_loss = compute_rotation_angle(T, tag2cams_tensor)
        trans_loss = (T[...,0:3,3] - tag2cams_tensor[...,0:3,3]).norm(dim=-1)
        # if i % 1000 == 0:
        #     import IPython; IPython.embed()
        rot_loss_mean = rot_loss.mean()
        trans_loss_mean = trans_loss.mean()
        
        loss = (rot_loss_mean + trans_loss_mean)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
        print(i, loss.item(), rot_loss_mean.item(), trans_loss_mean.item())
        print(T_plate.cpu())
        print(x.item(), y.item(), theta.item(), omega.item(), phi.item())

    torch.save([vw_plate, x, y, theta, omega, phi, T, tag2cams_tensor, timestamps_tensor, rot_loss, trans_loss], f"{args.experiment_directory}/result.pt")

    # import IPython; IPython.embed()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # # scale the plot
    ax.set_xlim(-0.2, 0)
    ax.set_ylim(0, 0.2)
    ax.set_zlim(0.7, 0.9)
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    H = T.detach().cpu().numpy()[100:101]
    trajectory = Trajectory(H, show_direction=False, s=0.2, c="k")
    trajectory.add_trajectory(ax)

    H = np.array(tag2cams, dtype=np.float32)[100:101]
    trajectory = Trajectory(H, show_direction=False, s=0.2, c="r")
    trajectory.add_trajectory(ax)

    frame = Frame(np.eye(4), label="camera frame", s=0.5)
    frame.add_frame(ax)

    plt.show()

    # print(se3_exp_map(torch.tensor([[4, 5, 6, 1, 2, 3]], dtype=torch.float32)).transpose(-1, -2))
    # print(pt.transform_from_exponential_coordinates(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)))