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

wv_plate = torch.zeros([1, 6], dtype=torch.float32, requires_grad=True)

T_plate = se3_exp_map(wv_plate)

x = torch.zeros([], dtype=torch.float32, requires_grad=True)
y = torch.zeros([], dtype=torch.float32, requires_grad=True)
theta = torch.zeros([], dtype=torch.float32, requires_grad=True)

T_tag_rot = torch.tensor([
    [torch.cos(theta), -torch.sin(theta), 0, x],
    [torch.sin(theta), torch.cos(theta), 0, y],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

omega = torch.zeros([], dtype=torch.float32, requires_grad=True)

def sparse_dot(A, B):
    # T = [
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 0, 0, 1],
    # ]
    
    # for i in range(4):
    #     for j in range(4):
    #         for k in range(4):
    #             T[i, j] += A[i, k] * B[k, j]
    
    return [
        [A[0][0]*B[0][0]+A[0][1]*B[1][0]+A[0][2]*B[2][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]+A[0][2]*B[2][1], A[0][0]*B[0][2]+A[0][1]*B[1][2]+A[0][2]*B[2][2], A[0][0]*B[0][3]+A[0][1]*B[1][3]+A[0][2]*B[2][3]+A[0][3]],
        [A[1][0]*B[0][0]+A[1][1]*B[1][0]+A[1][2]*B[2][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]+A[1][2]*B[2][1], A[1][0]*B[0][2]+A[1][1]*B[1][2]+A[1][2]*B[2][2], A[1][0]*B[0][3]+A[1][1]*B[1][3]+A[1][2]*B[2][3]+A[1][3]],
        [A[2][0]*B[0][0]+A[2][1]*B[1][0]+A[2][2]*B[2][0], A[2][0]*B[0][1]+A[2][1]*B[1][1]+A[2][2]*B[2][1], A[2][0]*B[0][2]+A[2][1]*B[1][2]+A[2][2]*B[2][2], A[2][0]*B[0][3]+A[2][1]*B[1][3]+A[2][2]*B[2][3]+A[2][3]],
        [0, 0, 0, 1],
    ]

optimizer = torch.optim.Adam([wv_plate, x, y, theta, omega], lr=1)

for i in range(100):
    print(1)
    for timestamp, tag2cam in zip(timestamps, tag2cams):
        T_rot_plate = [
            [torch.cos(omega * timestamp), -torch.sin(omega * timestamp), 0, 0],
            [torch.sin(omega * timestamp), torch.cos(omega * timestamp), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

        T = sparse_dot(sparse_dot(T_tag_rot, T_rot_plate), T_plate[0])
        
        T[0][1].backward()
        
        import IPython; IPython.embed()
        
        # loss = (T[0][1] - tag2cam[0][1]) ** 2 + (T[0][2] - tag2cam[0][2]) ** 2 + (T[0][3] - tag2cam[0][3]) ** 2
        # loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    # print(i, loss.item())
    
    








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