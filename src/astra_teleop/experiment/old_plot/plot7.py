import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from pytransform3d.plot_utils import Trajectory, Frame
from pytransform3d.rotations import passive_matrix_from_angle, R_id
from pytransform3d.transformations import transform_from, concat
import json

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# # scale the plot
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

tag2cams = []

with open("data.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        
        timestamp = data["timestamp"]
        tag2cam_left = data["tag2cam_left"]
        tag2cam_right = data["tag2cam_right"]
        
        if tag2cam_right is not None:
            tag2cam_right = np.array(tag2cam_right, dtype=np.float32)
            tag2cams.append(tag2cam_right)

tag2cam = tag2cams[0]

R = pr.matrix_from_axis_angle(np.array([0, 0, 1, np.pi / 2], dtype=np.float32))

T = transform_from(R, np.zeros(3))

frame = Frame(np.eye(4), label="camera frame", s=0.5)
frame.add_frame(ax)

frame = Frame(tag2cam, label="t0", s=0.5)
frame.add_frame(ax)

frame = Frame(concat(T, tag2cam), label="t1", s=0.5)
frame.add_frame(ax)

frame = Frame(tag2cam @ T, label="t2", s=0.5)
frame.add_frame(ax)

plt.show()
