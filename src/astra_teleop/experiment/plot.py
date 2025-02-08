import json
import matplotlib.pyplot as plt
import numpy as np
import torch
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
import math

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_directory", help="Experiment directory.", default="./experiment_results")
    args = parser.parse_args()

    vw_plate, x, y, theta, omega, phi, T, tag2cams, timestamps, rot_loss, trans_loss = torch.load(f"{args.experiment_directory}/result.pt")

    tag2cams = tag2cams.detach().cpu().numpy()
    T = T.detach().cpu().numpy()
    timestamps = timestamps.detach().cpu().numpy()
    rot_loss = rot_loss.detach().cpu().numpy()
    trans_loss = trans_loss.detach().cpu().numpy()

    # show two figures
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(timestamps, rot_loss / math.pi * 180)
    ax1.set_xlabel("timestamps")
    ax1.set_ylabel("rotation loss (deg)")
    ax1.set_title("rotation loss")
    fig1.savefig(f"{args.experiment_directory}/rotation_loss.pdf")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(timestamps, trans_loss * 1000)    
    ax2.set_xlabel("timestamps")
    ax2.set_ylabel("translation loss (mm)")
    ax2.set_title("translation loss")
    fig2.savefig(f"{args.experiment_directory}/translation_loss.pdf")

    # import IPython; IPython.embed()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(tag2cams[:, 0, 3], tag2cams[:, 1, 3], tag2cams[:, 2, 3])

    ax.scatter(T[:, 0, 3], T[:, 1, 3], T[:, 2, 3])

    H = T
    trajectory = Trajectory(H, show_direction=False, s=0.2, c="k")
    trajectory.add_trajectory(ax)

    H = tag2cams
    trajectory = Trajectory(H, show_direction=False, s=0.2, c="r")
    trajectory.add_trajectory(ax)

    frame = Frame(np.eye(4), label="camera frame", s=0.5)
    frame.add_frame(ax)

    plt.show()
