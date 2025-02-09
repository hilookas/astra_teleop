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
    vw_plate, x, y, theta, omega, phi, T, tag2cams, timestamps, rot_error, trans_error = torch.load(f"/home/ubuntu/astra_teleop_robicu/src/experiment_results_robocu_subpix_1080p_01/result.pt")

    timestamps_robocu = timestamps.detach().cpu().numpy()
    rot_error_robocu = rot_error.detach().cpu().numpy()
    trans_error_robocu = trans_error.detach().cpu().numpy()
    
    vw_plate, x, y, theta, omega, phi, T, tag2cams, timestamps, rot_error, trans_error = torch.load(f"/home/ubuntu/astra_teleop_vanilla/src/experiment_results_vanilla_1080p_subpix/result.pt")

    timestamps_vanilla = timestamps.detach().cpu().numpy()
    rot_error_vanilla = rot_error.detach().cpu().numpy()
    trans_error_vanilla = trans_error.detach().cpu().numpy()


    # show two figures
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(timestamps_robocu, rot_error_robocu / math.pi * 180)
    ax1.plot(timestamps_vanilla, rot_error_vanilla / math.pi * 180)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Rotation Error (deg)")
    ax1.set_title("Rotation Error")
    ax1.legend(["RoboCu", "Vanilla"], loc="upper right")
    fig1.savefig(f"rotation_error.pdf")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(timestamps_robocu, trans_error_robocu * 1000)
    ax2.plot(timestamps_vanilla, trans_error_vanilla * 1000)
    ax2.legend(["RoboCu", "Vanilla"], loc="upper right")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Translation Error (mm)")
    ax2.set_title("Translation Error")
    fig2.savefig(f"translation_error.pdf")
    
    print(f"RoboCu: rot {rot_error_robocu.mean() / math.pi * 180}deg, trans {trans_error_robocu.mean() * 1000}mm")
    print(f"Vanilla: rot {rot_error_vanilla.mean() / math.pi * 180}deg, trans {trans_error_vanilla.mean() * 1000}mm")

# import torch; vw_plate, x, y, theta, omega, phi, T, tag2cams, timestamps, rot_error, trans_error = torch.load(f"result.pt"); 
# print(rot_error.mean())
# print(trans_error.mean())