import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_exp_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    timestamps = np.array([item[0] for item in raw], dtype=float)
    tag2cam_left = np.array([item[1] for item in raw], dtype=float)
    real_pos = np.array([item[2] for item in raw], dtype=float)
    cmd_pos = np.array([item[3] for item in raw], dtype=float)

    return timestamps, tag2cam_left, real_pos, cmd_pos


def find_latest_exp_data(pattern: str = "exp_data_*.json") -> str:
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    return max(candidates, key=os.path.getmtime)


def compute_translation_delta_norm(tag2cam_left: np.ndarray) -> np.ndarray:
    if tag2cam_left.ndim != 3 or tag2cam_left.shape[1:] != (4, 4):
        raise ValueError(
            f"tag2cam_left shape must be (N, 4, 4), got {tag2cam_left.shape}"
        )

    translation = tag2cam_left[:, :3, 3]
    base_translation = translation[0]
    delta = translation - base_translation
    return np.linalg.norm(delta, axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Plot real/cmd lift position and tag translation delta norm"
    )
    parser.add_argument(
        "-f",
        "--file",
        default=None,
        help="Path to exp_data JSON. If omitted, uses latest exp_data_*.json",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Path to save figure (e.g. plot.png). If omitted, only show figure",
    )
    args = parser.parse_args()

    file_path = args.file or find_latest_exp_data()
    timestamps, tag2cam_left, real_pos, cmd_pos = load_exp_data(file_path)

    if len(timestamps) == 0:
        raise ValueError(f"No frame data found in {file_path}")

    t = timestamps - timestamps[0]
    tag_delta_norm = compute_translation_delta_norm(tag2cam_left)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(t, real_pos, label="real_pos (gt)", linewidth=1.6)
    ax1.plot(t, cmd_pos, label="cmd_pos", linewidth=1.2, alpha=0.9)
    ax1.set_ylabel("Position (m)")
    ax1.set_title("Lift Position")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(t, tag_delta_norm, label="||tag_t - tag_t0||", linewidth=1.6)
    ax2.plot(t, real_pos, label="real_pos (gt)", linewidth=1.2, alpha=0.9)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Displacement (m)")
    ax2.set_title("Tag Translation Delta Norm vs GT")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(os.path.basename(file_path))
    fig.tight_layout()

    fig.savefig("resolution.pdf")
    print(f"Saved figure to: resolution.pdf")

    plt.show()


if __name__ == "__main__":
    main()
