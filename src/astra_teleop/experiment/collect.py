import argparse
import json
from ..process import get_process
import time
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device name.", default="/dev/video0")
    parser.add_argument("-c", "--calibration_directory", help="Calibration directory.", default="./calibration_images")
    parser.add_argument("-e", "--experiment_directory", help="Experiment directory.", default="./experiment_results")
    args = parser.parse_args()
    
    Path(args.experiment_directory).mkdir(parents=True, exist_ok=True)
    
    with open(f"{args.experiment_directory}/data.jsonl", "w") as f:
        try:
            process = get_process(args.device, args.calibration_directory, debug=False)
            while True:
                tag2cam_left, tag2cam_right = process()
                f.write(json.dumps({
                    "timestamp": time.time(), 
                    "tag2cam_left": tag2cam_left.tolist() if tag2cam_left is not None else None,
                    "tag2cam_right": tag2cam_right.tolist() if tag2cam_right is not None else None,
                }) + "\n")
        except KeyboardInterrupt:
            pass
