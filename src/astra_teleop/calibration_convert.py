import cv2
from pathlib import Path

def calibration_convert(video_path='video.mp4', calibration_directory="./calibration_images", skip=30):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    Path(calibration_directory).mkdir(parents=True, exist_ok=True)

    num_images = 0

    cnt = 0

    while True:
        ret, color_image = cap.read()

        if cnt % skip == 0:
            num_images = num_images + 1
            file_name = Path(calibration_directory) / (str(num_images).zfill(4) + '.png')
            print('save', file_name)
            cv2.imwrite(file_name, color_image)

        cnt += 1

# Example usage
if __name__ == "__main__":
    calibration_convert()