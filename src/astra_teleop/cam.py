import cv2
import cv2
import argparse

# Open camera
def open_cam(device, image_height=1080, image_width=1920, frames_per_second=30):
    cam = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fourcc_value = cv2.VideoWriter_fourcc(*'MJPG')

    cam.set(cv2.CAP_PROP_FOURCC, fourcc_value)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cam.set(cv2.CAP_PROP_FPS, frames_per_second)
    
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)   
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cam.get(cv2.CAP_PROP_FPS)
    
    print(f"camera: {width}x{height}@{fps}")
    
    assert width == image_width
    assert height == image_height
    assert fps == frames_per_second
    
    return cam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Device name.", default="/dev/video8")
    args = parser.parse_args()

    # Open camera
    cam = open_cam(args.device)

    while True:
        ret, rgb_image = cam.read()
        
        rgb_image = cv2.circle(rgb_image, (1920 // 2, 1080 // 2), 100, (0, 0, 255), 2)
        
        cv2.imshow("rgb_image", rgb_image)
        if cv2.waitKey(1) == 27:
            raise Exception("Stop")
            