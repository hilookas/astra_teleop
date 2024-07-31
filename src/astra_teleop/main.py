import cv2

cam = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

image_width = 1920
image_height = 1080

image_size = (image_height, image_width)

frames_per_second = 30

fourcc_value = cv2.VideoWriter_fourcc(*'MJPG')

cam.set(cv2.CAP_PROP_FOURCC, fourcc_value)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[0])
cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[1])            
cam.set(cv2.CAP_PROP_FPS, frames_per_second)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_detection_parameters = cv2.aruco.DetectorParameters()
# aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_detection_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
# aruco_detection_parameters.cornerRefinementWinSize = 2
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_detection_parameters)

while True:
    ret, frame = cam.read() 

    rgb_image = frame.copy()
    
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    aruco_corners, aruco_ids, aruco_rejected_image_points = detector.detectMarkers(gray_image)

    cv2.aruco.drawDetectedMarkers(rgb_image, aruco_corners, aruco_ids)
    
    cv2.imshow('Image', rgb_image)
    cv2.waitKey(1) # Must wait, otherwise imshow will show black screen