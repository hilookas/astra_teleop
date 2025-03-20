import cv2
import cv2.aruco as aruco

# See: https://github.com/hello-robot/stretch_dex_teleop/blob/main/webcam_calibration_create_board.py

# Aruco Board
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
aruco_board = cv2.aruco.CharucoBoard(
    size = (5,7),
    squareLength = 0.04,
    markerLength = 0.02,
    dictionary = aruco_dict
)

########
# From
# https://papersizes.online/paper-size/letter/
#
# "Letter size in pixels when using 600 DPI: 6600 x 5100 pixels."
########

########
# From
# https://docs.opencv.org/4.8.0/d4/db2/classcv_1_1aruco_1_1Board.html
#
# Parameters
# outSize	size of the output image in pixels.
# img	        output image with the aruco_board. The size of this image will be outSize and the aruco_board will be on the center, keeping the aruco_board proportions.
# marginSize	minimum margins (in pixels) of the aruco_board in the output image
# borderBits	width of the marker borders.
########

image_size = (5100, 6600)
margin_size = int(image_size[1]/20)
border_bits = 1

aruco_board_image = aruco_board.generateImage(
    outSize=image_size,
    marginSize=margin_size,
    borderBits=border_bits)

cv2.imwrite('calibration_board.png', aruco_board_image)