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

import numpy as np
from PIL import Image

a4_res = 11.811 # px/mm # 300 ppi
a4_res_ppi = 300 # px/inch # 300 ppi
a4_size_mm = (297, 210) # mm
a4_size = (int(a4_size_mm[0] * a4_res), int(a4_size_mm[1] * a4_res))

# margin_size = 0
margin_size1 = (210 - 0.04 * 5 * 1000) * a4_res / 2 # px
margin_size2 = (297 - 0.04 * 7 * 1000) * a4_res / 2 # px
margin_size = int(min(margin_size1, margin_size2))
border_bits = 1

aruco_board_image = aruco_board.generateImage(
    outSize=(a4_size[1], a4_size[0]),
    marginSize=margin_size,
    borderBits=border_bits)

Image.fromarray(aruco_board_image).save("calibration_board.pdf", "PDF", resolution=a4_res_ppi)
