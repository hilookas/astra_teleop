import cv2
import numpy as np
from PIL import Image

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

a4_res = 11.811 # px/mm # 300 ppi
a4_res_ppi = 300 # px/inch # 300 ppi
a4_size_mm = (297, 210) # mm 
a4_size = (int(a4_size_mm[0] * a4_res), int(210 * a4_res)) 

grid_size_mm = 5 # mm
grid_size = int(5 * 8 * a4_res) # px

def get_marker_image(id):
    marker_image = cv2.aruco.generateImageMarker(dictionary, id, 4 + 2, borderBits=1)
    marker_image = np.pad(marker_image, ((1, 1), (1, 1)), constant_values=255) # pad white border
    assert(marker_image.shape == (8, 8))
    # cv2.imwrite("marker2.png", marker_image)

    marker_image = cv2.resize(marker_image, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
    
    marker_image = np.pad(marker_image, ((1, 1), (1, 1)), constant_values=0) # black cut line
    
    return marker_image

buf2 = []
for i in range(7):
    buf = []
    for j in range(5):
        idx = i * 5 + j
        buf.append(get_marker_image(idx))
    buf2.append(np.concat(buf, axis=1))
marker_array_image = np.concat(buf2, axis=0)
# 1 2 3 4 5
# 6 7 8 9 10
# ...

margin = np.array(a4_size) - marker_array_image.shape

marker_array_image = np.pad(marker_array_image, (
    (int(margin[0] / 2), margin[0] - int(margin[0] / 2)), 
    (int(margin[1] / 2), margin[1] - int(margin[1] / 2))), constant_values=255)

Image.fromarray(marker_array_image).save("marker_array_front.pdf", "PDF", resolution=a4_res_ppi)




def get_marker_image_back(id):
    marker_image = np.ones((grid_size, grid_size), np.uint8) * 255
    
    marker_image = cv2.circle(marker_image, (grid_size - 50, 50), 5, (127, 127, 127), 4)
    
    marker_image = cv2.putText(marker_image, str(id), (int(grid_size / 2), int(grid_size / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (127, 127, 127), 4)
    
    marker_image = np.pad(marker_image, ((1, 1), (1, 1)), constant_values=0) # black cut line
    
    return marker_image

buf2 = []
for i in range(7):
    buf = []
    for j in reversed(range(5)):
        idx = i * 5 + j
        buf.append(get_marker_image_back(idx))
    buf2.append(np.concat(buf, axis=1))
marker_array_image = np.concat(buf2, axis=0)
# 1 2 3 4 5
# 6 7 8 9 10
# ...

margin = np.array(a4_size) - marker_array_image.shape

marker_array_image = np.pad(marker_array_image, (
    (int(margin[0] / 2), margin[0] - int(margin[0] / 2)), 
    (int(margin[1] / 2), margin[1] - int(margin[1] / 2))), constant_values=255)

Image.fromarray(marker_array_image).save("marker_array_back.pdf", "PDF", resolution=a4_res_ppi)
