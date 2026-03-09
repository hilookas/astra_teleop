import json
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# # scale the plot
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# ax.set_zlim(-10, 10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

with open("data.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        
        timestamp = data["timestamp"]
        tag2cam_left = data["tag2cam_left"]
        tag2cam_right = data["tag2cam_right"]
                
        if tag2cam_right is not None:
            tag2cam_right = np.array(tag2cam_right, dtype=np.float32)
            # plot the coordinate system using tag2cam_right as a transformation matrix using matplotlib
            # print(tag2cam_right[0, 0], tag2cam_right[0, 1], tag2cam_right[0, 2])
            ax.scatter(tag2cam_right[0, 0], tag2cam_right[0, 1], tag2cam_right[0, 2])

plt.show()
# wait for 10 seconds
# plt.pause(10)
