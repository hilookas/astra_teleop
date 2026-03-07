import json

from lift_controller import LiftController
import time

from astra_teleop.process import open_cam, calibration_load, get_detect, get_solve, cv2

debug = True

def main():
    # Open camera
    cam = open_cam("/dev/video14")

    lift_controller_right = LiftController("/dev/ttyUSB1")

    # Load calibration
    camera_matrix, distortion_coefficients = calibration_load("calibration_images")

    detect = get_detect()
    solve = get_solve()

    for i in range(100):
        ret, rgb_image = cam.read()
        lift_controller_right.set_pos(0.000) # wait for stable
        print("wait for stable", i)

    ts_start = time.time()

    exp_data = []

    cnt = 0

    last_time = time.time()
    while True:
        ts = time.time()
        ret, rgb_image = cam.read()
        real_pos = lift_controller_right.get_pos()[0]

        # 步进波形

        # 0-1s 0.0
        # 1-2s 0.001
        # 2-3s 0.002

        cmd_pos = int((ts - ts_start)/2) * 0.001
        lift_controller_right.set_pos(cmd_pos)

        if (ts - ts_start) > 60: break # run for 20s

        if debug:
            # draw results
            debug_image = rgb_image.copy()
        else:
            debug_image = None

        t0 = time.perf_counter()
        aruco_corners, aruco_ids = detect(
            rgb_image,
            debug,
            debug_image,
        ) # 10ms@1080p
        t1 = time.perf_counter()

        tag2cam_left, tag2cam_right = solve(
            camera_matrix, distortion_coefficients,
            aruco_corners, aruco_ids,
            debug,
            debug_image,
        ) # 1ms@1080p
        t2 = time.perf_counter()

        if debug:
            cv2.imshow('Debug Image', debug_image)
            if (cv2.waitKey(1) == 27): # Must wait, otherwise imshow will show black screen
                raise Exception("Stop")

        print(f"detect time: {t1 - t0}")
        print(f"solve time: {t2 - t1}")
        print(f"total time: {t2 - t0}")

        print(ts, tag2cam_left, real_pos, cmd_pos)

        cv2.imwrite(f"debug/debug_{cnt:04d}.jpg", debug_image)
        cnt += 1

        exp_data.append((ts, tag2cam_left.tolist(), real_pos, cmd_pos))

        this_time = time.time()
        print("fps: ", 1 / (this_time - last_time))
        last_time = this_time

    breakpoint()
    json.dump(exp_data, open(f"exp_data_{int(time.time())}.json", "w"))

if __name__ == '__main__':
    main()
