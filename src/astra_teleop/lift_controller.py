import serial
import threading
import numpy as np
import struct
import time
import sys

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LiftController:
    COMM_LEN = 2 + 16
    COMM_HEAD = 0x5A
    COMM_TYPE_PING = 0x00
    COMM_TYPE_PONG = 0x01
    COMM_TYPE_CTRL = 0x02
    COMM_TYPE_FEEDBACK = 0x03

    STEPPER_PULSE_PER_REV = 800 # pulse/rev
    RAIL_MM_PER_REV = 10 # mm/rev
    RAIL_MAX_LENGTH_MM = 1190
    STEPPER_MAX_PULSE = (RAIL_MAX_LENGTH_MM / RAIL_MM_PER_REV * STEPPER_PULSE_PER_REV) # 80000

    @staticmethod
    def to_si_unit(x):
        return x / LiftController.STEPPER_PULSE_PER_REV * LiftController.RAIL_MM_PER_REV / 1000

    @staticmethod
    def to_raw_unit(x):
        return int(x * 1000 / LiftController.RAIL_MM_PER_REV * LiftController.STEPPER_PULSE_PER_REV)

    def __init__(self, name):
        logger.info(f"Using device {name}")

        self.state_cb = None
        self.pong_cb = None

        self.ser = serial.Serial(name, 921600, timeout=None)
        # self.ser.rts = False

        self.lock = threading.Lock()

        self.last_position = None
        self.last_velocity = None
        self.last_effort = None
        self.last_time = None

        self.write_lock = threading.Lock()

        self.error_cb = None

        self.quit = threading.Event()

        self.t = threading.Thread(target=self.recv_thread, daemon=True)
        self.t.start()

        while self.last_position is None: # wait for init done
            time.sleep(0.1)

    def recv_thread(self):
        try:
            while not self.quit.is_set():
                data = self.ser.read(1)
                if not (data[0] == self.COMM_HEAD): # 逐步同步
                    sys.stdout.buffer.write(data)
                    sys.stdout.flush()
                    continue

                data += self.ser.read(self.COMM_LEN - 1)
                assert(len(data) == self.COMM_LEN)

                if data[1] == self.COMM_TYPE_PONG:
                    if self.pong_cb is not None:
                        self.pong_cb(struct.unpack('>xxHHHHHHxxxx', data))
                elif data[1] == self.COMM_TYPE_FEEDBACK:
                    position = self.to_si_unit(*struct.unpack('>xxIxxxxxxxxxxxx', data))
                    this_time = time.time()
                    with self.lock:
                        if self.last_time is None:
                            self.last_position = position
                            self.last_velocity = 0
                            self.last_effort = 0
                            self.last_time = this_time - 1 # in case of dividing 0
                        delta_time = this_time - self.last_time
                        velocity = (position - self.last_position) / delta_time
                        effort = (velocity - self.last_velocity) / delta_time # without bias (gravity) and mass
                        self.last_position = position
                        self.last_velocity = velocity
                        self.last_effort = effort
                        self.last_time = this_time
                        if self.state_cb is not None:
                            self.state_cb(self.last_position, self.last_velocity, self.last_effort, self.last_time)
                else:
                    # sys.stdout.buffer.write(data)
                    # sys.stdout.flush()
                    pass
        finally:
            logger.info("thread exiting")

    def get_pos(self): # 一帧可能会被用多次，但是绝对不会卡住
        with self.lock:
            return self.last_position, self.last_velocity, self.last_effort, self.last_time

    def write(self, encoded_data):
        with self.write_lock:
            self.ser.write(encoded_data)

    def set_pos(self, pos):
        if not (0 <= pos <= self.RAIL_MAX_LENGTH_MM / 1000):
            logger.error(f"Joint #{1} reach limit, min: {0}, max: {self.RAIL_MAX_LENGTH_MM / 1000}, current pos: {pos}")
            if self.error_cb is not None:
                self.error_cb(f"Joint #{1} reach limit")
        else:
            self.write(struct.pack('>BBIxxxxxxxxxxxx', self.COMM_HEAD, self.COMM_TYPE_CTRL, self.to_raw_unit(pos)))

    def stop(self):
        if not self.quit.is_set():
            self.quit.set()
            self.ser.close()

    def __del__(self):
        self.stop()