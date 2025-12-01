# left_hand_control.py

THROTTLE_MIN = 0
THROTTLE_MAX = 100

YAW_MIN = -100
YAW_MAX = 100


def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def control_throttle(gesture_value):
    gesture_value = max(0.0, min(100.0, gesture_value))  # Since script returns throttle in % already
    throttle = map_value(gesture_value, 0.0, 100.0, THROTTLE_MIN, THROTTLE_MAX)
    return int(throttle)


def control_yaw(gesture_value):
    gesture_value = max(-100.0, min(100.0, gesture_value))  # Script returns yaw from -100..100
    yaw = map_value(gesture_value, -100.0, 100.0, YAW_MIN, YAW_MAX)
    return int(yaw)
