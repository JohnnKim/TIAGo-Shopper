"""
This file contains the mapping code for the grocery shopper.
All code written in this file was written by Andrew Rho.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)
MIN_LIDAR_DISTANCE = 0.2 # Meters

MAP_WIDTH = 900
MAP_HEIGHT = 483
MAP_SCALE = 30  # pixels per meter

def main(robot, robot_parts, keyboard, lidar, gps, compass, display, map_array, timestep, lidar_sensor_readings, lidar_offsets, vL, vR):
    # --- Get Actual Display Dimensions ---
    display_width = display.getWidth()
    display_height = display.getHeight()
    autonomous_mode = False
    a_key_last_state = False
    prev_side_right_min = 999

    escape_direction = 0
    escaping_steps = 0

    # Main Loop
    while robot.step(timestep) != -1:
        # --- Get sensor readings ---
        lidar_sensor_readings = lidar.getRangeImage()
        lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

        # --- Get current pose ---
        gps_values = gps.getValues()
        compass_values = compass.getValues()

        pose_x = gps_values[0]
        pose_y = gps_values[1]

        north_x = compass_values[0]
        north_z = compass_values[1]
        compass_based_angle_raw = math.atan2(-north_x, north_z) - math.pi / 2
        compass_based_angle = (compass_based_angle_raw + math.pi) % (2 * math.pi) - math.pi
        pose_theta_raw = -compass_based_angle - math.pi / 2
        pose_theta = (pose_theta_raw + math.pi) % (2 * math.pi) - math.pi

        # --- Convert World Pose to Map Coordinates ---
        map_x_float = pose_x * MAP_SCALE + MAP_WIDTH / 2
        map_y_float = -pose_y * MAP_SCALE + MAP_HEIGHT / 2
        map_x = min(max(0, int(map_x_float)), MAP_WIDTH - 1)
        map_y = min(max(0, int(map_y_float)), MAP_HEIGHT - 1)

        # --- Handle Keyboard Input ---
        key = keyboard.getKey()
        TURN_SPEED = MAX_SPEED * 0.5
        ACCEL_RATE = 0.1
        prev_vL = vL
        prev_vR = vR

        if key == keyboard.LEFT:
            vL = -TURN_SPEED
            vR = TURN_SPEED
        elif key == keyboard.RIGHT:
            vL = TURN_SPEED
            vR = -TURN_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            filtered_map = np.zeros(map_array.shape, dtype=np.uint8)
            filtered_map[map_array > 0.5] = 1
            np.save("map.npy", filtered_map)
            print("Map file saved")
        elif key == ord('L'):
            map_array = np.load("map.npy")
            height, width = map_array.shape
            for y in np.arange(height):
                for x in np.arange(width):
                    if map_array[y, x] > 0:
                        display.setColor(int(0xFFFFFF))
                    else:
                        display.setColor(int(0x000000))
                    display.drawPixel(x, y)
            print("Map loaded")
        elif key == ord('A'):
            if not a_key_last_state:
                autonomous_mode = not autonomous_mode
                if autonomous_mode:
                    print("Autonomous mode: ON")
                else:
                    print("Autonomous mode: OFF")
            a_key_last_state = True
        else:
            a_key_last_state = False
            vL *= 0.9
            vR *= 0.9

        # --- Autonomous Behavior ---
        if autonomous_mode:
            FRONT_ANGLE_WIDTH = 100
            SIDE_ANGLE_WIDTH = 120
            FRONT_CENTER_WIDTH = 120
            center_idx = len(lidar_sensor_readings) // 2

            front_left = lidar_sensor_readings[max(0, center_idx - FRONT_ANGLE_WIDTH):center_idx - 10]
            front_center = lidar_sensor_readings[center_idx - FRONT_CENTER_WIDTH//2:center_idx + FRONT_CENTER_WIDTH//2]
            front_right = lidar_sensor_readings[center_idx + 10:min(len(lidar_sensor_readings), center_idx + FRONT_ANGLE_WIDTH)]
            side_left = lidar_sensor_readings[max(0, center_idx - SIDE_ANGLE_WIDTH):center_idx - FRONT_ANGLE_WIDTH]
            side_right = lidar_sensor_readings[center_idx + FRONT_ANGLE_WIDTH:min(len(lidar_sensor_readings), center_idx + SIDE_ANGLE_WIDTH)]

            BUFFER = 0.5

            front_left_min = min(max(rho - BUFFER, 0) for rho in front_left) if front_left else 999
            front_center_min = min(max(rho - BUFFER, 0) for rho in front_center)
            front_center_avg = np.mean([max(rho - BUFFER, 0) for rho in front_center])
            front_right_min = min(max(rho - BUFFER, 0) for rho in front_right) if front_right else 999
            side_left_min = min(max(rho - BUFFER, 0) for rho in side_left) if side_left else 999
            side_right_min = min(max(rho - BUFFER, 0) for rho in side_right) if side_right else 999

            SAFE_THRESHOLD = 0.8

            if escaping_steps > 0:
                escaping_steps -= 1
                if escaping_steps > 15:
                    print("ESCAPING: BACKING UP")
                    vL = -MAX_SPEED * 0.5
                    vR = -MAX_SPEED * 0.5
                else:
                    if escape_direction == -1:
                        print("ESCAPING: TURN LEFT")
                        vL = -MAX_SPEED * 0.5
                        vR = MAX_SPEED * 0.5
                    else:
                        print("ESCAPING: TURN RIGHT")
                        vL = MAX_SPEED * 0.5
                        vR = -MAX_SPEED * 0.5

            else:
                if (front_center_min < 0.5 or front_center_avg < 1.0 or front_left_min < 0.4 or front_right_min < 0.4) and (side_left_min < 0.4 or side_right_min < 0.4):
                    print("FULL BLOCKED - INITIATE ESCAPE")
                    escaping_steps = 30  # About 1.5 seconds if timestep is 32 ms
                    escape_direction = np.random.choice([-1, 1])
                elif front_center_min < 0.5 or front_center_avg < 1.0 or front_left_min < 0.4 or front_right_min < 0.4:
                    print("FRONT BLOCKED - SPIN ESCAPE")
                    vL = -MAX_SPEED
                    vR = MAX_SPEED
                elif side_left_min < 0.4:
                    print("SIDE LEFT CLOSE - HARD RIGHT")
                    vL = MAX_SPEED * 0.8
                    vR = MAX_SPEED * 0.2
                elif side_right_min < 0.4:
                    if side_right_min < prev_side_right_min - 0.05:
                        print("SIDE RIGHT GETTING CLOSER - HARD LEFT")
                        vL = MAX_SPEED * 0.2
                        vR = MAX_SPEED * 0.8
                    else:
                        print("SIDE RIGHT STABLE - DRIVE FORWARD")
                        vL = MAX_SPEED * 0.5
                        vR = MAX_SPEED * 0.5
                elif front_center_min < 0.8 or front_left_min < 0.8 or front_right_min < 0.8:
                    if front_left_min < SAFE_THRESHOLD and front_right_min >= SAFE_THRESHOLD:
                        print("LEFT WALL DANGER - CURVE RIGHT")
                        vL = MAX_SPEED * 0.6
                        vR = MAX_SPEED * 0.3
                    elif front_right_min < SAFE_THRESHOLD and front_left_min >= SAFE_THRESHOLD:
                        print("RIGHT WALL DANGER - CURVE LEFT")
                        vL = MAX_SPEED * 0.3
                        vR = MAX_SPEED * 0.6
                    else:
                        print("FRONT APPROACHING - CURVE BASED ON OPENNESS")
                        left_openness = sum(front_left)
                        right_openness = sum(front_right)
                        if left_openness > right_openness:
                            vL = MAX_SPEED * 0.6
                            vR = MAX_SPEED * 0.3
                        else:
                            vL = MAX_SPEED * 0.3
                            vR = MAX_SPEED * 0.6
                else:
                    left_openness = sum(front_left)
                    right_openness = sum(front_right)
                    center_openness = sum(front_center)

                    if left_openness > right_openness and left_openness > center_openness:
                        bias = 0.1
                        print("EXPLORING LEFT (OPEN SPACE)")
                    elif right_openness > left_openness and right_openness > center_openness:
                        bias = -0.1
                        print("EXPLORING RIGHT (OPEN SPACE)")
                    else:
                        bias = 0
                        print("EXPLORING CENTER (OPEN SPACE)")

                    if random.random() < 0.02:
                        wiggle = random.choice([-0.2, 0.2])
                        bias += wiggle
                        print("WIGGLING: ", "LEFT" if wiggle < 0 else "RIGHT")

                    bias = max(min(bias, 0.3), -0.3)  # clamp bias between -0.3 and +0.3

                    vL = MAX_SPEED * (0.5 + bias)
                    vR = MAX_SPEED * (0.5 - bias)
                    prev_side_right_min = side_right_min

        # --- Smooth velocity transition ---
        vL = prev_vL + ACCEL_RATE * (vL - prev_vL)
        vR = prev_vR + ACCEL_RATE * (vR - prev_vR)

        # --- Draw Robot on Display ---
        if 0 <= map_x < display_width and 0 <= map_y < display_height:
            display.setColor(int(0x00FF00))
            display_angle = pose_theta
            direction_len = 10
            dir_end_x = int(map_x + direction_len * math.cos(display_angle))
            dir_end_y = int(map_y - direction_len * math.sin(display_angle))
            if 0 <= dir_end_x < display_width and 0 <= dir_end_y < display_height:
                display.drawLine(map_x, map_y, dir_end_x, dir_end_y)
            else:
                display.drawPixel(map_x, map_y)

        # --- Process Lidar Data ---
        for i, rho in enumerate(lidar_sensor_readings):
            if rho < MIN_LIDAR_DISTANCE or rho > LIDAR_SENSOR_MAX_RANGE:
                continue

            alpha = lidar_offsets[i]
            world_angle_beam = pose_theta - alpha
            wx = pose_x + rho * math.cos(world_angle_beam)
            wy = pose_y + rho * math.sin(world_angle_beam)

            map_x_lidar_float = wx * MAP_SCALE + MAP_WIDTH / 2
            map_y_lidar_float = -wy * MAP_SCALE + MAP_HEIGHT / 2
            map_x_lidar = min(max(0, int(map_x_lidar_float)), MAP_WIDTH - 1)
            map_y_lidar = min(max(0, int(map_y_lidar_float)), MAP_HEIGHT - 1)

            if 0 <= map_y_lidar < MAP_HEIGHT and 0 <= map_x_lidar < MAP_WIDTH:
                map_array[map_y_lidar, map_x_lidar] = min(map_array[map_y_lidar, map_x_lidar] + 0.05, 1.0)
                g = map_array[map_y_lidar, map_x_lidar]
                gray_value = max(0, min(int(g * 255), 255))
                gray_color = (gray_value << 16) + (gray_value << 8) + gray_value
                display_x = map_x_lidar
                display_y = map_y_lidar
                if 0 <= display_x < display_width and 0 <= display_y < display_height:
                    display.setColor(int(gray_color))
                    display.drawPixel(display_x, display_y)

        # --- Set Wheel Velocities ---
        robot_parts["wheel_left_joint"].setVelocity(vL)
        robot_parts["wheel_right_joint"].setVelocity(vR)
