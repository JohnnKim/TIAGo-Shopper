"""
This file contains the mapping code for the grocery shopper.
All code written in this file was written by Andrew Rho.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)
MIN_LIDAR_DISTANCE = 0.2 # Meters - Minimum distance to consider a reading valid (Increased)

MAP_WIDTH = 900
MAP_HEIGHT = 483
MAP_SCALE = 30 # pixels per meter



"""
Entry point for the mapping code.
Takes in a lot of things from main controller.
"""
def main(robot, keyboard, lidar, gps, compass, display, map_array, timestep, lidar_sensor_readings, lidar_offsets, vL, vR):
    # --- Get Actual Display Dimensions ---
    display_width = display.getWidth()
    display_height = display.getHeight()
    # print(f"Display dimensions: {display_width} x {display_height}") # Keep commented unless needed

    # Main Loop
    while robot.step(timestep) != -1:
        # --- Get sensor readings ---
        lidar_sensor_readings = lidar.getRangeImage()
        lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

        # --- Get current pose ---
        gps_values = gps.getValues()
        compass_values = compass.getValues() # Get compass readings ONCE

        # World coordinates from GPS
        pose_x = gps_values[0]
        pose_y = gps_values[1]

        # --- Calculate Robot Orientation (pose_theta) ---
        # Step 1: Calculate intermediate angle based on compass readings
        north_x = compass_values[0]
        north_z = compass_values[1]
        # Angle derived directly from compass (relative to some frame, proved inconsistent with world East)
        compass_based_angle_raw = math.atan2(-north_x, north_z) - math.pi / 2
        # Normalize intermediate angle to [-pi, pi]
        compass_based_angle = (compass_based_angle_raw + math.pi) % (2 * math.pi) - math.pi

        # Step 2: Convert intermediate angle to true world angle (relative to World +X East)
        # Based on observed relationship: theta_calc = -WB_angle - 90 => WB_angle = -theta_calc - 90
        pose_theta_raw = -compass_based_angle - math.pi / 2
        # Normalize final pose_theta to [-pi, pi]
        pose_theta = (pose_theta_raw + math.pi) % (2 * math.pi) - math.pi
        # ------------------------------------------------

        # --- Print the FINAL pose_theta for comparison with Webots ---
        print(f"DEBUG_CALC: Final pose_theta = {math.degrees(pose_theta):.1f} deg")
        # -------------------------------------------------------------

        # --- Convert World Pose to Map Coordinates ---
        # (This part remains unchanged)
        map_x_float = pose_x * MAP_SCALE + MAP_WIDTH / 2
        map_y_float = -pose_y * MAP_SCALE + MAP_HEIGHT / 2 # Flipped Y-axis
        map_x = min(max(0, int(map_x_float)), MAP_WIDTH - 1)
        map_y = min(max(0, int(map_y_float)), MAP_HEIGHT - 1)

        # --- Handle Keyboard Input ---
        # (Keep keyboard handling logic)
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
        else:
            vL *= 0.9
            vR *= 0.9
        
        vL = prev_vL + ACCEL_RATE * (vL - prev_vL)
        vR = prev_vR + ACCEL_RATE * (vR - prev_vR)

        # --- Draw Robot on Display ---
        # Uses the FINAL, corrected pose_theta
        if 0 <= map_x < display_width and 0 <= map_y < display_height:
            display.setColor(int(0x00FF00))
            display_angle = pose_theta # Use final angle
            direction_len = 10
            dir_end_x = int(map_x + direction_len * math.cos(display_angle))
            dir_end_y = int(map_y - direction_len * math.sin(display_angle)) # Flipped Y
            if 0 <= dir_end_x < display_width and 0 <= dir_end_y < display_height:
                display.drawLine(map_x, map_y, dir_end_x, dir_end_y)
            else:
                 display.drawPixel(map_x, map_y)

        # --- Process Lidar Data ---
        first_lidar_point_processed = False
        for i, rho in enumerate(lidar_sensor_readings):
            if rho < MIN_LIDAR_DISTANCE or rho > LIDAR_SENSOR_MAX_RANGE:
                continue

            alpha = lidar_offsets[i]
            # World angle uses the FINAL, corrected pose_theta
            world_angle_beam = pose_theta - alpha # Use final angle, SUBTRACT alpha

            # (wx, wy calculation remains unchanged)
            wx = pose_x + rho * math.cos(world_angle_beam)
            wy = pose_y + rho * math.sin(world_angle_beam)

            # (map coord calculation remains unchanged)
            map_x_lidar_float = wx * MAP_SCALE + MAP_WIDTH / 2
            map_y_lidar_float = -wy * MAP_SCALE + MAP_HEIGHT / 2 # Flipped Y-axis
            map_x_lidar = min(max(0, int(map_x_lidar_float)), MAP_WIDTH - 1)
            map_y_lidar = min(max(0, int(map_y_lidar_float)), MAP_HEIGHT - 1)

            # --- Update Occupancy Map ---
            # (Keep occupancy map update and drawing logic)
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
        # robot.setVelocities(vL, vR) # Make sure you call this function!
        pass # End of loop pass



