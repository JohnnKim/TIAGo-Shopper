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

### This is where you should write all of your code (arguments are flexible, they were just examples, make sure to update them here and in the grocery_shopper.py file)
def main(robot, keyboard, lidar, gps, compass, display, map_array, timestep, lidar_sensor_readings, lidar_offsets):
    # Enable LiDAR
    lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
    lidar.enable(timestep)
    lidar.enablePointCloud()

    # Enable display
    display = robot.getDevice("display")

    # --- Get Actual Display Dimensions ---
    display_width = display.getWidth()
    display_height = display.getHeight()
    print(f"Display dimensions: {display_width} x {display_height}")

    # Calculate scaling factors for drawing
    scale_x_display = display_width / MAP_WIDTH
    scale_y_display = display_height / MAP_HEIGHT
    print(f"Display scaling factors: sx={scale_x_display:.3f}, sy={scale_y_display:.3f}")

    # Odometry
    pose_x = 0
    pose_y = 0
    pose_theta = 0

    vL = 0
    vR = 0


    # Verify initial map dimensions passed from grocery_shopper.py
    print(f"Initial map_array shape received: {map_array.shape}")
    if map_array.shape[0] != MAP_HEIGHT or map_array.shape[1] != MAP_WIDTH:
         print(f"WARNING: map_array shape {map_array.shape} does not match expected ({MAP_HEIGHT}, {MAP_WIDTH}).")
    map = np.zeros(shape=(MAP_HEIGHT, MAP_WIDTH)) # Use internal map

    gripper_status="closed"
    
    # Draw initial crosshairs
    # display.setColor(int(0x000000))
    # display.fillRectangle(0, 0, display_width, display_height)
    center_x_display = int((MAP_WIDTH/2) * scale_x_display)
    center_y_display = int((MAP_HEIGHT/2) * scale_y_display)
    # display.setColor(int(0x0000FF))  # Blue for center
    # display.drawLine(center_x_display-10, center_y_display, center_x_display+10, center_y_display)
    # display.drawLine(center_x_display, center_y_display-10, center_x_display, center_y_display+10)

    # Main Loop
    while robot.step(timestep) != -1:
        # --- Clear display and draw crosshairs each frame --- 
        display.setColor(int(0x000000))
        display.fillRectangle(0, 0, display_width, display_height)
        display.setColor(int(0x0000FF))  # Blue for center
        display.drawLine(center_x_display-10, center_y_display, center_x_display+10, center_y_display)
        display.drawLine(center_x_display, center_y_display-10, center_x_display, center_y_display+10)
        # -----------------------------------------------------
        
        # Get sensor readings
        lidar_sensor_readings = lidar.getRangeImage()
        lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
        
        # Get current pose from GPS and Compass
        gps_values = gps.getValues()
        compass_values = compass.getValues()
        
        # World coordinates
        pose_x = gps_values[0]  # East-West
        pose_y = gps_values[1]  # North-South (using Z for ground plane Y)

        # Compass vector (points North in world frame)
        north_x = compass_values[0]
        north_z = compass_values[1] # Use Z component for North projection
        pose_theta = math.atan2(-north_x, north_z) # atan2(sin, cos)
        
        # pose_theta = north_angle # Removed the +/- pi/2 offset -- OLD INCORRECT WAY
        pose_theta = (pose_theta + math.pi) % (2 * math.pi) - math.pi # Normalize

        # Debug pose info - uncomment if needed
        # print(f"World Pose: x={pose_x:.2f}, y={pose_y:.2f} | North Angle: {math.degrees(north_angle):.1f}° | Pose Theta: {math.degrees(pose_theta):.1f}°")

        # Convert World Pose to Map Array Coordinates (Map Origin at Top-Left)
        # Map Center corresponds to World (0,0)
        map_x_float = pose_x * MAP_SCALE + MAP_WIDTH / 2
        # Keep Y mapping WITHOUT negation: World +Y (North) maps to Map +Y (Down)
        map_y_float = pose_y * MAP_SCALE + MAP_HEIGHT / 2

        # Clamp to map array boundaries
        map_x = min(max(0, int(map_x_float)), MAP_WIDTH - 1)
        map_y = min(max(0, int(map_y_float)), MAP_HEIGHT - 1)

        # Debug map coords - uncomment if needed
        # print(f"Robot Map Coords: map_x={map_x}, map_y={map_y}")

        # Handle keyboard input for manual control
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
            # Save the map
            filtered_map = np.zeros(map.shape, dtype=np.uint8)
            filtered_map[map > 0.5] = 1
            np.save("grocery_map.npy", filtered_map)
            print("Map saved to grocery_map.npy")
        elif key == ord('L'):
            # Load a saved map
            try:
                map = np.load("grocery_map.npy")
                print("Map loaded from grocery_map.npy")
                # Redraw entire loaded map onto display
                display.setColor(int(0x000000))
                display.fillRectangle(0, 0, display_width, display_height)
                # Redraw crosshairs
                display.setColor(int(0x0000FF))
                display.drawLine(center_x_display-10, center_y_display, center_x_display+10, center_y_display)
                display.drawLine(center_x_display, center_y_display-10, center_x_display, center_y_display+10)
                # Draw map points
                height, width = map.shape
                for y_idx in range(height):
                    for x_idx in range(width):
                        if map[y_idx, x_idx] > 0:
                            display_x = int(x_idx * scale_x_display)
                            display_y = int(y_idx * scale_y_display)
                            if 0 <= display_x < display_width and 0 <= display_y < display_height:
                                display.setColor(int(0xFFFFFF)) # White for loaded obstacles
                                display.drawPixel(display_x, display_y)
            except Exception as e:
                print(f"Error loading or drawing map: {e}")
        elif key == ord('C'):
            # Clear the internal map and the display
            map = np.zeros(shape=(MAP_HEIGHT, MAP_WIDTH))
            display.setColor(int(0x000000))
            display.fillRectangle(0, 0, display_width, display_height)
            # Redraw crosshairs
            display.setColor(int(0x0000FF))
            display.drawLine(center_x_display-10, center_y_display, center_x_display+10, center_y_display)
            display.drawLine(center_x_display, center_y_display-10, center_x_display, center_y_display+10)
            print("Map cleared")
        else: 
            # Smooth deceleration when no key is pressed
            vL *= 0.9
            vR *= 0.9
        
        # Apply acceleration smoothing
        vL = prev_vL + ACCEL_RATE * (vL - prev_vL)
        vR = prev_vR + ACCEL_RATE * (vR - prev_vR)
        
        # Draw robot position
        robot_display_x = int(map_x * scale_x_display)
        robot_display_y = int(map_y * scale_y_display)
        if 0 <= robot_display_x < display_width and 0 <= robot_display_y < display_height:
            display.setColor(int(0x00FF00))  # Green for robot
            display.drawOval(robot_display_x, robot_display_y, 3, 3) # Draw circle at center

            # Draw robot orientation line
            # Since map Y now aligns with display Y (both grow down), use pose_theta directly
            display_angle = pose_theta
            direction_len = 10 # pixels long
            dir_end_x = int(robot_display_x + direction_len * math.cos(display_angle))
            dir_end_y = int(robot_display_y + direction_len * math.sin(display_angle)) # Use '+' for sin component
            if 0 <= dir_end_x < display_width and 0 <= dir_end_y < display_height:
                display.drawLine(robot_display_x, robot_display_y, dir_end_x, dir_end_y)

        # Mark current robot cell in map array for testing
        if 0 <= map_y < MAP_HEIGHT and 0 <= map_x < MAP_WIDTH:
            # Optionally mark robot's path, or leave this out if not desired
            # map[map_y, map_x] = 0.5 # Example: mark path with a different value
            pass # Keep map focused on obstacles

        # Process lidar readings using adapted Lab5 logic
        for i, rho in enumerate(lidar_sensor_readings):
            # Use MIN_LIDAR_DISTANCE from mapping.py
            if rho < MIN_LIDAR_DISTANCE or rho > LIDAR_SENSOR_MAX_RANGE:
                continue

            # Angle of lidar beam relative to robot's forward axis
            alpha = lidar_offsets[i]

            # Skip front-facing sensors (e.g., +/- 45 degrees)
            if abs(alpha) < math.pi / 4:
                 continue

            # Calculate point in robot frame (relative to robot center)
            # Lab5 convention: +x forward, +y left
            # Our convention: +x forward, +y ??? (Let's assume +y left like Lab5 for now)
            # rx = math.cos(alpha) * rho
            # ry = math.sin(alpha) * rho # If alpha=0 is forward, angle increases counter-clockwise (left)

            # Adapt transformation using current pose_theta (already correct world angle)
            # Need world angle of the beam
            world_angle_beam = pose_theta + alpha

            # Endpoint of beam in world coordinates (relative to world origin)
            wx = pose_x + rho * math.cos(world_angle_beam)
            wy = pose_y + rho * math.sin(world_angle_beam) # Use '+' for sin component as in original mapping.py

            # Convert World Point to Map Array Coordinates (Map Origin at Top-Left)
            # Map Center corresponds to World (0,0)
            map_x_lidar_float = wx * MAP_SCALE + MAP_WIDTH / 2
            map_y_lidar_float = wy * MAP_SCALE + MAP_HEIGHT / 2 # Keep Y mapping consistent with robot pose

            # Clamp to map array boundaries
            map_x_lidar = min(max(0, int(map_x_lidar_float)), MAP_WIDTH - 1)
            map_y_lidar = min(max(0, int(map_y_lidar_float)), MAP_HEIGHT - 1)

            # Update occupancy map (increase probability, capped at 1.0)
            # Use a smaller increment for smoother mapping
            if 0 <= map_y_lidar < MAP_HEIGHT and 0 <= map_x_lidar < MAP_WIDTH:
                map[map_y_lidar, map_x_lidar] = min(map[map_y_lidar, map_x_lidar] + 0.05, 1.0) # Increment occupied cell

                # Draw the detected point on the display using grayscale
                g = map[map_y_lidar, map_x_lidar]
                gray_value = int(g * 255)
                # Ensure value is within 0-255 range
                gray_value = max(0, min(gray_value, 255))
                # Convert to RGB integer format for display.setColor()
                gray_color = (gray_value << 16) + (gray_value << 8) + gray_value

                # Convert map coordinates to display coordinates
                display_x = int(map_x_lidar * scale_x_display)
                display_y = int(map_y_lidar * scale_y_display)

                # Draw only if within display bounds
                if 0 <= display_x < display_width and 0 <= display_y < display_height:
                    display.setColor(int(gray_color))
                    display.drawPixel(display_x, display_y)


        # Robot position drawing is already handled above the lidar loop

    pass
