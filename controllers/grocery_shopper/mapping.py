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

### This is where you should write all of your code (arguments are flexible, they were just examples, make sure to update them here and in the grocery_shopper.py file)
def main(robot, keyboard, lidar, gps, compass, display, map_array, timestep ):
    # Enable LiDAR
    lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
    lidar.enable(timestep)
    lidar.enablePointCloud()

    # Enable display
    display = robot.getDevice("display")

    # Odometry
    pose_x = 0
    pose_y = 0
    pose_theta = 0

    vL = 0
    vR = 0

    lidar_sensor_readings = [] # List to hold sensor readings
    lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
    lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

    map = None
    map = np.zeros((map_array.shape[0], map_array.shape[1]))

    gripper_status="closed"

    # Main Loop
    while robot.step(timestep) != -1:
        # Get sensor readings
        lidar_sensor_readings = lidar.getRangeImage()
        lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
        
        # Convert to map coordinates
        map_x = min(max(0, int(360 - abs(pose_x * 30))), 359)
        map_y = min(max(0, int(abs(pose_y * 30))), 359)
        
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
                
                # Display the loaded map
                display.setColor(int(0x000000))
                display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
                
                height, width = map.shape
                for y in range(height):
                    for x in range(width):
                        if map[y, x] > 0:
                            display.setColor(int(0xFFFFFF))
                            display.drawPixel(x, y)
            except:
                print("Error loading map file")
        elif key == ord('C'):
            # Clear the map
            map = np.zeros(shape=[360, 360])
            display.setColor(int(0x000000))
            display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
            print("Map cleared")
        else: 
            # Smooth deceleration when no key is pressed
            vL *= 0.9
            vR *= 0.9
        
        # Apply acceleration smoothing
        vL = prev_vL + ACCEL_RATE * (vL - prev_vL)
        vR = prev_vR + ACCEL_RATE * (vR - prev_vR)
        
        # Update map based on lidar readings
        # Mark current position as free
        map[map_y, map_x] = 0
        
        # Process lidar readings to update map
        for i, distance in enumerate(lidar_sensor_readings):
            if distance < LIDAR_SENSOR_MAX_RANGE:
                # Calculate angle of this lidar beam
                angle = pose_theta + lidar_offsets[i]
                
                # Calculate endpoint of beam in world coordinates
                endpoint_x = pose_x + distance * math.cos(angle)
                endpoint_y = pose_y + distance * math.sin(angle)
                
                # Convert to map coordinates
                endpoint_map_x = min(max(0, int(360 - abs(endpoint_x * 30))), 359)
                endpoint_map_y = min(max(0, int(abs(endpoint_y * 30))), 359)
                
                # Mark as occupied
                map[endpoint_map_y, endpoint_map_x] = 1
                
                # Visualize on display
                display.setColor(int(0xFF0000))  # Red for obstacles
                display.drawPixel(endpoint_map_x, endpoint_map_y)
    pass
