"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from localization import main as odometry
from mapping import main as mapping
from vision import main as vision
from manipulation import main as manipulation
from planning import main as planning
from planning import visualize_path
from planning import world_to_map_coords
from planning import rrt_star
from planning import inflate_map
from follow import follow_waypoints
from localization import main as odometry

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

MAP_WIDTH = 900
MAP_HEIGHT = 483

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.2, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)


# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()


# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable display
display = robot.getDevice("display")

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 5
vR = 5

prev_time = robot.getTime()

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis


map_array = np.zeros((483,900))

gripper_status="closed"

# === Toggles ===
use_mapping = False
use_vision = False # this might cook your PC right now
use_manipulation = False
use_planning = True
use_odometry = False

# Initialize waypoint index
waypoint_index = 0

# Check if path.npy exists, if not, calculate the path
if use_planning:
    print("Planning enabled, commencing")
    
    # Load the map first
    map_array = np.load("map.npy")
    print(f"Loaded map.npy with shape {map_array.shape}")

    print("Path file not found, calculating path using RRT...")

    # Start and goal in world coordinates (meters from center)
    start_world = (-5.0, 0.0)
    goal_world = (0.7, 0.8)

    # Inflate the loaded map
    map_array_inflated = inflate_map(map_array)
    
    start_pixel = world_to_map_coords(start_world[0], start_world[1])
    goal_pixel = world_to_map_coords(goal_world[0], goal_world[1])

    #TODO: Check if start and goal are inside obstacles and convert to GPS cords and not use relative units cause gross.
    path = rrt_star(map_array_inflated, start_pixel, goal_pixel) # RRT uses inflated map

    

    if path:
        print(f"Path generated with {len(path)} waypoints.")
        np.save("path.npy", path)
    else:
        print("RRT failed to generate a path.")

    # Load the path if it exists (or was just created)
    if os.path.exists("path.npy"):
        path = np.load("path.npy").tolist() # Load path as list of tuples
    else:
        path = [] # Ensure path is an empty list if file doesn't exist

    # Visualize the path on the *original* map
    map_rgb = np.stack([map_array]*3, axis=-1).astype(float) # Use original map for visualization

    # Draw the path in red
    if path:
        map_rgb_with_path = visualize_path(path, map_rgb.copy())
    else:
        map_rgb_with_path = map_rgb # Show map without path if none found

    # Draw origin + gridlines
    plt.imshow(map_rgb_with_path, interpolation='none')

    # Draw red crosshair at world (0,0)
    plt.axhline(y=MAP_HEIGHT // 2, color='red', linestyle='--', linewidth=1)
    plt.axvline(x=MAP_WIDTH // 2, color='red', linestyle='--', linewidth=1)
    plt.text(MAP_WIDTH // 2 + 5, MAP_HEIGHT // 2 + 15, "(0,0)", color='red', fontsize=8)

    # Optional 1-meter grid every 30px
    for x in range(0, MAP_WIDTH, 30):
        plt.axvline(x=x, color='lightgray', linewidth=0.3)
    for y in range(0, MAP_HEIGHT, 30):
        plt.axhline(y=y, color='lightgray', linewidth=0.3)

    plt.title("Obstacle Map with RED Path")
    plt.show()

    np.save("path_visualized.npy", map_rgb_with_path)

else:
    # Default to an empty path if planning is disabled
    path = []


# Main Loop
while robot.step(timestep) != -1:
    # Get updated odometry
    pose_x, pose_y, pose_theta, prev_time = odometry(robot, compass, AXLE_LENGTH, gps, display, vL, vR, prev_time, pose_x, pose_y, pose_theta)

    if gps:
        gps_values = gps.getValues()
        pose_x = gps_values[0]
        pose_y = gps_values[1]
    
    if compass:
        compass_values = compass.getValues()
        pose_theta = math.atan2(compass_values[0], compass_values[1])

    if use_mapping:
        mapping(robot, keyboard, lidar, gps, compass, display, map_array, timestep, lidar_sensor_readings, lidar_offsets, vL, vR)

    if use_vision:
        vision(robot, camera, display, keyboard, timestep, robot_parts)

    if use_manipulation:
        manipulation(robot, display, robot_parts, gripper_status, right_gripper_enc, left_gripper_enc, timestep, 0, 0, 0)

    if use_odometry:
        odometry(robot, compass, AXLE_LENGTH, gps, display, vL, vR, prev_time, pose_x, pose_y, pose_theta)

    if use_planning and len(path) > 0:
            # Get current position for path following
            vL, vR, waypoint_index = follow_waypoints(path, robot_parts, vL, vR, waypoint_index, pose_theta, pose_x, pose_y, gps)

    if use_planning and waypoint_index >= len(path) and len(path) > 0:
        print("Path completed! Robot stopped.")
        vL = 0
        vR = 0
        
    # Set wheel speeds
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
