"""grocery controller."""

# Apr 1, 2025
# Written by Mitch Siegel, Andrew Rho, John Kim

from controller import Robot
import math
import numpy as np
from localization import main as odometry
from mapping import main as mapping
from vision import main as vision
from manipulation import main as manipulation
from planning import world_to_map_coords
from planning import rrt_basic
from planning import inflate_map
from planning import visualize_map_and_path_on_display
from planning import smooth_path
from follow import follow_waypoints
from localization import main as odometry
from planning import visualize_path
from localization import rotateToAngle

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

# We are using a GPS and compass to disentangle mapping and localization, and to prevent odometry from accumulating error
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable display
display = robot.getDevice("display")

# Enable keyboard
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

# Initialize map
map_array = np.zeros((483,900))

gripper_status="closed"

# === Toggles ===
use_mapping = False
use_vision = False
use_manipulation = False
use_planning = True
use_odometry = False

# Initialize waypoint index
waypoint_index = 0
path_index = 0
paths = []

#harded cube waypoints (uses odometry)
All_points = [ # verified = I have picked up the cube from that point
    (3.53305, 6.23893), #item(11) - verified
    (-3.26316, 2.65703), #item(8) - verified
    (-2.68907, 1.1591), #item(7) and item(5) - verified 
    (0.810102, 1.21804), #item(6) - verified
    (2.6, 2.7), #item - verified
    (3.53022, 2.777), #item (10) - verified
    (2.36729,-2.8), #item(3) - verified
    (4.45738, -2.8), #item(12)
    (-6.18925, 0) #ending point
]
rotation_points = [ #90 = left, 0 = forward, -90 = right
    90, #item(11)
    90, #item(8)
    -90, #item(7) and item(5)
    -90, #item(6)
    90, #item
    90, #item(10)
    -90, #item(3)
    -90, #item(12)
    0 #ending point

]
#ensure all points are valid
map_array = np.load("map.npy")
map_array_inflated = inflate_map(map_array)
All_points = [world_to_map_coords(point[0], point[1]) for point in All_points]

# display the map and all points
visualize_map_and_path_on_display(display, map_array_inflated, All_points, path_color=0xFF0000)
# check if all points are valid
for point in All_points:
    if map_array_inflated[point[1], point[0]] != 0:
        print(f"ERROR: Point {point} is inside an inflated obstacle")
        exit()
    else:
        print(f"Point {point} is valid")

if use_planning:
    # Load the map first
    map_array = np.load("map.npy")
    map_array_inflated = inflate_map(map_array)

    while robot.step(timestep) != -1 and math.isnan(gps.getValues()[0]):
        print("Waiting for GPS data...")

    start_pixel = world_to_map_coords(gps.getValues()[0], gps.getValues()[1])

    # Keep track of all path segments for visualization and execution
    all_paths_pixels = [] # Store paths in pixel coordinates for visualization


    current_start_pixel = start_pixel
    for i in range(len(All_points)):
        print(f"Generating path {i+1} of {len(All_points)}")

        goal_pixel = All_points[i] # Use the map coordinates directly

        # Debug: show inflated map occupancy at start and goal pixels
        sx, sy = current_start_pixel
        gx, gy = goal_pixel
        val_start = map_array_inflated[sy, sx]
        val_goal = map_array_inflated[gy, gx]
        print(f"DEBUG: segment {i+1} start {current_start_pixel} val {val_start}, goal {goal_pixel} val {val_goal}", flush=True)

        # Check if start and goal are valid before planning
        if map_array_inflated[current_start_pixel[1], current_start_pixel[0]] != 0:
            print(f"ERROR: Start point {current_start_pixel} is inside an inflated obstacle for segment {i+1}, skipping segment.")
            continue
        if map_array_inflated[goal_pixel[1], goal_pixel[0]] != 0:
            print(f"ERROR: Goal point {goal_pixel} is inside an inflated obstacle for segment {i+1}, skipping segment.")
            continue

        # RRT* uses the inflated map for planning :)
        path_segment_pixels = rrt_basic(map_array_inflated, current_start_pixel, goal_pixel, max_iter=3000)

        if path_segment_pixels:
            # Smooth the path after generating it
            smoothed_path_segment = smooth_path(path_segment_pixels, map_array_inflated)
            if smoothed_path_segment:
                all_paths_pixels.append(smoothed_path_segment)
                paths.append(smoothed_path_segment) # Use the smoothed path for execution
                current_start_pixel = goal_pixel # Update start for next segment based on the original goal pixel
            else:
                # Fallback to original path if smoothing somehow failed (shouldn't happen with current logic)
                all_paths_pixels.append(path_segment_pixels)
                paths.append(path_segment_pixels)
                current_start_pixel = goal_pixel
        # If RRT failed, don't add anything and print error (handled by RRT function)


    # Combine all pixel paths into one list for visualization
    full_pixel_path = [point for segment in all_paths_pixels for point in segment]

    if full_pixel_path:
        print("Visualizing map and full path on robot display...")
        # Use the original map for visualization, not the inflated one
        visualize_map_and_path_on_display(display, map_array, full_pixel_path, path_color=0xFF0000) # Red path
    else:
        print("No path generated, visualizing map only.")
        # Still display the map even if no path was found
        visualize_map_and_path_on_display(display, map_array, [], path_color=0xFF0000)


    # Assign the first path segment to 'path' for the follow_waypoints logic
    if paths:
        path = paths[path_index] # Start with the first path
        waypoint_index = 0 # cut off the first point to avoid circling the starting point


# Main Loop
while robot.step(timestep) != -1:
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

    if use_odometry:
        odometry(robot, compass, AXLE_LENGTH, gps, display, vL, vR, prev_time, pose_x, pose_y, pose_theta)


    if use_planning:
        if path and waypoint_index >= len(path): # Check if index is at or beyond the end
            if path_index < len(paths) - 1: # Check if there are more path segments to follow
                print(f"Moving to path segment {path_index + 1}/{len(paths)}")
                # Get updated odometry
                # # rotate to the correct angle facing the shelf
                x, y, theta = rotateToAngle(rotation_points[path_index], robot, compass, timestep, robot_parts, vL, vR, prev_time, pose_x, pose_y, pose_theta, AXLE_LENGTH)
                pose_x = x #since this function updates the odometry from within, we bubble up the changes to the odometry variables
                pose_y = y
                pose_theta = theta
                # give control over to user to pick up the item
                if use_manipulation:
                    manipulation(robot,robot_parts, timestep)

                waypoint_index = 0 # Reset waypoint index for the new path
                path_index += 1 #increment the path index to move to the next path segment
                path = paths[path_index] # Load the next path segment
            else:
                #print("All path segments completed! Robot stopped.")
                vL = 0 # Stop left wheel
                vR = 0 # Stop right wheel

                exit()
        
        # Follow the current path segment if it's not completed
        # Add a check to ensure path exists and waypoint_index is valid before calling follow_waypoints
        if path and waypoint_index < len(path):
            vL, vR, waypoint_index = follow_waypoints(path, robot_parts, vL, vR, waypoint_index, pose_theta, pose_x, pose_y, gps)
        elif not path and not (path_index < len(paths) - 1):
            # This case handles when the last path is completed and vL, vR are already set to 0
            pass # Already stopped, do nothing
        # Consider adding an else case here if needed, e.g., if path is empty initially


    # Set wheel speeds based on the outcome of the planning/following logic
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)