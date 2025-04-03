"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np

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

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

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

# TODO: Read GPS data and convert to local reference frame
# TODO: Read compass data and determine heading direction

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# TODO: Collect LIDAR data to identify obstacles and free space

# Map Initialization
map = None  # This will hold the occupancy grid representation

# TODO: Convert LIDAR data into an occupancy grid map
# TODO: Store and update the map dynamically as new sensor data arrives

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

prev_time = robot.getTime()

# Kalman filter variables (for sensor fusion, if needed later)
# TODO: Implement Kalman filter if improved accuracy is required

# Implemented April 2 - John
# Function to get heading from compass
def get_heading():
    compass_data = compass.getValues()
    heading = math.atan2(compass_data[0], compass_data[1])
    return heading

# Implemented April 2 - John
# Function to update odometry from wheel velocities
def update_odometry(vL, vR, dt):
    global pose_x, pose_y, pose_theta
    v = (vL + vR) / 2.0
    w = (vR - vL) / AXLE_LENGTH
    pose_theta += w * dt
    pose_x += v * dt * math.cos(pose_theta)
    pose_y += v * dt * math.sin(pose_theta)

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None

# TODO: Implement odometry calculations using wheel encoder data
# TODO: Integrate inertial sensor (if available) to improve localization accuracy
# TODO: Apply noise filtering (Kalman Filter or Complementary Filter)

# ------------------------------------------------------------------
# Helper Functions

gripper_status="closed"

# Main Loop
while robot.step(timestep) != -1:
    # Implemented April 2 - John
    # Compute time step
    current_time = robot.getTime()
    dt = current_time - prev_time
    prev_time = current_time
    
    # Read GPS data
    gps_position = gps.getValues()
    
    # Read Compass data
    heading = get_heading()
    
    # Update odometry
    update_odometry(vL, vR, dt)
    
    # TODO: Validate the map using test cases to make sure accuracy is good
    
    # Print localization info for debugging
    print(f"GPS: {gps_position}, Heading: {heading}, Odometry: (x: {pose_x}, y: {pose_y}, theta: {pose_theta})")
    
    # TODO: Implement sensor fusion to correct odometry drift
    # TODO: Store GPS and odometry data for debugging
    
    # Setting wheel speeds
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
