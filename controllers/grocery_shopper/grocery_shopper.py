"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np
from localization import main as odometry
from mapping import main as mapping
from vision import main as vision
from manipulation import main as manipulation
from planning import main as planning
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
use_vision = True #this might cook ur pc right now
use_manipulation = False
use_planning = False
use_odometry = False

# Main Loop
while robot.step(timestep) != -1:
    # try:
    #     waypoints = np.load("path.npy")
    #     print(f"Loaded {len(waypoints)} waypoints from planning.np")
    # except Exception as e:
    #     print(f"Error loading path.npy: {e}")
    #     waypoints = []

    # # Placeholder: Print or process waypoints (No path-following yet)
    # if waypoints:
    #     print("Waypoints loaded, ready for path following.")
    # else:
    #     print("No waypoints found to follow.")
    # thing is 16.1x30
    if use_mapping:
        mapping(robot, keyboard, lidar, gps, compass, display, map_array, timestep, lidar_sensor_readings, lidar_offsets, vL, vR)

    if use_vision:
        vision(robot, camera, display, keyboard, timestep, robot_parts)

    if use_manipulation:
        manipulation(robot, display, robot_parts, gripper_status, right_gripper_enc, left_gripper_enc, timestep, 0,0,0)

    if use_planning:
        planning(robot, display)
    
    if use_odometry:
        odometry(robot, compass, AXLE_LENGTH, gps, display, vL, vR, prev_time, pose_x, pose_y, pose_theta)
    
    # set wheel speeds
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
