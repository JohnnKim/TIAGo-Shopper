# Grocery Shopper Robot Controller

This project contains the controller code for a simulated Tiago robot designed to perform grocery shopping tasks within the Webots environment.
## Setup

1.  **Install Dependencies:** Ensure you have Python installed. Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Webots Environment:** This controller is designed to run within a Webots simulation environment containing a Tiago robot and the necessary world setup.
3.  **Required Files:**
    *   Make sure the YOLO model file (`model.pt`) is present in the same directory as the controller.
    *   A pre-generated map, `map.npy`, is also provided. The mapping module can also generate this file.

#### Toggles:
*   `use_mapping`: Enable/disable mapping mode.
*   `use_vision`: Enable/disable vision mode.
*   `use_manipulation`: Enable/disable manipulation mode.
*   `use_planning`: Enable/disable path planning.
*   `use_odometry`: Enable/disable odometry.

#### Main Loop:

The robot operates by path following to hard coded cube points. Once arrived at the location, it will rotate to face the cube, and then give control over to the manipulation module to pick up the cube (if `use_manipulation` is enabled it will go into keyboard IK mode). Once the user has picked up the cube, it will navigate to the next point.


## Modules

The controller is organized into several Python modules:

*   `grocery_shopper.py`: The main entry point and orchestrator for the robot's behavior. It initializes the robot, sensors, actuators, and integrates the functionality from other modules. Handles the main control loop, waypoint navigation, and switching between different modes (mapping, vision, manipulation, etc.).
*   `vision.py`: Handles object detection. It includes functions for detecting objects based on color (HSV thresholding) and using a YOLOv11n model (`model.pt`). Also contains code for collecting training data for the YOLO model.
*   `manipulation.py`: Controls the robot's arm and gripper. It utilizes Inverse Kinematics (IK) via the `ikpy` library to move the arm to target positions and provides functions for opening/closing the gripper and manual arm control via keyboard input.
*   `localization.py`: Manages the robot's pose estimation. It calculates odometry based on wheel encoder readings and fuses this with GPS and Compass data for more accurate localization. Includes functions for normalizing angles and rotating the robot to a specific heading.
*   `mapping.py`: Responsible for creating a 2D occupancy grid map of the environment using LiDAR data. It includes logic for autonomous exploration (wall following/obstacle avoidance) and allows saving/loading the map (`map.npy`).
*   `planning.py`: Implements pathfinding and navigation algorithms. It includes functions for inflating the map (creating a C-space) to account for robot size, finding paths using the Rapidly-exploring Random Tree (RRT) algorithm, smoothing the generated paths, and converting between world and map coordinates. Also provides visualization utilities for the map and path on the robot's display.