# Grocery Shopper Robot Controller

This project contains the controller code for a simulated Tiago robot designed to perform grocery shopping tasks within the Webots environment.

---

## Overview

The robot autonomously navigates a simulated grocery environment, using mapping, vision, and manipulation to retrieve objects from shelves and deliver them to an endpoint. It combines classic robotics techniques (mapping, path planning, and odometry) with modern approaches like object detection and inverse kinematics.

---

## Setup Instructions

### 1. Install Dependencies

Ensure Python 3.8+ is installed. Then install required packages:

```bash
pip install -r requirements.txt
```

### 2. Webots Simulation

Ensure you have the Webots simulator and a world that includes:

* A Tiago robot
* Grocery shelf objects
* An open indoor environment

### 3. Required Files

* `model.pt`: YOLO model for object recognition (place in root directory)
* `map.npy`: (Optional) pre-generated environment map

> If not present, `mapping.py` will generate the map.

---

## Feature Toggles

Inside `grocery_shopper.py`, these flags enable/disable modules:

*   `use_mapping`: Enable/disable mapping mode.
*   `use_vision`: Enable/disable vision mode.
*   `use_manipulation`: Enable/disable manipulation mode.
*   `use_planning`: Enable/disable path planning.
*   `use_odometry`: Enable/disable odometry.

---

## How It Works

The robot:

1. Loads a map and plans paths to hard-coded shelf locations.
2. Uses LiDAR + RRT for global planning and smoothing.
3. Follows waypoints using GPS + compass for localization.
4. Upon arrival, rotates to a pre-specified angle.
5. Enters manual manipulation mode (keyboard-based) for item pickup.
6. Continues to the next point until all are completed.

---

## File Descriptions

### Core Controller

* `grocery_shopper.py`: Main entry point. Initializes devices, toggles modules, runs control loop, and integrates all subsystems.

### Modules

* **`vision.py`**: Object detection using:

  * HSV color segmentation
  * YOLOv11n (`model.pt`)
  * Collects training data for custom objects

* **`manipulation.py`**: Robot arm control via IK.

  * Keyboard-based inverse kinematics using `ikpy`
  * Gripper opening/closing
  * Manual + automatic item retrieval

* **`localization.py`**: Calculates robot position.

  * Combines GPS and compass
  * Provides pose tracking, odometry reset, and heading control

* **`mapping.py`**: Builds 2D occupancy map.

  * Uses LiDAR for obstacle detection
  * Supports exploration mode (wall following)
  * Saves as `map.npy`

* **`planning.py`**: Handles navigation.

  * RRT-based path planner
  * Path smoothing
  * Coordinate transforms (world ↔ map)
  * Display visualization

* **`follow.py`**: Path-following controller.

  * Follows smoothed waypoints based on pose
  * Outputs left/right wheel speeds

---

## Planning Configuration

* All target points (`All_points`) are defined in world coordinates.
* Orientation at target (`rotation_points`) is used to align with shelf.
* Before planning, the system validates that all waypoints are in free space.

---

## Robot Behavior

* Starts at robot spawn location
* Navigates to each hard-coded shelf
* Faces shelf using `rotateToAngle()`
* Waits for item pickup via manual IK
* Proceeds to next point
* Stops at endpoint when complete

---

## Visual Debugging

* Paths and map are drawn on the Tiago display
* Each segment is color-coded
* Obstacle errors (e.g., point inside inflated map) are reported with coordinates

---

## Notes

* The LiDAR angle range is clipped to ignore the robot's chassis (using 667 bins).
* GPS must initialize before planning can begin.
* Map inflation accounts for robot footprint in the environment.

---

## Authors

* Mitch Siegel
* Andrew Rho
* John Kim

April 2025
