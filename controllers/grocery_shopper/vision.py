"""
This file contains the vision module for the grocery shopper.
It contains the functions for the camera to detect objects.
All functions are written by Mitch Siegel
"""


import cv2
import numpy as np
import math

KNOWN_CUBE_HEIGHT_M = 0.07
KNOWN_CUBE_WIDTH_M = 0.07
COLOR_RANGES_HSV = {

    'red': { #red boxes
        'lower1': np.array([0, 120, 70]), 'upper1': np.array([10, 255, 255]),
        'lower2': np.array([170, 120, 70]), 'upper2': np.array([180, 255, 255])
    },
    'yellow': { #yellow cubes - Tighter Range
        'lower1': np.array([20, 140, 140]), 'upper1': np.array([35, 255, 255])
    },
    'blue': { #shelves
        'lower1': np.array([100, 150, 50]), 'upper1': np.array([140, 255, 255])
    },
    'bright_green': { #green cubes
        'lower1': np.array([50, 120, 120]), 'upper1': np.array([75, 255, 255])
    },
    'white': {
        'lower1': np.array([0, 0, 200]), 'upper1': np.array([180, 30, 255])
    }
}

IMAGE_SCALE_FACTOR = 100
FOCAL_LENGTH_PIXELS = 77.05111391211969

# Camera offsets relative to the robot base frame (derived from lab5_joint.py)
# These might need tuning for the specific robot setup
CAMERA_OFFSET_X = 0# Corresponds to -Z_cam contribution
CAMERA_OFFSET_Y = 0  # Corresponds to -X_cam contribution
CAMERA_OFFSET_Z = 0.97 # Corresponds to Y_cam contribution (0.97 + 0.2 from lab5)


class DetectedObject:
    """Represents a detected object with its properties and distance."""
    def __init__(self, x, y, w, h, cx, cy, area, color):
        self.x = x  # Pixel coord (top-left)
        self.y = y  # Pixel coord (top-left)
        self.w = w  # Pixel width
        self.h = h  # Pixel height
        self.cx = cx # Pixel coord (center)
        self.cy = cy # Pixel coord (center)
        self.area = area
        self.color = color
        self.distance = None # Z_cam (distance along camera forward axis)
        self.is_target = False
        # Robot frame coordinates (relative to base link or IK origin)
        self.X_robot = None
        self.Y_robot = None
        self.Z_robot = None

    def update_distance(self, focal_length_px=FOCAL_LENGTH_PIXELS, known_height_m=KNOWN_CUBE_HEIGHT_M):
        """Calculates and updates the distance to the object (Z_cam)."""
        if self.h is None or self.h <= 0:
            self.distance = None
            return

        if focal_length_px is None or focal_length_px <= 0:
            print("Distance Calc Error: Invalid FOCAL_LENGTH_PIXELS.")
            self.distance = None
            return

        try:
            scaled_focal_length = focal_length_px

            self.distance = (known_height_m * scaled_focal_length) / self.h
            
        except ZeroDivisionError:
            print(f"Distance Calc Warning: Division by zero for object at ({self.cx}, {self.cy}).")
            self.distance = None

    def update_robot_coords(self, image_width, image_height, focal_length_px=FOCAL_LENGTH_PIXELS):
        """Calculates the object's position in the robot's base coordinate frame."""
        if self.distance is None or self.distance <= 0:
            self.X_robot, self.Y_robot, self.Z_robot = None, None, None
            return

        if focal_length_px is None or focal_length_px <= 0:
             print("Robot Coord Calc Error: Invalid FOCAL_LENGTH_PIXELS.")
             self.X_robot, self.Y_robot, self.Z_robot = None, None, None
             return

        # Calculate image center based on *original* dimensions
        center_x_pixel = image_width / 2.0
        center_y_pixel = image_height / 2.0

        # Calculate camera frame coordinates based on pinhole model
        Z_cam = self.distance # Forward distance
        X_cam = (self.cx - center_x_pixel) * Z_cam / focal_length_px # Right displacement
        Y_cam = -(self.cy - center_y_pixel) * Z_cam / focal_length_px # Down displacement if cy > center_y


        self.X_robot = Z_cam + CAMERA_OFFSET_X
        self.Y_robot = -X_cam + CAMERA_OFFSET_Y
        self.Z_robot = Y_cam + CAMERA_OFFSET_Z

        # print(f"Obj {self.color}: Z_cam={Z_cam:.2f}, X_cam={X_cam:.2f}, Y_cam={Y_cam:.2f} -> Robot X={self.X_robot:.2f}, Y={self.Y_robot:.2f}, Z={self.Z_robot:.2f} (Proposed)")

    def __repr__(self):
        robot_coords = f"Robot(X={self.X_robot:.2f}, Y={self.Y_robot:.2f}, Z={self.Z_robot:.2f})" if self.X_robot is not None else "Robot(None)"
        return (f"DetectedObject(color='{self.color}', center_px=({self.cx},{self.cy}), "
                f"area={self.area}, distance={self.distance:.2f}m, {robot_coords}, target={self.is_target})")

# While this function is not used, it is here for reference
def get_focal_length_pixels(camera):
    fov_rad = camera.getFov()
    image_width_px = camera.getWidth()
    if fov_rad <= 0 or image_width_px <= 0: return None
    return (image_width_px / 2.0) / math.tan(fov_rad / 2.0)


def detect_objects_by_color(bgr_image, target_color, min_contour_area=50):
    """
    Detects objects of a specific color in a BGR image using HSV thresholding.

    Args:
        bgr_image: The input image in BGR format (NumPy array).
        target_color (str): The name of the color to detect (e.g., 'red', 'yellow').
                              Must be a key in COLOR_RANGES_HSV.
        min_contour_area (int): The minimum area for a contour to be considered an object.

    Returns:
        tuple: A tuple containing:
            - list[DetectedObject]: A list of detected objects.
            - np.ndarray: The binary mask used for detection. Returns None if color invalid.
    """
    if target_color not in COLOR_RANGES_HSV:
        print(f"Error: Color '{target_color}' not defined in COLOR_RANGES_HSV.")
        return [], None

    # Convert BGR to HSV
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Get ranges for the target color
    ranges = COLOR_RANGES_HSV[target_color]

    # Create mask(s)
    mask1 = cv2.inRange(hsv_image, ranges['lower1'], ranges['upper1'])
    if 'lower2' in ranges: # Check if a second range exists (like for red)
        mask2 = cv2.inRange(hsv_image, ranges['lower2'], ranges['upper2'])
        full_mask = mask1 + mask2
    else:
        full_mask = mask1

    # Find contours
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours and gather object info
    detected_objects = []
    # print(f"Found {len(contours)} initial contours for '{target_color}'.") # Less verbose logging
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            obj = DetectedObject(
                x=x, y=y, w=w, h=h,
                cx=center_x, cy=center_y,
                area=area, color=target_color
            )
            detected_objects.append(obj)

    return detected_objects, full_mask

def draw_detected_objects(detected_objects, detected_image):
    """Draws bounding boxes and info for detected objects on the image."""
    color_map = {
        'red': (0, 0, 255),      # BGR Red
        'yellow': (0, 255, 255), # BGR Yellow
        'blue': (255, 0, 0),     # BGR Blue
        'bright_green': (0, 255, 0), # BGR Green
        'white': (255, 255, 255) # BGR White
    }
    default_color = (0, 0, 0) # Black for unknown colors

    for obj in detected_objects:
        # Use object pixel coordinates *after* scaling for drawing
        draw_x = int(obj.x) # Assuming coordinates were already scaled before calling this
        draw_y = int(obj.y)
        draw_w = int(obj.w)
        draw_h = int(obj.h)
        draw_cx = int(obj.cx)
        draw_cy = int(obj.cy)

        draw_color = color_map.get(obj.color, default_color)
        # Draw rectangle
        cv2.rectangle(detected_image, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), draw_color, 2)

        # Put text for center pixel coordinates and distance
        info_text1 = f"({draw_cx},{draw_cy})"
        if obj.distance is not None:
             info_text1 += f" {obj.distance:.2f}m"
        cv2.putText(detected_image, info_text1, (draw_x, draw_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)

        # Put text for robot coordinates if available
        if obj.X_robot is not None:
            info_text2 = f"R({obj.X_robot:.2f},{obj.Y_robot:.2f},{obj.Z_robot:.2f})"
            cv2.putText(detected_image, info_text2, (draw_x, draw_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)


        # Indicate target object
        if obj.is_target:
             cv2.putText(detected_image, "TARGET", (draw_cx - 30, draw_cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def show_image(image, window_name="Detected Objects"):
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


def main(robot, camera, display):
    
    # Get original image
    image_bytes = camera.getImage()
    orig_height = camera.getHeight()
    orig_width = camera.getWidth()
    image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape((orig_height, orig_width, 4))
    bgr_image_orig = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
    
    # Detect objects in original image
    colors_to_detect = ['yellow', 'bright_green']
    min_area_threshold = 10
    all_detected_objects = []
    
    for color in colors_to_detect:
        detected_objects, _ = detect_objects_by_color(bgr_image_orig, color, min_area_threshold)
        if detected_objects:
            all_detected_objects.extend(detected_objects)
    
    # Calculate distances and robot coordinates using original image dimensions

    focal_length_to_use = FOCAL_LENGTH_PIXELS # Default


    for obj in all_detected_objects:
        # Pass calculated focal length
        obj.update_distance(focal_length_px=focal_length_to_use)
        # Calculate robot coords using original dimensions and calculated focal length
        obj.update_robot_coords(orig_width, orig_height, focal_length_px=focal_length_to_use)

    # Now scale the image for display purposes only
    scale_percent = IMAGE_SCALE_FACTOR
    new_width = int(bgr_image_orig.shape[1] * scale_percent / 100)
    new_height = int(bgr_image_orig.shape[0] * scale_percent / 100)
    display_image = cv2.resize(bgr_image_orig, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Scale object pixel coordinates just for drawing on the scaled display image

    for obj in all_detected_objects:
        # Create a temporary copy or update in place if you don't need original pixels later
        # Here we update in place for simplicity before drawing
        obj.x = int(obj.x * scale_percent / 100)
        obj.y = int(obj.y * scale_percent / 100)
        obj.w = int(obj.w * scale_percent / 100)
        obj.h = int(obj.h * scale_percent / 100)
        obj.cx = int(obj.cx * scale_percent / 100)
        obj.cy = int(obj.cy * scale_percent / 100)
        # Note: obj.distance, obj.X_robot etc. remain unscaled (real-world values)


    # Now you can access obj.X_robot, obj.Y_robot, obj.Z_robot for control logic
    # Example: Find the closest yellow cube in robot coordinates
    closest_yellow = None
    min_dist_sq = float('inf')
    for obj in all_detected_objects:
        if obj.color == 'yellow' and obj.X_robot is not None:
             # Use X_robot (forward distance) or Euclidean distance in robot frame
             dist_sq = obj.X_robot**2 + obj.Y_robot**2 + obj.Z_robot**2
             if dist_sq < min_dist_sq:
                 min_dist_sq = dist_sq
                 closest_yellow = obj

    if closest_yellow is not None:
        closest_yellow.is_target = True

    # Draw detections (using the now-scaled pixel coords) on the scaled display image
    draw_detected_objects(all_detected_objects, display_image)
    show_image(display_image, window_name="Shelf Image with Detections")
    if closest_yellow is not None:
        return (closest_yellow.X_robot, closest_yellow.Y_robot, closest_yellow.Z_robot) # Or some default value if no target found
    else:
        return (None, None, None)

def calculate_camera_coords(cx, cy, h, real_height_m, focal_length_px, image_width, image_height):
    """
    Calculates camera frame coordinates (Z_cam, X_cam, Y_cam) from object pixel info.

    Assumes Camera Frame: +Z Forward, +X Right, +Y Up
    """
    if h is None or h <= 0:
        print("Coord Calc Warning: Invalid pixel height (h).")
        return None, None, None
    if real_height_m is None or real_height_m <= 0:
         print("Coord Calc Warning: Invalid real_height_m.")
         return None, None, None
    if focal_length_px is None or focal_length_px <= 0:
         print("Coord Calc Warning: Invalid focal_length_px.")
         return None, None, None

    # Calculate distance (Forward, Z_cam)
    try:
        Z_cam = (real_height_m * focal_length_px) / h
    except ZeroDivisionError:
        print("Coord Calc Warning: Division by zero calculating Z_cam.")
        return None, None, None

    # Calculate image center
    center_x_pixel = image_width / 2.0
    center_y_pixel = image_height / 2.0

    # Calculate lateral displacement (Right, X_cam)
    X_cam = (cx - center_x_pixel) * Z_cam / focal_length_px

    # Calculate vertical displacement (Up, Y_cam)
    # Note the sign convention used in your original code: -(cy - center) results in +Y Up
    Y_cam = -(cy - center_y_pixel) * Z_cam / focal_length_px

    return Z_cam, X_cam, Y_cam


