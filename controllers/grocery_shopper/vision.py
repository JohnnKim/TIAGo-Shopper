import cv2
import numpy as np


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
            - list: A list of dictionaries, each representing a detected object.
                    Each dict contains 'x', 'y', 'w', 'h', 'cx', 'cy', 'area'.
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
    print(f"Found {len(contours)} initial contours for '{target_color}'.")
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            detected_objects.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': center_x, 'cy': center_y, 'area': area
            })
            print(f"  - Detected {target_color.upper()} object at center ({center_x}, {center_y}), Area: {area}")

    return detected_objects, full_mask


def main(robot, camera, display): # none so we can run standalone
    # --- Handle camera image ---

    image_path = camera.getImage() # Or 'shelf2.png', or the full path if needed
    image_bytes = camera.getImage()
    height = camera.getHeight()
    width = camera.getWidth()
    image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 4))
    bgr_image = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

    # --- scale up image ---
    scale_percent = 150
    new_width = int(bgr_image.shape[1] * scale_percent / 100)
    new_height = int(bgr_image.shape[0] * scale_percent / 100)
    dim = (new_width, new_height)
    resized_bgr = cv2.resize(bgr_image, dim, interpolation = cv2.INTER_AREA) # Use INTER_AREA for shrinking, INTER_LINEAR or INTER_CUBIC for enlarging

    if bgr_image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    bgr_image = resized_bgr
    print(f"Loaded image shape: {bgr_image.shape}") # Print dimensions (height, width, channels)
    detected_image = bgr_image.copy() # Start with a copy to draw on



    target_color_to_detect = 'yellow'
    second_target_color_to_detect = 'bright_green'
    min_area_threshold = 50
    detected_objects, color_mask = detect_objects_by_color(bgr_image, target_color_to_detect, min_area_threshold)
    detected_objects_2, color_mask_2 = detect_objects_by_color(bgr_image, second_target_color_to_detect, min_area_threshold)

    detected_objects.extend(detected_objects_2) # Combine the two lists
    # Print summary and draw results
    print(f"\nDetected {len(detected_objects)} {target_color_to_detect} objects meeting criteria.")


    detected_objects = [obj for obj in detected_objects if obj['area'] > 10] #get rid of tiny objects

    if detected_objects:
        # Define drawing color (optional, could be based on detected color)
        draw_color = (0, 255, 255) # Yellow for yellow objects
        if target_color_to_detect == 'red':
            draw_color = (0, 0, 255) # Red for red objects
        elif target_color_to_detect == 'blue':
             draw_color = (255, 0, 0) # Blue for blue objects

        for obj in detected_objects:
            print(f"Detected {target_color_to_detect.upper()} object at center ({obj['cx']},{obj['cy']}), Area: {obj['area']}")
            # Draw rectangle
            cv2.rectangle(detected_image, (obj['x'], obj['y']), (obj['x'] + obj['w'], obj['y'] + obj['h']), draw_color, 2)
            # Put text for center coordinates
            cv2.putText(detected_image, f"({obj['cx']},{obj['cy']})", (obj['x'], obj['y'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)

    # 3. Display the result
    if color_mask is not None:
        cv2.imshow(f"{target_color_to_detect.capitalize()} Color Mask", color_mask) # Show the mask
    if color_mask_2 is not None:
        cv2.imshow(f"{second_target_color_to_detect.capitalize()} Color Mask", color_mask_2) # Show the mask
    cv2.imshow("Shelf Image with Detections", detected_image)
    cv2.waitKey(1) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows() # Close the window when done
    return (10,10)
