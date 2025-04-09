import cv2
import numpy as np
import os
import glob

def extract_yellow_alpha(image_path):
    """Loads an image, extracts bright yellow areas, and saves the alpha mask."""
    # Define the bright yellow color range in HSV
    # Using the range from vision.py: {'lower1': np.array([20, 140, 140]), 'upper1': np.array([35, 255, 255])}
    lower_yellow = np.array([20, 140, 140])
    upper_yellow = np.array([35, 255, 255])

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for the yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Generate the output filename
    directory = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_filename = os.path.join(directory, f"{name}_alpha{ext}")

    # Save the alpha mask
    cv2.imwrite(output_filename, mask)
    print(f"Saved alpha mask to {output_filename}")

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all shelf*.png images in the script's directory
    image_pattern = os.path.join(script_dir, "shelf*.png")
    image_paths = glob.glob(image_pattern)

    if not image_paths:
        print(f"No images matching '{image_pattern}' found.")
    else:
        print(f"Found images: {image_paths}")
        for img_path in image_paths:
            # Exclude any previously generated alpha masks
            if "_alpha" not in img_path:
                extract_yellow_alpha(img_path)
