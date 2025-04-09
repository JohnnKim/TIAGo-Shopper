import cv2
import os 
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the input and output image using absolute paths
input_path = os.path.join(script_dir, "shelf_alpha.png")
output_path = os.path.join(script_dir, "shelf_alpha_with_boxes.png")

# Define bounding boxes: (x1, y1, x2, y2)
# Each tuple is a box: top-left corner (x1, y1), bottom-right corner (x2, y2)
bounding_boxes = [
    (615, 270, 645, 290), (615, 570, 645, 590)
]

# Load the image
image = cv2.imread(input_path)

# Color for the bounding box (BGR): Yellow
box_color = (0, 255, 255)
thickness = 2

box_color_2 = (0, 0, 255) # Red
box_color_3 = (0, 255, 0) # Green

# Draw each bounding box
for (x1, y1, x2, y2) in bounding_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)

# Save the result
cv2.imwrite(output_path, image)

print(f"Bounding boxes drawn and saved to '{output_path}'")
