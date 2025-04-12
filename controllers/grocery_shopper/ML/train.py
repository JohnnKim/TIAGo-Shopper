import os
from ultralytics import YOLO
import traceback # Added for detailed error printing

print("-" * 30)
print("DEBUG: Checking dataset contents...")
dataset_path_relative = "." # Path relative to where train.py is run (just the current directory)

# --- Calculate Absolute Paths --- #
script_dir = os.path.dirname(os.path.abspath(__file__)) # Get directory of the current script
abs_dataset_path = os.path.abspath(os.path.join(script_dir, dataset_path_relative))
data_config_path_relative = os.path.join(dataset_path_relative, "data.yml")
abs_data_config_path = os.path.abspath(os.path.join(script_dir, data_config_path_relative))

print(f"Script directory: {script_dir}")
print(f"Absolute dataset path: {abs_dataset_path}")
print(f"Absolute data config path: {abs_data_config_path}")

if not os.path.isdir(abs_dataset_path):
    print(f"ERROR: The dataset directory '{abs_dataset_path}' does not exist or is not a directory.")
elif not os.path.isfile(abs_data_config_path):
     print(f"ERROR: The data config file '{abs_data_config_path}' does not exist.")
else:
    print("Dataset directory and config file found. Listing files:")
    for root, dirs, files in os.walk(abs_dataset_path):
        # Skip hidden files/dirs like .DS_Store
        files = [f for f in files if not f.startswith('.')]
        dirs[:] = [d for d in dirs if not d.startswith('.')] # Modifying dirs[:] affects os.walk traversal

        for filename in files:
            # Get the path relative to the dataset directory for cleaner output
            relative_file_path = os.path.relpath(os.path.join(root, filename), abs_dataset_path)
            print(f"  - {relative_file_path}")

print("-" * 30)

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS using the ABSOLUTE path to data.yml
print(f"Starting training with data config: {abs_data_config_path}")
try:
    results = model.train(data=abs_data_config_path, epochs=100, imgsz=640, device="mps")
    print("Training completed successfully.")
    # You can access results attributes if needed, e.g., results.save_dir
except Exception as e:
    print(f"ERROR during model training: {e}")
    traceback.print_exc() # Print detailed traceback