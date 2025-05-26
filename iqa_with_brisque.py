import traceback
import os
import time
import pandas as pd
from PIL import Image
import torch
import pyiqa

# Set up the BRISQUE metric
tic = time.perf_counter()
print("Loading BRISQUE model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
brisque_metric = pyiqa.create_metric('brisque', device=device)
print("Model loaded")

# Create lists to store data
valid_images = []
valid_scores = []
corrupted_images = []
corrupted_errors = []

# Define the folder with images
image_folder = r"D:\iNaturalist\test_3000"
try:
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
                   if img.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))]
except FileNotFoundError:
    print(f"Error: The directory {image_folder} does not exist.")
    image_paths = []

# Loop through each image and calculate BRISQUE score
for image_path in image_paths:
    print(f"\nChecking out: {image_path}")
    try:
        # Load and ensure proper format
        img = Image.open(image_path)

        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Verify image integrity
        img.verify()
        img = Image.open(image_path).convert('RGB')  # Reopen after verification

        # Ensure minimum size (BRISQUE needs sufficient image content)
        if img.size[0] < 32 or img.size[1] < 32:
            raise ValueError(f"Image too small: {img.size}")

        # Calculate BRISQUE score
        score = brisque_metric(img).item()
        print(f"BRISQUE score: {score:.4f} (Lower is better)")

        # Append to lists
        valid_images.append(image_path)
        valid_scores.append(score)

    except Exception as e:
        print(f"Bad image detected: {image_path} (Error: {str(e)})")
        print(traceback.format_exc())
        corrupted_images.append(image_path)
        corrupted_errors.append(str(e))

# Create DataFrames and save results
valid_data = {'Image_Path': valid_images, 'BRISQUE_score': valid_scores}
valid_df = pd.DataFrame(valid_data)
corrupted_data = {'Image_Path': corrupted_images, 'Error': corrupted_errors}
corrupted_df = pd.DataFrame(corrupted_data)

valid_csv_file = '../valid_brisque_results.csv'
valid_df.to_csv(valid_csv_file, index=False)
print(f"\nValid results saved to {valid_csv_file}!")

corrupted_csv = "brisque_corrupted_images.csv"
corrupted_df.to_csv(corrupted_csv, index=False)
print(f"Corrupted images logged to {corrupted_csv}! Check it out!")

toc = time.perf_counter()
print(f"IQA with BRISQUE finished in {toc - tic:0.4f} seconds")



