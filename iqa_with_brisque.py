import traceback
import os
import time
import pandas as pd
from PIL import Image
import numpy as np
import brisque  # Using the brisque package

# Set up the BRISQUE metric
tic = time.perf_counter()
print("Loading BRISQUE model")
# Initialize BRISQUE object
brisque_obj = brisque.BRISQUE(url=False)
print("Model loaded")

# Create lists to store data
valid_images = []
valid_scores = []
corrupted_images = []
corrupted_errors = []

# Define the folder with images
image_folder = r"E:\iNaturalist\images"
try:
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
                   if img.lower().endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))]
except FileNotFoundError:
    print(f"Error: The directory {image_folder} does not exist.")
    # Create an empty list if the directory doesn't exist (for testing purposes)
    image_paths = []

# Loop through each image and calculate BRISQUE score
for image_path in image_paths:
    print(f"\nChecking out: {image_path}")
    try:
        # Load the image using PIL
        img = Image.open(image_path).convert('RGB')  # Convert to RGB if it isn't
        img.verify()  # Verify image integrity
        img = Image.open(image_path)  # Reopen after verification

        # Convert to numpy array for brisque
        img_np = np.array(img)

        # Calculate BRISQUE score
        score = brisque_obj.score(img_np)
        print(f"BRISQUE score: {score:.4f} (Lower is better)")

        # Append to lists
        valid_images.append(image_path)
        valid_scores.append(score)
    except Exception as e:
        # Catch errors related to corrupted images or processing failures
        print(f"Bad image detected: {image_path} (Error: {str(e)})")
        print(traceback.format_exc())  # Print the full traceback for debugging
        # Append to corrupted list
        corrupted_images.append(image_path)
        corrupted_errors.append(str(e))

# Create a Pandas DataFrame
valid_data = {'Image_Path': valid_images, 'BRISQUE_score': valid_scores}
valid_df = pd.DataFrame(valid_data)
corrupted_data = {'Image_Path': corrupted_images, 'Error': corrupted_errors}
corrupted_df = pd.DataFrame(corrupted_data)

# Save the DataFrame to a CSV file
valid_csv_file = '../valid_brisque_results.csv'
valid_df.to_csv(valid_csv_file, index=False)  # index=False avoids adding an extra index column
print(f"\nValid results saved to {valid_csv_file}!")

corrupted_csv = "brisque_corrupted_images.csv"
corrupted_df.to_csv(corrupted_csv, index=False)
print(f"Corrupted images logged to {corrupted_csv}! Check it out!")

# Output the time
toc = time.perf_counter()
print(f"IQA with BRISQUE finished in {toc - tic:0.4f} seconds")
