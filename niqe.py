import traceback

import pyiqa
import os
from PIL import Image
import torch
import time
import pandas as pd

# Set up the NIQE metric
tic = time.perf_counter()
print("Loading NIQE model")
# Create a NIQE metric object. 'device' tells it to use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
niqe_metric = pyiqa.create_metric('niqe', device=device)
print("Model loaded")


# Create list to store data
valid_images = []
valid_scores = []
corrupted_images = []
corrupted_errors = []

# Define the folder with images
image_folder = r"D:\iNaturalist\test_3000"
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
              if img.lower().endswith(('.jpg','.jpeg','.png'))]

# Loop through each image and calculate NIQE score
for image_path in image_paths:
    print(f"\nChecking out: {image_path}")
    try:
        # Load the image using PIL (Python Image Library)
        img = Image.open(image_path).convert('RGB') # Convert to RGB if it isn't
        img.verify()  # Verify image integrity
        img = Image.open(image_path).convert('RGB')
        # Calculate NIQE score (pyiqa handles the conversion to tensor internally)
        score = niqe_metric(img).item() # .item() gets the raw number form the tensor
        print(f"NIQE score: {score:.4f} (Lower is better)")

        # Append to lists
        valid_images.append(image_path)
        valid_scores.append(score)
    except Exception as e:
        # Catch errors related to corrupted images or processing failures
        print(f"Bad image detected: {image_path} (Error: {str(e)})")
        # Append to currupted list
        corrupted_images.append(image_path)
        corrupted_errors.append(str(e))



# Create a Pandas DataFrame
valid_data = {'Image_Path': valid_images, 'NIQE_score': valid_scores}
valid_df = pd.DataFrame(valid_data)
corrupted_data = {'Image_Path': corrupted_images, 'Error: ': corrupted_errors}
corrupted_df = pd.DataFrame(corrupted_data)

# Save the DataFrame to a CSV file
valid_csv_file = 'valid_niqe_results3000.csv'
valid_df.to_csv(valid_csv_file, index=False) # index=False avoids adding an extra index column
print(f"\nValid results saved to {valid_csv_file}!")

corrupted_csv = "niqe_corrupted_images3000.csv"
corrupted_df.to_csv(corrupted_csv, index=False)
print(f"Corrupted images logged to {corrupted_csv}! Check it out!")


# Output the time
toc = time.perf_counter()
print(f"IQA with NIQE finished in {toc - tic:0.4f}")