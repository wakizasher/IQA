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


# Define the folder with images
image_folder = r"D:\iNaturalist\test_images"
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
              if img.endswith(('.jpg', '.png','.JPG','.png','.PNG'))]


# Loop through each image and calculate NIQE score
for image_path in image_paths:
    print(f"\nChecking out: {image_path}")
    # Load the image using PIL (Python Image Library)
    img = Image.open(image_path).convert('RGB') # Convert to RGB if it isn't
    # Calculate NIQE score (pyiqa handles the conversion to tensor internally)
    score = niqe_metric(img).item() # .item() gets the raw number form the tensor
    print(f"NIQE score: {score:.4f} (Lower is better)")


# Create DataFrame and sort by quality score (worse quality first)
df = pd.DataFrame()

toc = time.perf_counter()
print(f"IQA with NIQE finished in {toc - tic:0.4f}")