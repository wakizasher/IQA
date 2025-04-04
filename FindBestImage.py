import cv2  # OpenCV for image processing
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# --- Configuration ---
IMAGE_DIRECTORY = r"D:\iNaturalist\images" # Your image folder path
FLOWER_NAMES = ["Bellis_perennis", "Leucanthemum_vulgare", "Matricaria_chamomilla"] # Categories
NUM_CANDIDATES = 10 # How many top candidates to suggest per category
# Optional: Resize images slightly before calculating metrics for speed.
# Set to None to use original size (more accurate sharpness but slower).
RESIZE_DIM_FOR_METRICS = (600, 600)

# --- Quality Metrics Function ---

def calculate_quality_metrics(image_path, resize_dim=None):
    """
    Calculates sharpness (Laplacian variance), brightness (mean),
    and contrast (std dev) for a given image.
    Returns a dictionary with metrics or None if an error occurs.
    """
    try:
        # Load image in grayscale using OpenCV (efficient for these metrics)
        # Using str() because OpenCV sometimes prefers string paths
        img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Check if image loading failed
        if img_gray is None:
            # print(f"Warning: Could not load {image_path}. Skipping.")
            return None

        if resize_dim:
            # Resize for speed. INTER_AREA is generally good for downscaling.
            img_gray = cv2.resize(img_gray, resize_dim, interpolation=cv2.INTER_AREA)

        # 1. Sharpness: Variance of Laplacian
        #    cv2.CV_64F specifies a higher precision data type for the result
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

        # 2. Brightness: Mean pixel intensity (0=black, 255=white)
        brightness = np.mean(img_gray)

        # 3. Contrast: Standard deviation of pixel intensities
        contrast = np.std(img_gray)

        # Basic check for completely black or white images (which might have 0 contrast/laplacian)
        if brightness < 1 or brightness > 254 : # Very dark or very bright
             laplacian_var = 0 # Penalize completely uniform images

        return {
            "sharpness": laplacian_var,
            "brightness": brightness,
            "contrast": contrast
        }
    except Exception as e:
        # Catch potential errors during processing
        print(f"Error calculating metrics for {image_path}: {e}")
        return None

# --- Main Candidate Finding Logic ---

def find_best_candidates(image_dir, flower_names, num_candidates, resize_dim=None):
    """
    Finds top N candidate reference images per category based on calculated quality metrics.
    """
    data_dir = Path(image_dir)
    all_candidates = {} # Dictionary to store results {flower_name: [candidates]}

    if not data_dir.is_dir():
        print(f"Error: Image directory not found: {data_dir}")
        return None

    # Process each flower category separately
    for flower_name in flower_names:
        print(f"\n--- Processing category: {flower_name} ---")
        category_images = []
        # Find all images belonging to this category (check both jpg and jpeg)
        for ext in ['.jpg', '.jpeg']:
            pattern = f"{flower_name}_*{ext}"
            # Use Path.glob to find matching files
            category_images.extend(list(data_dir.glob(pattern)))

        if not category_images:
            print(f"No images found matching pattern '{flower_name}_*.jpg/jpeg'. Skipping.")
            continue

        print(f"Found {len(category_images)} images for {flower_name}. Calculating metrics...")

        image_scores = [] # List to store results for this category
        # Calculate metrics for each image in the category
        for img_path in tqdm(category_images, desc=f"Metrics for {flower_name}"):
            metrics = calculate_quality_metrics(img_path, resize_dim)
            # Only add if metrics were successfully calculated
            if metrics:
                # Store the path along with its metrics
                image_scores.append({"path": img_path, **metrics})

        if not image_scores:
            print(f"Could not calculate metrics for any image in {flower_name}.")
            continue

        # --- Rank the images ---
        # This is where you decide how to prioritize. A common approach is:
        # Prioritize SHARPNESS, but maybe filter out images that are extremely dark/bright or flat.
        # Simple approach: Sort primarily by sharpness (descending).

        # Add filters (optional): You could uncomment and adjust these lines
        # image_scores = [s for s in image_scores if 50 < s['brightness'] < 220] # Filter out very dark/bright
        # image_scores = [s for s in image_scores if s['contrast'] > 15]       # Filter out very low contrast

        # Sort by sharpness score in descending order (highest sharpness first)
        sorted_images = sorted(image_scores, key=lambda x: x['sharpness'], reverse=True)

        # Select the top N candidates
        top_candidates = sorted_images[:num_candidates]
        all_candidates[flower_name] = top_candidates

        # Print the top candidates found for this category
        print(f"\nTop {min(num_candidates, len(top_candidates))} candidates for {flower_name} (ranked primarily by sharpness):")
        for i, candidate in enumerate(top_candidates):
            print(f"  {i+1}. Path: {candidate['path']}")
            # Print the metrics for context
            print(f"     Sharpness: {candidate['sharpness']:.2f}, Brightness: {candidate['brightness']:.2f}, Contrast: {candidate['contrast']:.2f}")

    return all_candidates

# --- Execution ---
if __name__ == "__main__":
    print("Starting automated search for candidate reference images...")
    # Make sure current time and location are up-to-date if needed elsewhere
    # (Not directly used in this script's logic, but good practice)
    # Current time: Thursday, April 3, 2025 at 5:20:48 PM CEST
    # Location: Eindhoven, North Brabant, Netherlands

    candidate_results = find_best_candidates(IMAGE_DIRECTORY, FLOWER_NAMES, NUM_CANDIDATES, RESIZE_DIM_FOR_METRICS)

    print("\n" + "="*60)
    print("--- Automated Candidate Search Complete ---")
    print("="*60)

    if candidate_results:
        print(f"\nFound top candidate images for {len(candidate_results)} categories.")
        print("\n**VERY IMPORTANT: Please MANUALLY REVIEW these candidates!**")
        print("This script ranks images based on automatic metrics (sharpness, etc.),")
        print("but these metrics CAN BE FOOLED (e.g., sharp background, poor composition).")
        print("\n**Your next step:**")
        print("1. Look at the suggested image files listed above for each category.")
        print("2. Choose the ONE image per category that YOU judge to be the best")
        print("   overall reference considering focus, lighting, clarity, composition,")
        print("   and how representative it is of the flower.")
        print("3. Use the paths of your final chosen images in the previous SSIM script.")

        # You can uncomment this section to print just the paths for easy copying:
        # print("\nCandidate paths for final manual review:")
        # for flower, candidates in candidate_results.items():
        #      print(f"\nCategory: {flower}")
        #      for c in candidates:
        #          print(f"  {c['path']}")

    else:
        print("\nNo candidate images were identified. Please check the configuration and")
        print("ensure images matching the expected patterns exist in the directory.")
        print("Review any error messages printed during the process.")