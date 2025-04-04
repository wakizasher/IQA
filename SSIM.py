# Imports remain the same
import time
from pathlib import Path
import numpy as np
from skimage import io, transform, color
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
import re # Import regular expressions for filename parsing

# --- Configuration ---

# 1. Define Paths
# !! IMPORTANT: Use raw string (r"...") for Windows paths !!
IMAGE_DIRECTORY = r"D:\iNaturalist\images"

# 2. Define Reference Images (UPDATED WITH YOUR SELECTIONS)
REFERENCE_IMAGE_PATHS = {
    "Bellis_perennis": r"D:\iNaturalist\images\Bellis_perennis_88084.jpg",
    "Leucanthemum_vulgare": r"D:\iNaturalist\images\Leucanthemum_vulgare_95402.jpeg",
    "Matricaria_chamomilla": r"D:\iNaturalist\images\Matricaria_chamomilla_113208.jpg",
}
FLOWER_NAMES = list(REFERENCE_IMAGE_PATHS.keys()) # Extracts ["Bellis_perennis", ...]

# 3. Define Processing Parameters
TARGET_SIZE = (256, 256) # Resize images for consistent comparison and speed

# 4. Define SSIM Threshold (STARTING POINT - ADJUST AFTER FIRST RUN)
SSIM_THRESHOLD = 0.5  # <<< ADJUST THIS VALUE AFTER REVIEWING RESULTS
# Helper Function (load_and_preprocess_image remains the same)
def load_and_preprocess_image(image_path, target_size):
    # ... (previous code for this function is unchanged) ...
    try:
        img = io.imread(image_path)
        if img.shape[-1] == 4: img = color.rgba2rgb(img)
        elif len(img.shape) == 2: img = color.gray2rgb(img)
        img_gray = color.rgb2gray(img)
        img_resized = transform.resize(img_gray, target_size, anti_aliasing=True)
        img_resized = img_resized.astype(np.float32)
        return img_resized
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- Main Processing Logic ---

def assess_images_per_category(image_dir, ref_image_paths_dict, target_size):
    """Calculates SSIM score for images against category-specific references."""
    data_dir = Path(image_dir)
    results = {}
    preprocessed_references = {} # To store loaded reference images

    # 1. Validate Image Directory
    if not data_dir.is_dir():
        print(f"Error: Image directory not found: {data_dir}")
        return None

    # 2. Load and Preprocess ALL Reference Images
    print("Loading reference images...")
    all_refs_loaded = True
    for flower_name, ref_path_str in ref_image_paths_dict.items():
        ref_path = Path(ref_path_str)
        # Basic check if the placeholder name is still there
        if f"REFERENCE_{flower_name}.jpg" in ref_path_str:
             print("="*60)
             print(f"ERROR: You MUST set the reference image path for '{flower_name}'")
             print(f"       in the 'REFERENCE_IMAGE_PATHS' dictionary.")
             print(f"       Current value: {ref_path_str}")
             print("="*60)
             all_refs_loaded = False
             continue # Continue checking other paths but mark as failed

        if not ref_path.is_file():
            print(f"Error: Reference image for {flower_name} not found: {ref_path}")
            all_refs_loaded = False
            continue

        print(f"  - Loading reference for {flower_name} from {ref_path}")
        ref_img = load_and_preprocess_image(ref_path, target_size)
        if ref_img is None:
            print(f"  - Failed to load or process reference for {flower_name}.")
            all_refs_loaded = False
        else:
            preprocessed_references[flower_name] = ref_img
            print(f"  - Reference for {flower_name} processed.")

    if not all_refs_loaded or not preprocessed_references:
        print("One or more reference images failed to load. Exiting.")
        return None
    if len(preprocessed_references) != len(ref_image_paths_dict):
         print("Warning: Not all expected reference images were loaded successfully.")
         # Decide if you want to proceed or exit - let's proceed but warn
         print("Proceeding with successfully loaded references only...")


    # 3. Find all image files
    print("\nScanning for image files...")
    image_extensions = {'.jpg', '.jpeg'} # Updated based on user info
    # We can make the glob slightly more specific if needed, but general works fine.
    all_image_files = [p for ext in image_extensions for p in data_dir.glob(f'*{ext}')]

    # Filter out the reference images themselves
    reference_file_paths = {Path(p) for p in ref_image_paths_dict.values()}
    image_files_to_process = [p for p in all_image_files if p not in reference_file_paths]


    if not image_files_to_process:
        print(f"No images found in {data_dir} (excluding references). Check directory/extensions.")
        return None

    print(f"Found {len(image_files_to_process)} images to assess.")

    # 4. Iterate, Determine Category, Preprocess Target, and Calculate SSIM
    start_time = time.time()
    skipped_unknown_category = 0
    processed_count = 0

    for img_path in tqdm(image_files_to_process, desc="Calculating SSIM"):
        # --- Determine flower category from filename ---
        filename = img_path.name
        current_flower_name = None
        for known_name in preprocessed_references.keys(): # Check against loaded references
             # Check if filename starts with the known flower name followed by '_'
            if filename.startswith(known_name + "_"):
                current_flower_name = known_name
                break # Found the match

        if not current_flower_name:
            # Optional: Use regex for more robust parsing if needed
            # match = re.match(r"([a-zA-Z_]+)_\d+\.(jpg|jpeg)", filename)
            # if match:
            #     potential_name = match.group(1)
            #     if potential_name in preprocessed_references:
            #         current_flower_name = potential_name
            #     else:
            #          # Filename pattern matched, but flower name unknown/ref not loaded
            #          # print(f"Warning: Unknown flower category '{potential_name}' for {img_path}. Skipping.")
            #          skipped_unknown_category += 1
            #          results[str(img_path)] = -3.0 # Special code for category mismatch
            #          continue
            # else:
            #     # Filename doesn't match expected pattern at all
            #     # print(f"Warning: Could not determine category for {img_path}. Skipping.")
            #     skipped_unknown_category += 1
            #     results[str(img_path)] = -3.0 # Special code for category mismatch
            #     continue

             # If the simple startswith check fails, log it and skip.
             # print(f"Warning: Could not determine category for {img_path} using known prefixes. Skipping.")
             skipped_unknown_category += 1
             results[str(img_path)] = -3.0 # Use a different code for category errors vs load errors
             continue


        # --- Proceed with SSIM calculation ---
        target_img = load_and_preprocess_image(img_path, target_size)

        if target_img is None:
            results[str(img_path)] = -2.0 # Code for loading/processing error
            continue

        # Select the correct reference image
        reference_img = preprocessed_references[current_flower_name]

        # Calculate SSIM
        score = ssim(reference_img, target_img, data_range=1.0)
        results[str(img_path)] = score
        processed_count += 1

    end_time = time.time()
    print(f"\nSSIM calculation finished in {end_time - start_time:.2f} seconds.")
    if skipped_unknown_category > 0:
         print(f"Skipped {skipped_unknown_category} images due to inability to determine category from filename or missing reference.")
    return results

# --- Execution and Thresholding ---

if __name__ == "__main__":
    print("Starting image quality assessment (per category)...")
    image_scores = assess_images_per_category(IMAGE_DIRECTORY, REFERENCE_IMAGE_PATHS, TARGET_SIZE)

    if image_scores:
        # Count total assessed (excluding category errors)
        total_assessed = sum(1 for score in image_scores.values() if score > -3.0)
        print(f"\nProcessed {total_assessed} images with valid categories.")

        # 8. Thresholding - Same principle, potentially different thresholds per category?
        # For now, let's use ONE threshold, but you could adapt this.
        SSIM_THRESHOLD = 0.5  # <<< STARTING POINT - ADJUST THIS (might need to be higher now)

        # Filter images below the threshold (excluding processing and category errors)
        bad_quality_images = {
            path: score for path, score in image_scores.items()
            if score > -2.0 and score < SSIM_THRESHOLD # Exclude -2 (load error) and -3 (category error)
        }
        error_loading_images = {
            path: score for path, score in image_scores.items()
            if score == -2.0
        }
        error_category_images = {
             path: score for path, score in image_scores.items()
             if score == -3.0
        }


        print(f"\n--- Filtering Results (Threshold = {SSIM_THRESHOLD}) ---")
        print(f"Found {len(bad_quality_images)} images below SSIM threshold.")
        print(f"Found {len(error_loading_images)} images that failed processing.")
        print(f"Found {len(error_category_images)} images that failed category identification.")


        # 9. Display Sample Results (Optional) - Same logic as before
        if bad_quality_images:
            print("\nExamples of images flagged as potentially low quality (lowest scores first):")
            sorted_bad_images = sorted(bad_quality_images.items(), key=lambda item: item[1])
            for i, (path, score) in enumerate(sorted_bad_images):
                if i < 15:
                    print(f"  - Score: {score:.4f} | Path: {path}")
                else:
                    break
        else:
             print("\nNo images were found below the current SSIM threshold.")


        if error_loading_images:
             print("\nImages that failed during loading/processing:")
             for path in error_loading_images: print(f"  - {path}")
        if error_category_images:
             print("\nImages that failed category identification:")
             for path in error_category_images: print(f"  - {path}")

        # 10. Guidance on Setting the Threshold - Same principle, but scores might be higher now
        print("\n--- IMPORTANT: How to Set the SSIM Threshold ---")
        print(f"1. The current threshold (SSIM_THRESHOLD = {SSIM_THRESHOLD}) is a starting guess.")
        print(f"   (Note: Scores might be generally higher now since comparisons are within the same flower type).")
        print("2. MANUALLY INSPECT flagged images (low scores) and images just above/below the threshold.")
        print("3. ADJUST the 'SSIM_THRESHOLD' value and rerun until satisfied.")
        print("4. Consider if you need *different thresholds* for each flower type (more complex).")
        print("5. A histogram remains helpful:")

        # 11. Plot Histogram (Optional but Recommended) - Same logic
        all_valid_scores = [score for score in image_scores.values() if score > -2.0] # Include scores >= 0
        if all_valid_scores:
            plt.figure(figsize=(12, 6))
            plt.hist(all_valid_scores, bins=100, color='skyblue', edgecolor='black')
            plt.title('Distribution of SSIM Scores (Compared to Category-Specific Reference)')
            plt.xlabel('SSIM Score')
            plt.ylabel('Number of Images')
            plt.axvline(SSIM_THRESHOLD, color='red', linestyle='dashed', linewidth=1, label=f'Current Threshold ({SSIM_THRESHOLD})')
            plt.legend()
            plt.grid(axis='y', alpha=0.75)
            plt.tight_layout()
            try:
                plt.show()
                print("\nHistogram displayed. Close the plot window to continue.")
            except Exception as e:
                print(f"\nCould not display histogram automatically ({e}).")
                # Consider saving: plt.savefig("ssim_histogram_per_category.png")
        else:
            print("\nNo valid scores to plot in a histogram.")

        # 12. Next Steps (Example Actions) - Same logic
        print("\n--- Next Steps ---")
        print(" - Refine SSIM_THRESHOLD.")
        print(" - Investigate category identification errors if any occurred.")
        print(" - Implement actions (move, list, delete flagged files).")

    else:
        print("\nImage assessment could not be completed due to initial errors (e.g., loading references).")