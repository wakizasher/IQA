import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import logging
import traceback
from skimage import io
from skimage.util import img_as_float
from scipy.signal import convolve2d
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='brisque_analysis.log'
)

# Directory containing the images
image_dir = r"D:\iNaturalist\images"
# Output CSV file path
output_csv = "image_quality_brisque.csv"


def compute_features(img):
    """
    Compute BRISQUE-inspired features for image quality assessment.
    This is a simplified implementation of the BRISQUE algorithm.

    Args:
        img: Grayscale image (float, values in [0, 1])

    Returns:
        float: Quality score
    """
    # Check if image is valid
    if img.ndim != 2:
        raise ValueError("Image must be grayscale")

    # Compute local statistics (mean and variance)
    window = np.ones((7, 7)) / 49.0
    mu = convolve2d(img, window, mode='same', boundary='symm')
    mu_sq = mu * mu
    sigma = np.sqrt(abs(convolve2d(img * img, window, mode='same', boundary='symm') - mu_sq))

    # Compute features
    # MSCN coefficients
    mscn = (img - mu) / (sigma + 0.0000001)

    # Calculate histogram statistics
    mscn_flat = mscn.flatten()
    mean_mscn = np.mean(mscn_flat)
    var_mscn = np.var(mscn_flat)
    skew_mscn = np.mean((mscn_flat - mean_mscn) ** 3) / (var_mscn ** 1.5 + 0.000001)
    kurt_mscn = np.mean((mscn_flat - mean_mscn) ** 4) / (var_mscn ** 2 + 0.000001) - 3

    # Simple score based on MSCN statistics (higher is worse quality)
    # This is a simplified approach - real BRISQUE uses an SVR model
    score = 100 * (abs(skew_mscn) + (kurt_mscn - 3) + var_mscn)

    return score


def calculate_simplified_brisque(image_path):
    """
    Calculate a simplified BRISQUE-inspired score for a single image.

    Args:
        image_path (str): Path to the image file

    Returns:
        tuple: (image_filename, quality_score, error_message)
    """
    try:
        # Read image with scikit-image
        img = io.imread(image_path)

        # Convert to grayscale if needed
        if img.ndim == 3 and img.shape[2] == 3:
            # Simple grayscale conversion
            gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        elif img.ndim == 2:
            gray = img
        else:
            return os.path.basename(image_path), None, f"Unsupported image format: {img.shape}"

        # Normalize to [0,1]
        gray_norm = img_as_float(gray)

        # Calculate quality score
        score = compute_features(gray_norm)

        return os.path.basename(image_path), score, None
    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"Error processing {image_path}: {str(e)}\n{error_detail}")
        return os.path.basename(image_path), None, str(e)


def process_images_parallel(image_dir, max_workers=8):
    """
    Process all images in parallel using ThreadPoolExecutor.

    Args:
        image_dir (str): Directory containing images
        max_workers (int): Maximum number of worker threads

    Returns:
        list: List of (image_filename, quality_score, error_message) tuples
    """
    # Get list of all image files
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                image_files.append(os.path.join(root, file))

    logging.info(f"Found {len(image_files)} image files")

    # Process a small sample first
    sample_size = min(5, len(image_files))
    logging.info(f"Testing with {sample_size} sample images first")
    for i in range(sample_size):
        result = calculate_simplified_brisque(image_files[i])
        logging.info(f"Sample {i + 1}: {result}")

    # Process images in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a mapping of futures to filenames
        future_to_file = {
            executor.submit(calculate_simplified_brisque, image_path): image_path
            for image_path in image_files
        }

        # Process as completed with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_file),
                           total=len(image_files),
                           desc="Processing images"):
            try:
                results.append(future.result())
            except Exception as e:
                file_path = future_to_file[future]
                error_detail = traceback.format_exc()
                logging.error(f"Exception processing {file_path}: {str(e)}\n{error_detail}")
                results.append((os.path.basename(file_path), None, str(e)))

    return results


def main():
    """
    Main function to process images and save results to CSV.
    """
    try:
        logging.info(f"Starting simplified BRISQUE analysis on {image_dir}")

        # Process all images and get scores
        results = process_images_parallel(image_dir)

        # Filter results
        valid_results = [(img, score) for img, score, error in results if score is not None]
        failed_images = [(img, error) for img, score, error in results if score is None]

        # Create DataFrame and sort by quality score (worse quality first)
        if valid_results:
            df = pd.DataFrame(valid_results, columns=['Image', 'Quality_Score'])
            df = df.sort_values('Quality_Score', ascending=False)
            df.to_csv(output_csv, index=False)

            logging.info(f"Analysis complete. Results saved to {output_csv}")
        else:
            logging.error("No valid results obtained. Check the log for details.")

        # Create error summary
        if failed_images:
            error_df = pd.DataFrame(failed_images, columns=['Image', 'Error'])
            error_df.to_csv("failed_images.csv", index=False)

            # Count error types
            error_counts = {}
            for _, error in failed_images:
                error_type = str(error).split(':')[0] if ':' in str(error) else str(error)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            logging.info("Error summary:")
            for error_type, count in error_counts.items():
                logging.info(f"  {error_type}: {count} images")

        # Print summary
        logging.info(f"Successfully processed {len(valid_results)} images")
        logging.info(f"Failed to process {len(failed_images)} images")

        print(f"Analysis complete. Results saved to {output_csv}")
        print(f"Successfully processed {len(valid_results)} images")
        print(f"Failed to process {len(failed_images)} images")

        # If we have failures but some successes, print statistics
        if valid_results:
            print("\nStatistics:")
            print(f"Average Quality score: {df['Quality_Score'].mean():.2f}")
            print(f"Worst quality image: {df.iloc[0]['Image']} (Score: {df.iloc[0]['Quality_Score']:.2f})")
            print(f"Best quality image: {df.iloc[-1]['Image']} (Score: {df.iloc[-1]['Quality_Score']:.2f})")
        elif failed_images:
            print("\nAll images failed. Top error types:")
            for error_type, count in list(error_counts.items())[:3]:
                print(f"  {error_type}: {count} images")
            print("Check 'failed_images.csv' and 'brisque_analysis.log' for details")

    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"Error in main function: {str(e)}\n{error_detail}")
        print(f"Error: {str(e)}")
        print("Check 'brisque_analysis.log' for details")


if __name__ == "__main__":
    main()
