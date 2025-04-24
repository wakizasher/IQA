import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import logging
import traceback
from skimage import io
from skimage.util import img_as_float
from scipy.ndimage import gaussian_filter
from scipy import linalg
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='niqe_analysis.log'
)

# Directory containing the images
image_dir = r"D:\iNaturalist\test_images"
# Output CSV file path
output_csv = "image_quality_niqe.csv"


def extract_image_patches(img, patch_size=96, stride=8):
    """
    Extract patches from image for NIQE calculation.

    Args:
        img: Grayscale image
        patch_size: Size of square patches
        stride: Stride between patches

    Returns:
        list: List of patch arrays
    """
    h, w = img.shape
    patches = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i + patch_size, j:j + patch_size]
            # Skip low contrast patches
            if np.std(patch) < 0.01:
                continue
            patches.append(patch)

    return patches


def estimate_gaussian_parameters(patches):
    """
    Estimate multivariate Gaussian parameters from patches.

    Args:
        patches: List of image patches

    Returns:
        tuple: (mean_vector, covariance_matrix)
    """
    features = []

    for patch in patches:
        # Compute local mean and variance
        mu = gaussian_filter(patch, sigma=7 / 6)
        sigma = np.sqrt(np.abs(gaussian_filter(patch ** 2, sigma=7 / 6) - mu ** 2))

        # Normalize patch
        normalized = (patch - mu) / (sigma + 1e-7)

        # Simple features: mean, variance, skewness, kurtosis
        mean_val = np.mean(normalized)
        var_val = np.var(normalized)
        skew_val = np.mean((normalized - mean_val) ** 3) / (var_val ** 1.5 + 1e-12)
        kurt_val = np.mean((normalized - mean_val) ** 4) / (var_val ** 2 + 1e-12) - 3

        # Add these basic features
        feature = [mean_val, var_val, skew_val, kurt_val]
        features.append(feature)

    if not features:
        return None, None

    features_array = np.array(features)
    mean_vector = np.mean(features_array, axis=0)
    cov_matrix = np.cov(features_array, rowvar=False)

    return mean_vector, cov_matrix


def calculate_niqe(image_path):
    """
    Calculate a simplified NIQE score for a single image.

    Args:
        image_path (str): Path to the image file

    Returns:
        tuple: (image_filename, niqe_score, error_message)
    """
    try:
        # Read image with scikit-image
        img = io.imread(image_path)

        # Convert to grayscale if needed
        if img.ndim == 3 and img.shape[2] == 3:
            gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        elif img.ndim == 2:
            gray = img
        else:
            return os.path.basename(image_path), None, f"Unsupported image format: {img.ndim}"

        # Normalize to [0,1]
        gray_norm = img_as_float(gray)

        # Check if image is large enough
        min_size = 96  # Minimum size for patch extraction
        if gray_norm.shape[0] < min_size or gray_norm.shape[1] < min_size:
            return os.path.basename(image_path), None, f"Image too small: {gray_norm.shape}"

        # Extract patches
        patches = extract_image_patches(gray_norm)
        if not patches:
            return os.path.basename(image_path), None, "No valid patches extracted"

        # Estimate parameters from patches
        mean_vector, cov_matrix = estimate_gaussian_parameters(patches)
        if mean_vector is None:
            return os.path.basename(image_path), None, "Failed to estimate parameters"

        # Reference parameters for pristine natural images
        # These are simplified for illustration - in a real implementation,
        # they would be computed from a database of pristine images
        pristine_mean = np.array([0.0, 0.1, 0.0, 3.0])  # Example values
        pristine_cov = np.eye(4) * 0.1

        # Calculate NIQE score (Mahalanobis distance between pristine and distorted model)
        diff = mean_vector - pristine_mean
        invcov = linalg.pinv((pristine_cov + cov_matrix) / 2)
        quality_score = np.sqrt(diff.dot(invcov).dot(diff.T))

        return os.path.basename(image_path), float(quality_score), None
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
        result = calculate_niqe(image_files[i])
        logging.info(f"Sample {i + 1}: {result}")

    # Process images in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a mapping of futures to filenames
        future_to_file = {
            executor.submit(calculate_niqe, image_path): image_path
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
        logging.info(f"Starting simplified NIQE analysis on {image_dir}")

        # Process all images and get scores
        results = process_images_parallel(image_dir)

        # Filter results
        valid_results = [(img, score) for img, score, error in results if score is not None]
        failed_images = [(img, error) for img, score, error in results if score is None]

        # Create DataFrame and sort by quality score (worse quality first)
        if valid_results:
            df = pd.DataFrame(valid_results, columns=['Image', 'NIQE_Score'])
            df = df.sort_values('NIQE_Score', ascending=False)
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
            print(f"Average NIQE score: {df['NIQE_Score'].mean():.2f}")
            print(f"Worst quality image: {df.iloc[0]['Image']} (Score: {df.iloc[0]['NIQE_Score']:.2f})")
            print(f"Best quality image: {df.iloc[-1]['Image']} (Score: {df.iloc[-1]['NIQE_Score']:.2f})")
        elif failed_images:
            print("\nAll images failed. Top error types:")
            for error_type, count in list(error_counts.items())[:3]:
                print(f"  {error_type}: {count} images")
            print("Check 'failed_images.csv' and 'niqe_analysis.log' for details")

    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"Error in main function: {str(e)}\n{error_detail}")
        print(f"Error: {str(e)}")
        print("Check 'niqe_analysis.log' for details")


if __name__ == "__main__":
    main()