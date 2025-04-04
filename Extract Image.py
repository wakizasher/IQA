import pandas as pd
import requests
import os
from urllib.parse import urlparse

def download_images_with_labels(csv_file, image_column='image_url', scientific_name_column='scientific_name', output_folder='images'):
    """
    Downloads images from URLs in a CSV file, and saves them to a specified output folder.

    Args:
        csv_file (str): Path to the CSV file.
        image_column (str): Name of the column containing image URLs.
        scientific_name_column (str): Name of the column containing scientific names.
        output_folder (str): Path to the directory where images should be saved.
    """
    try:
        # 1. Read the CSV file using pandas
        df = pd.read_csv(csv_file)

        # 2. Create the output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 3. Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # 4. Get the image URL and scientific name from the current row
            image_url = row[image_column]
            scientific_name = row[scientific_name_column]

            try:
                # 5. Download the image
                response = requests.get(image_url, stream=True, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes

                # 6. Extract the file extension from the URL
                parsed_url = urlparse(image_url)
                file_extension = os.path.splitext(parsed_url.path)[1]
                if not file_extension:
                    file_extension = '.jpg'  # Default to .jpg if no extension found

                # 7. Sanitize the scientific name to create a valid filename
                safe_scientific_name = ''.join(c if c.isalnum() else '_' for c in scientific_name)

                # 8. Construct the filename with scientific name and unique index
                image_name = f"{safe_scientific_name}_{index}{file_extension}"
                filepath = os.path.join(output_folder, image_name)

                # 9. Save the image to the output folder
                with open(filepath, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

                # 10. Print a success message
                print(f"Downloaded and saved: {image_name}")

            except requests.exceptions.RequestException as e:
                # 11. Handle download errors
                print(f"Error downloading image from {image_url}: {e}")
            except Exception as e:
                # 12. Handle other errors
                print(f"An unexpected error occurred: {e}")

    except FileNotFoundError:
        # 13. Handle the case where the CSV file is not found
        print(f"Error: CSV file not found at {csv_file}")
    except KeyError as e:
        # 14. Handle the case where a required column is missing
        print(f"Error: Column not found in CSV file: {e}")
    except Exception as e:
        # 15. Handle any other exceptions
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # 16. Specify the path to your CSV file
    csv_file_path = 'plants.csv'  # Replace with your CSV file path

    # 17. Specify the path to your external drive (where you want to save the images)
    external_drive_path = r"D:\iNaturalist\images"

    # 18. Call the function to download and save the images
    download_images_with_labels(csv_file_path,
                               image_column='image_url',  # Make sure this matches your CSV column name
                               scientific_name_column='scientific_name',  # Make sure this matches your CSV column name
                               output_folder=external_drive_path)

    print("Image downloading process completed.")

