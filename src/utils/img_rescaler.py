import os
import sys
from PIL import Image


def resize_images_in_folder(folder_path, output_folder, max_size=(800, 800)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Scan the folder for image files
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        output_filepath = os.path.join(output_folder, filename)

        # Check if the file is an image
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Open the image
                img = Image.open(filepath)
                # Resize the image while preserving aspect ratio
                img.thumbnail(max_size)
                # Save the resized image
                img.save(output_filepath)
                print(f"Resized {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py input_folder [output_folder]")
        sys.exit(1)

    input_folder = sys.argv[1]
    if len(sys.argv) == 3:
        output_folder = sys.argv[2]
    else:
        output_folder = input_folder

    resize_images_in_folder(input_folder, output_folder)
