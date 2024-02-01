import os
from PIL import Image
import csv

def resize_image(image_path, output_size=(28, 28)):
    img = Image.open(image_path).convert('L')  # 'L' for grayscale
    img = img.resize(output_size, Image.LANCZOS)
    return img

def process_folder(folder_path, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header to CSV file
        csv_writer.writerow(['label'] + [f'pixel_{i}' for i in range(28 * 28)])

        # Create a mapping from labels to numeric values
        label_mapping = {}
        numeric_label = 0

        # Process each subfolder
        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if os.path.isdir(label_path):
                label_mapping[label] = numeric_label
                numeric_label += 1

                for filename in os.listdir(label_path):
                    image_path = os.path.join(label_path, filename)
                    
                    # Resize the image to 28x28 pixels
                    resized_image = resize_image(image_path)

                    # Extract pixels and flatten the image
                    pixels = list(resized_image.getdata())

                    # Write label and pixels to CSV file
                    csv_writer.writerow([label_mapping[label]] + pixels)

if __name__ == "__main__":
    dataset_folder = "Numbers"  # Replace with your dataset folder path
    output_csv_file = "dataset_numbers.csv"  # Replace with your desired output CSV file path

    process_folder(dataset_folder, output_csv_file)
