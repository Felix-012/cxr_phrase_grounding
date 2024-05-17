import pandas as pd
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
from radbert_pipe import FrozenCustomPipe
import os


def load_and_save_images(csv_file, output_folder):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image path in the DataFrame
    for index, row in df.iterrows():
        image_path = row['path']  # assuming the column name that contains the image paths is 'rel_path'
        try:
            # Load the image
            with Image.open(image_path) as img:
                # Define the new path for the image
                # This example keeps the original filename but you can modify this logic if needed
                _, filename = os.path.split(image_path)
                new_path = os.path.join(output_folder, filename)

                # Save the image to the new location
                img.save(new_path)
                print(f"Image saved to {new_path}")
        except IOError:
            print(f"Error occurred while opening or saving the image from {image_path}")


def generate_images(csv_file, output_folder):
    # Load the CSV file containing the image descriptions
    df = pd.read_csv(csv_file)

    # Initialize the pipeline, make sure to replace `model_id` with your actual model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = FrozenCustomPipe().pipe
    pipeline.load_lora_weights("/vol/ideadata/ce90tate/cxr_phrase_grounding/finetune/lora/radbert/checkpoint-23000")
    pipeline = pipeline.to(device)

    # Generate an image for each entry in the DataFrame
    for index, row in df.iterrows():
        # Assume that the column containing the text descriptions is named 'description'
        impression = row['impression']

        # Generate the image
        image = pipeline(impression).images[0]

        # Save the image
        image_path = f"{output_folder}/image_{index + 1}.png"
        image.save(image_path)
        print(f"Saved {image_path}")


# Example usage
#generate_images("/vol/ideadata/ce90tate/data/mimic/p19_5k_preprocessed_evenly.csv",
 #               "/vol/ideadata/ce90tate/data/mimic/test/generated_images")

load_and_save_images("/vol/ideadata/ce90tate/data/mimic/p19_5k_preprocessed_evenly.csv",
                     "/vol/ideadata/ce90tate/data/mimic/test/sample_images")
