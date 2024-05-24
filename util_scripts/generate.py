import os
from PIL import Image
from custom_pipe import FrozenCustomPipe
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor


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




def generate_image(index, impression, pipeline, output_folder):
    # Generate the image
    image = pipeline(impression).images[0]

    # Save the image
    image_path = f"{output_folder}/image_{index + 1}.png"
    image.save(image_path)
    print(f"Saved {image_path}")

def generate_images(csv_file, output_folder, start_index=0, num_threads=4):
    # Load the CSV file containing the image descriptions
    df = pd.read_csv(csv_file)

    # Initialize the pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = FrozenCustomPipe(path="/vol/ideadata/ce90tate/cxr_phrase_grounding/components").pipe
    pipeline.load_lora_weights("/vol/ideadata/ce90tate/cxr_phrase_grounding/finetune/lora/radbert/checkpoint-30000")
    pipeline = pipeline.to(device)

    # Use ThreadPoolExecutor to handle multiple threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks to the executor
        futures = []
        for index, row in df.iterrows():
            if index >= start_index:
                impression = row['impression']
                futures.append(executor.submit(generate_image, index, impression, pipeline, output_folder))

        # Wait for all futures to complete
        for future in futures:
            future.result()




# Example usage
#generate_images("/vol/ideadata/ce90tate/data/mimic/p19_5k_preprocessed_evenly.csv",
#               "/vol/ideadata/ce90tate/data/mimic/test/generated_images", start_index=2438)

#load_and_save_images("/vol/ideadata/ce90tate/data/mimic/p19_5k_preprocessed_evenly.csv",
#                    "/vol/ideadata/ce90tate/data/mimic/test/sample_images")

pipe = FrozenCustomPipe(path="/vol/ideadata/ce90tate/cxr_phrase_grounding/components").pipe
pipe.load_lora_weights("/vol/ideadata/ce90tate/cxr_phrase_grounding/finetune/lora/radbert/checkpoint-30000")
image = pipe("front view, pneumonia lower left lung", num_inference_steps=30, cross_attention_kwargs={"scale": 0.95}).images[0]
image.save("pneumonia_12.jpg")