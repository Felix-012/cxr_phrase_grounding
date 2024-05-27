import argparse
import os
from PIL import Image
from torchvision.utils import save_image

from custom_pipe import FrozenCustomPipe
import pandas as pd
import torch
import torch.distributed as dist



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




def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def generate_images_distributed(csv_file, output_folder, start_index=0, rank=0, world_size=1):
    # Setup distributed processing
    setup(rank, world_size)



    # Load the CSV file containing the image descriptions
    df = pd.read_csv(csv_file)

    # Initialize and distribute the pipeline
    device = f'cuda:{rank}'
    pipeline = FrozenCustomPipe(path="/vol/ideadata/ce90tate/cxr_phrase_grounding/components").pipe
    pipeline.load_lora_weights("/vol/ideadata/ce90tate/cxr_phrase_grounding/finetune/lora/radbert/checkpoint-30000")
    pipeline.to(device)

    # Generate an image for each entry in the DataFrame starting from the specified index
    for index, row in enumerate(df.itertuples(), start=1):
        if index < start_index or index % world_size != rank:
            continue  # Skip rows that do not align with this GPU's turn

        # Assume that the column containing the text descriptions is named 'description'
        impression = getattr(row, 'impression')

        # Generate the image
        image = pipeline(impression).images[0]

        # Save the image
        image_path = f"{output_folder}/image_{index + 1}.png"
        image.save(image_path)
        print(f"Saved {image_path}")

    # Cleanup distributed processing
    cleanup()




def main():
    parser = argparse.ArgumentParser(description="Distributed Image Generation")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing image descriptions.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder where images will be saved.')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for image generation.')
    parser.add_argument('--local_rank', type=int, help='Local rank passed by torch.distributed.launch.')

    args = parser.parse_args()

    # Assuming world_size is set by the environment, we can fetch it
    world_size = torch.cuda.device_count()

    generate_images_distributed(args.csv_file, args.output_folder, args.start_index, int(os.environ['LOCAL_RANK']), world_size)

if __name__ == '__main__':
    main()

#load_and_save_images("/vol/ideadata/ce90tate/data/mimic/p19_5k_preprocessed_evenly.csv",
#                    "/vol/ideadata/ce90tate/data/mimic/test/sample_images")

# pipe = FrozenCustomPipe(path="/vol/ideadata/ce90tate/cxr_phrase_grounding/components").pipe
# pipe.load_lora_weights("/vol/ideadata/ce90tate/cxr_phrase_grounding/finetune/lora/radbert/checkpoint-30000")
# image = pipe("front view, pneumonia lower left lung", num_inference_steps=30, cross_attention_kwargs={"scale": 0.95}).images[0]
# image.save("pneumonia_12.jpg")