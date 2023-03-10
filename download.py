# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from diffusers import StableDiffusionInstructPix2PixPipeline
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16);

if __name__ == "__main__":
    download_model()
