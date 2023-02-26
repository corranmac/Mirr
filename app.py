from diffusers import UniPCMultistepScheduler, StableDiffusionInstructPix2PixPipeline, StableDiffusionImg2ImgPipeline,EulerAncestralDiscreteScheduler
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global pipe
    
    device = 0 if torch.cuda.is_available() else -1
    pix2pix=True
    model_ckpt = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.safety_checker=None


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', "")
    guidance = model_inputs.get('guidance', None)
    image_guidance = model_inputs.get('image_guidance', None)
    steps = mode_inputs.get('steps',5)
    image = mode_inputs.get('image',None)
    seed = mode_inputs.get('seed',42)
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    img = Image.open(io.BytesIO(file.read()))

    output = pipe(prompt,image,steps, guidance_scale,image_guidance_scale,generator).images[0] #PIL image

    # Return the results as a dictionary
    return output
