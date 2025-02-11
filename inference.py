from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel
)
import numpy as np
import torch

class FewStepPipeline:
    def __init__(
        self,
        unet_path,
        model_id="stabilityai/stable-diffusion-2-1-base",
        device="cuda",
        weight_dtype=torch.float16,
    ):
        self.device = device
        self.weight_dtype = weight_dtype

        unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder='unet').eval()
        unet = unet.to(device, dtype=weight_dtype)
        unet.requires_grad_(False)

        noise_scheduler = DDIMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id, unet=unet, scheduler=noise_scheduler, torch_dtype=weight_dtype, safety_checker=None
        ).to(device)

    def __call__(
        self,
        prompt, 
        batch_size,
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=0
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        images = self.pipeline(
            prompt=prompt,
            num_images_per_prompt=batch_size,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images

        return images


# Helper functions
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_path", type=str, default="reg-sbv2-reparam-dists-recon")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--prompt", type=str, default="a photo of a cat")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

if __name__ == "__main__":
    import os
    from PIL import Image

    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    pipe = FewStepPipeline(unet_path=args.unet_path)
    
    images = pipe(
        args.prompt,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )

    grid = image_grid(images, 1, args.batch_size)
    grid.save(f"{args.save_dir}/{args.prompt}.jpg")
