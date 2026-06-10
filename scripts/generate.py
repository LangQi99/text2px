"""
Inference / Generation script for Text2Px.
Generate pixel art from text descriptions.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.dit import Text2PxDiT
from model.diffusion import GaussianDiffusion
from model.tokenizer import CharTokenizer


def generate(checkpoint_path, prompts, output_dir="outputs", scale_factor=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model_config = config['model'].copy()
    model_config['vocab_size'] = checkpoint['tokenizer_vocab_size']
    model = Text2PxDiT(model_config).to(device)

    state_key = 'ema_model_state_dict' if 'ema_model_state_dict' in checkpoint else 'model_state_dict'
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    tokenizer = CharTokenizer(max_len=config['model']['max_text_len'])
    tokenizer_path = os.path.join(config['data']['dataset_dir'], 'tokenizer.json')
    tokenizer.load(tokenizer_path)

    diffusion = GaussianDiffusion(
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    )

    os.makedirs(output_dir, exist_ok=True)

    tokens_list = []
    masks_list = []
    for text in prompts:
        tokens_list.append(torch.tensor(tokenizer.encode(text), dtype=torch.long))
        masks_list.append(torch.tensor(tokenizer.get_mask(text), dtype=torch.bool))

    tokens = torch.stack(tokens_list).to(device)
    masks = torch.stack(masks_list).to(device)

    print(f"Generating {len(prompts)} images...")
    samples = diffusion.sample(model, tokens, masks, image_size=16, channels=4)
    samples = (samples + 1) / 2.0
    samples = samples.clamp(0, 1)

    for i, (sample, text) in enumerate(zip(samples, prompts)):
        img_array = (sample.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='RGBA')
        img_upscaled = img.resize(
            (16 * scale_factor, 16 * scale_factor), Image.NEAREST
        )
        filename = f"{text.replace(' ', '_')}.png"
        img_upscaled.save(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate.py <checkpoint_path> <prompt1> [prompt2] ...")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    prompts = sys.argv[2:]
    generate(checkpoint_path, prompts)
