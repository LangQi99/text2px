import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.npy_dataset import denormalize_rgba
from diffusion import create_diffusion
from model.reference_dit import create_reference_dit


def encode_prompt(tokenizer, prompt, length):
    encoded = tokenizer.encode(prompt)
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    pad = tokenizer.token_to_id("<|pad|>")
    ids = [bos] + encoded.ids[: length - 2] + [eos]
    return ids + [pad] * (length - len(ids))


def save_contact_sheet(images, labels, path, scale=8):
    cell = 16 * scale
    label_h = 28
    sheet = Image.new("RGBA", (cell * len(images), cell + label_h), (24, 24, 24, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (img, label) in enumerate(zip(images, labels)):
        up = img.resize((cell, cell), Image.NEAREST)
        sheet.paste(up, (idx * cell, 0), up)
        draw.text((idx * cell + 4, cell + 6), label[:18], fill=(235, 235, 235, 255))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/reference/final-model.pt")
    parser.add_argument("--tokenizer", default="ref_artifacts/token-7524n.json")
    parser.add_argument("--prompts", nargs="+", default=["box"])
    parser.add_argument("--out-dir", default="outputs/reference")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    metadata = ckpt.get("metadata", {})
    model_name = args.model or metadata.get("model", "T2P-DiT-nano")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    token_length = int(metadata.get("token_length", 8))
    mean = metadata.get("rgba_mean")
    std = metadata.get("rgba_std")
    if mean is None or std is None:
        raise ValueError("Checkpoint metadata must include rgba_mean and rgba_std")

    model = create_reference_dit(
        model_name,
        vocab_size=metadata.get("vocab_size", tokenizer.get_vocab_size()),
        token_seq_len=token_length,
        num_timesteps=metadata.get("num_timestep", 300),
    ).to(device)
    key = "ema" if args.use_ema and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.eval()

    token_ids = torch.tensor(
        [encode_prompt(tokenizer, prompt, token_length) for prompt in args.prompts],
        dtype=torch.long,
        device=device,
    )
    noise = torch.randn(len(args.prompts), 4, 16, 16, device=device)
    z = torch.cat([noise, noise], dim=0)
    token_cfg = torch.cat([token_ids, torch.zeros_like(token_ids)], dim=0)
    diffusion = create_diffusion(str(metadata.get("num_timestep", 300)))
    model_kwargs = dict(token_ids=token_cfg, cfg_scale=args.cfg_scale)

    with torch.no_grad():
        if args.sampler == "ddim":
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )
        else:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )
    samples, _ = samples.chunk(2, dim=0)
    arrays = denormalize_rgba(samples, mean, std)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for prompt, arr in zip(args.prompts, arrays):
        img = Image.fromarray(arr, mode="RGBA")
        path = out_dir / f"{prompt.replace(' ', '_')}.png"
        img.save(path)
        images.append(img)
        print(path)
    save_contact_sheet(images, args.prompts, out_dir / "contact_sheet.png")
    print(out_dir / "contact_sheet.png")


if __name__ == "__main__":
    main()
