import argparse
import json
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.npy_dataset import NpyRGBA16Dataset
from diffusion import create_diffusion
from model.reference_dit import REFERENCE_DIT_CONFIGS, create_reference_dit


def update_ema(ema_model, model, decay=0.9999):
    with torch.no_grad():
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for param in model.parameters():
        param.requires_grad = flag


def encode_prompt(tokenizer, prompt, length):
    encoded = tokenizer.encode(prompt)
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    pad = tokenizer.token_to_id("<|pad|>")
    ids = [bos] + encoded.ids[: length - 2] + [eos]
    return ids + [pad] * (length - len(ids))


def select_indices(names, include, limit):
    if not include:
        return np.arange(min(limit, len(names))) if limit else np.arange(len(names))
    keywords = [word.lower() for word in include]
    selected = [
        idx for idx, name in enumerate(names)
        if any(word in str(name).lower() for word in keywords)
    ]
    if limit:
        selected = selected[:limit]
    return np.array(selected, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ref_artifacts")
    parser.add_argument("--img-data", default=None, help="Override img data npy file")
    parser.add_argument("--token-ids", default=None, help="Override token ids npy file")
    parser.add_argument("--names", default=None, help="Override names npy file")
    parser.add_argument("--tokenizer", default=None, help="Override tokenizer json file")
    parser.add_argument("--model", choices=list(REFERENCE_DIT_CONFIGS), default="T2P-DiT-mini")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=3000)
    parser.add_argument("--subset-limit", type=int, default=0)
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--out-dir", default="checkpoints/reference")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir)
    token_ids = np.load(args.token_ids or root / "imgIndexIds-8l-95t.npy")
    img_data = np.load(args.img_data or root / "img16data-19180n.npy")
    names = np.load(args.names or root / "img16index-19180n.npy")
    tokenizer_path = args.tokenizer or root / "token-7524n.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    indices = select_indices(names, args.include, args.subset_limit)
    if len(indices) == 0:
        raise ValueError(f"No dataset rows matched include={args.include}")

    dataset = NpyRGBA16Dataset(token_ids[indices], img_data[indices])
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) >= args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = create_reference_dit(
        args.model,
        vocab_size=tokenizer.get_vocab_size(),
        token_seq_len=8,
        num_timesteps=300,
    ).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)

    diffusion = create_diffusion(timestep_respacing=[300])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    os.makedirs(args.out_dir, exist_ok=True)

    metadata = {
        "model": args.model,
        "vocab_size": tokenizer.get_vocab_size(),
        "token_length": 8,
        "rgba_mean": dataset.mean.tolist(),
        "rgba_std": dataset.std.tolist(),
        "num_timestep": 300,
        "dataset_size": len(dataset),
        "include": args.include,
        "tokenizer_file": str(tokenizer_path),
    }
    with open(Path(args.out_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"device={device}")
    print(f"dataset={len(dataset)} model_params={sum(p.numel() for p in model.parameters()):,}")
    print(f"examples={[str(names[i]) for i in indices[:10]]}")

    train_steps = 0
    started = time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for token_batch, x in loader:
            token_batch = token_batch.to(device)
            x = x.permute(0, 3, 1, 2).to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, dict(token_ids=token_batch))
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            train_steps += 1
            running += loss.item()
            if train_steps % args.log_every == 0:
                print(
                    f"step={train_steps:06d} epoch={epoch:03d} "
                    f"loss={running / args.log_every:.4f} sec={time() - started:.1f}",
                    flush=True,
                )
                running = 0.0

            if train_steps % args.ckpt_every == 0:
                save_checkpoint(args.out_dir, f"ckpt-step-{train_steps:07d}.pt", model, ema, opt, args, metadata, train_steps, epoch)

    save_checkpoint(args.out_dir, "final-model.pt", model, ema, opt, args, metadata, train_steps, args.epochs)
    print(f"saved {Path(args.out_dir) / 'final-model.pt'}")


def save_checkpoint(out_dir, name, model, ema, opt, args, metadata, train_steps, epoch):
    torch.save(
        {
            "train_steps": train_steps,
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "metadata": metadata,
            "vocab_size": metadata["vocab_size"],
        },
        Path(out_dir) / name,
    )


if __name__ == "__main__":
    main()
