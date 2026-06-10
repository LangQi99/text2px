"""
Training script for Text2Px model.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.dit import Text2PxDiT
from model.diffusion import GaussianDiffusion
from model.tokenizer import CharTokenizer
from data.dataset import MinecraftItemDataset
from scripts.prepare_dataset import download_client_jar, extract_from_jar


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ema_update(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def save_samples(model, diffusion, tokenizer, epoch, device, save_dir):
    model.eval()
    sample_texts = ["diamond sword", "iron pickaxe", "golden apple", "ender pearl"]
    os.makedirs(save_dir, exist_ok=True)

    tokens_list = []
    masks_list = []
    for text in sample_texts:
        tokens_list.append(torch.tensor(tokenizer.encode(text), dtype=torch.long))
        masks_list.append(torch.tensor(tokenizer.get_mask(text), dtype=torch.bool))

    tokens = torch.stack(tokens_list).to(device)
    masks = torch.stack(masks_list).to(device)

    samples = diffusion.sample(model, tokens, masks, image_size=16, channels=4)
    samples = (samples + 1) / 2.0
    samples = samples.clamp(0, 1)

    from PIL import Image
    import numpy as np
    for i, (sample, text) in enumerate(zip(samples, sample_texts)):
        img_array = (sample.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='RGBA')
        img_upscaled = img.resize((64, 64), Image.NEAREST)
        img_upscaled.save(os.path.join(save_dir, f"epoch{epoch}_{text.replace(' ', '_')}.png"))

    model.train()


def train():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = config['data']['dataset_dir']
    labels_path = os.path.join(data_dir, 'labels.json')
    if not os.path.exists(labels_path):
        print(f"Dataset not found at {data_dir}; downloading latest official Minecraft client jar.")
        jar_path = download_client_jar("release")
        labels = extract_from_jar(jar_path, data_dir)
        with open(labels_path, 'w') as f:
            import json
            json.dump(labels, f, indent=2, ensure_ascii=False)
        print(f"Prepared {len(labels)} Minecraft item textures.")

    tokenizer = CharTokenizer(max_len=config['model']['max_text_len'])

    import json
    with open(labels_path) as f:
        labels = json.load(f)
    tokenizer.fit(labels.values())
    tokenizer.save(os.path.join(data_dir, 'tokenizer.json'))

    model_config = config['model'].copy()
    model_config['vocab_size'] = tokenizer.vocab_size
    model = Text2PxDiT(model_config).to(device)
    ema_model = copy.deepcopy(model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    diffusion = GaussianDiffusion(
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    )

    dataset = MinecraftItemDataset(
        root_dir=data_dir,
        tokenizer=tokenizer,
        image_size=config['data']['image_size'],
        augment=config['data'].get('augment', False),
    )
    drop_last = len(dataset) > config['training']['batch_size'] * 2
    dataloader = DataLoader(
        dataset, batch_size=min(config['training']['batch_size'], len(dataset)),
        shuffle=True, num_workers=0, pin_memory=True, drop_last=drop_last
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    save_dir = "checkpoints"
    sample_dir = "samples"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Starting training for {config['training']['epochs']} epochs...")

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            images, tokens, masks = batch
            images = images.to(device)
            tokens = tokens.to(device)
            masks = masks.to(device)

            loss = diffusion.training_loss(model, images, tokens, masks)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()

            ema_update(ema_model, model, config['training']['ema_decay'])

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch}/{config['training']['epochs']} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if epoch % config['training']['sample_every'] == 0:
            save_samples(ema_model, diffusion, tokenizer, epoch, device, sample_dir)

        if epoch % config['training']['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'tokenizer_vocab_size': tokenizer.vocab_size,
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt"))
            torch.save(checkpoint, os.path.join(save_dir, "latest.pt"))

    print("Training complete!")
    final_checkpoint = {
        'model_state_dict': ema_model.state_dict(),
        'config': config,
        'tokenizer_vocab_size': tokenizer.vocab_size,
    }
    torch.save(final_checkpoint, os.path.join(save_dir, "final_model.pt"))


if __name__ == "__main__":
    train()
