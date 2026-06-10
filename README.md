# Text2Px

A lightweight **Diffusion Transformer (DiT)** model that generates 16x16 pixel art from text descriptions. Trained on Minecraft item textures.

## Architecture

Text2Px uses a custom DiT (Diffusion Transformer) architecture designed for tiny image generation:

- **Image Tokenization**: 16x16 RGBA images → 2x2 patches → 64 patch tokens
- **Text Conditioning**: Character-level tokenizer + small Transformer encoder → cross-attention
- **Diffusion Process**: Cosine noise schedule, epsilon prediction
- **Adaptive LayerNorm (AdaLN)**: Timestep-conditioned normalization in each block
- **Model Size**: ~9M parameters (lightweight, trainable on a single GPU or CPU)

```
Input Text → TextEncoder → Cross-Attention ↘
                                             DiT Blocks → Predicted Noise
Noisy Image → PatchEmbed + PosEmbed → AdaLN ↗
                                  ↑
                         Timestep Embedding
```

## Project Structure

```
text2px/
├── configs/
│   └── default.yaml         # Training configuration
├── model/
│   ├── __init__.py
│   ├── dit.py               # DiT model architecture
│   ├── diffusion.py         # Gaussian diffusion process
│   └── tokenizer.py         # Character-level tokenizer
├── data/
│   └── dataset.py           # Dataset loading
├── scripts/
│   ├── prepare_dataset.py   # Dataset preparation from Minecraft assets
│   ├── generate_synthetic_data.py  # Optional fallback toy dataset
│   ├── train.py             # Training script
│   ├── generate.py          # Inference / generation
│   └── web_ui.py            # Browser UI: text -> 16x16 pixel art
├── samples/                  # Generated samples during training
└── README.md
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

By default, the dataset script downloads the latest official Minecraft client jar from Mojang and extracts vanilla item textures:

```bash
python scripts/prepare_dataset.py
```

You can also pass a local Minecraft `.jar` file or extracted assets folder:

```bash
python scripts/prepare_dataset.py /path/to/minecraft.jar
python scripts/prepare_dataset.py /path/to/assets/minecraft/textures/item/
```

This creates `data/minecraft_items/` with images and labels.

### 3. Train

```bash
python scripts/train.py configs/default.yaml
```

If `data/minecraft_items/labels.json` is missing, `train.py` automatically downloads the latest official client jar and prepares the dataset first.

Training progress is saved to `checkpoints/` and sample generations to `samples/`.

### 4. Generate

```bash
python scripts/generate.py checkpoints/final_model.pt "diamond sword" "golden apple" "ender pearl"
```

Generated images are saved to `outputs/`.

### 5. Web UI

This repository includes a trained checkpoint at `checkpoints/final_model.pt`.

```bash
python scripts/web_ui.py checkpoints/final_model.pt 5000
```

Open `http://localhost:5000`, type text, and generate a 16x16 RGBA pixel item. The web path uses a DDIM sampler for interactive generation speed.

## Requirements

```
torch>=2.0
Pillow
numpy
pyyaml
tqdm
```

## Design Choices

- **16x16 resolution**: Matches Minecraft's native item texture size — no upscaling/downscaling needed
- **RGBA channels**: Preserves transparency information for proper item rendering
- **Character-level tokenizer**: Simple, no pretrained model dependency, sufficient for item names
- **Cosine schedule**: Better for small images than linear schedule
- **Small model**: Designed to train on a single consumer GPU in reasonable time
- **Cross-attention for text**: More expressive than class-conditional approaches, allows free-form descriptions

## Acknowledgments

Architecture inspired by:
- [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [PixArt-α: Fast Training of Diffusion Transformer](https://arxiv.org/abs/2310.00426)

## License

MIT
