"""
Text2Px: A lightweight Diffusion Transformer for text-to-pixel-art generation.
Generates 16x16 RGBA pixel art from text descriptions.
"""
import torch
import torch.nn as nn
import math


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, num_heads=4, num_layers=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, mask=None):
        x = self.token_embed(tokens) + self.pos_embed[:, :tokens.shape[1]]
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embed)


class AdaLN(nn.Module):
    def __init__(self, hidden_size, cond_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj = nn.Linear(cond_size, hidden_size * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, text_dim):
        super().__init__()
        self.adaln1 = AdaLN(hidden_size, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.adaln2 = AdaLN(hidden_size, hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size) if text_dim != hidden_size else nn.Identity()

    def forward(self, x, cond, text_features, text_mask=None):
        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h

        text_k = self.text_proj(text_features)
        h = self.cross_norm(x)
        h, _ = self.cross_attn(h, text_k, text_k, key_padding_mask=text_mask)
        x = x + h

        h = self.adaln2(x, cond)
        h = self.mlp(h)
        x = x + h
        return x


class Text2PxDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config['image_size']
        self.patch_size = config['patch_size']
        self.in_channels = config['in_channels']
        self.hidden_size = config['hidden_size']

        num_patches = (self.image_size // self.patch_size) ** 2
        patch_dim = self.patch_size * self.patch_size * self.in_channels

        self.patch_embed = nn.Linear(patch_dim, self.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_size))

        self.time_embed = TimestepEmbedding(self.hidden_size)

        self.text_encoder = TextEncoder(
            vocab_size=config['vocab_size'],
            embed_dim=config['text_embed_dim'],
            max_len=config['max_text_len'],
        )

        self.blocks = nn.ModuleList([
            DiTBlock(
                self.hidden_size, config['num_heads'],
                config['mlp_ratio'], config['text_embed_dim']
            ) for _ in range(config['depth'])
        ])

        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, patch_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h = w = self.image_size // p
        C = self.in_channels
        x = x.reshape(-1, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(-1, C, self.image_size, self.image_size)
        return x

    def forward(self, x, t, tokens, token_mask=None):
        x = self.patchify(x)
        x = self.patch_embed(x) + self.pos_embed

        t_emb = self.time_embed(t)
        text_features = self.text_encoder(tokens, mask=token_mask)

        for block in self.blocks:
            x = block(x, t_emb, text_features, text_mask=token_mask)

        x = self.final_norm(x)
        x = self.out_proj(x)
        x = self.unpatchify(x)
        return x
