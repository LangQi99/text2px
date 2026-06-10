"""
Reference-style Text-to-Pixel DiT.

This mirrors the important design choices from LangQi99/Txt2PixelDiT:
- fixed 2D sin-cos position embeddings
- token sequence encoder with classifier-free token dropout
- timestep + text embedding summed into a single condition vector
- AdaLN-Zero gated DiT blocks
- optional learned sigma output for OpenAI-style diffusion losses
"""
import math

import numpy as np
import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=16, patch_size=2, in_chans=4, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, seq_len=8, hidden_size=384, dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.null_token_emb = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def token_drop(self, token_ids):
        if self.dropout_prob <= 0:
            return token_ids
        drop_mask = torch.rand(token_ids.shape[0], device=token_ids.device) < self.dropout_prob
        return torch.where(drop_mask.unsqueeze(1), torch.zeros_like(token_ids), token_ids)

    def forward(self, token_ids, train=True):
        if train:
            token_ids = self.token_drop(token_ids)
        token_emb = self.token_emb(token_ids)
        null_mask = (token_ids == 0).all(dim=1)
        token_emb = torch.where(
            null_mask[:, None, None],
            self.null_token_emb.expand_as(token_emb),
            token_emb,
        )
        encoded = self.token_encoder(token_emb)
        return encoded.mean(dim=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=128, max_period=1000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class ReferenceTxt2ImgDiT(nn.Module):
    def __init__(
        self,
        vocab_size,
        token_seq_len=8,
        input_size=16,
        patch_size=2,
        in_channels=4,
        hidden_size=384,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        token_dropout_prob=0.1,
        learn_sigma=True,
        num_timesteps=300,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.input_size = input_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, max_period=num_timesteps)
        self.token_embedder = TokenEmbedder(
            vocab_size=vocab_size,
            seq_len=token_seq_len,
            hidden_size=hidden_size,
            dropout_prob=token_dropout_prob,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, hidden_size),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.token_embedder.token_emb.weight, std=0.02)
        nn.init.normal_(self.token_embedder.null_token_emb, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        channels = self.out_channels
        patch = self.patch_size
        height = width = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], height, width, patch, patch, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], channels, height * patch, width * patch)

    def forward(self, x, t, token_ids):
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.token_embedder(token_ids, self.training)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final_layer(x, c))

    def forward_with_cfg(self, x, t, token_ids, cfg_scale=3.0):
        half = x[: len(x) // 2]
        combined_x = torch.cat([half, half], dim=0)
        combined_t = torch.cat([t[: len(t) // 2], t[: len(t) // 2]], dim=0)
        null_token_ids = torch.zeros_like(token_ids[: len(token_ids) // 2])
        combined_token_ids = torch.cat([token_ids[: len(token_ids) // 2], null_token_ids], dim=0)
        model_out = self.forward(combined_x, combined_t, combined_token_ids)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([guided, guided], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


REFERENCE_DIT_CONFIGS = {
    "T2P-DiT-nano": dict(hidden_size=128, depth=4, num_heads=4),
    "T2P-DiT-tiny": dict(hidden_size=256, depth=6, num_heads=4),
    "T2P-DiT-mini": dict(hidden_size=384, depth=8, num_heads=8),
    "T2P-DiT-small": dict(hidden_size=512, depth=10, num_heads=8),
}


def create_reference_dit(model_name, vocab_size, **kwargs):
    config = REFERENCE_DIT_CONFIGS[model_name].copy()
    config.update(kwargs)
    return ReferenceTxt2ImgDiT(vocab_size=vocab_size, **config)
