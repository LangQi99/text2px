"""
Gaussian Diffusion process with cosine schedule.
"""
import torch
import torch.nn.functional as F
import math


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule="cosine"):
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        self.timesteps = timesteps
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_minus_one = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def training_loss(self, model, x_start, tokens, token_mask=None):
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t, tokens, token_mask)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, tokens, token_mask=None):
        betas_t = self._extract(self.betas.to(x.device), t, x.shape)
        sqrt_one_minus_alpha_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod.to(x.device), t, x.shape
        )
        sqrt_recip_alpha_t = self._extract(
            self.sqrt_recip_alphas_cumprod.to(x.device), t, x.shape
        )
        sqrt_recip_minus_one_t = self._extract(
            self.sqrt_recip_alphas_cumprod_minus_one.to(x.device), t, x.shape
        )

        pred_noise = model(x, t, tokens, token_mask)
        x_start_pred = sqrt_recip_alpha_t * x - sqrt_recip_minus_one_t * pred_noise
        x_start_pred = torch.clamp(x_start_pred, -1, 1)

        posterior_mean = (
            self._extract(self.alphas_cumprod_prev.to(x.device), t, x.shape).sqrt() * betas_t
            / (1 - self._extract(self.alphas_cumprod.to(x.device), t, x.shape))
        ) * x_start_pred + (
            self._extract(self.alphas.to(x.device), t, x.shape).sqrt()
            * (1 - self._extract(self.alphas_cumprod_prev.to(x.device), t, x.shape))
            / (1 - self._extract(self.alphas_cumprod.to(x.device), t, x.shape))
        ) * x

        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_var = self._extract(self.posterior_variance.to(x.device), t, x.shape)
            return posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            return posterior_mean

    @torch.no_grad()
    def sample(self, model, tokens, token_mask=None, image_size=16, channels=4, cfg_scale=0.0):
        device = next(model.parameters()).device
        batch_size = tokens.shape[0]
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)

        for t_val in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, tokens, token_mask)

        return x

    @torch.no_grad()
    def sample_ddim(self, model, tokens, token_mask=None, image_size=16, channels=4, steps=50, eta=0.0):
        """Fast deterministic DDIM sampler for interactive generation."""
        device = next(model.parameters()).device
        batch_size = tokens.shape[0]
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)

        steps = max(2, min(int(steps), self.timesteps))
        time_pairs = torch.linspace(self.timesteps - 1, 0, steps, device=device).long().tolist()

        for idx, t_val in enumerate(time_pairs):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            pred_noise = model(x, t, tokens, token_mask)

            alpha_t = self._extract(self.alphas_cumprod.to(device), t, x.shape)
            x_start = (x - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x_start = torch.clamp(x_start, -1, 1)

            if idx == len(time_pairs) - 1:
                x = x_start
                continue

            t_next = torch.full((batch_size,), time_pairs[idx + 1], device=device, dtype=torch.long)
            alpha_next = self._extract(self.alphas_cumprod.to(device), t_next, x.shape)

            sigma = eta * torch.sqrt(
                (1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)
            )
            c = torch.sqrt(torch.clamp(1 - alpha_next - sigma ** 2, min=0.0))
            noise = torch.randn_like(x) if eta > 0 else 0
            x = torch.sqrt(alpha_next) * x_start + c * pred_noise + sigma * noise

        return x
