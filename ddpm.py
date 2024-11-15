import torch
import numpy as np


class DDPMSampler:
    def __init__(
        self,
        generator,
        num_training_steps=1000,
        beta_start=0.00085,
        beta_end=0.0120,
    ):
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )  # Scaled Linear Schedule

        # For Multi Steps
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)

        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(
            np.arange(0, num_training_steps)[::-1].copy()
        )  # From 999 to 0

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        # Since we only step 50 times, we space the time steps that
        # one step = step_ratio *  time steps
        step_ratio = self.num_training_steps // num_inference_steps

        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep):
        step_ratio = self.num_training_steps // self.num_inference_steps
        prev_t = timestep - step_ratio
        return prev_t

    def _get_variance(self, timestep):
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = variance.clamp(min=1e-20)
        return variance

    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep, latent, model_output):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 1. Compute the predicted original sample
        pred_original_sample = (
            latent - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5

        # 2. Compute the coefficients from pred_original_sample and current x_t
        pred_original_sample_coeff = (
            alpha_prod_t_prev**0.5 * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        # 3. Compute the predicted previous sample mean
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latent
        )

        # 4. Get Variance
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )
            variance = (self._get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(self, original_sample, timesteps):
        device = original_sample.device
        d_type = original_sample.dtype
        alpha_cumprod = self.alpha_cumprod.to(device, dtype=d_type)
        timesteps = timesteps.to(device)

        sqrt_alpha_prod = torch.sqrt(alpha_cumprod[timesteps])
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()  # Why need flat
        while len(sqrt_alpha_prod.shape) < len(original_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (
            1 - alpha_cumprod[timesteps]
        ) ** 0.5  # Standard Deviation
        sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(
            original_sample.shape, generator=self.generator, device=device, dtype=d_type
        )
        noisy_samples = (
            sqrt_alpha_prod * original_sample + (sqrt_one_minus_alpha_prod) * noise
        )
        return noisy_samples
