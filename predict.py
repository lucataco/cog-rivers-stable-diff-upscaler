from cog import BasePredictor, Input, Path
import PIL
import time
import torch
import tempfile
import numpy as np
from PIL import Image
import k_diffusion as K
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF
# Local files
from make_upscaler import make_upscaler_model
from text_encoder import CFGUpscaler, CLIPEmbedder, CLIPTokenizerTransform
from utils import download_from_huggingface, load_model_from_config

device = torch.device("cuda")
cached_vae = None

def condition_up(prompts):
    device = torch.device("cuda")
    tok_up = CLIPTokenizerTransform()
    text_encoder_up = CLIPEmbedder(device=device)
    return text_encoder_up(tok_up(prompts))

def run(seed):
    timestamp = int(time.time())
    if not seed:
        print('No seed was provided, using the current time.')
        seed = timestamp
    print(f'Generating with seed={seed}')
    seed_everything(seed)

def do_sample(noise, sigma_max, sigma_min, steps, device, sampler, model_wrap, tol_scale, eta, extra_args):
    # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
    sigmas = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps+1).exp().to(device)
    if sampler == 'k_euler':
        return K.sampling.sample_euler(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args)
    elif sampler == 'k_euler_ancestral':
        return K.sampling.sample_euler_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
    elif sampler == 'k_dpm_2_ancestral':
        return K.sampling.sample_dpm_2_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
    elif sampler == 'k_dpm_fast':
        return K.sampling.sample_dpm_fast(model_wrap, noise * sigma_max, sigma_min, sigma_max, steps, extra_args=extra_args, eta=eta)
    elif sampler == 'k_dpm_adaptive':
        sampler_opts = dict(s_noise=1., rtol=tol_scale * 0.05, atol=tol_scale / 127.5, pcoeff=0.2, icoeff=0.4, dcoeff=0)
        return K.sampling.sample_dpm_adaptive(model_wrap, noise * sigma_max, sigma_min, sigma_max, extra_args=extra_args, eta=eta, **sampler_opts)


class Predictor(BasePredictor):
    def setup(self):
        global cached_vae
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = make_upscaler_model('config_laion_text_cond_latent_upscaler_2.json', 'laion_text_cond_latent_upscaler_2_1_00470000_slim.pth')
        vae_560k_model_path = download_from_huggingface("stabilityai/sd-vae-ft-ema-original", "vae-ft-ema-560000-ema-pruned.ckpt")
        vae_model_560k = load_model_from_config("latent-diffusion/models/first_stage_models/kl-f8/config.yaml", vae_560k_model_path)
        self.vae_model_560k = vae_model_560k.to(device)
        self.model = self.model.to("cuda")

    # The arguments and types the model takes as input
    def predict(self,
        image: Path = Input(description="Input image"),
        prompt: str = "the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas",
        guidance_scale: float = Input(description="Guidance scale (0 - 10)", ge=0, le=10, default=1),
        noise_aug_type: str = Input(description="Sampler to use", choices=["gaussian", "fake"], default='gaussian'),
        noise_aug_level: float = Input(description="Amount of noise to augment", ge=0, le=1, default=0),
        sampler: str = Input(description="Sampler to use", choices=["k_euler", "k_euler_ancestral", "k_dpm_2_ancestral", "k_dpm_fast", "k_dpm_adaptive"], default='k_dpm_adaptive'),
        steps: int = Input(description=" # inference steps", ge=0, le=100, default=50),
        tol_scale: float = Input(description="k_dpm_adaptive uses an an adaptive solver with error tolerance tol_scale, all other use a fixed # of steps", ge=0, le=10, default=0.25),
        eta: float = Input(description="Amount of noise to add per step (0-deterministic). Used in all samples except k_euler", ge=0, le=10, default=1),
        seed: int = Input(description="Seed (0 = random, maximum: 2147483647)", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        num_samples = 1
        batch_size = 1
        SD_Q = 0.18215
        # Noise levels from stable diffusion.
        sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512
        # Set the input image
        input_image = Image.open(str(image)).convert("RGB")
        uc = condition_up(batch_size * [""])
        c = condition_up(batch_size * [prompt])
        vae = self.vae_model_560k
        image = TF.to_tensor(input_image).to(device) * 2 - 1
        low_res_latent = vae.encode(image.unsqueeze(0)).sample() * SD_Q
        [_, C, H, W] = low_res_latent.shape
        model_wrap = CFGUpscaler(self.model, uc, cond_scale=guidance_scale)
        low_res_sigma = torch.full([batch_size], noise_aug_level, device=device)
        x_shape = [batch_size, C, 2*H, 2*W]
        output_path = ''
        for _ in range((num_samples-1)//batch_size + 1):
            if noise_aug_type == 'gaussian':
                latent_noised = low_res_latent + noise_aug_level * torch.randn_like(low_res_latent)
            elif noise_aug_type == 'fake':
                latent_noised = low_res_latent * (noise_aug_level ** 2 + 1)**0.5
            extra_args = {'low_res': latent_noised, 'low_res_sigma': low_res_sigma, 'c': c}
            noise = torch.randn(x_shape, device=device)
            up_latents = do_sample(noise, sigma_max, sigma_min, steps, device, sampler, model_wrap, tol_scale, eta, extra_args)
            pixels = vae.decode(up_latents/SD_Q)
            pixels = pixels.add(1).div(2).clamp(0,1)
            # Save img
            img = TF.to_pil_image(make_grid(pixels, batch_size))
            output_path = Path(tempfile.mkdtemp()) / "output.png"
            img.save(output_path)

        run(seed)
        return  Path(output_path)