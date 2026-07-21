#!/usr/bin/env python3
from dataset.tempo_features import make_tempo_condition
from audio_diffusion.vae1d import AudioVAE1D
from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from einops import rearrange
import torchaudio
import wandb

from dataset.dataset import SampleDataset

from audio_diffusion.models import DiffusionAttnUnet1D
from audio_diffusion.utils import ema_update
from viz.viz import audio_spectrogram_image


# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(
    model,
    x,
    steps,
    eta,
    cond=None,
    guidance_scale=1.0,
):
    """
    Draw samples from a model given starting noise.

    cond shape:
        [B, condition_channels, latent_time]
    """
    ts = x.new_ones([x.shape[0]])

    t = torch.linspace(
        1,
        0,
        steps + 1,
        device=x.device,
    )[:-1]

    t = get_crash_schedule(t)

    alphas, sigmas = get_alphas_sigmas(t)

    for i in trange(steps):
        current_t = ts * t[i]

        with torch.cuda.amp.autocast():
            if cond is not None and guidance_scale != 1.0:
                v_uncond = model(
                    x,
                    current_t,
                    cond=torch.zeros_like(cond),
                ).float()

                v_cond = model(
                    x,
                    current_t,
                    cond=cond,
                ).float()

                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            else:
                v = model(
                    x,
                    current_t,
                    cond=cond,
                ).float()

        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        if i < steps - 1:
            ddim_sigma = (
                eta
                * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt()
                * (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
            )

            adjusted_sigma = (
                sigmas[i + 1] ** 2
                - ddim_sigma ** 2
            ).sqrt()

            x = (
                pred * alphas[i + 1]
                + eps * adjusted_sigma
            )

            if eta:
                x += torch.randn_like(x) * ddim_sigma

    return pred



class DiffusionUncond(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.global_args = global_args
        self.rng = torch.quasirandom.SobolEngine(
        1,
        scramble=True,
        seed=global_args.seed
        )
        self.ema_decay = global_args.ema_decay

    # -----------------------------
    # Load trained VAE
    # -----------------------------
        self.vae = AudioVAE1D(
        in_channels=2,
        latent_channels=global_args.latent_channels,
        base_channels=global_args.vae_base_channels,
       )

        ckpt = torch.load(global_args.vae_ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        vae_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("vae."):
                vae_state_dict[key.replace("vae.", "")] = value

        if len(vae_state_dict) == 0:
            raise RuntimeError(
            "No VAE weights found in checkpoint. Expected keys starting with 'vae.'. "
            "Make sure vae_ckpt_path points to a checkpoint created by train_vae.py."
            )

        self.vae.load_state_dict(vae_state_dict)
        self.vae.eval()

        for p in self.vae.parameters():
            p.requires_grad = False

    # -----------------------------
    # Latent diffusion model
    # -----------------------------
        self.diffusion = DiffusionAttnUnet1D(
        global_args,
        io_channels=global_args.latent_channels,
        depth=6,
        n_attn_layers=3,
       )

    # -----------------------------
    # EMA copy of diffusion model
    # -----------------------------
        self.diffusion_ema = deepcopy(self.diffusion)

        for p in self.diffusion_ema.parameters():
            p.requires_grad = False

        self.diffusion_ema.eval()
        
    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=4e-5)
  
    def training_step(self, batch, batch_idx):
        reals = batch[0]

        # tempo_vec = None

        # # If tempo conditioning is enabled, batch must contain:
        # # batch[0] = audio
        # # batch[1] = tempo condition, shape [B, 1]
        # if getattr(self.global_args, "use_tempo_conditioning", False):
        #     if len(batch) < 3:
        #         raise RuntimeError(
        #             "Tempo conditioning is enabled, but dataset did not return "
        #             "(audio, tempo_cond, audio_filename). Check dataset.py."
        #         )

        #     tempo_vec = batch[1].to(self.device)
        tempo_vec = None
        tempo_cond = None
        tempo_enabled = getattr(self.global_args, "latent_dim", 0) > 0

        if tempo_enabled:
            if len(batch) < 3:
                raise RuntimeError(
                    "Tempo conditioning is enabled because latent_dim > 0, "
                    "but dataset did not return (audio, tempo_cond, audio_filename). "
                    "Check dataset.py and defaults.ini."
                )

            tempo_vec = batch[1].to(self.device)

        with torch.no_grad():
            mu, logvar = self.vae.encode(reals)
            latents = mu

        t = self.rng.draw(latents.shape[0])[:, 0].to(self.device)
        t = get_crash_schedule(t)

        alphas, sigmas = get_alphas_sigmas(t)

        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]

        noise = torch.randn_like(latents)

        noised_latents = latents * alphas + noise * sigmas

        targets = noise * alphas - latents * sigmas

        tempo_cond = None

        if tempo_vec is not None:
            # Optional classifier-free guidance dropout
            drop_prob = getattr(self.global_args, "cond_drop_prob", 0.1)

            if drop_prob > 0:
                keep_mask = (
                    torch.rand(
                        tempo_vec.shape[0],
                        1,
                        device=self.device,
                    )
                    > drop_prob
                ).float()

                tempo_vec = tempo_vec * keep_mask

            # [B, 1] -> [B, 1, latent_time]
            tempo_cond = tempo_vec[:, :, None].expand(
                -1,
                -1,
                noised_latents.shape[-1],
            )

        # Safety check
        if getattr(self.global_args, "latent_dim", 0) > 0 and tempo_cond is None:
            raise RuntimeError(
                "Model was created with latent_dim > 0, but tempo_cond is None. "
                "The UNet expects tempo conditioning but did not receive it."
            )

        with torch.cuda.amp.autocast():
            v = self.diffusion(
                noised_latents,
                t,
                cond=tempo_cond,
            )

            mse_loss = F.mse_loss(v, targets)
            loss = mse_loss

        self.log_dict(
            {
                "train/loss": loss.detach(),
                "train/mse_loss": mse_loss.detach(),
            },
            prog_bar=True,
            on_step=True,
        )

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.diffusion, self.diffusion_ema, decay)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.num_demos = global_args.num_demos
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.sample_rate = global_args.sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

    # VAE downsamples by 16:
    # waveform [B, 2, sample_size] -> latent [B, latent_channels, sample_size // 16]
        latent_samples = self.demo_samples // 16

        noise = torch.randn(
        [
            self.num_demos,
            module.global_args.latent_channels,
            latent_samples,
        ],
        device=module.device,
        )
        
        try:
        # Sample in latent space
            target_bpm = getattr(
            module.global_args,
            "demo_bpm",
            90.0,
            )

            tempo_cond = None

            if getattr(module.global_args, "use_tempo_conditioning", False):
                tempo_vec = make_tempo_condition(
                target_bpm,
                min_bpm=module.global_args.min_bpm,
                max_bpm=module.global_args.max_bpm,
            )

            tempo_vec = torch.tensor(
            tempo_vec,
            dtype=torch.float32,
            device=module.device,
            )

            tempo_vec = tempo_vec[None, :].repeat(
            self.num_demos,
            1,
            )

            tempo_cond = tempo_vec[:, :, None].expand(
            -1,
            -1,
            latent_samples,
            )
            latent_fakes = sample(
            module.diffusion_ema,
            noise,
            self.demo_steps,
            0,
            cond=tempo_cond,
            guidance_scale=1.0,
            )

        # Decode latent audio back to waveform
            fakes = module.vae.decode(latent_fakes)

        # Put demos together into one long stereo audio file
            fakes = rearrange(fakes, "b d n -> d (b n)")

            log_dict = {}

            filename = f"demo_{trainer.global_step:08}.wav"

            fakes = fakes.clamp(-1, 1)
            fakes = fakes.mul(32767).to(torch.int16).cpu()

            torchaudio.save(filename, fakes, self.sample_rate)

            log_dict["demo"] = wandb.Audio(
            filename,
            sample_rate=self.sample_rate,
            caption="Demo",
            )

            log_dict["demo_melspec_left"] = wandb.Image(
            audio_spectrogram_image(fakes)
            )

            trainer.logger.experiment.log(
            log_dict,
            step=trainer.global_step
           )

        except Exception as e:
            print(f"{type(e).__name__}: {e}", file=sys.stderr)

def main():

    args = get_all_args()

    # if getattr(args, "use_tempo_conditioning", False):
    args.latent_dim = 1
    # else:
    #     args.latent_dim = 0

    save_path = None if args.save_path == "" else args.save_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name, log_model='all' if args.save_wandb=='all' else None)

    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1, dirpath=save_path)
    demo_callback = DemoCallback(args)

    diffusion_model = DiffusionUncond(args)

    wandb_logger.watch(diffusion_model)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        # num_nodes = args.num_nodes,
        # strategy='ddp',
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=150000,
    )

    diffusion_trainer.fit(diffusion_model, train_dl, ckpt_path=args.ckpt_path or None)

if __name__ == '__main__':
    main()

