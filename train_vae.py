#!/usr/bin/env python3

from pathlib import Path
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
import pytorch_lightning as pl
import torchaudio
import wandb

from prefigure.prefigure import get_all_args, push_wandb_config
from dataset.dataset import SampleDataset
from audio_diffusion.vae1d import AudioVAE1D
from viz.viz import audio_spectrogram_image

def stft_magnitude(x, fft_size, hop_size, win_length):
    """
    x: [B, C, T]
    returns magnitude spectrogram
    """
    b, c, t = x.shape
    x = x.reshape(b * c, t)

    window = torch.hann_window(win_length, device=x.device)

    stft = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True,
    )

    return torch.abs(stft)


def multi_resolution_stft_loss(pred, target):
    """
    Multi-resolution STFT loss for waveform reconstruction.
    pred:   [B, 2, T]
    target: [B, 2, T]
    """

    resolutions = [
        (1024, 256, 1024),
        (2048, 512, 2048),
        (512, 128, 512),
    ]

    loss = 0.0

    for fft_size, hop_size, win_length in resolutions:
        pred_mag = stft_magnitude(pred, fft_size, hop_size, win_length)
        target_mag = stft_magnitude(target, fft_size, hop_size, win_length)

        spectral_convergence = torch.norm(
            target_mag - pred_mag,
            p="fro",
        ) / (torch.norm(target_mag, p="fro") + 1e-8)

        log_mag_loss = torch.mean(
            torch.abs(
                torch.log(target_mag + 1e-7) - torch.log(pred_mag + 1e-7)
            )
        )

        loss = loss + spectral_convergence + log_mag_loss

    return loss / len(resolutions)


class AudioVAEModule(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()
        self.save_hyperparameters()
        self.global_args = global_args

        self.vae = AudioVAE1D(
            in_channels=2,
            latent_channels=global_args.latent_channels,
            base_channels=global_args.vae_base_channels,
        )

        self.kl_weight = global_args.kl_weight
        self.stft_weight = global_args.vae_stft_weight
        self.lr = global_args.vae_lr

    def configure_optimizers(self):
        return optim.AdamW(self.vae.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=1e-4)

    def training_step(self, batch, batch_idx):
        audio = batch[0]

        recon, mu, logvar = self.vae(audio)

        l1_loss = F.l1_loss(recon, audio)
        stft_loss = multi_resolution_stft_loss(recon, audio)

        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        loss = l1_loss + self.stft_weight * stft_loss + self.kl_weight * kl_loss

        with torch.no_grad():
            recon_mse = F.mse_loss(recon, audio)
            latent_rms = torch.sqrt(torch.mean(mu ** 2))

        self.log_dict(
            {
                "vae/loss": loss.detach(),
                "vae/l1_loss": l1_loss.detach(),
                "vae/stft_loss": stft_loss.detach(),
                "vae/kl_loss": kl_loss.detach(),
                "vae/recon_mse": recon_mse.detach(),
                "vae/latent_rms": latent_rms.detach(),
            },
            prog_bar=True,
            on_step=True,
        )

        return loss


class VAEDemoCallback(pl.Callback):
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.sample_rate = global_args.sample_rate
        self.last_demo_step = -1

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0:
            return

        if self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        audio = batch[0][:1].to(module.device)
        recon, _, _ = module.vae(audio)

        original = audio[0].detach().clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        reconstructed = recon[0].detach().clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        original_file = f"vae_original_{trainer.global_step:08}.wav"
        recon_file = f"vae_recon_{trainer.global_step:08}.wav"

        torchaudio.save(original_file, original, self.sample_rate)
        torchaudio.save(recon_file, reconstructed, self.sample_rate)

        trainer.logger.experiment.log(
            {
                "vae/original_audio": wandb.Audio(original_file, sample_rate=self.sample_rate),
                "vae/reconstructed_audio": wandb.Audio(recon_file, sample_rate=self.sample_rate),
                "vae/recon_spectrogram": wandb.Image(audio_spectrogram_image(reconstructed)),
            },
            step=trainer.global_step,
        )


def main():
    args = get_all_args()

    save_path = None if args.save_path == "" else args.save_path

    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(
        train_set,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )

    wandb_logger = pl.loggers.WandbLogger(
        project=args.name,
        log_model="all" if args.save_wandb == "all" else None,
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every,
        save_top_k=-1,
        dirpath=save_path,
    )

    demo_callback = VAEDemoCallback(args)

    model = AudioVAEModule(args)

    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        precision=16,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[ckpt_callback, demo_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    trainer.fit(model, train_dl, ckpt_path=args.ckpt_path or None)


if __name__ == "__main__":
    main()