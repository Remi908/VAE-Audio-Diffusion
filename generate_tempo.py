import argparse

import torch
import torchaudio

from train_uncond import sample
from audio_diffusion.models import DiffusionAttnUnet1D
from audio_diffusion.vae1d import AudioVAE1D
from dataset.tempo_features import make_tempo_condition


def load_vae(
    vae_ckpt_path,
    latent_channels,
    vae_base_channels,
):
    vae = AudioVAE1D(
        in_channels=2,
        latent_channels=latent_channels,
        base_channels=vae_base_channels,
    )

    ckpt = torch.load(
        vae_ckpt_path,
        map_location="cpu",
    )

    state_dict = ckpt["state_dict"]

    vae_state_dict = {
        k.replace("vae.", ""): v
        for k, v in state_dict.items()
        if k.startswith("vae.")
    }

    if len(vae_state_dict) == 0:
        raise RuntimeError(
            "No VAE weights found. Expected keys starting with 'vae.'."
        )

    vae.load_state_dict(vae_state_dict)
    vae.eval().cuda()

    return vae


def load_diffusion(
    diffusion_ckpt_path,
    latent_channels,
    latent_dim,
):
    class Args:
        pass

    args = Args()
    args.latent_dim = latent_dim

    model = DiffusionAttnUnet1D(
        args,
        io_channels=latent_channels,
        depth=6,
        n_attn_layers=3,
    )

    ckpt = torch.load(
        diffusion_ckpt_path,
        map_location="cpu",
    )

    ema_state_dict = {
        k.replace("diffusion_ema.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("diffusion_ema.")
    }

    if len(ema_state_dict) == 0:
        raise RuntimeError(
            "No EMA diffusion weights found. Expected keys starting with 'diffusion_ema.'."
        )

    model.load_state_dict(ema_state_dict)
    model.eval().cuda()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diffusion-ckpt",
        required=True,
    )

    parser.add_argument(
        "--vae-ckpt",
        required=True,
    )

    parser.add_argument(
        "--target-bpm",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=65536,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--latent-channels",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--vae-base-channels",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--min-bpm",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--max-bpm",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--out-prefix",
        default="generated_tempo",
    )

    args = parser.parse_args()

    latent_dim = 1
    latent_downsample_factor = 16
    latent_samples = args.sample_size // latent_downsample_factor

    vae = load_vae(
        args.vae_ckpt,
        latent_channels=args.latent_channels,
        vae_base_channels=args.vae_base_channels,
    )

    diffusion = load_diffusion(
        args.diffusion_ckpt,
        latent_channels=args.latent_channels,
        latent_dim=latent_dim,
    )

    noise = torch.randn(
        [
            args.num_samples,
            args.latent_channels,
            latent_samples,
        ],
        device="cuda",
    )

    tempo_vec = make_tempo_condition(
        args.target_bpm,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
    )

    tempo_vec = torch.tensor(
        tempo_vec,
        dtype=torch.float32,
        device="cuda",
    )

    tempo_vec = tempo_vec[None, :].repeat(
        args.num_samples,
        1,
    )

    tempo_cond = tempo_vec[:, :, None].expand(
        -1,
        -1,
        latent_samples,
    )

    with torch.no_grad():
        latent_fakes = sample(
            diffusion,
            noise,
            steps=args.steps,
            eta=args.eta,
            cond=tempo_cond,
            guidance_scale=args.guidance_scale,
        )

        audio = vae.decode(latent_fakes)
        audio = audio.clamp(-1, 1).cpu()

    for i in range(args.num_samples):
        path = (
            f"{args.out_prefix}_"
            f"{int(args.target_bpm)}bpm_"
            f"{i+1}.wav"
        )

        torchaudio.save(
            path,
            audio[i],
            args.sample_rate,
        )

        print(f"Saved: {path}")


if __name__ == "__main__":
    main()