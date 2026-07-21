#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys

import torch
import matplotlib.pyplot as plt
from torch.utils import data

from prefigure.prefigure import get_all_args
from dataset.dataset import SampleDataset
from train_uncond import DiffusionUncond, get_alphas_sigmas, get_crash_schedule


def extract_epoch_from_ckpt(ckpt_path, ckpt):
    """
    Prefer checkpoint['epoch'].
    If missing, try to parse epoch from filename.
    """
    if "epoch" in ckpt:
        return int(ckpt["epoch"])

    filename = os.path.basename(ckpt_path)

    match = re.search(r"epoch[=\-_](\d+)", filename)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))

    return -1


def load_model_from_checkpoint(ckpt_path, args, device):
    model = DiffusionUncond(args)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    epoch = extract_epoch_from_ckpt(ckpt_path, ckpt)

    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"Epoch: {epoch}")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()

    return model, epoch


@torch.no_grad()
def analyze_noise_by_timestep(model, dataloader, device, num_batches=20, num_bins=10):
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1)

    actual_sums = torch.zeros(num_bins)
    pred_sums = torch.zeros(num_bins)
    mse_sums = torch.zeros(num_bins)
    counts = torch.zeros(num_bins)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        reals = batch[0].to(device)

        # waveform -> VAE latent
        mu, logvar = model.vae.encode(reals)
        latents = mu

        batch_size = latents.shape[0]

        # evenly spread raw timesteps across [0.01, 0.99]
        t_raw = torch.linspace(
            0.01,
            0.99,
            batch_size,
            device=device,
        )

        t_raw = t_raw[torch.randperm(batch_size, device=device)]

        # same schedule used in training
        t = get_crash_schedule(t_raw)

        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]

        actual_noise = torch.randn_like(latents)

        noised_latents = latents * alphas + actual_noise * sigmas

        pred_v = model.diffusion(noised_latents, t)

        # Convert predicted velocity to predicted noise:
        # eps = x_t * sigma + v * alpha
        pred_noise = noised_latents * sigmas + pred_v * alphas

        actual_noise_rms = torch.sqrt(torch.mean(actual_noise ** 2, dim=(1, 2)))
        pred_noise_rms = torch.sqrt(torch.mean(pred_noise ** 2, dim=(1, 2)))
        noise_mse = torch.mean((pred_noise - actual_noise) ** 2, dim=(1, 2))

        for i in range(num_bins):
            left = bin_edges[i].to(device)
            right = bin_edges[i + 1].to(device)

            if i == num_bins - 1:
                mask = (t_raw >= left) & (t_raw <= right)
            else:
                mask = (t_raw >= left) & (t_raw < right)

            if mask.any():
                actual_sums[i] += actual_noise_rms[mask].detach().cpu().sum()
                pred_sums[i] += pred_noise_rms[mask].detach().cpu().sum()
                mse_sums[i] += noise_mse[mask].detach().cpu().sum()
                counts[i] += mask.detach().cpu().sum()

        print(f"Processed batch {batch_idx + 1}/{num_batches}")

    valid = counts > 0

    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).numpy()

    actual_avg = torch.zeros(num_bins)
    pred_avg = torch.zeros(num_bins)
    mse_avg = torch.zeros(num_bins)

    actual_avg[valid] = actual_sums[valid] / counts[valid]
    pred_avg[valid] = pred_sums[valid] / counts[valid]
    mse_avg[valid] = mse_sums[valid] / counts[valid]

    return {
        "bin_centers": bin_centers,
        "actual_noise_rms": actual_avg.numpy(),
        "pred_noise_rms": pred_avg.numpy(),
        "noise_mse": mse_avg.numpy(),
        "counts": counts.numpy(),
    }


def summarize_timestep_results(results):
    counts = torch.tensor(results["counts"])
    actual = torch.tensor(results["actual_noise_rms"])
    pred = torch.tensor(results["pred_noise_rms"])
    mse = torch.tensor(results["noise_mse"])

    valid = counts > 0

    return {
        "actual_noise_rms": actual[valid].mean().item(),
        "pred_noise_rms": pred[valid].mean().item(),
        "noise_mse": mse[valid].mean().item(),
    }


def plot_timestep_results(results, output_path):
    x = results["bin_centers"]

    plt.figure(figsize=(10, 6))
    plt.plot(x, results["actual_noise_rms"], marker="o", label="Actual noise RMS")
    plt.plot(x, results["pred_noise_rms"], marker="o", label="Predicted noise RMS")
    plt.xlabel("Raw diffusion timestep t")
    plt.ylabel("Noise RMS")
    plt.title("Predicted Noise vs Actual Noise across Timesteps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print("Saved timestep noise plot to:", output_path)

    mse_path = output_path.replace(".png", "_mse.png")

    plt.figure(figsize=(10, 6))
    plt.plot(x, results["noise_mse"], marker="o", label="Noise MSE")
    plt.xlabel("Raw diffusion timestep t")
    plt.ylabel("MSE")
    plt.title("Predicted Noise MSE across Timesteps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mse_path, dpi=200)
    print("Saved timestep MSE plot to:", mse_path)


def find_checkpoints(ckpt_dir):
    patterns = [
        os.path.join(ckpt_dir, "*.ckpt"),
        os.path.join(ckpt_dir, "**", "*.ckpt"),
    ]

    ckpts = []
    for pattern in patterns:
        ckpts.extend(glob.glob(pattern, recursive=True))

    ckpts = sorted(list(set(ckpts)))

    if len(ckpts) == 0:
        raise RuntimeError(f"No .ckpt files found in {ckpt_dir}")

    return ckpts


def analyze_noise_by_epoch(ckpt_paths, args, dataloader, device, num_batches=5, num_bins=10):
    epoch_results = []

    for ckpt_path in ckpt_paths:
        model, epoch = load_model_from_checkpoint(ckpt_path, args, device)

        results = analyze_noise_by_timestep(
            model,
            dataloader,
            device,
            num_batches=num_batches,
            num_bins=num_bins,
        )

        summary = summarize_timestep_results(results)

        epoch_results.append(
            {
                "epoch": epoch,
                "ckpt_path": ckpt_path,
                "actual_noise_rms": summary["actual_noise_rms"],
                "pred_noise_rms": summary["pred_noise_rms"],
                "noise_mse": summary["noise_mse"],
            }
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    epoch_results = sorted(epoch_results, key=lambda x: x["epoch"])

    return epoch_results


def plot_epoch_results(epoch_results, output_path):
    epochs = [x["epoch"] for x in epoch_results]
    actual = [x["actual_noise_rms"] for x in epoch_results]
    pred = [x["pred_noise_rms"] for x in epoch_results]
    mse = [x["noise_mse"] for x in epoch_results]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, actual, marker="o", label="Actual noise RMS")
    plt.plot(epochs, pred, marker="o", label="Predicted noise RMS")
    plt.xlabel("Epoch")
    plt.ylabel("Noise RMS")
    plt.title("Predicted Noise vs Actual Noise across Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print("Saved epoch noise plot to:", output_path)

    mse_path = output_path.replace(".png", "_mse.png")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mse, marker="o", label="Noise MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Noise Prediction MSE across Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(mse_path, dpi=200)
    print("Saved epoch MSE plot to:", mse_path)

    csv_path = output_path.replace(".png", ".csv")

    with open(csv_path, "w") as f:
        f.write("epoch,actual_noise_rms,pred_noise_rms,noise_mse,ckpt_path\n")
        for row in epoch_results:
            f.write(
                f"{row['epoch']},"
                f"{row['actual_noise_rms']},"
                f"{row['pred_noise_rms']},"
                f"{row['noise_mse']},"
                f"{row['ckpt_path']}\n"
            )

    print("Saved epoch results CSV to:", csv_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=str, default="defaults.ini")

    # Single-checkpoint plot: noise vs timestep
    parser.add_argument("--ckpt-path", type=str, default=None)

    # Multi-checkpoint plot: noise vs epoch
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--ckpt-glob", type=str, default=None)

    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--num-bins", type=int, default=10)

    parser.add_argument("--output-timestep", type=str, default="noise_vs_timestep.png")
    parser.add_argument("--output-epoch", type=str, default="noise_vs_epoch.png")

    parser.add_argument("--cpu", action="store_true")

    cli_args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0], "--config-file", cli_args.config_file]
    args = get_all_args()
    args.latent_dim = 0

    device = torch.device(
        "cpu" if cli_args.cpu or not torch.cuda.is_available() else "cuda"
    )

    print("Using device:", device)

    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args)

    train_dl = data.DataLoader(
        train_set,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ---------------------------------
    # Plot noise vs timestep
    # ---------------------------------
    if cli_args.ckpt_path is not None:
        model, epoch = load_model_from_checkpoint(
            cli_args.ckpt_path,
            args,
            device,
        )

        results = analyze_noise_by_timestep(
            model,
            train_dl,
            device,
            num_batches=cli_args.num_batches,
            num_bins=cli_args.num_bins,
        )

        print("Counts per timestep bin:", results["counts"])

        plot_timestep_results(
            results,
            cli_args.output_timestep,
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------
    # Plot noise vs epoch
    # ---------------------------------
    if cli_args.ckpt_dir is not None or cli_args.ckpt_glob is not None:
        if cli_args.ckpt_glob is not None:
            ckpt_paths = sorted(glob.glob(cli_args.ckpt_glob))
        else:
            ckpt_paths = find_checkpoints(cli_args.ckpt_dir)

        print(f"Found {len(ckpt_paths)} checkpoints")

        epoch_results = analyze_noise_by_epoch(
            ckpt_paths,
            args,
            train_dl,
            device,
            num_batches=cli_args.num_batches,
            num_bins=cli_args.num_bins,
        )

        for row in epoch_results:
            print(
                f"epoch={row['epoch']}, "
                f"actual={row['actual_noise_rms']:.4f}, "
                f"pred={row['pred_noise_rms']:.4f}, "
                f"mse={row['noise_mse']:.4f}"
            )

        plot_epoch_results(
            epoch_results,
            cli_args.output_epoch,
        )

    if cli_args.ckpt_path is None and cli_args.ckpt_dir is None and cli_args.ckpt_glob is None:
        raise RuntimeError(
            "Please provide --ckpt-path for timestep plot, "
            "or --ckpt-dir / --ckpt-glob for epoch plot."
        )


if __name__ == "__main__":
    main()