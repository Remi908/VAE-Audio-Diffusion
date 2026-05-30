import torch
from train_uncond import sample
from audio_diffusion.models import DiffusionAttnUnet1D
import torchaudio
from copy import deepcopy

# === Settings ===
#logs/remi_audio_run/version_0/checkpoints/epoch=43-step=1000.ckpt"
checkpoint_path = "/data/Remi/HarmonAI/sample-generator-main-1/checkpoint/epoch=249-step=10000.ckpt"
sample_rate = 16000
sample_size = 65536
num_samples = 1
steps = 100
eta = 0.0  # set to 1.0 for stochastic sampling

# === Dummy args (only needed to build model)
class Args:
    latent_dim = 0
global_args = Args()

# === Load model architecture
model = DiffusionAttnUnet1D(global_args, io_channels=2, n_attn_layers=4)

# === Load EMA weights from checkpoint
ckpt = torch.load(checkpoint_path, map_location="cpu")
ema_state_dict = {
    k.replace("diffusion_ema.", ""): v
    for k, v in ckpt["state_dict"].items()
    if k.startswith("diffusion_ema.")
}
model.load_state_dict(ema_state_dict)
model.eval().cuda()

def spectral_gate_torch(wav, sr, n_fft=2048, hop=512,
                        noise_quantile=0.10, strength_db=10.0, floor_db=-35.0):
    """
    wav: [C, T] float tensor on CPU
    Returns: [C, T] denoised
    """
    # STFT
    window = torch.hann_window(n_fft, device=wav.device)
    X = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window,
                   return_complex=True)  # [C, F, Frames]
    mag = X.abs()
    phase = X / (mag + 1e-8)

    # Pick quiet frames by energy
    frame_energy = mag.mean(dim=1).mean(dim=0)  # [Frames]
    thresh = torch.quantile(frame_energy, noise_quantile)
    quiet = frame_energy <= thresh

    # Noise estimate per freq bin (median over quiet frames)
    noise_mag = mag[:, :, quiet].median(dim=-1, keepdim=True).values + 1e-8  # [C,F,1]

    # Convert params
    strength = 10 ** (strength_db / 20.0)
    floor = 10 ** (floor_db / 20.0)

    # Soft mask
    ratio = mag / (noise_mag * strength)
    mask = ratio / (1.0 + ratio)            # smooth 0..1
    mask = floor + (1.0 - floor) * mask     # never fully zero

    Y = (mag * mask) * phase
    y = torch.istft(Y, n_fft=n_fft, hop_length=hop, window=window, length=wav.shape[-1])

    return y.clamp(-1, 1)

# Optional: keep your HPF (doesn't remove hiss, but safe for rumble)
def highpass(wav, sr, cutoff=30.0):
    return torchaudio.functional.highpass_biquad(wav, sr, cutoff)

# === Generate samples
with torch.no_grad():
    noise = torch.randn([num_samples, 2, sample_size]).cuda()
    samples = sample(model, noise, steps=steps, eta=eta).clamp(-1, 1).cpu()
    samples = samples.float().cpu()

# === Save output
for i in range(num_samples):
    raw = samples[i]  # [2, T] on CPU

    # Filtered version
    wav = highpass(raw, sample_rate, 30.0)
    wav = spectral_gate_torch(
        wav, sample_rate,
        n_fft=2048, hop=512,
        noise_quantile=0.10,
        strength_db=8.0,
        floor_db=-30.0
    )

    raw_path = f"generated_sample_{i+1}_raw.wav"
    filt_path = f"generated_sample_{i+1}_filtered.wav"

    torchaudio.save(raw_path, raw, sample_rate)
    torchaudio.save(filt_path, wav, sample_rate)

    print(f"✅ Saved: {raw_path}")
    print(f"✅ Saved: {filt_path}")
