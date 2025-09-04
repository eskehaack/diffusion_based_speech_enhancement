from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from scipy.io import wavfile

from .unet import UNet1D
from .noise_scheduler import NoiseScheduler


class SoundDataset(Dataset):
    def __init__(
        self, directory, desired_sampling_rate=8000, max_length=None, audio_length=8000
    ):
        self.files = list(Path(directory).rglob("*.wav"))
        self.sr = desired_sampling_rate
        self.max_length = max_length
        self.audio_length = audio_length

        self._preprocess()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

    def _preprocess(self):
        batch = []
        for i, file in enumerate(self.files):
            # Set max limit of files
            if i >= self.max_length:
                break

            # Read soundfile
            sr, aud = wavfile.read(file)
            # Downsample
            aud = aud[:: int(sr / self.sr)]
            # Change datatype
            aud = np.array(aud).astype(np.float32)
            # Standardize
            aud = (aud - aud.mean()) / aud.std()
            # Pad
            if len(aud) < 8000:
                aud = np.pad(aud, (0, self.audio_length - len(aud)))
            else:
                aud = aud[: self.audio_length]
            # Add to batch
            batch.append(aud)

        self.X = torch.stack([torch.from_numpy(x) for x in batch])


class KLGaussian:
    # KL(N(mu_pred, exp(logvar_pred)) || N(0, 1))

    def get_loss(
        self, mean_pred: float, logvar_pred: float, epsilon_target: torch.Tensor
    ):
        assert (
            mean_pred.shape == logvar_pred.shape == epsilon_target.shape
        ), f"{mean_pred.shape},{logvar_pred.shape},{epsilon_target.shape}"
        return (
            0.5
            * (
                logvar_pred.exp()
                + (mean_pred - epsilon_target) ** 2
                - 1.0
                - logvar_pred
            ).mean()
        )


# assuming hydra
def reconstruct(
    log_dir_str: str, device: str = "cuda"
) -> tuple[NoiseScheduler, UNet1D]:
    log_dir = Path(log_dir_str)
    cfg = yaml.safe_load(open(log_dir / ".hydra" / "config.yaml", "r"))
    ns = NoiseScheduler(
        **dict(
            filter(lambda x: x[0] != "_target_", cfg["noise_scheduler"].items()),
            device=device,
        )
    )
    model = UNet1D(**dict(filter(lambda x: x[0] != "_target_", cfg["model"].items())))
    model.load_state_dict(torch.load(log_dir / "params.pt"))
    return ns, model


@torch.no_grad()
def denoise(log_dir_str: str, noisy_data=np.ndarray) -> list[torch.Tensor]:
    ns, model = reconstruct(log_dir_str, device="cpu")
    model.eval()
    x_last = torch.tensor(noisy_data.astype(np.float32))
    samples = [x_last]
    for t in reversed(range(ns.num_timesteps)):
        t = ns.num_timesteps - 1
        mean, logvar = model(samples[-1], torch.tensor(t))
        std = torch.exp(0.5 * logvar)
        residual = mean + std * torch.randn_like(mean)
        samples.append(ns.remove_noise(samples[-1], residual, t))
    return samples


def load_single_audio(file: str) -> np.ndarray:
    desired_sampling_rate = 8_000
    sr, aud = wavfile.read(file)
    aud = aud[:: int(sr / desired_sampling_rate)].astype(np.float64)
    aud = (aud - aud.mean()) / aud.std()
    aud = np.pad(aud, (0, 8000 - len(aud)))
    return aud
