import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print(f'pyTorch version: {torch.__version__}')
NOISE_CHANNELS = 1
NOISE_SCALE = 1.5
LABEL_SMOOTH = 0.9   # Fixed: Discriminator sees real targets as 0.9

SPEC_H = 20
SPEC_W = 29

# don't need this in final version
DATASET_SIZE = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
#  LIGHTWEIGHT ARCHITECTURE (LAPTOP EDITION)
# ============================================================================

def weights_init(m):
    name = m.__class__.__name__
    if 'Conv' in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class LightweightCNNGenerator(nn.Module):
    """
    A very fast, shallow CNN that keeps the spatial dimensions intact.
    Uses only 16 feature maps to keep memory and compute extremely low.
    """

    def __init__(self, in_channels, out_channels=1, features=16):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, features, kernel_size=3, padding="same"),
            nn.BatchNorm2d(features),
            nn.GELU(),

            # Layer 2 (Hidden)
            nn.Conv2d(features, features, kernel_size=3, padding="same"),
            nn.BatchNorm2d(features),
            nn.GELU(),

            # Layer 3 (Output)
            nn.Conv2d(features, out_channels, kernel_size=3, padding="same"),
            nn.Sigmoid()  # Bound to [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = LightweightCNNGenerator(in_channels=1, out_channels=1)

    def forward(self, x_noisy):
        return self.net(x_noisy)


def _weights_path(filename):
    """Resolve weights path. Checks CWD, ML_models/ subdirs, then alongside this module."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    candidates = [
        filename,
        os.path.join('ML_models', filename),
        os.path.join(project_root, 'ML_models', filename),
        os.path.join(project_root, filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return filename


WEIGHTS_BY_BAND_SIDE = {
    ('K', 'HFS'): 'v2.2_K_HFS_1500.pth',
}


class MLdenoising():
    def __init__(self, band, side):
        self.band = band
        self.side = side
        weights_name = WEIGHTS_BY_BAND_SIDE.get((band, side))
        if weights_name is None:
            raise FileNotFoundError(
                f"No ML denoiser weights available for band={band}, side={side}"
            )
        self.denoiser = Denoiser().to(DEVICE)
        weights = _weights_path(weights_name)
        major, minor = map(int, torch.__version__.split('+')[0].split('.')[:2])
        if (major, minor) >= (2, 0):
            self.denoiser.load_state_dict(torch.load(weights, map_location=DEVICE))
        else:
            self.denoiser.load_state_dict(torch.load(weights))
        self.denoiser.eval()

    def run(self, input_spectrogram):
        if isinstance(input_spectrogram, np.ndarray):
            if len(input_spectrogram.shape) == 3:
                if input_spectrogram.shape[1] != SPEC_H or input_spectrogram.shape[2] != SPEC_W:
                    raise ValueError(f"Input spectrogram must have shape (N, {SPEC_H}, {SPEC_W}), got {input_spectrogram.shape}")
                self.arr = input_spectrogram

            if len(input_spectrogram.shape) == 2:
                if input_spectrogram.shape[0] != SPEC_H or input_spectrogram.shape[1] != SPEC_W:
                    raise ValueError(f"Input spectrogram must have shape ({SPEC_H}, {SPEC_W}), got {input_spectrogram.shape}")
                self.arr = np.expand_dims(input_spectrogram, axis=0)
        else:
            raise TypeError(f"Expected np.ndarray, got {type(input_spectrogram)}")

        size = len(self.arr)
        noisy_specs = self.normalize_array(self.arr.astype(np.float32))

        final_output = np.zeros((size, SPEC_H, SPEC_W))

        with torch.no_grad():
            for i in range(size):
                noisy_spec = torch.FloatTensor(noisy_specs[i]).unsqueeze(0).unsqueeze(0).to(DEVICE)
                denoised_data = self.denoiser(noisy_spec)
                final_output[i] = denoised_data.cpu().numpy()
        return final_output

    def normalize_array(self, arr):
        """
        Applies min-max normalization to a 3D array (N, H, W) independently for each N.
        """
        arr_min = arr.min(axis=(1, 2), keepdims=True)
        arr_max = arr.max(axis=(1, 2), keepdims=True)

        diff = arr_max - arr_min
        diff[diff == 0] = 1.0

        return (arr - arr_min) / diff
