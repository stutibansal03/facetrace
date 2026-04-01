import numpy as np
from PIL import Image
from facetrace.config import FFT_HIGH_FREQ_RATIO


def compute_fft_features(pil_image: Image.Image) -> list[float]:
    gray = np.array(pil_image.convert("L"), dtype=np.float32)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    total_energy = np.sum(magnitude)

    high_cutoff = int(min(cy, cx) * FFT_HIGH_FREQ_RATIO)
    mask_high = np.ones((h, w), dtype=bool)
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    mask_high[dist < (min(cy, cx) - high_cutoff)] = False
    high_energy = np.sum(magnitude[mask_high])

    high_freq_ratio = high_energy / (total_energy + 1e-8)

    mid_inner = int(min(cy, cx) * 0.3)
    mid_outer = int(min(cy, cx) * 0.7)
    mask_mid = (dist >= mid_inner) & (dist < mid_outer)
    mid_energy = np.sum(magnitude[mask_mid])
    mid_high_ratio = mid_energy / (total_energy + 1e-8)

    return [float(high_freq_ratio), float(mid_high_ratio)]
