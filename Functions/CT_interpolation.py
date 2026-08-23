import numpy as np
from pathlib import Path
from scipy.interpolate import CloughTocher2DInterpolator
 
from AEP import channels, get_aep_coords

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "Data" / "processed"

GRID_SIZE = 16
MARGIN = 0.1
NORMALIZATION_FACTOR = 1e5

def build_brain_maps(power_tensor, coords_2d, grid_size=GRID_SIZE, margin=MARGIN):
    n_subjects, n_electrodes, n_freqs = power_tensor.shape
 
    u_min, u_max = coords_2d[:, 0].min() - margin, coords_2d[:, 0].max() + margin
    v_min, v_max = coords_2d[:, 1].min() - margin, coords_2d[:, 1].max() + margin
 
    grid_u, grid_v = np.meshgrid(
        np.linspace(u_min, u_max, grid_size),
        np.linspace(v_min, v_max, grid_size)
    )
 
    brain_maps = np.zeros((n_subjects, grid_size, grid_size, n_freqs))
 
    for subj_idx in range(n_subjects):
        for freq_idx in range(n_freqs):
            values = power_tensor[subj_idx, :, freq_idx]
 
            interpolator = CloughTocher2DInterpolator(coords_2d, values)
            interpolated = interpolator(grid_u, grid_v)
            interpolated = np.nan_to_num(interpolated, nan=0.0)
 
            brain_maps[subj_idx, :, :, freq_idx] = interpolated
 
    return brain_maps

if __name__ == "__main__":
    power_tensor = np.load(PROCESSED_DIR / "power_tensor.npy")
    coords_2d = get_aep_coords()  
    brain_maps = build_brain_maps(power_tensor, coords_2d)
    brain_maps = brain_maps / NORMALIZATION_FACTOR
    np.save(PROCESSED_DIR / "brain_maps.npy", brain_maps)
