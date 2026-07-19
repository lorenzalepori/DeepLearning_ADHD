import mne
import numpy as np

channels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']

def get_electrode_coords_3d():
    montage = mne.channels.make_standard_montage("standard_1020")
    coords = montage.get_positions()["ch_pos"]
    coords_array = np.array([coords[ch] for ch in channels])
    return coords_array / np.linalg.norm(coords_array, axis=1)[:, np.newaxis]

def aep_projection(xyz):
    x, y, z = xyz
    phi = np.arcsin(z)
    lam = np.arctan2(y, x)
    c = np.pi/2 - phi
    return c * np.sin(lam), -c * np.cos(lam)

def get_aep_coords():
    coords_3d = get_electrode_coords_3d()
    return np.array([aep_projection(xyz) for xyz in coords_3d])
