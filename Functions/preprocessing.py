import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import welch

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "unprocessed" / "adhdata.csv"
channels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
                'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']

SF = 128  # Sampling frequency

# Load the data
df = pd.read_csv(DATA_PATH)
# Convert ADHD class label to binary
df['is_ADHD'] = (df['Class'] == 'ADHD').astype(int)
# Group by subject
subjects = df.groupby('ID')

matrices = []
adhd_labels = []
subject_ids = []

for subject_id, group in subjects:
    signal = group[channels].values.T  # Transpose to get shape
    label = group['is_ADHD'].iloc[0]  # Get the label for the subject
    channel_powers = []

    for channel_index in range(signal.shape[0]):
        freq, psd = welch(signal[channel_index], fs=SF, nperseg=256)
        band_powers = []

        for low in range(1, 41):
            mask = (freq >= low) & (freq < low + 1)
            band_powers.append(psd[mask].sum())
        channel_powers.append(band_powers)

    sub_matrix = np.array(channel_powers)
    matrices.append(sub_matrix)
    adhd_labels.append(label)
    subject_ids.append(subject_id)

power_tensor = np.stack(matrices)
labels = np.array(adhd_labels)

OUTPUT_DIR = BASE_DIR / "Data" / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

np.save(OUTPUT_DIR / "power_tensor.npy", power_tensor)
np.save(OUTPUT_DIR / "labels.npy", labels)
np.save(OUTPUT_DIR / "subject_ids.npy", np.array(subject_ids))
