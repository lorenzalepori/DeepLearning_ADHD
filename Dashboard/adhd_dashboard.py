# Interactive Dashboard for ADHD Model using Siamese Network
import numpy as np
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title = "ADHD Diagnosis using Siamese Network",
    layout = "wide",
)

with st.sidebar:
    st.markdown("## Menu")
    st.markdown("[What is ADHD?](#what-is-adhd)")
    st.markdown("[EEG-based diagnosis proposal](#eeg-diagnosis-proposal-overview)")
    st.markdown("[Data](#data)")
    st.markdown("[Bibliography](#bibliography)")

st.markdown("""
<style>
[data-testid="stTab"] p {
    font-size: 20px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"C:\Users\ASUS\OneDrive\Desktop\DSAI\DeepLearning\Functions\Data\processed")


# Bands division
BANDS = {"Alpha": (8,12), "Beta": (12,35), "Delta": (0,4),  "Theta": (4,8), "Gamma": (35,40)}

# Data load
@st.cache_data
def load_data():
    brain_maps = np.load(DATA_DIR / "brain_maps.npy")
    labels = np.load(DATA_DIR / "labels.npy")
    subject_ids = np.load(DATA_DIR / "subject_ids.npy")
    power_tensor = np.load(DATA_DIR / "power_tensor.npy")
    return brain_maps, labels, subject_ids, power_tensor

try:
    brain_maps, labels, subject_ids, power_tensor = load_data()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    missing_file = str(e)


import base64
def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_image(BASE_DIR / "extra_images" / "brain_drawing.jpg")

st.markdown(f"""
<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
    <img src="data:image/png;base64,{img_base64}" width="120">
    <h1 style="color: #0f6d95; margin: 0;">ADHD Diagnosis using Siamese Network</h1>
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.error(
        f"File not found: {missing_file}\n\n"
        "Please ensure that the required data files are present and in the correct directory." \
    )
    st.stop()

st.divider()

st.markdown("""## What is ADHD?""")
st.markdown("""
ADHD (Attention-Deficit Hyperactivity Disorder) is one of the most common childhood-onset neuropsychiatric disorders, with a chronic course affecting functioning across the lifespan.

In the school-age range, ADHD mostly impacts:

- Academic performance
- Self-esteem
- Peer relationships
""")
st.divider()

st.markdown("## EEG-based diagnosis proposal")

st.markdown("""
ADHD detection based on EEG signals is a recently growing research area.

The model analyzes the **Power Spectral Density (PSD)** of EEG recordings,
mapped into 2D brain topography images obtained via a combination of AEP and Clough-Tocher interpolation across five frequency bands, and compares pairs of subjects through
a Siamese neural network trained to recognize patterns shared among ADHD
cases.

Our extension consists on introducing a multi-head self-attention mechanism that learns which bands
are most discriminative for each subject, instead of concatenating them.
This allows us not to rely solely on the Grad-GAM explainability method to identify the most relevant frequency bands.""")

st.divider()

st.markdown("## Data")
"""The dataset consists of EEG recordings from children aged 7-12 years old. The EEG signals were recorded using a 19-channel setup, following the standard 10-20 system. The dataset is free and available for download on [IEEE DataPort](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children) 
(a mirror copy is also available on 
[Kaggle](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd))"""

st.divider()

n_subjects = len(labels)
n_adhd = int((labels == 1).sum())
n_control = int((labels == 0).sum())

c1, c2, c3 = st.columns(3)
c1.metric("Total Subjects", n_subjects)
c2.metric("ADHD", n_adhd)
c3.metric("Control", n_control)

tab_maps, tab_spectralpower = st.tabs(["Brain Maps", "Spectral Power"])

st.divider()
st.markdown("## Bibliography")
st.markdown(""" - Kieling, Renata, and Luis A. Rohde - "ADHD in children and adults: diagnosis and prognosis." *Behavioral neuroscience of attention deficit hyperactivity disorder and its treatment (2011): 1-16.* 
                - Latifi, Amini & Motie Nasrabadi (2024) - "Siamese based deep neural network for ADHD detection
                  using EEG signal" """)

st.divider()
st.markdown(
    "<p style='text-align: center; color: gray; margin-top: -10px;'>Deep Learning project by Elia Crimi and Lorenza Lepori<br>"
    "Academic Year 2025-2026</p>",
    unsafe_allow_html=True
)