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

st.markdown("""
<style>
[data-testid="stTab"] p {
    font-size: 20px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

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


st.markdown(
    "<h1 style='text-align: center; color: #0f6d95;'> ADHD Diagnosis using Siamese Network</h1>",
    unsafe_allow_html=True
)
if not data_loaded:
    st.error(
        f"File not found: {missing_file}\n\n"
        "Please ensure that the required data files are present and in the correct directory."
    )
    st.stop()

st.divider()

n_subjects = len(labels)
n_adhd = int((labels == 1).sum())
n_control = int((labels == 0).sum())

c1, c2, c3 = st.columns(3)
c1.metric("Total Subjects", n_subjects)
c2.metric("ADHD", n_adhd)
c3.metric("Control", n_control)

st.divider()

tab_maps, tab_spectralpower = st.tabs(["Brain Maps", "Spectral Power"])