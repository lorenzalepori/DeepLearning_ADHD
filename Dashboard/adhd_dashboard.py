# Interactive Dashboard for ADHD Model using Siamese Network
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title = "ADHD Diagnosis using Siamese Network",
    layout = "wide",
)

# Bands division
BANDS = {"Alpha": (8,12), "Beta": (12,35), "Delta": (0,4),  "Theta": (4,8), "Gamma": (35,40)}

# Data load
@st.cache_data
def load_data():
    brain_maps = np.load("brain_maps.npy")
    labels = np.load("labels.npy")
    subject_ids = np.load("subject_ids.npy")
    power_tensor = np.load("power_tensor.npy")
    return brain_maps, labels, subject_ids, power_tensor

try:
    brain_maps, labels, subject_ids, power_tensor = load_data()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    missing_file = str(e)


st.title("ADHD Diagnosis using Siamese Network")
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
c1.metric("total patients", n_subjects)
c2.metric("ADHD", n_adhd)
c3.metric("Control", n_control)

st.divider()

tab_maps, tab_spectralpower = st.tabs(["Brain Maps", "Spectral Power"])