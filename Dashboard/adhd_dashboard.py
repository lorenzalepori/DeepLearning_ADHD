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

st.title("ADHD Diagnosis using Siamese Network")
