# Interactive Dashboard for ADHD Model using Siamese Network
import numpy as np
from pathlib import Path
import streamlit as st
from plotly.subplots import make_subplots
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

st.markdown("### Explore a subject")

label_lookup = {sid: ("ADHD" if lab == 1 else "Control") for sid, lab in zip(subject_ids, labels)}

_adhd_ids_all = [sid for sid in subject_ids if label_lookup[sid] == "ADHD"]
_control_ids_all = [sid for sid in subject_ids if label_lookup[sid] == "Control"]
demo_subject_ids = _adhd_ids_all[:3] + _control_ids_all[:3]

col_sel, col_badge = st.columns([3, 1])
with col_sel:
    selected_subject = st.selectbox(
        "Choose a subject",
        options=demo_subject_ids,
        format_func=lambda sid: f"{sid} ({label_lookup[sid]})",
    )
selected_label = label_lookup[selected_subject]
with col_badge:
    badge_color = "#d1495b" if selected_label == "ADHD" else "#2a9d8f"
    st.markdown(
        f"<div style='margin-top:28px; padding:8px 16px; border-radius:8px; "
        f"background-color:{badge_color}; color:white; text-align:center; "
        f"font-weight:700;'>{selected_label}</div>",
        unsafe_allow_html=True,
    )

selected_idx = int(np.where(subject_ids == selected_subject)[0][0])

tab_maps, tab_spectralpower = st.tabs(["Brain Maps", "Spectral Power"])

with tab_maps:
    st.markdown(f"**Sub-band brain maps for subject `{selected_subject}` ({selected_label})**")
    st.caption(
        "Each map is the average power over the electrode topography for that "
        "frequency band. Brighter = higher power (same convention as Fig. 6 of the paper)."
    )

    band_cols = st.columns(len(BANDS))
    for col, (band_name, (low, high)) in zip(band_cols, BANDS.items()):
        band_map = brain_maps[selected_idx, :, :, low:high].mean(axis=-1)
        fig = go.Figure(data=go.Heatmap(
            z=band_map,
            colorscale="Viridis",
            showscale=False,
        ))
        fig.update_layout(
            title=f"{band_name} ({low}-{high} Hz)",
            width=220, height=220,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange="reversed"),
        )
        col.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Effect size (Cohen's d) per band — the fair comparison across bands**")
    st.caption(
        "Cohen's d standardizes the ADHD-Control difference by how much subjects "
        "naturally vary within each group at that pixel: d = (mean_ADHD - "
        "mean_Control) / pooled_std. Unlike the raw difference above, this uses "
        "the **same fixed color scale for every band**, so bands really are "
        "comparable here. Rule of thumb: |d| around 0.2 = small, 0.5 = medium, "
        "0.8+ = large effect. If Theta shows the largest, most spatially "
        "structured |d| here, that matches the paper's finding."
    )

    adhd_mask = labels == 1
    control_mask = labels == 0
    band_items_group = list(BANDS.items())
    D_SCALE = 0.8  # scala fissa e condivisa tra tutte le bande

    effect_fig = make_subplots(
        rows=1, cols=len(band_items_group),
        subplot_titles=[f"{name} ({low}-{high} Hz)" for name, (low, high) in band_items_group],
        horizontal_spacing=0.02,
    )

    for col_i, (band_name, (low, high)) in enumerate(band_items_group, start=1):
        adhd_band_maps = brain_maps[adhd_mask][:, :, :, low:high].mean(axis=-1)
        control_band_maps = brain_maps[control_mask][:, :, :, low:high].mean(axis=-1)

        mean_adhd = adhd_band_maps.mean(axis=0)
        mean_control = control_band_maps.mean(axis=0)
        std_adhd = adhd_band_maps.std(axis=0, ddof=1)
        std_control = control_band_maps.std(axis=0, ddof=1)

        n1, n2 = adhd_band_maps.shape[0], control_band_maps.shape[0]
        pooled_std = np.sqrt(
            ((n1 - 1) * std_adhd**2 + (n2 - 1) * std_control**2) / (n1 + n2 - 2)
        )
        cohens_d = (mean_adhd - mean_control) / (pooled_std + 1e-8)

        effect_fig.add_trace(
            go.Heatmap(
                z=cohens_d, colorscale="RdBu_r",
                zmin=-D_SCALE, zmax=D_SCALE,
                showscale=(col_i == len(band_items_group)),
            ),
            row=1, col=col_i,
        )

    effect_fig.update_xaxes(showticklabels=False)
    effect_fig.update_yaxes(showticklabels=False, autorange="reversed")
    effect_fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))

    st.plotly_chart(effect_fig, use_container_width=True)
    st.caption(
        "Red = ADHD systematically higher than Control at that pixel (relative "
        "to within-group variability), blue = the opposite. Same scale "
        f"(-{D_SCALE} to +{D_SCALE}) across all 5 bands, so color intensity is "
        "directly comparable band to band. Rule of thumb: |d| around 0.2 = small, "
        "0.5 = medium, 0.8+ = large effect."
    )

with tab_spectralpower:
    st.markdown(f"**Power spectral density (1-40 Hz) for subject `{selected_subject}` ({selected_label})**")

    channels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
                'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']
    freqs = np.arange(1, 41)

    view_mode = st.radio(
        "View",
        options=["Average across electrodes", "Per-electrode heatmap", "Per-electrode lines"],
        horizontal=True,
    )

    subject_power = power_tensor[selected_idx]  # shape (19, 40)

    if view_mode == "Average across electrodes":
        avg_power = subject_power.mean(axis=0)
        fig = px.line(x=freqs, y=avg_power, labels={"x": "Frequency (Hz)", "y": "Power"})
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Per-electrode heatmap":
        fig = go.Figure(data=go.Heatmap(
            z=subject_power,
            x=freqs,
            y=channels,
            colorscale="Viridis",
        ))
        fig.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title="Electrode",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        selected_channels = st.multiselect(
            "Electrodes to show", options=channels, default=["Fz", "Cz", "Pz", "O1", "O2"]
        )
        fig = go.Figure()
        for ch in selected_channels:
            ch_idx = channels.index(ch)
            fig.add_trace(go.Scatter(x=freqs, y=subject_power[ch_idx], mode="lines", name=ch))
        fig.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Power", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("## Bibliography")
st.markdown(""" - Kieling, Renata, and Luis A. Rohde - "ADHD in children and adults: diagnosis and prognosis." *Behavioral neuroscience of attention deficit hyperactivity disorder and its treatment (2011): 1-16.* """)
st.markdown(""" - Alfeld (June 1984) - "A Trivariate Clough-Tocher Scheme for Thetraedal Data" """) 
st.markdown(""" - Bahivan, Rish, Ywasin, Codella (2016) - "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks" """)
st.markdown(""" - Latifi, Amini & Motie Nasrabadi (2024) - "Siamese based deep neural network for ADHD detection
                  using EEG signal" """)

st.divider()
st.markdown(
    "<p style='text-align: center; color: gray; margin-top: -10px;'>Deep Learning project by Elia Crimi and Lorenza Lepori<br>"
    "Academic Year 2025-2026</p>",
    unsafe_allow_html=True
)