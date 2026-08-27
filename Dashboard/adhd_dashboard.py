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
    st.markdown("[Data processing](#data-processing)")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;[ AEP projection](#aep-projection)", unsafe_allow_html=True)
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;[ Clough-Tocher interpolation](#clough-tocher-interpolation)", unsafe_allow_html=True)
    st.markdown("[Network](#full-network)", unsafe_allow_html=True)
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;[ Locally connected 2D](#locally-connected-2d)", unsafe_allow_html=True)
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;[ Multi-head attention](#multi-head-attention)", unsafe_allow_html=True)
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;[ Siamese comparison](#siamese-comparison)", unsafe_allow_html=True)
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;[ Majority vote](#majority-vote)", unsafe_allow_html=True)
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

st.divider()

st.markdown("### Collection")
st.markdown("""The data were collected using a 19-channel EEG system, with electrodes placed according to the international 10-20 system.r eyes closed.""")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(BASE_DIR / "extra_images" / "electrodes.png", caption="Electrode placement according to the 10-20 system, plotted using plotly.graph_objects.",width=500)
st.divider()

st.markdown("### Raw EEG signal")
st.markdown(
    "Before any processing, this is what the EEG actually looks like: raw "
    "voltage over time, one line per electrode (19 in total), stacked "
    "vertically with an offset so the traces don't overlap."
)

st.image(BASE_DIR / "extra_images" / "normalized_electrodes.png", caption="Raw EEG signal (19 electrodes, 1-40 Hz) for a single subject.")

st.divider()

st.markdown("## Data Processing")
st.markdown("### Power Spectral Density (PSD)")
st.markdown(r"""
Before any spatial mapping can happen, the raw EEG time-series for each electrode
must be converted into a measure of **how much energy is present at each frequency**.
This is the Power Spectral Density (PSD), and it's the very first processing step,
applied independently to each electrode.

**1. The raw signal.** For a given electrode, the recording is simply a sequence of
$N$ voltage samples over time:
""")
st.latex(r"x[0], x[1], \dots, x[N-1]")
st.markdown(r"""
**2. Discrete Fourier Transform (DFT).** Any such signal can be decomposed into a sum
of pure sine/cosine waves at different frequencies. The DFT extracts, for each candidate frequency $k$, a complex number that
tells us how strongly that frequency is present in the signal:
""")
st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i 2\pi k n / N}")
st.markdown(r"""
**3. From complex number to power.** $X[k]$ has both a magnitude and a phase. Since we
only care about *how much* energy is present — not the phase of the oscillation — we
take the squared magnitude:
""")
st.latex(r"P[k] = |X[k]|^2 = \text{Re}(X[k])^2 + \text{Im}(X[k])^2")
st.markdown(r"""
**4. One value per Hz.** Repeating this for every electrode and for frequencies from
1 to 40 Hz produces the `power_tensor` array you can explore below: a
**19 (electrodes) × 40 (Hz)** matrix per subject. This is the raw material that AEP and
Clough-Tocher will later turn into 2D brain maps — the frequency axis becomes the 40
image channels, and each electrode's power value becomes one scattered point to be
projected and interpolated.
""")

st.divider()

st.markdown("### AEP projection")
st.markdown("""In order to build the brain maps, first we had to turn 3D coordinates of the electrodes into 2D coordinates.
    The technique we used is called Azimuthal Equidistant Projection and it's used in carthography.<br>
        It works as follows:<br>
        - Takes the 3D coordinates and normalizes it over the surface of a unitary sphere <br>
        - Spplies the projection formula to convert the spherical coordinates into planar ones, centering the projection on the vertex.<br>
        The end result od AEP is:""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
     st.image(BASE_DIR / "extra_images" / "AEP.png")

with st.expander("The math behind AEP"):
    st.markdown(r"""
    We can look at the last step of AEP as very similar to the transformation of polar coordinates.
    First thing first we normalize each electrode onto a unit sphere:""")
    st.latex(r"\hat x=\frac{x}{r}; \hspace{.3cm} \hat y=\frac{y}{r}; \hspace{.3cm} \hat z = \frac{z}{r}; \hspace{.3cm} r=\sqrt{x^2+y^2+z^2}")

    st.markdown(r"""
    **1. Angles instead of coordinates.** We can identify each position with two angles, much
   like latitude and longitude on the Earth. The first angle $\phi$ is the angular distance from the vertex. The second angle $\lambda$ is the direction around the vertex.
    """)
    st.latex(r"\phi = \arcsin(\hat{z}) \qquad \lambda = \text{atan2}(\hat{y}, \hat{x})")
    col_space1, col1, col_space2, col2, col_space3 = st.columns([1, 2, 1, 2, 1])
    with col1:
        st.image(BASE_DIR / "extra_images" / "phi.jpg", width=300)
    with col2:
        st.image(BASE_DIR / "extra_images" / "lambda.jpg", width=300)
    st.markdown(r"""

    **2. The trick.** Use $\phi$ directly as the radius in the 2D map, and
    keep $\lambda$ as the angle:
    """)
    st.latex(r"\rho = \phi")
    st.image(BASE_DIR / "extra_images" / "rho.png")
    st.markdown(r"""
    This is what makes the projection *equidistant*: an electrode
    twice as far from the vertex lands twice as far from the
    center of the map.

    **3. Converting to (x, y)** The last step is
    the same formula you'd use for any point in a 2D
    plane to turn coordinates in polar form into Cartesian coordinates:
    """)
    st.latex(r"x = \rho \cos(\lambda) = \phi \cos(\lambda) \hspace{.4cm} \qquad y = \rho \sin(\lambda) = \phi \sin(\lambda)")
    st.markdown(r"""
    AEP has *one limitation*: distances from the vertex
    are exact by construction, but distances between two electrodes are approximate, since a flat plane can't perfectly
    preserve curved, on-sphere distances between points that are both far
    from the pole.
    """)

st.divider()

st.markdown("### Clough-Tocher interpolation")
st.markdown("""
At this point we know the power value at 19 scattered points but
 everything else on the 16×16 grid is empty. Clough-Tocher
interpolation fills those gaps with a smooth surface.

It works as follows:
- Connect the 19 projected points into a way such that each triangle has
  three known electrodes at its corners.
- Split each triangle into **3 smaller sub-triangles**, meeting at the
  triangle's centroid.""")
st.image(BASE_DIR / "extra_images" / "ct.png", width=700)
st.markdown(r"""
- Fit a smooth curved patch over each sub-triangle so that it matches the known
  power values exactly at the electrode corners.
- Stitch all the patches together so the final surface is smooth.

The result is a full 16×16 image where every pixel has a plausible power value,
not just the 19 pixels that happen to sit under an electrode.
""")


with st.expander("Math behind Clough-Tocher"):
    st.markdown(r"""
    **1. Barycentric coordinates.** Inside a triangle with corners
    $V_1, V_2, V_3$, any point can be written as a weighted mix of the three
    corners:
    """)
    st.latex(r"P = b_1 V_1 + b_2 V_2 + b_3 V_3, \qquad b_1+b_2+b_3=1")
    st.markdown(r"""
    $(b_1, b_2, b_3)$ say how "close" $P$ is to each corner.

    **2. Splitting into 3 micro-triangles.** The triangle's centroid
    $C = (V_1+V_2+V_3)/3$ becomes a fourth vertex, splitting the triangle into
    three smaller ones: $(V_1,V_2,C)$, $(V_2,V_3,C)$, $(V_3,V_1,C)$.

    **3. A cubic polynomial per micro-triangle.** On each of the three
    micro-triangles, the interpolated value is a cubic polynomial
    in the barycentric coordinates:
    """)
    st.latex(r"f(P)=f(b_1,b_2,b_3) = \sum_{i+j+k=3} c_{ijk}\, b_1^i b_2^j b_3^k")
    st.markdown(r"""
    The coefficients $c_{ijk}$ are fixed and automatically computed when solving a linear system of equations.

    **4. Enforcing continuity.** Three independently-fit
    cubic patches could still meet at slightly different angles along their
    shared edges. Clough-Tocher adds extra linear
    constraints between neighboring coefficients so that **both the value and
    the slope** match exactly across every internal edge and across every edge shared with a neighboring electrode
    triangle. The result is a surface that is smooth, not just
    continuous.

    **5. Making the image.** Given the 16x16 grid of pixel, for each pixel we first find which triangle it sits in, then which of the three micro-triangles, and finally evaluate the cubic polynomial at that point to get the interpolated power value.
    """)

st.markdown("### Explore a subject")

label_lookup = {sid: ("ADHD" if lab == 1 else "Control") for sid, lab in zip(subject_ids, labels)}

_control_ids_all = [sid for sid in subject_ids if label_lookup[sid] == "Control"]
demo_subject_ids = _control_ids_all[:6]

col_sel, col_badge = st.columns([3, 1])
with col_sel:
    selected_subject = st.selectbox(
        "Choose a subject",
        options=demo_subject_ids,
        format_func=lambda sid: f"{sid} ({label_lookup[sid]})",
    )
selected_label = label_lookup[selected_subject]
with col_badge:
    badge_color = "#2a9d8f"
    st.markdown(
        f"<div style='margin-top:28px; padding:8px 16px; border-radius:8px; "
        f"background-color:{badge_color}; color:white; text-align:center; "
        f"font-weight:700;'>{selected_label}</div>",
        unsafe_allow_html=True,
    )

selected_idx = int(np.where(subject_ids == selected_subject)[0][0])

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
        colorscale="gray",
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


st.markdown("### Network")
st.markdown("""
Putting AEP, Clough-Tocher, and the Siamese architecture together, here's the
complete path from the 5 band maps of a single subject to the final ADHD/Control
prediction.
""")
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(BASE_DIR / "extra_images" / "siamese_network_schema_en.png")

st.markdown("### Locally connected 2D")
st.markdown(""" We can look at a locally connected layer as something in between a convolutional layer and a fully connected layer.
In particular, it's similar to a CNN in the sense that it looks at local patches of the input, but it differs in that it does not share weights across different spatial locations. 
This can be particularly useful because the spatial structure of the input data is important and varies across different regions, meaning our inputs are not invariant for translation.
""")
st.markdown("""In this layer we have:<br>
- **Input**: 40 frequency images divided into 5 bands -> 5 tensors of shape 16x16×4, 16×16×4, 16×16×4, 16×16×23, 16×16×5 <br>
- 5 locally connected layers with kernel size = 5<br>
- **Output**: 5 tensors of shape 12x12×1, 12×12×1, 12×12×1, 12×12×1, 12×12×1 
""", unsafe_allow_html=True) 

st.markdown("### *Global average pooling*")
st.markdown("""The global average pooling layer is used to reduce the spatial dimensions of the feature maps produced by the locally connected layers.
It computes the average of each feature map, resulting in a single value for each feature map, losing all the spatial information. it's not really a layer since it computes a simple average.
""")

st.markdown("### *Dense layer*")
st.markdown("""
A Dense layer is the classic fully-connected layer. Given
an input vector $x$ of size $D$ and $M$ output neurons, each output is:
""")
st.latex(r"h_j = a\left(\beta_j + \sum_{i=1}^{D} \omega_{ji}\, x_i\right), \qquad j=1,\dots,M")
st.markdown("""
The weights $\\omega_{ji}$ start out random and are adjusted during training,
via backpropagation and gradient descent.
""")

with st.expander("Every Dense layer in the network"):
    st.markdown("""
    | Where | Input → Output | Activation | Role |
    |---|---|---|---|
    | Band tokenization | 1 → 16 | tanh | Expands the pooled scalar into a 16-value token |
    | Attention feed-forward (1st) | 16 → 16 | ReLU | Processes each token after attention |
    | Attention feed-forward (2nd) | 16 → 16 | linear | Re-projects before the residual sum |
    | Importance score | 16 → 1 | sigmoid | Produces the per-band importance weight |
    | Final embedding | ~256 → 16 | tanh | Produces the subject's 16-dim embedding |
    """)

st.markdown("### Multi-head attention")
st.markdown("""
So far each band has been processed in complete isolation and here comes the innovation of the project.
Attention lets each band look at the other four and
decide how much to "borrow" from each, before moving on.

Each band produces three transformed versions of its 16-value token: <br>

- **Query** -> what it's looking for <br>
- **Key** -> how it makes itself findable <br>
- **Value** -> the information it actually carries<br>
Comparing every band's Query against every other band's Key gives a 5×5 relevance score, normalized
with softmax so each band's weights sum to 1.

With `num_heads=2`, this whole process runs twice in parallel. The two
results are combined into a single output the same size as the original
token, which is then added back to it so no band's
original information is lost. We use the importance scores to weight the previous 5 maps.
""", unsafe_allow_html=True)

with st.expander("Math behind multi-head attention"):
    st.markdown(r"""
    **1. Query, Key, Value.** For each band $i$, three learned projections of
    its token:
    """)
    st.latex(r"q_i = W_Q \cdot \text{token}_i \qquad k_i = W_K \cdot \text{token}_i \qquad v_i = W_V \cdot \text{token}_i")
    st.markdown(r"""
    **2. Relevance scores.** How much band $i$'s query matches band $j$'s key,
    for every pair of bands — a 5×5 matrix:
    """)
    st.latex(r"\text{score}_{ij} = q_i \cdot k_j")
    st.markdown(r"""
    **3. Softmax normalization.** Scaled by $\sqrt{d_k}$ and normalized so each
    band's weights over the other bands sum to 1:
    """)
    st.latex(r"\alpha_{ij} = \text{softmax}_j\left(\frac{\text{score}_{ij}}{\sqrt{d_k}}\right)")
    st.markdown(r"""
    **4. Weighted combination.** Band $i$'s new representation is a weighted
    average of every band's Value:
    """)
    st.latex(r"\text{attn\_output}_i = \sum_{j=1}^{5} \alpha_{ij}\, v_j")
    st.markdown(r"""
    This layer also returns the raw attention scores, the
    5×5 relevance matrix per head.
    """)

st.markdown("### *Concatenation + Conv2D×2*")
st.markdown("""
The 5 weighted band mapsare stacked into a single 12×12×5 tensor. Two standard Convolution layers then process this
combined map. Two Convolution layers in a row let the second layer indirectly "see" a wider
    area of the map than the first, building up spatial context gradually.
 """)


st.markdown("### Siamese comparison")
st.markdown("""
Everything above — locally-connected layers, attention, concatenation,
Conv2D×2, Dense — is one single network (the base network). To compare two
subjects, that exact same network, with the exact same weights, is run twice:
once on subject A, once on subject B. Using shared weights is what makes the
comparison fair — both subjects are judged by the same "measuring stick".
""")

st.markdown("### Euclidean distance")
st.markdown("""
The two 16-dimensional embeddings are compared with a simple euclidean
distance. During training, this distance is pushed toward 0 for pairs of the
same class, and pushed apart for pairs of different classes.
""")

st.markdown("### Majority vote")
st.markdown("""
At evaluation time, a new subject is compared only against the ADHD subjects
from the training set.
Each pairwise distance becomes a binary vote and the final prediction is simply whichever vote wins the majority.
""")

st.markdown("### Prediction")
st.markdown("""
The end result: **ADHD** or **Control** for the held-out subject.
""")








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