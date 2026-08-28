# ADHD Detection from EEG using a Siamese Network with Attention

Deep learning project that detects ADHD from EEG recordings, extending the Siamese
architecture proposed by Latifi et al. (2024) with a multi-head attention mechanism
over frequency bands.

## Pipeline

```
Raw EEG (19 channels)
  → Power Spectral Density (1-40 Hz)
  → AEP projection (electrodes 3D → 2D)
  → Clough-Tocher interpolation → 16×16×40 brain map
  → split into 5 frequency bands
  → Siamese network (LocallyConnected2D → attention → Conv2D×2 → Dense)
  → embedding → euclidean distance → majority vote vs ADHD references
  → prediction: ADHD or Control
```

## What's different from the paper

Multi-head attention over the 5 bands produces a per-band importance score used to
weight the maps before merging — replacing the paper's post-hoc Grad-CAM step.

## References

- Latifi, Amini & Motie Nasrabadi (2024). *Siamese based deep neural network for ADHD detection using EEG signal.*
- Bashivan et al. (2016). *Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks.*
- Alfeld (1984). *A Trivariate Clough-Tocher Scheme for Tetrahedral Data.*

## Authors

Elia Crimi, Lorenza Lepori — Deep Learning, A.Y. 2025-2026.