# Paper conformance

The release centralizes manuscript constants in `glicore/glicore_config.py`.

- Evidence: `softplus(logits / temperature) + 1e-4`.
- Temperature: `softplus(raw) + 0.01`, capped at 5.
- FAEC: patch side 8, real FFT, low-frequency factor 2.0, residual factor 0.1.
- FECE: BraTS ET threshold 0.35, WT boundary refinement, correction cap 0.5.
- Transfer masks: confidence-weighted foreground and multi-class interfaces.
- AdaTER: 5-voxel morphology and a learned sigmoid class-relation matrix.
- PACER calibration: 16 mini-batches and target gradient ratios 0.5/0.2.
- PACER relation loss uses margin 0.3 and at most 2,048 adjacent pairs.
- PACER is disabled by the standard inference `forward(x)` path.
