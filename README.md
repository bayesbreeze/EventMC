# EventMC: training-free motion correction for interleaved 3D brain MRI

EventMC (event-guided motion correction) jointly estimates a motion-corrected image and rigid head motion from multi-coil
3D Cartesian k-space acquired in interleaved segments. It needs no training data.

Each segment's pose is fitted against a reference image reconstructed with a total-variation (TV) prior from a *stable
interval* — consecutive segments between detected motion events, around an anchor segment. The image and motion are then
refined with all segments from low to high spatial frequencies.

> W. Jiang, H. Chen, T. Ding, F. Liu, N. Ye, H. Sun. *Training-free motion-corrected 3D brain MRI reconstruction via
> event-guided references.* Submitted to IEEE ISBI 2027.

## Installation

Requires Python 3.10+ and PyTorch (a CUDA GPU is strongly recommended for full-size data).

```bash
git clone https://github.com/bayesbreeze/EventMC.git
cd EventMC
pip install -r requirements.txt
```

## Quick start

```python
import numpy as np
from eventmc import run_eventmc

# y:        complex (C, D, H, W) multi-coil 3D k-space (coils, kz, ky, kx), zeros where not sampled
# segments: list of 1D int arrays, the ky indices of each segment in acquisition order
out = run_eventmc(y, segments, maps=None, voxel_mm=1.0)

image = out['image']   # complex (D, H, W), motion-corrected, in the anchor segment's head position
poses = out['poses']   # (N_s, 6): rotation vector (deg) and translation (mm) of each segment relative to the anchor
```

Run the synthetic example (an ellipsoid phantom with two abrupt movements) to check the installation:

```bash
python examples/synthetic_demo.py                  # EventMC (multi-anchor)
python examples/synthetic_demo.py --single         # EventMC-S only (faster)
```

## Input requirements and assumptions

| Item | Requirement |
|---|---|
| Acquisition | 3D Cartesian; each shot acquires one ky index (all kz, kx sampled for that ky are on the 2D (kz, ky) mask); shots are grouped into time-contiguous **segments** |
| Segments | each segment must sample both low and high spatial frequencies, i.e. its ky lines are spread over k-space (interleaved ordering); one rigid pose per segment |
| `mask2d` | boolean (kz, ky) sampling pattern; inferred from the nonzero entries of `y` if omitted |
| `maps` | coil sensitivities (C, D, H, W), assumed scanner-fixed; estimated from the low-frequency k-space centre of `y` if omitted |
| Voxels | isotropic, size `voxel_mm`; poses are in degrees and millimetres about the FOV centre |
| Scaling | with `normalize=True` (default), `y` is scaled so that the 99.5th percentile of the uncorrected magnitude is 1, the scale for which `tv_weight` is set |

## Method and code

| Paper (Section / Algorithm 1) | Code |
|---|---|
| Forward model, Eq. (1); objective, Eq. (2) | `core.Core.E`, `rows_k`, `warp` (Fourier shears and phase ramps) |
| ImgEst (TV reconstruction with fixed poses) | `core.Core.tv_recon` |
| MotEst (per-segment pose fit) | `core.Core.pose_refine` |
| Pre-run and anchor selection (Sec. 3.1, line 2) | `anchor.prerun`, `anchor.guard` |
| Reference segments, Eq. (3) | `stages.stable_interval`, `stages.stage_a` |
| Pose fitting and events (nearest-to-farthest fitting, three starts, events, smoothing) | `core.Core.centre_out`, `core.Core.detect`, `core.smooth` |
| Stage B: coarse-to-fine refinement (Sec. 3.2, lines 9–13) | `stages.stage_b` |
| Final reconstruction (line 14) | `stages.final_reconstruction` |
| Multi-anchor extension (Sec. 3.3) | `pipeline.half_resolution`, `pipeline.eventmc_half`, `pipeline.run_eventmc` |

`run_eventmc(..., multi_anchor=False)` gives a single run of Algorithm 1 (EventMC-S).

## Parameters used in the paper

| Parameter | Value | Where |
|---|---|---|
| TV weight λ | 1e-3 (simulated CC359 data); 3e-3 (PMoC3D, chosen by k-space cross-validation on motion-free scans) | `tv_weight` |
| Stage A rounds | 5 | `stages.STAGE_A_ROUNDS` |
| Stage B schedule | f = 0.25, 0.5, 1, 1 | `stages.STAGE_B_SCHEDULE` |
| L-BFGS iterations | 15 per image update, 30 per segment pose fit, 30 for the final reconstruction | `core.REF_ITERS`, `core.POSE_ITERS`, `core.FINAL_ITERS` |
| Event threshold | m_j > max(3σ_j, 0.6 mm) | `core.Core.detect` |
| Guard | 3 sub-windows per segment, 24 slices, z > 3, up to 5 rounds | `anchor.py` |

## Runtime

For 12 coils, a 170 × 218 × 256 grid and 13 segments, a single run (EventMC-S) took a median of under 7 min and the full
EventMC under 12 min on one NVIDIA H100, excluding data loading.

## Limitations

* The TV prior limits performance under heavy undersampling (about four-fold and above).
* One pose per segment cannot represent motion within a segment, and frequent movements leave no long stable interval for
  the anchor.
* Candidates in the multi-anchor extension are ranked by data residual, a heuristic that does not guarantee the lowest image
  error.

This repository contains the method. The data simulation, baselines and evaluation used for the paper's experiments are not
included.

## Citation

```bibtex
@inproceedings{jiang2027eventmc,
  title     = {Training-free motion-corrected 3D brain MRI reconstruction via event-guided references},
  author    = {Jiang, Wei and Chen, Hongli and Ding, Tianyi and Liu, Feng and Ye, Nan and Sun, Hongfu},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027},
  note      = {Submitted}
}
```

## License

MIT, see [LICENSE](LICENSE).
