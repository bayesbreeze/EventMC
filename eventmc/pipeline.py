"""EventMC-S (Algorithm 1) and EventMC (multi-anchor extension, Section 3.3)."""
import time

import numpy as np
import torch

from .anchor import prerun
from .model import EncodingModel
from .params import TV_WEIGHT
from .stages import stage_a, stage_b, final_reconstruction


def half_resolution(model, y):
    """Half-resolution copy of an acquisition: the central half of each k-space axis (data and sampling pattern), voxels
    twice as large, coil sensitivities re-estimated from the cropped data (appendix A.5). Poses (degrees, mm) carry over
    unchanged. The data are multiplied by sqrt(8) to keep the orthonormal FFT scaling."""
    C, D, H, W = y.shape
    d0, h0, w0 = D // 2 - D // 4, H // 2 - H // 4, W // 2 - W // 4
    Dc, Hc, Wc = D // 2, H // 2, W // 2
    y_half = y[:, d0:d0 + Dc, h0:h0 + Hc, w0:w0 + Wc] * np.sqrt(8.)
    segments = [np.array([r - h0 for r in seg if h0 <= r < h0 + Hc], int) for seg in model.segments]
    mask = model.mask2d[d0:d0 + Dc, h0:h0 + Hc]
    half = EncodingModel(y_half.shape, torch.ones((C, Dc, Hc, Wc), dtype=torch.complex64), segments, mask,
                         voxel_mm=2 * model.voxel_mm, tv_weight=model.tv_weight, device=model.device)
    half.set_maps(half.estimate_maps(y_half))
    return half, y_half


def eventmc_s(model, y, anchor, log=print):
    """One run of Algorithm 1 with a given anchor: Stage A at full resolution, Stage B, final reconstruction.
    Returns (poses, image)."""
    t0 = time.monotonic()
    model.anchor = anchor
    poses, events = stage_a(model, y)
    log(f'  Stage A (anchor {anchor}): events {sorted(events)}, {time.monotonic() - t0:.0f} s')
    poses = stage_b(model, y, poses)
    log(f'  Stage B: {time.monotonic() - t0:.0f} s')
    image = final_reconstruction(model, y, poses)
    log(f'  final reconstruction: {time.monotonic() - t0:.0f} s')
    return poses, image


def eventmc_half(model, y, anchor, log=print):
    """Multi-anchor candidate: Stage A at half resolution, then Stage B and the final reconstruction at full resolution.
    Returns (poses, image)."""
    t0 = time.monotonic()
    half, y_half = half_resolution(model, y)
    half.anchor = anchor
    poses, events = stage_a(half, y_half)
    del half, y_half
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f'  Stage A, half resolution (anchor {anchor}): events {sorted(events)}, {time.monotonic() - t0:.0f} s')
    model.anchor = anchor
    poses = stage_b(model, y, poses)
    image = final_reconstruction(model, y, poses)
    log(f'  Stage B + final reconstruction: {time.monotonic() - t0:.0f} s')
    return poses, image


def run_eventmc(y, segments, maps=None, mask2d=None, voxel_mm=1.0, tv_weight=TV_WEIGHT, multi_anchor=True,
                use_guard=True, normalize=True, device=None, log=print):
    """Correct rigid head motion in an interleaved 3D Cartesian acquisition.

    Parameters
    ----------
    y : complex array (C, D, H, W) -- multi-coil 3D k-space (coils, kz, ky, kx), zeros where not sampled.
    segments : list of 1D int arrays -- ky indices of each segment, in acquisition order (time-contiguous groups of
        shots).
    maps : complex array (C, D, H, W), optional -- coil sensitivities; estimated from the data if None.
    mask2d : bool array (D, H), optional -- sampled (kz, ky) points; inferred from the nonzero entries of y if None.
    voxel_mm : isotropic voxel size in mm.
    tv_weight : TV weight lambda for data normalised so that the 99.5th percentile of |uncorrected image| is 1.
    multi_anchor : run the multi-anchor extension (EventMC); False gives a single run of Algorithm 1 (EventMC-S).
    use_guard : run the within-segment check in the pre-run (needs several shots per segment).
    normalize : rescale y as above before estimation (set False if y is already normalised).

    Returns
    -------
    dict with 'image' (complex (D, H, W), in the scale of the input), 'poses' (N_s, 6: rotation vector in degrees,
    translation in mm, relative to the anchor segment), 'anchor', 'selected' candidate, 'residuals' of all candidates,
    'prerun' and 'candidates' (poses and anchor of each candidate).
    """
    y = torch.as_tensor(y).to(device or ('cuda' if torch.cuda.is_available() else 'cpu'), torch.complex64)
    if mask2d is None:
        mask2d = (y.abs().sum((0, 3)) > 0).cpu().numpy()
    if maps is None:
        model = EncodingModel(y.shape, torch.ones_like(y), segments, mask2d, voxel_mm, tv_weight, device=y.device.type)
        model.set_maps(model.estimate_maps(y))
    else:
        model = EncodingModel(y.shape, maps, segments, mask2d, voxel_mm, tv_weight, device=y.device.type)
    scale = 1.0
    if normalize:
        scale = float(torch.quantile(model.coil_combine(y).abs().flatten()[::97], .995))
        y = y / scale

    t0 = time.monotonic()
    pre = prerun(model, y, use_guard=use_guard)
    log(f'pre-run: anchor {pre["anchor"]}, candidates {pre["candidates"]}, events {pre["events"]}, '
        f'Guard-flagged {pre["guard_flagged"]}, {time.monotonic() - t0:.0f} s')

    anchor = pre['anchor']
    candidates = {}
    log('EventMC-S (full resolution):')
    poses, image = eventmc_s(model, y, anchor, log=log)
    candidates[f'full@{anchor}'] = dict(poses=poses, image=image, anchor=anchor,
                                        residual=model.data_residual(image, poses, y))
    if multi_anchor:
        for a in [anchor] + pre['candidates'][1:2]:           # candidate (ii): same anchor; (iii): second anchor
            log(f'candidate at half resolution, anchor {a}:')
            poses, image = eventmc_half(model, y, a, log=log)
            candidates[f'half@{a}'] = dict(poses=poses, image=image, anchor=a,
                                           residual=model.data_residual(image, poses, y))

    selected = min(candidates, key=lambda k: candidates[k]['residual'])
    log(f'selected {selected} (residuals: ' + ', '.join(f'{k} {v["residual"]:.4g}' for k, v in candidates.items())
        + f'); total {time.monotonic() - t0:.0f} s')
    best = candidates[selected]
    return dict(image=(best['image'] * scale).cpu(), poses=np.asarray(best['poses']), anchor=best['anchor'],
                selected=selected, residuals={k: v['residual'] for k, v in candidates.items()}, prerun=pre,
                candidates={k: dict(poses=np.asarray(v['poses']), anchor=v['anchor']) for k, v in candidates.items()})
