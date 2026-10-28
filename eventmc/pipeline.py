"""EventMC-S (Algorithm 1) and EventMC (multi-anchor extension, Section 3.3)."""
import time
import numpy as np
import torch

from .core import Core, TV_LAM
from .anchor import prerun
from .stages import stage_a, stage_b, final_reconstruction, STAGE_A_ROUNDS, STAGE_B_SCHEDULE


def half_resolution(fine, y):
    """Half-resolution copy of an acquisition: the central half of each k-space axis (data and sampling pattern), voxels
    twice as large, coil sensitivities re-estimated from the cropped data. Poses (degrees, mm) transfer unchanged."""
    C, D, H, W = y.shape; d0, h0, w0 = D // 2 - D // 4, H // 2 - H // 4, W // 2 - W // 4; Dc, Hc, Wc = D // 2, H // 2, W // 2
    yc = y[:, d0:d0 + Dc, h0:h0 + Hc, w0:w0 + Wc] * np.sqrt(8.)
    segments = [np.array([r - h0 for r in b if h0 <= r < h0 + Hc], int) for b in fine.blocks]
    mask = fine.M2np[d0:d0 + Dc, h0:h0 + Hc]
    c = Core(yc.shape, torch.ones((C, Dc, Hc, Wc), dtype=torch.complex64), segments, mask, voxel_mm=2 * fine.voxel_mm,
             tv_weight=fine.tv_weight, device=fine.dev)
    c.use_maps(c.maps_from_data(yc))
    return c, yc


def eventmc_s(c, y, anchor, rounds=STAGE_A_ROUNDS, schedule=STAGE_B_SCHEDULE, log=print):
    """One run of Algorithm 1 with a given anchor (Stage A at full resolution, Stage B, final TV reconstruction)."""
    t0 = time.monotonic(); c.anchor = anchor
    XA, events = stage_a(c, y, rounds); log(f'  Stage A (anchor {anchor}): events {sorted(events)}, {time.monotonic() - t0:.0f} s')
    X = stage_b(c, y, XA, schedule); log(f'  Stage B: {time.monotonic() - t0:.0f} s')
    x = final_reconstruction(c, y, X); log(f'  final reconstruction: {time.monotonic() - t0:.0f} s')
    return X, x


def eventmc_half(c, y, anchor, rounds=STAGE_A_ROUNDS, schedule=STAGE_B_SCHEDULE, log=print):
    """Multi-anchor candidate: Stage A at half resolution, then Stage B and the final reconstruction at full resolution."""
    t0 = time.monotonic(); h, yh = half_resolution(c, y); h.anchor = anchor
    XA, events = stage_a(h, yh, rounds); del h, yh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f'  Stage A, half resolution (anchor {anchor}): events {sorted(events)}, {time.monotonic() - t0:.0f} s')
    c.anchor = anchor; X = stage_b(c, y, XA, schedule); x = final_reconstruction(c, y, X)
    log(f'  Stage B + final reconstruction: {time.monotonic() - t0:.0f} s')
    return X, x


def run_eventmc(y, segments, maps=None, mask2d=None, voxel_mm=1.0, tv_weight=TV_LAM, multi_anchor=True,
                normalize=True, device=None, log=print):
    """Correct rigid head motion in an interleaved 3D Cartesian acquisition.

    Parameters
    ----------
    y : complex array (C, D, H, W) -- multi-coil 3D k-space (coils, kz, ky, kx), zeros where not sampled.
    segments : list of 1D int arrays -- ky indices of each segment, in acquisition order (time-contiguous groups of shots).
    maps : complex array (C, D, H, W), optional -- coil sensitivities; estimated from the data if None.
    mask2d : bool array (D, H), optional -- sampled (kz, ky) points; inferred from the nonzero entries of y if None.
    voxel_mm : isotropic voxel size in mm.
    tv_weight : TV weight lambda for data normalised so that the 99.5th percentile of |uncorrected image| is 1.
    multi_anchor : run the multi-anchor extension (EventMC); False gives a single run (EventMC-S).
    normalize : rescale y as above before estimation (set False if y is already normalised).

    Returns
    -------
    dict with 'image' (complex (D, H, W), in the scale of the input), 'poses' (N_s, 6: rotation vector in degrees, translation
    in mm, relative to the anchor segment), 'anchor', 'selected' candidate, 'residuals' of all candidates, and 'prerun'.
    """
    y = torch.as_tensor(y).to(device or ('cuda' if torch.cuda.is_available() else 'cpu'), torch.complex64)
    if mask2d is None:
        mask2d = (y.abs().sum((0, 3)) > 0).cpu().numpy()
    if maps is None:
        c = Core(y.shape, torch.ones_like(y), segments, mask2d, voxel_mm, tv_weight, device=y.device.type)
        c.use_maps(c.maps_from_data(y))
    else:
        c = Core(y.shape, maps, segments, mask2d, voxel_mm, tv_weight, device=y.device.type)
    scale = 1.0
    if normalize:
        scale = float(torch.quantile(c.comb(y).abs().flatten()[::97], .995)); y = y / scale
    t0 = time.monotonic(); pre = prerun(c, y)
    log(f'pre-run: anchor {pre["anchor"]}, candidates {pre["candidates"]}, events {pre["events"]}, '
        f'Guard-flagged {pre["guard_flagged"]}, {time.monotonic() - t0:.0f} s')
    a1 = pre['anchor']; cands = {}
    log('EventMC-S (full resolution):'); X, x = eventmc_s(c, y, a1, log=log)
    cands['full@%d' % a1] = dict(X=X, x=x, anchor=a1, D=c.residual(x, X, y))
    if multi_anchor:
        for a in [a1] + pre['candidates'][1:2]:
            log(f'candidate at half resolution, anchor {a}:'); Xk, xk = eventmc_half(c, y, a, log=log)
            cands['half@%d' % a] = dict(X=Xk, x=xk, anchor=a, D=c.residual(xk, Xk, y))
    sel = min(cands, key=lambda k: cands[k]['D'])
    log(f'selected {sel} (residuals: ' + ', '.join(f'{k} {v["D"]:.4g}' for k, v in cands.items()) + f'); total {time.monotonic() - t0:.0f} s')
    best = cands[sel]
    return dict(image=(best['x'] * scale).cpu(), poses=np.asarray(best['X']), anchor=best['anchor'], selected=sel,
                residuals={k: v['D'] for k, v in cands.items()}, prerun=pre,
                candidates={k: dict(poses=np.asarray(v['X']), anchor=v['anchor']) for k, v in cands.items()})
