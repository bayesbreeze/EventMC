"""Stage A (fitting to event-guided references), Stage B (coarse-to-fine refinement) and the final reconstruction,
Algorithm 1, lines 3-14."""
import numpy as np
import torch

from .events import detect_events, smooth_poses
from .params import GROUPS, STAGE_A_ROUNDS, STAGE_B_SCHEDULE, IMAGE_ITERS, POSE_ITERS, FINAL_ITERS


def _fit_detect_smooth(model, y, references, previous):
    """One Stage A pass after the references are built: outward pose fitting, event detection and smoothing that does
    not cross events. Returns (smoothed poses, raw fits, events)."""
    raw = model.fit_poses_outward(y, references, previous)
    events, _ = detect_events(raw)
    return smooth_poses(raw, events, model.anchor), raw, events


def first_round(model, y):
    """Round 1 (also the pre-run): no events are known yet, so the stable interval is all segments and each reference
    omits only the target's group s mod 4 (Eq. (3)). Starts from zero motion and the uncorrected image.
    Returns (smoothed poses, raw fits, events)."""
    x0 = model.coil_combine(y)
    zero = np.zeros((model.n_segments, 6))
    group_refs = [model.reconstruct(x0, zero, y, [s for s in range(model.n_segments) if s % GROUPS != g])
                  for g in range(GROUPS)]
    references = [group_refs[s % GROUPS] for s in range(model.n_segments)]
    return _fit_detect_smooth(model, y, references, zero)


def stable_interval(model, events):
    """A: the anchor's interval between events, without the segments next to an event; the anchor is always kept
    (Eq. (A.4))."""
    next_to_event = {s for j in events for s in (j, j + 1)} - {model.anchor}
    # interval index of each segment: the number of events before it
    interval = np.cumsum([0] + [int(s - 1 in events) for s in range(1, model.n_segments)])
    return [s for s in range(model.n_segments) if interval[s] == interval[model.anchor] and s not in next_to_event]


def stage_a(model, y, rounds=STAGE_A_ROUNDS):
    """Stage A with the anchor model.anchor fixed. Round 1 as first_round(); Rounds 2..rounds rebuild the references
    from the stable interval A with the current poses (Eq. (3)): targets outside A use one reference from all of A;
    targets in A use A without their group s mod 4. Returns (poses, events).

    Image initialisation (appendix A.3): the reference from A starts from x0 in every round; the group references
    start from x0 in Round 2 and from the same group's reference of the previous round afterwards."""
    x0 = model.coil_combine(y)
    poses, _, events = first_round(model, y)
    group_refs = [x0] * GROUPS
    for _ in range(2, rounds + 1):
        A = stable_interval(model, events)
        ref_all = model.reconstruct(x0, poses, y, A)
        group_refs = [model.reconstruct(group_refs[g], poses, y, [s for s in A if s % GROUPS != g])
                      for g in range(GROUPS)]
        references = [group_refs[s % GROUPS] if s in A else ref_all for s in range(model.n_segments)]
        poses, _, events = _fit_detect_smooth(model, y, references, poses)
    return poses, events


def central_band(model, fraction):
    """The k-space selector P_f: samples within the central fraction f of each k-space axis, as (mask over (kz, kx),
    ky indices of each segment inside the band)."""
    kz = (torch.arange(model.D, device=model.device) - model.D // 2).abs().float()
    kx = (torch.arange(model.W, device=model.device) - model.W // 2).abs().float()
    kmask = ((kz[:, None] <= fraction * model.D / 2) & (kx[None, :] <= fraction * model.W / 2)).to(torch.complex64)
    rows = [np.array([r for r in seg if abs(r - model.H // 2) <= fraction * model.H / 2]) for seg in model.segments]
    return kmask, rows


def stage_b(model, y, poses, schedule=STAGE_B_SCHEDULE):
    """Stage B (Section 3.2): alternate ImgEst and MotEst with all segments on the central band P_f, f increasing.
    One shared image, started from the uncorrected image; one start per pose fit; no events or smoothing."""
    all_segments = list(range(model.n_segments))
    poses = poses.copy()
    x = model.coil_combine(y)
    for fraction, rounds in schedule:
        band = central_band(model, fraction)
        for _ in range(rounds):
            x = model.reconstruct(x, poses, y, all_segments, IMAGE_ITERS, band=band)
            updated = poses.copy()
            for s in all_segments:
                if s != model.anchor:
                    updated[s] = model.fit_pose(x, poses[s], y, s, POSE_ITERS, band=band)
            poses = updated
    return poses


def final_reconstruction(model, y, poses, iters=FINAL_ITERS):
    """Algorithm 1, line 14: TV reconstruction from all samples with the poses fixed, starting from the uncorrected
    image."""
    return model.reconstruct(model.coil_combine(y), poses, y, list(range(model.n_segments)), iters)
