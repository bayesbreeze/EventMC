"""Stage A (fitting to event-guided references) and Stage B (coarse-to-fine refinement), Algorithm 1 lines 3-13."""
import numpy as np
import torch

from .core import FOLDS, REF_ITERS, POSE_ITERS, smooth

STAGE_A_ROUNDS = 5
STAGE_B_SCHEDULE = ((0.25, 1), (0.5, 1), (1.0, 2))   # (central fraction f, rounds): f = 0.25, 0.5, 1, 1


def first_round(c, y):
    """One Stage A round without events: each reference omits only the target's s mod 4 group (Eq. (3) with A = all
    segments). Starts from zero poses and the uncorrected image. Returns smoothed poses, raw fits and detected events."""
    x0 = c.comb(y); X = np.zeros((c.NBK, 6)); refs = [x0] * FOLDS
    refs = [c.tv_recon(refs[f], X, y, [b for b in range(c.NBK) if b % FOLDS != f], REF_ITERS) for f in range(FOLDS)]
    P, _ = c.centre_out(y, refs, X)
    ev, _ = c.detect(np.array([P[i + 1] - P[i] for i in c.KEEP])); evp = {c.KEEP[q] for q in ev}
    X = smooth(c.NBK, {}, {b: P[b] for b in range(c.NBK) if b != c.anchor}, events=evp, anchor=c.anchor)
    return X, P, evp


def stable_interval(c, evp):
    """A: the anchor's interval (between events) without the segments next to an event; the anchor is always kept."""
    mixed = {q for e in evp for q in (e, e + 1)} - {c.anchor}; cl = np.zeros(c.NBK, int); k = 0
    for b in range(1, c.NBK):
        if (b - 1) in evp:
            k += 1
        cl[b] = k
    return [b for b in range(c.NBK) if cl[b] == cl[c.anchor] and b not in mixed]


def stage_a(c, y, rounds=STAGE_A_ROUNDS):
    """Stage A with the anchor c.anchor fixed. Round 1 as first_round(); later rounds rebuild the references from the stable
    interval A with the current poses (S_s = A for s outside A, A minus the target's s mod 4 group otherwise), refit the
    poses from the anchor outwards, redetect events and smooth except across events."""
    allb = list(range(c.NBK)); x0 = c.comb(y)
    X, P, evp = first_round(c, y); refs_f = [x0] * FOLDS
    for r in range(2, rounds + 1):
        A = stable_interval(c, evp)
        refA = c.tv_recon(x0, X, y, A, REF_ITERS)
        refs_f = [c.tv_recon(refs_f[f], X, y, [b for b in A if b % FOLDS != f], REF_ITERS) for f in range(FOLDS)]
        refs = [refs_f[b % FOLDS] if b in A else refA for b in allb]
        P, _ = c.centre_out(y, refs, X)
        ev, _ = c.detect(np.array([P[q + 1] - P[q] for q in c.KEEP])); evp = {c.KEEP[q] for q in ev}
        X = smooth(c.NBK, {}, {b: P[b] for b in allb if b != c.anchor}, events=evp, anchor=c.anchor)
    return X, evp


def stage_b(c, y, X0, schedule=STAGE_B_SCHEDULE):
    """Stage B: alternate ImgEst and MotEst with all segments on the central fraction f of each k-space axis, f increasing
    (low frequencies are less sensitive to small pose errors). One shared image, started from the uncorrected image."""
    D, H, W = c.D, c.H, c.W; allb = list(range(c.NBK)); X = X0.copy(); x = c.comb(y)
    kz = (torch.arange(D, device=c.dev) - D // 2).abs().float(); kx = (torch.arange(W, device=c.dev) - W // 2).abs().float()
    for frac, rounds in schedule:
        km = ((kz[:, None] <= frac * D / 2) & (kx[None, :] <= frac * W / 2)).to(torch.complex64)
        rows = [np.array([r for r in c.blocks[b] if abs(c.row_ky[r] - H // 2) <= frac * H / 2]) for b in allb]
        for _ in range(rounds):
            x = c.tv_recon(x, X, y, allb, REF_ITERS, kmask=km, rows=rows); Pn = X.copy()
            for b in allb:
                if b != c.anchor:
                    Pn = c.pose_refine(x, Pn, y, [b], 2 * POSE_ITERS, kmask=km, rows=rows)
            X = Pn
    return X


def final_reconstruction(c, y, X, iters=None):
    """Algorithm 1, line 14: TV reconstruction from all samples with the poses fixed, starting from the uncorrected image."""
    from .core import FINAL_ITERS
    return c.tv_recon(c.comb(y), X, y, list(range(c.NBK)), FINAL_ITERS if iters is None else iters)
