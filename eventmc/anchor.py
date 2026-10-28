"""Pre-run and anchor selection (Section 3.1, Pre-run; Algorithm 1 line 2).

The pre-run runs one Stage A round from zero motion with the temporary anchor N_s // 2 and detects events. Segments next to
an event, or with likely motion within them (Guard: a third of their data poorly predicted from all other data by 2D TV
reconstruction), are excluded; the anchor is the middle remaining segment of the interval with the most remaining segments.
The second-largest interval gives the second candidate anchor used by the multi-anchor extension (Section 3.3)."""
import numpy as np
import torch

from .fourier import fftc, ifftc
from .stages import first_round

N_SLICES = 24        # head slices along the readout used by the Guard
GUARD_ROUNDS = 5     # cumulative-exclusion rounds
GUARD_Z = 3.0        # robust z threshold
WINDOWS = 3          # sub-windows per segment

F2 = lambda x: fftc(x, dims=(-2, -1))
IF2 = lambda x: ifftc(x, dims=(-2, -1))


class GuardFailed(RuntimeError):
    pass


def tv2(u):
    dx = u[:, 1:, :-1] - u[:, :-1, :-1]; dy = u[:, :-1, 1:] - u[:, :-1, :-1]
    return torch.sqrt(dx.abs().square() + dy.abs().square() + 1e-6).sum()


class Slices:
    """Hybrid-space slices (inverse FFT along the readout): at zero pose each slice is a 2D (kz, ky) multi-coil problem."""

    def __init__(s, c, y, n_slices=N_SLICES):
        e = c.comb(y).abs().square().sum((0, 1)); idx = torch.nonzero(e > .05 * e.max()).flatten()
        sel = idx[torch.linspace(0, len(idx) - 1, n_slices).round().long()]
        s.sel = sel; s.Y = ifftc(y, dims=(3,))[..., sel].permute(3, 0, 1, 2).contiguous(); s.S = c.maps[..., sel].permute(3, 0, 1, 2).contiguous()
        s.den = s.S.abs().square().sum(1).clamp_min(1e-6); s.M2 = c.M2.real; s.lam = c.tv_weight

    def mask(s, rows):
        m = torch.zeros(s.Y.shape[-1], device=s.Y.device); m[torch.as_tensor(np.asarray(rows, int), device=s.Y.device)] = 1.
        return s.M2 * m[None, :]

    def recon(s, M, iters=40):
        u0 = (s.S.conj() * IF2(s.Y * M)).sum(1) / s.den
        v = torch.stack((u0.real, u0.imag)).detach().clone().requires_grad_(True)
        opt = torch.optim.LBFGS([v], max_iter=iters, history_size=8, line_search_fn='strong_wolfe', tolerance_grad=0, tolerance_change=0)

        def closure():
            opt.zero_grad(); u = torch.complex(v[0], v[1])
            l = .5 * ((F2(s.S * u[:, None]) - s.Y) * M).abs().square().sum() + s.lam * tv2(u); l.backward()
            return l
        opt.step(closure)
        return torch.complex(v[0], v[1]).detach()

    def residual(s, u, Mw):
        with torch.no_grad():
            r = ((F2(s.S * u[:, None]) - s.Y) * Mw).abs().square().sum((1, 2, 3)); ref = (s.Y * Mw).abs().square().sum((1, 2, 3))
            return r, ref


def zphase(le, phase, seg, excl):
    """Robust z of log residuals against sub-windows at the same position in segments at least two segments away."""
    z = np.zeros(len(le))
    for w in range(len(le)):
        nb = [v for v in range(len(le)) if phase[v] == phase[w] and abs(seg[v] - seg[w]) >= 2 and v not in excl]
        med = np.median(le[nb]); mad = 1.4826 * np.median(np.abs(le[nb] - med)); z[w] = (le[w] - med) / max(mad, .1)
    return z


def guard(c, y, max_rounds=GUARD_ROUNDS, z_threshold=GUARD_Z):
    """Segments with likely motion within them: each segment is split into three sub-windows in time; each sub-window is
    predicted from all other (non-excluded) sub-windows by 2D TV reconstruction; z > 3 flags it. Returns flagged segments."""
    tor = np.asarray(c.time_of_row); wins = []; phase = []; seg = []
    for b in range(c.NBK):
        rb = np.array(sorted(c.blocks[b], key=lambda r: tor[r]))
        for j, part in enumerate(np.array_split(rb, WINDOWS)):
            wins.append(part); phase.append(j); seg.append(b)
    sl = Slices(c, y)
    if len(sl.sel) == 0:
        raise GuardFailed('no head slices')
    excl = set(); rounds = []
    for rd in range(1, max_rounds + 1):
        E = []
        for w in range(len(wins)):
            c.check(); ctx = [r for v, rw in enumerate(wins) if v != w and v not in excl for r in rw]
            if not ctx:
                raise GuardFailed(f'context exhausted (round {rd}, window {w})')
            u = sl.recon(sl.mask(ctx), 40); r_, ref = sl.residual(u, sl.mask(wins[w])); E.append(float(r_.sum() / ref.sum()))
        E = np.array(E)
        if not np.isfinite(E).all() or (E <= 0).any():
            raise GuardFailed(f'nonfinite residual (round {rd})')
        for w in range(len(wins)):
            if not [v for v in range(len(wins)) if phase[v] == phase[w] and abs(seg[v] - seg[w]) >= 2 and v not in excl]:
                raise GuardFailed(f'empty calibration neighbourhood w={w}')
        z = zphase(np.log(E), phase, seg, excl)
        if not np.isfinite(z).all():
            raise GuardFailed('nonfinite z')
        flag = sorted(int(w) for w in np.nonzero(z > z_threshold)[0]); rounds.append(flag)
        if set(flag) <= excl:
            break
        excl |= set(flag)
    return sorted({seg[w] for w in rounds[-1]})


def intervals(NBK, events, excluded):
    """Event-separated intervals, each reduced to its remaining segments (event-adjacent and excluded segments removed)."""
    cl = [0] * NBK; k = 0
    for b in range(1, NBK):
        if (b - 1) in events:
            k += 1
        cl[b] = k
    mixed = {q for e in events for q in (e, e + 1)} | set(excluded)
    return [[b for b in range(NBK) if cl[b] == q and b not in mixed] for q in sorted(set(cl))]


def prerun(c, y):
    """Pre-run (Algorithm 1, line 2). Returns the anchor, up to two candidate anchors (largest and second-largest interval),
    the pre-run events and the Guard-flagged segments."""
    c.anchor = c.NBK // 2
    _, _, events = first_round(c, y)
    bad = guard(c, y)
    groups = [g for g in intervals(c.NBK, events, bad) if g]
    order = sorted(range(len(groups)), key=lambda i: -len(groups[i]))      # stable: earlier interval wins ties
    candidates = [groups[i][len(groups[i]) // 2] for i in order[:2]]
    anchor = candidates[0] if candidates else c.NBK // 2
    return dict(anchor=int(anchor), candidates=[int(a) for a in candidates], events=sorted(int(e) for e in events),
                guard_flagged=[int(b) for b in bad])
