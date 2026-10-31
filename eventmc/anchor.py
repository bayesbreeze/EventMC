"""Pre-run and anchor selection (Section 3.1, "Pre-run"; Algorithm 1, line 2; appendix A.2).

The pre-run runs one Stage A round from zero motion with the temporary anchor N_s // 2 and detects events. Segments
next to an event, and segments with likely motion within them (the within-segment check, "Guard"), are removed; the
anchor is the middle remaining segment of the interval with the most remaining segments. The interval with the
second-most remaining segments gives the second anchor used by the multi-anchor extension (Section 3.3)."""
import numpy as np
import torch

from .fourier import fftc, ifftc
from .params import GUARD_SLICES, GUARD_WINDOWS, GUARD_Z, GUARD_MAX_ROUNDS, GUARD_ITERS
from .stages import first_round


class GuardFailed(RuntimeError):
    """The within-segment check could not be evaluated (e.g. no head slices, or no data left to predict from)."""


def _fft2(x):
    return fftc(x, dims=(-2, -1))


def _ifft2(x):
    return ifftc(x, dims=(-2, -1))


def _tv2(u):
    """Smoothed isotropic 2D total variation of a stack of slices (S, kz, ky)."""
    dz = u[:, 1:, :-1] - u[:, :-1, :-1]
    dy = u[:, :-1, 1:] - u[:, :-1, :-1]
    return torch.sqrt(dz.abs().square() + dy.abs().square() + 1e-6).sum()


class GuardSlices:
    """Readout positions used by the within-segment check. After an inverse FFT along the readout (hybrid space), each
    readout position is a 2D multi-coil (kz, ky) problem when there is no motion."""

    def __init__(self, model, y, n_slices=GUARD_SLICES):
        energy = model.coil_combine(y).abs().square().sum((0, 1))                 # per readout position
        head = torch.nonzero(energy > .05 * energy.max()).flatten()
        selected = head[torch.linspace(0, len(head) - 1, n_slices).round().long()]
        self.selected = selected
        self.y = ifftc(y, dims=(3,))[..., selected].permute(3, 0, 1, 2).contiguous()   # (slices, C, kz, ky)
        self.maps = model.maps[..., selected].permute(3, 0, 1, 2).contiguous()
        self.maps_energy = self.maps.abs().square().sum(1).clamp_min(1e-6)
        self.sampled = model.sampled.real
        self.tv_weight = model.tv_weight

    def mask(self, ky):
        """Sampling mask (kz, ky) restricted to the given ky indices."""
        m = torch.zeros(self.y.shape[-1], device=self.y.device)
        m[torch.as_tensor(np.asarray(ky, int), device=self.y.device)] = 1.
        return self.sampled * m[None, :]

    def reconstruct(self, mask, iters=GUARD_ITERS):
        """Zero-motion 2D TV reconstruction of all slices from the samples in `mask`."""
        u0 = (self.maps.conj() * _ifft2(self.y * mask)).sum(1) / self.maps_energy
        v = torch.stack((u0.real, u0.imag)).detach().clone().requires_grad_(True)
        opt = torch.optim.LBFGS([v], max_iter=iters, history_size=8, line_search_fn='strong_wolfe',
                                tolerance_grad=0, tolerance_change=0)

        def closure():
            opt.zero_grad()
            u = torch.complex(v[0], v[1])
            data = .5 * ((_fft2(self.maps * u[:, None]) - self.y) * mask).abs().square().sum()
            loss = data + self.tv_weight * _tv2(u)
            loss.backward()
            return loss
        opt.step(closure)
        return torch.complex(v[0], v[1]).detach()

    def prediction_error(self, u, mask):
        """Squared error of predicting the samples in `mask` from the slices u, and the energy of those samples."""
        with torch.no_grad():
            error = ((_fft2(self.maps * u[:, None]) - self.y) * mask).abs().square().sum((1, 2, 3))
            energy = (self.y * mask).abs().square().sum((1, 2, 3))
            return error, energy


def _robust_z(log_errors, position, segment, excluded):
    """Robust z-score of each sub-window's log error against the non-excluded sub-windows at the same position within
    their segment, in segments at least two segments away (Eq. (A.5))."""
    z = np.zeros(len(log_errors))
    for w in range(len(log_errors)):
        peers = [v for v in range(len(log_errors))
                 if position[v] == position[w] and abs(segment[v] - segment[w]) >= 2 and v not in excluded]
        med = np.median(log_errors[peers])
        mad = 1.4826 * np.median(np.abs(log_errors[peers] - med))
        z[w] = (log_errors[w] - med) / max(mad, .1)
    return z


def guard(model, y, max_rounds=GUARD_MAX_ROUNDS, z_threshold=GUARD_Z):
    """Within-segment check: split each segment into three sub-windows in time, predict each sub-window from all
    other non-excluded sub-windows by 2D TV, and flag sub-windows whose robust z-score exceeds 3. Flagged sub-windows
    are excluded and the check is repeated until no new sub-window is flagged. Returns the segments with a sub-window
    flagged in the last round."""
    windows, position, segment = [], [], []
    for s, rows in enumerate(model.segments):                 # rows are in acquisition order
        for j, part in enumerate(np.array_split(np.array(rows), GUARD_WINDOWS)):
            windows.append(part)
            position.append(j)
            segment.append(s)
    slices = GuardSlices(model, y)
    if len(slices.selected) == 0:
        raise GuardFailed('no head slices')
    excluded, flagged_per_round = set(), []
    for rnd in range(1, max_rounds + 1):
        errors = []
        for w in range(len(windows)):
            context = [r for v, rows in enumerate(windows) if v != w and v not in excluded for r in rows]
            if not context:
                raise GuardFailed(f'context exhausted (round {rnd}, window {w})')
            u = slices.reconstruct(slices.mask(context))
            error, energy = slices.prediction_error(u, slices.mask(windows[w]))
            errors.append(float(error.sum() / energy.sum()))
        errors = np.array(errors)
        if not np.isfinite(errors).all() or (errors <= 0).any():
            raise GuardFailed(f'nonfinite residual (round {rnd})')
        for w in range(len(windows)):
            if not [v for v in range(len(windows))
                    if position[v] == position[w] and abs(segment[v] - segment[w]) >= 2 and v not in excluded]:
                raise GuardFailed(f'empty calibration neighbourhood w={w}')
        z = _robust_z(np.log(errors), position, segment, excluded)
        if not np.isfinite(z).all():
            raise GuardFailed('nonfinite z')
        flagged = sorted(int(w) for w in np.nonzero(z > z_threshold)[0])
        flagged_per_round.append(flagged)
        if set(flagged) <= excluded:
            break
        excluded |= set(flagged)
    return sorted({segment[w] for w in flagged_per_round[-1]})


def remaining_intervals(n_segments, events, excluded):
    """Event-separated intervals, each reduced to its remaining segments (event-adjacent and excluded segments
    removed)."""
    interval = np.cumsum([0] + [int(s - 1 in events) for s in range(1, n_segments)])
    removed = {s for j in events for s in (j, j + 1)} | set(excluded)
    return [[s for s in range(n_segments) if interval[s] == k and s not in removed] for k in sorted(set(interval))]


def prerun(model, y, use_guard=True):
    """Pre-run (Algorithm 1, line 2), run with the temporary anchor N_s // 2 (left set on the model).

    Returns dict(anchor, candidates, events, guard_flagged): the anchor, up to two candidate anchors (largest and
    second-largest remaining interval), the pre-run events and the segments flagged by the within-segment check. The
    check is skipped with use_guard=False, as for PMoC3D in the paper (four shots per segment are too few to split)."""
    model.anchor = model.n_segments // 2
    _, _, events = first_round(model, y)
    flagged = guard(model, y) if use_guard else []
    groups = [g for g in remaining_intervals(model.n_segments, events, flagged) if g]
    order = sorted(range(len(groups)), key=lambda i: -len(groups[i]))        # stable: the earlier interval wins ties
    candidates = [groups[i][len(groups[i]) // 2] for i in order[:2]]
    anchor = candidates[0] if candidates else model.n_segments // 2
    return dict(anchor=int(anchor), candidates=[int(a) for a in candidates], events=sorted(int(e) for e in events),
                guard_flagged=[int(s) for s in flagged])
