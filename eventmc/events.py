"""Event detection and event-aware pose smoothing (Section 3.1, "Pose fitting and events"; appendix A.4).

These functions work on segment poses only: NumPy arrays of shape (N_s, 6), with the rotation vector in degrees and
the translation in mm.
"""
import numpy as np

from .params import (PROBE_RADIUS_MM, EVENT_MAD_FACTOR, EVENT_FLOOR_MM, EVENT_NEIGHBOURS,
                     SMOOTH_SIGMA, SMOOTH_LAMBDA, SMOOTH_DELTA, SMOOTH_MAX_ITERS)

# Six probe points on the coordinate axes, PROBE_RADIUS_MM from the FOV centre.
PROBE_POINTS = np.concatenate([np.eye(3) * PROBE_RADIUS_MM, -np.eye(3) * PROBE_RADIUS_MM])


def rotation_matrix(rotvec_deg):
    """Rotation matrix of a rotation vector in degrees (Rodrigues' formula)."""
    w = np.deg2rad(np.asarray(rotvec_deg, float))
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rms_displacement(pose_change):
    """Root-mean-square displacement (mm) of the six probe points under a pose change (Eq. (A.7)).

    This puts rotations and translations on one scale: a translation of d mm gives d, and a rotation of 1 degree about a
    coordinate axis gives about 0.86 mm."""
    moved = PROBE_POINTS @ rotation_matrix(pose_change[:3]).T - PROBE_POINTS + pose_change[3:]
    return float(np.sqrt(np.mean(np.sum(moved ** 2, 1))))


def detect_events(poses):
    """Boundaries j (between segments j and j + 1) whose pose jump m_j exceeds max(3 sigma_j, 0.6 mm).

    sigma_j is 1.4826 times the median absolute deviation of m over up to three boundaries on each side of j.
    Returns (set of event boundaries, array of m_j)."""
    m = np.array([rms_displacement(d) for d in np.diff(poses, axis=0)])
    events = set()
    for j in range(len(m)):
        lo, hi = max(0, j - EVENT_NEIGHBOURS), min(len(m), j + EVENT_NEIGHBOURS + 1)
        neighbours = [m[k] for k in range(lo, hi) if k != j]
        sigma = 1.4826 * np.median(np.abs(neighbours - np.median(neighbours)))
        if m[j] > max(EVENT_MAD_FACTOR * sigma, EVENT_FLOOR_MM):
            events.add(j)
    return events, m


def smooth_poses(raw_poses, events, anchor, sigma=SMOOTH_SIGMA, lam=SMOOTH_LAMBDA, delta=SMOOTH_DELTA,
                 max_iters=SMOOTH_MAX_ITERS):
    """Event-aware smoothing of the raw segment poses, with the anchor pose fixed to zero (Eq. (A.8)).

    Each of the six components is smoothed independently: least squares fidelity to the raw fits (weight 1 / sigma^2)
    plus a Huber penalty (threshold delta, weight lam) on second temporal differences, solved by iteratively reweighted
    least squares. Second differences that touch an event are dropped, so the smoothing never averages across a
    movement."""
    n = len(raw_poses)
    free = [s for s in range(n) if s != anchor]
    column = {s: k for k, s in enumerate(free)}

    def row(coefs, value, scale):
        r = np.zeros(n - 1)
        for s, a in coefs:
            if s != anchor:
                r[column[s]] += a
        return r * scale, value * scale

    smoothed = np.zeros((n, 6))
    for c in range(6):
        weights = np.ones(max(n - 2, 0))
        for _ in range(max_iters):
            rows, rhs = [], []
            for s in free:                                            # fidelity to the raw fits
                r, v = row([(s, 1.)], raw_poses[s][c], 1 / sigma)
                rows.append(r)
                rhs.append(v)
            for j in range(1, n - 1):                                 # second differences centred at j
                if (j - 1) in events or j in events:
                    continue
                r, v = row([(j + 1, 1.), (j, -2.), (j - 1, 1.)], 0., np.sqrt(lam * weights[j - 1]))
                rows.append(r)
                rhs.append(v)
            solution = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
            x = np.zeros(n)
            x[free] = solution
            second_diff = x[2:] - 2 * x[1:-1] + x[:-2]
            new_weights = np.minimum(1., delta / np.maximum(np.abs(second_diff), 1e-12))   # Huber IRLS weights
            if np.allclose(new_weights, weights, atol=1e-6):
                break
            weights = new_weights
        smoothed[:, c] = x
    return smoothed
