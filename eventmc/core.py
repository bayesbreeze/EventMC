"""Forward model, image update (ImgEst), motion update (MotEst), event detection and event-aware smoothing.

Conventions
-----------
* k-space data ``y`` has shape (C, D, H, W): coils, then the three k-space axes (kz, ky, kx). Each shot acquires one ky
  index (all kz, kx); a *segment* is a list of ky indices acquired consecutively in time with one rigid pose.
* A pose is (rotation vector in degrees, translation in millimetres), about the FOV centre; the object is moved by
  x -> R (x - c) + c + t. Voxels are assumed isotropic with size ``voxel_mm``.
* Rigid transforms are applied without interpolation: rotations by three-pass Fourier shears, translations by k-space
  phase ramps (periodic boundary).

The numerical code is the implementation used for the ISBI 2027 paper (Algorithm 1 and Eq. (2)), reorganised into a
package; only interfaces, names and comments were changed.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation  # noqa: F401  (kept for users converting poses)
import torch

from .fourier import fftc, ifftc, tv3

# Algorithm constants used in the paper (Section 3 and Algorithm 1).
TV_LAM = 1e-3          # TV weight lambda for normalised data (Eq. (2)); 3e-3 was used for PMoC3D
REF_ITERS = 15         # L-BFGS iterations per reference / image update (ImgEst)
POSE_ITERS = 15        # MotEst uses 2 * POSE_ITERS iterations per segment
FINAL_ITERS = 30       # final TV reconstruction (Algorithm 1, line 14)
FOLDS = 4              # segments are grouped by s mod 4 when excluding the target from its reference (Eq. (3))
ALIAS_MAX_PERIOD = 5.  # extra AP starts for segments whose AP period is shorter than this (never triggered for interleaving)

POINTS = np.concatenate([np.eye(3) * 60, -np.eye(3) * 60])   # six probe points 60 mm from the centre (RMS displacement)

# event-aware smoothing (Pose fitting and events, Section 3.1)
SIG_REL, SIG_ABS, DELTA, LAM = .4, .3, .3, 1 / .15 ** 2


class Budget(RuntimeError):
    pass


def rot_np(rv):
    """Rotation matrix from a rotation vector in degrees (Rodrigues)."""
    w = np.deg2rad(np.asarray(rv, float)); th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mag6(v):
    """Root-mean-square displacement (mm) of the six probe points under pose difference v."""
    return float(np.sqrt(np.mean(np.sum((POINTS @ rot_np(v[:3]).T - POINTS + v[3:]) ** 2, 1))))


def smooth(N, rel, absobs, sig_rel=SIG_REL, sig_abs=SIG_ABS, lam=LAM, delta=DELTA, iters=30, events=(), keep=(), anchor=0):
    """Event-aware smoothing of segment poses (N, 6) with the anchor fixed to zero.

    Data terms: absolute observations ``absobs`` {segment: pose} (and optional relative ones ``rel``); prior: Huber penalty on
    second temporal differences (IRLS), switched off for differences that touch an event in ``events``. Each of the six
    components is solved independently."""
    X = np.zeros((N, 6)); free = [b for b in range(N) if b != anchor]; col = {b: k for k, b in enumerate(free)}
    for c in range(6):
        w = np.ones(max(N - 2, 0))
        for _ in range(iters):
            rows = []; y = []

            def row(coefs, val, scale):
                r = np.zeros(N - 1)
                for b, a in coefs:
                    if b != anchor:
                        r[col[b]] += a
                rows.append(r * scale); y.append(val * scale)
            for i, z in rel.items():
                row([(i + 1, 1.), (i, -1.)], z[c], 1 / sig_rel)
            for b, p in absobs.items():
                if b != anchor:
                    row([(b, 1.)], p[c], 1 / sig_abs)
            for j in range(1, N - 1):
                if ((j - 1) in events or j in events) and j not in keep:
                    continue
                row([(j + 1, 1.), (j, -2.), (j - 1, 1.)], 0., np.sqrt(lam * w[j - 1]))
            sol = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)[0]; x = np.zeros(N); x[free] = sol
            d = x[2:] - 2 * x[1:-1] + x[:-2]; w_new = np.minimum(1., delta / np.maximum(np.abs(d), 1e-12))
            if np.allclose(w_new, w, atol=1e-6):
                break
            w = w_new
        X[:, c] = x
    return X


class Core:
    """Encoding operator and the per-segment image/motion updates for one acquisition.

    Parameters
    ----------
    shape : (C, D, H, W) of the k-space data.
    maps : coil sensitivities, complex (C, D, H, W), scanner-fixed (they do not move with the head).
    segments : list of 1D integer arrays; segment s holds the ky indices acquired in segment s, in acquisition order.
    mask2d : boolean (D, H) array of sampled (kz, ky) points (all kx are acquired for a sampled point).
    voxel_mm : isotropic voxel size in mm (poses are in mm and degrees).
    tv_weight : TV weight lambda of Eq. (2) for normalised data.
    """

    def __init__(s, shape, maps, segments, mask2d, voxel_mm=1.0, tv_weight=TV_LAM, device=None):
        s.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu'); s.deadline = None
        s.C, s.D, s.H, s.W = shape; D, H, W = s.D, s.H, s.W
        s.voxel_mm = float(voxel_mm); s.tv_weight = float(tv_weight)
        M = np.asarray(mask2d, bool); s.measured_kz = M.any(1); s.M2np = M
        s.M2 = torch.tensor(M, device=s.dev).to(torch.complex64)
        s.mk = torch.tensor(s.measured_kz, device=s.dev).to(torch.complex64)[None, :, None, None]
        s.cen = np.array([(D - 1) / 2, (H - 1) / 2, (W - 1) / 2]); s.cen_t = s.tt(s.cen)
        s.hh = torch.arange(H, device=s.dev, dtype=torch.float64) - H // 2
        s.row_ky = np.arange(H); s.n_ids = H
        s.blocks = [np.asarray(b, int) for b in segments]; s.NBK = len(s.blocks)
        s.anchor = s.NBK // 2; s.KEEP = list(range(s.NBK - 1))
        s.time_of_row = np.full(H, -1, int); order = np.concatenate(s.blocks); s.time_of_row[order] = np.arange(len(order))
        s.use_maps(torch.as_tensor(maps).to(s.dev, torch.complex64))

    # ---------- basics ----------
    def tt(s, v):
        return torch.tensor(np.asarray(v), device=s.dev, dtype=torch.float32)

    def check(s):
        if s.deadline is not None and time.monotonic() > s.deadline:
            raise Budget('budget')

    def use_maps(s, maps):
        s.maps = maps; s.den = maps.abs().square().sum(0).clamp_min(1e-6)

    def rot_t(s, rv):
        w = torch.deg2rad(rv); z = w.new_zeros(())
        K = torch.stack([z, -w[2], w[1], w[2], z, -w[0], -w[1], w[0], z]).reshape(3, 3)
        return torch.linalg.matrix_exp(K)

    # ---------- rigid transform (Fourier shears + phase ramp) ----------
    def warp(s, x, rv, t):
        """Move image x by pose (rv deg, t mm)."""
        return s.warp_fft(x, rv, t / s.voxel_mm)

    def _fft_grid(s):
        if not hasattr(s, '_fftcache'):
            D, H, W = s.D, s.H, s.W; f64 = dict(device=s.dev, dtype=torch.float64)
            Np = Nf = (D, H, W); coords = [torch.arange(n, **f64) - s.cen[d] for d, n in enumerate(Np)]
            s._fftcache = (Np, Nf, 1, coords, [torch.arange(n, **f64) - n // 2 for n in Nf])
        return s._fftcache

    @staticmethod
    def _bshape(v, d):
        shp = [1, 1, 1]; shp[d] = -1
        return v.reshape(shp)

    def _fshear(s, v, ax, by, amt, G):
        _, Nf, g, coords, freqs = G; sh = g * (-amt * coords[by])
        ph = torch.exp(-2j * np.pi * s._bshape(freqs[ax], ax) * s._bshape(sh, by) / Nf[ax]).to(v.dtype)
        return ifftc(fftc(v, dims=(ax,)) * ph, dims=(ax,))

    def warp_fft(s, x, rv, t):
        """Sample x at M (r - c - t) + c, M = rot(rv)^T, without interpolation: M = R0 R1 R2 (rotations about the array axes),
        each applied as three Fourier shears, then a phase-ramp translation (t in voxels)."""
        G = s._fft_grid(); Np, Nf, g, coords, freqs = G; v = x
        rv = rv.to(torch.float64); t = t.to(torch.float64)
        if rv.requires_grad or bool(rv.abs().max() > 0):
            M = s.rot_t(rv).T
            angles = (torch.atan2(-M[1, 2], M[2, 2]), torch.asin(-M[0, 2]), torch.atan2(-M[0, 1], M[0, 0]))
            for axis, phi in zip((0, 1, 2), angles):
                p, q = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]; al = -torch.tan(phi / 2); be = torch.sin(phi)
                v = s._fshear(v, p, q, al, G); v = s._fshear(v, q, p, be, G); v = s._fshear(v, p, q, al, G)
        if t.requires_grad or bool(t.abs().max() > 0):
            V = fftc(v)
            for d in range(3):
                V = V * torch.exp(-2j * np.pi * s._bshape(freqs[d], d) * g * t[d] / Nf[d]).to(V.dtype)
            v = ifftc(V)
        return v

    # ---------- encoding ----------
    def E(s, rows):
        """Partial DFT along ky for the given ky indices (rows)."""
        H = s.H; ky = s.row_ky[np.asarray(rows)]
        Er = (torch.exp(-2j * np.pi * (torch.tensor(np.asarray(ky), device=s.dev, dtype=torch.float64) - H // 2)[:, None]
                        * s.hh[None] / H) / np.sqrt(H)).to(torch.complex64)
        Er.ky = torch.as_tensor(np.asarray(ky), device=s.dev)
        return Er

    def rows_k(s, x, Er):
        """k-space samples of image x on the ky rows of Er, all coils, masked by the (kz, ky) sampling pattern."""
        K = ((s.maps * x[None]).permute(0, 1, 3, 2) @ Er.T).permute(0, 1, 3, 2)
        K = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(K, dim=1), dim=1, norm='ortho'), dim=1)
        K = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(K, dim=3), dim=3, norm='ortho'), dim=3) * s.mk
        return K * s.M2[None, :, Er.ky, None]

    def comb(s, k):
        """Uncorrected coil-combined image of k-space k."""
        return (s.maps.conj() * ifftc(k[:, :, :s.H])).sum(0) / s.den

    def maps_from_data(s, y):
        """Coil sensitivities from the (motion-corrupted) data: Hann-windowed 24^3 k-space centre, normalised by its RSS."""
        y = y[:, :, :s.H]
        C, Dn, Hn, Wn = y.shape; L = [min(24, n) for n in (Dn, Hn, Wn)]      # 24^3 centre (smaller only for tiny grids)
        win = [torch.hann_window(l, periodic=False, device=s.dev) for l in L]
        low = torch.zeros_like(y); sl = [slice(n // 2 - l // 2, n // 2 - l // 2 + l) for n, l in zip((Dn, Hn, Wn), L)]
        w3 = win[0][:, None, None] * win[1][None, :, None] * win[2][None, None, :]
        low[:, sl[0], sl[1], sl[2]] = y[:, sl[0], sl[1], sl[2]] * w3; low = ifftc(low); rss = low.abs().square().sum(0).sqrt()
        return low / rss.clamp_min(rss.max() * 1e-3)

    # ---------- ImgEst: TV reconstruction with fixed poses ----------
    def tv_recon(s, x, X, y, use, iters, tv_lam=None, kmask=None, rows=None):
        """Approximately solve min_x 0.5 sum_{b in use} ||P(A_b(theta_b) x - y_b)||^2 + lambda TV(x) by L-BFGS from x.
        kmask (D, W) and rows (per-segment ky subsets) restrict the data to a central k-space band (Stage B)."""
        tv_lam = s.tv_weight if tv_lam is None else tv_lam
        v = torch.stack((x.real, x.imag)).detach().clone().requires_grad_(True)
        R = (lambda b: s.blocks[b]) if rows is None else (lambda b: rows[b])
        U = [(s.E(R(b)), y[:, :, R(b), :], s.tt(X[b])) for b in use if len(R(b))]
        M = 1. if kmask is None else kmask[None, :, None, :]
        opt = torch.optim.LBFGS([v], max_iter=iters, history_size=8, line_search_fn='strong_wolfe', tolerance_grad=0, tolerance_change=0)

        def closure():
            s.check(); opt.zero_grad(); reg = tv_lam * tv3(torch.complex(v[0], v[1])); reg.backward(); tot = float(reg.detach())
            for Eg, yg, pz in U:
                xb = torch.complex(v[0], v[1])
                l = .5 * ((s.rows_k(s.warp(xb, pz[:3], pz[3:]), Eg) - yg) * M).abs().square().sum(); l.backward(); tot += float(l.detach())
            return torch.tensor(tot)
        opt.step(closure)
        return torch.complex(v[0], v[1]).detach()

    # ---------- MotEst: per-segment pose fit against a fixed image ----------
    def pose_refine(s, x, X, y, which, iters=POSE_ITERS, kmask=None, rows=None):
        """Fit the poses of segments ``which`` (one at a time, anchor skipped) to image x, starting from X."""
        Xn = X.copy()
        for b in which:
            if b == s.anchor:
                continue
            rb = s.blocks[b] if rows is None else rows[b]
            if len(rb) == 0:
                continue
            Eg = s.E(rb); yg = y[:, :, rb, :]; p = s.tt(X[b]).requires_grad_(True); M = 1. if kmask is None else kmask[None, :, None, :]
            opt = torch.optim.LBFGS([p], max_iter=iters, history_size=10, line_search_fn='strong_wolfe', tolerance_grad=0, tolerance_change=0)

            def closure():
                s.check(); opt.zero_grad(); l = .5 * ((s.rows_k(s.warp(x, p[:3], p[3:]), Eg) - yg) * M).abs().square().sum(); l.backward()
                return l
            opt.step(closure); Xn[b] = p.detach().cpu().numpy()
        return Xn

    def block_loss(s, x, p, y, b):
        with torch.no_grad():
            return float(.5 * (s.rows_k(s.warp(x, s.tt(p[:3]), s.tt(p[3:])), s.E(s.blocks[b])) - y[:, :, s.blocks[b], :]).abs().square().sum())

    def centre_out(s, y, refs, X0, iters=2 * POSE_ITERS):
        """Fit segment poses from the nearest to the anchor to the farthest. Each segment tries three starts (its fitted
        neighbour closer to the anchor, its previous-round pose X0[b], and that neighbour plus the previous-round
        difference) and keeps the lowest data residual. refs: one reference per segment, or one per s mod 4 group."""
        P = np.zeros((s.NBK, 6)); choice = {}
        for b in sorted([q for q in range(s.NBK) if q != s.anchor], key=lambda q: (abs(q - s.anchor), q)):
            n = b + 1 if b < s.anchor else b - 1; ref = refs[b] if len(refs) == s.NBK else refs[b % FOLDS]; best = None
            for tag, init in (('neighbour', P[n]), ('previous', X0[b]), ('neighbour+difference', P[n] + X0[b] - X0[n])):
                Xi = P.copy(); Xi[b] = init; p = s.pose_refine(ref, Xi, y, [b], iters)[b]; l = s.block_loss(ref, p, y, b)
                if best is None or l < best[0]:
                    best = (l, p, tag)
            kc = abs(float(s.row_ky[s.blocks[b]].mean()) - s.H // 2); period = s.H / max(kc, 1e-6)
            if period < ALIAS_MAX_PERIOD:   # AP translation ambiguous modulo FOV_AP/|k_b|; not reached for interleaved segments
                base = best[1]
                for k in (-2, -1, 1, 2):
                    Xi = P.copy(); Xi[b] = base + np.r_[0, 0, 0, 0, k * period, 0]; p = s.pose_refine(ref, Xi, y, [b], iters)[b]
                    l = s.block_loss(ref, p, y, b)
                    if l < best[0]:
                        best = (l, p, best[2] + f'+alias{k:+d}')
            P[b] = best[1]; choice[b] = best[2]
        return P, choice

    # ---------- events ----------
    def detect(s, Z):
        """Events among adjacent-pose differences Z (N-1, 6): m_j > max(3 sigma_j, 0.6 mm), sigma_j = 1.4826 MAD of m over up
        to three boundaries on each side of j."""
        m = np.array([mag6(v) for v in Z]); ev = set()
        for q in range(len(Z)):
            nb = [m[k] for k in range(max(0, q - 3), min(len(Z), q + 4)) if k != q]; sl = 1.4826 * np.median(np.abs(nb - np.median(nb)))
            if m[q] > max(3 * sl, .6):
                ev.add(q)
        return ev, m

    def residual(s, x, X, y):
        """Full-data residual sum_b ||A_b(theta_b) x - y_b||^2 used to rank multi-anchor candidates."""
        with torch.no_grad():
            return float(sum(((s.rows_k(s.warp(x, s.tt(X[b][:3]), s.tt(X[b][3:])), s.E(s.blocks[b])) - y[:, :, s.blocks[b], :]).abs().square().sum())
                             for b in range(s.NBK)))
