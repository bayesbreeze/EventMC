"""Forward model of an interleaved 3D Cartesian acquisition with one rigid pose per segment, and the two updates of the
objective, Eq. (2): ImgEst (`reconstruct`) and MotEst (`fit_pose`, `fit_poses_outward`).

Conventions
-----------
* k-space ``y`` has shape (C, D, H, W): coils, then the k-space axes (kz, ky, kx). Each shot acquires one ky index,
  with all kx and the sampled kz of that ky. A *segment* is an array of ky indices acquired consecutively, in
  acquisition order.
* A pose is (rotation vector in degrees, translation in mm) about the FOV centre: the object is moved by
  u -> R (u - c) + c + t. Voxels are isotropic with size ``voxel_mm``.
* Rigid transforms are applied with Fourier operations: each axis rotation as three 1D Fourier shears and the
  translation as a k-space phase ramp (periodic boundary). Coil sensitivities are scanner-fixed: they do not move with
  the head.
"""
import numpy as np
import torch

from .fourier import fftc, ifftc, tv3
from .params import TV_WEIGHT, IMAGE_ITERS, POSE_ITERS, ALIAS_MAX_PERIOD


class EncodingModel:
    """Encoding operator of one acquisition and the image and motion updates that use it.

    Parameters
    ----------
    shape : (C, D, H, W) of the k-space data.
    maps : complex (C, D, H, W) coil sensitivities.
    segments : list of 1D integer arrays; segment s holds its ky indices in acquisition order.
    mask2d : boolean (D, H) array of sampled (kz, ky) points (all kx are acquired for a sampled point).
    voxel_mm : isotropic voxel size in mm.
    tv_weight : TV weight lambda of Eq. (2), for normalised data.
    """

    def __init__(self, shape, maps, segments, mask2d, voxel_mm=1.0, tv_weight=TV_WEIGHT, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.C, self.D, self.H, self.W = shape
        self.voxel_mm = float(voxel_mm)
        self.tv_weight = float(tv_weight)
        self.segments = [np.asarray(s, int) for s in segments]
        self.n_segments = len(self.segments)
        self.anchor = self.n_segments // 2                 # set by the pre-run (anchor.prerun)
        self.mask2d = np.asarray(mask2d, bool)
        self.sampled = torch.tensor(self.mask2d, device=self.device).to(torch.complex64)
        measured_kz = torch.tensor(self.mask2d.any(1), device=self.device).to(torch.complex64)
        self.measured_kz = measured_kz[None, :, None, None]
        self.centre = np.array([(self.D - 1) / 2, (self.H - 1) / 2, (self.W - 1) / 2])
        self.ky_offsets = torch.arange(self.H, device=self.device, dtype=torch.float64) - self.H // 2
        self.set_maps(torch.as_tensor(maps).to(self.device, torch.complex64))

    # ------------------------------------------------------------------ coil sensitivities and simple images
    def set_maps(self, maps):
        self.maps = maps
        self.maps_energy = maps.abs().square().sum(0).clamp_min(1e-6)

    def estimate_maps(self, y):
        """Coil sensitivities from the (motion-corrupted) data: Hann-windowed 24^3 k-space centre, normalised by its
        root-sum-of-squares (appendix A.1)."""
        C, D, H, W = y.shape
        size = [min(24, n) for n in (D, H, W)]             # 24^3 centre (smaller only for tiny grids)
        windows = [torch.hann_window(n, periodic=False, device=self.device) for n in size]
        centre = [slice(n // 2 - m // 2, n // 2 - m // 2 + m) for n, m in zip((D, H, W), size)]
        window3d = windows[0][:, None, None] * windows[1][None, :, None] * windows[2][None, None, :]
        low = torch.zeros_like(y)
        low[:, centre[0], centre[1], centre[2]] = y[:, centre[0], centre[1], centre[2]] * window3d
        low = ifftc(low)
        rss = low.abs().square().sum(0).sqrt()
        return low / rss.clamp_min(rss.max() * 1e-3)

    def coil_combine(self, k):
        """Zero-filled, sensitivity-weighted coil combination of k-space k (the uncorrected image x0, Eq. (A.1))."""
        return (self.maps.conj() * ifftc(k)).sum(0) / self.maps_energy

    # ------------------------------------------------------------------ rigid transform
    def _tensor(self, values):
        return torch.tensor(np.asarray(values), device=self.device, dtype=torch.float32)

    @staticmethod
    def _rotation_matrix(rotvec_deg):
        w = torch.deg2rad(rotvec_deg)
        z = w.new_zeros(())
        K = torch.stack([z, -w[2], w[1], w[2], z, -w[0], -w[1], w[0], z]).reshape(3, 3)
        return torch.linalg.matrix_exp(K)

    @staticmethod
    def _along(v, axis):
        """Reshape a 1D tensor to broadcast along one of the three image axes."""
        shape = [1, 1, 1]
        shape[axis] = -1
        return v.reshape(shape)

    def _grid(self):
        """Voxel coordinates relative to the centre and centred FFT frequencies, per axis (cached)."""
        if not hasattr(self, '_grid_cache'):
            f64 = dict(device=self.device, dtype=torch.float64)
            sizes = (self.D, self.H, self.W)
            coords = [torch.arange(n, **f64) - self.centre[d] for d, n in enumerate(sizes)]
            freqs = [torch.arange(n, **f64) - n // 2 for n in sizes]
            self._grid_cache = (sizes, coords, freqs)
        return self._grid_cache

    def _shear(self, v, axis, by, amount):
        """Shift each line along `axis` by `amount` times its coordinate along `by`, as a Fourier phase."""
        sizes, coords, freqs = self._grid()
        shift = -amount * coords[by]
        phase = torch.exp(-2j * np.pi * self._along(freqs[axis], axis) * self._along(shift, by) / sizes[axis])
        return ifftc(fftc(v, dims=(axis,)) * phase.to(v.dtype), dims=(axis,))

    def move(self, x, rotvec_deg, translation_mm):
        """Image x moved by the pose (rotation vector in degrees, translation in mm): x is sampled at
        M (r - c - t) + c with M = R^T, where M is factored into rotations about the three array axes, each applied as
        three Fourier shears."""
        sizes, coords, freqs = self._grid()
        rotvec = rotvec_deg.to(torch.float64)
        shift = (translation_mm / self.voxel_mm).to(torch.float64)          # in voxels
        v = x
        if rotvec.requires_grad or bool(rotvec.abs().max() > 0):
            M = self._rotation_matrix(rotvec).T
            angles = (torch.atan2(-M[1, 2], M[2, 2]), torch.asin(-M[0, 2]), torch.atan2(-M[0, 1], M[0, 0]))
            for axis, phi in zip((0, 1, 2), angles):
                p, q = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]
                a, b = -torch.tan(phi / 2), torch.sin(phi)
                v = self._shear(v, p, q, a)
                v = self._shear(v, q, p, b)
                v = self._shear(v, p, q, a)
        if shift.requires_grad or bool(shift.abs().max() > 0):
            V = fftc(v)
            for d in range(3):
                V = V * torch.exp(-2j * np.pi * self._along(freqs[d], d) * shift[d] / sizes[d]).to(V.dtype)
            v = ifftc(V)
        return v

    # ------------------------------------------------------------------ encoding
    def ky_operator(self, ky):
        """Partial DFT along ky for the given ky indices, and those indices as a tensor."""
        ky = np.asarray(ky)
        k = torch.tensor(ky, device=self.device, dtype=torch.float64) - self.H // 2
        dft = torch.exp(-2j * np.pi * k[:, None] * self.ky_offsets[None] / self.H) / np.sqrt(self.H)
        dft = dft.to(torch.complex64)
        return dft, torch.as_tensor(ky, device=self.device)

    def encode(self, x, ky_op):
        """Multi-coil k-space of image x on the ky lines of `ky_op`, masked by the (kz, ky) sampling pattern."""
        dft, ky = ky_op
        k = ((self.maps * x[None]).permute(0, 1, 3, 2) @ dft.T).permute(0, 1, 3, 2)
        k = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(k, dim=1), dim=1, norm='ortho'), dim=1)
        k = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(k, dim=3), dim=3, norm='ortho'), dim=3)
        k = k * self.measured_kz
        return k * self.sampled[None, :, ky, None]

    # ------------------------------------------------------------------ ImgEst
    def reconstruct(self, x_init, poses, y, segment_ids, iters=IMAGE_ITERS, band=None):
        """ImgEst: approximately solve min_x 0.5 sum_{s in segment_ids} ||P(A_s(theta_s) x - y_s)||^2 + lambda TV(x) by
        L-BFGS from x_init, with the poses fixed. `band` = (k-space mask (D, W), per-segment ky subsets) restricts the
        data to the central band P_f of Stage B; None uses all samples."""
        kmask, rows_of = (None, None) if band is None else band
        rows = (lambda s: self.segments[s]) if rows_of is None else (lambda s: rows_of[s])
        terms = [(self.ky_operator(rows(s)), y[:, :, rows(s), :], self._tensor(poses[s]))
                 for s in segment_ids if len(rows(s))]
        weight = 1. if kmask is None else kmask[None, :, None, :]
        v = torch.stack((x_init.real, x_init.imag)).detach().clone().requires_grad_(True)
        opt = torch.optim.LBFGS([v], max_iter=iters, history_size=8, line_search_fn='strong_wolfe',
                                tolerance_grad=0, tolerance_change=0)

        def closure():
            opt.zero_grad()
            reg = self.tv_weight * tv3(torch.complex(v[0], v[1]))
            reg.backward()
            total = float(reg.detach())
            for ky_op, y_s, pose in terms:                     # one backward pass per segment keeps memory low
                x = torch.complex(v[0], v[1])
                loss = .5 * ((self.encode(self.move(x, pose[:3], pose[3:]), ky_op) - y_s) * weight).abs().square().sum()
                loss.backward()
                total += float(loss.detach())
            return torch.tensor(total)
        opt.step(closure)
        return torch.complex(v[0], v[1]).detach()

    # ------------------------------------------------------------------ MotEst
    def fit_pose(self, image, init_pose, y, s, iters=POSE_ITERS, band=None):
        """MotEst for one segment: fit the pose of segment s to the fixed image by L-BFGS from init_pose."""
        kmask, rows_of = (None, None) if band is None else band
        rows = self.segments[s] if rows_of is None else rows_of[s]
        if len(rows) == 0:
            return np.asarray(init_pose)
        ky_op = self.ky_operator(rows)
        y_s = y[:, :, rows, :]
        weight = 1. if kmask is None else kmask[None, :, None, :]
        p = self._tensor(init_pose).requires_grad_(True)
        opt = torch.optim.LBFGS([p], max_iter=iters, history_size=10, line_search_fn='strong_wolfe',
                                tolerance_grad=0, tolerance_change=0)

        def closure():
            opt.zero_grad()
            loss = .5 * ((self.encode(self.move(image, p[:3], p[3:]), ky_op) - y_s) * weight).abs().square().sum()
            loss.backward()
            return loss
        opt.step(closure)
        return p.detach().cpu().numpy()

    def segment_loss(self, image, pose, y, s):
        """Data residual 0.5 ||P_{s}(A_s(theta) x - y)||^2 of segment s for a given image and pose."""
        with torch.no_grad():
            moved = self.move(image, self._tensor(pose[:3]), self._tensor(pose[3:]))
            model = self.encode(moved, self.ky_operator(self.segments[s]))
            return float(.5 * (model - y[:, :, self.segments[s], :]).abs().square().sum())

    def fit_poses_outward(self, y, references, previous):
        """Stage A pose fitting (Section 3.1): fit every segment except the anchor, from the nearest to the anchor to
        the farthest, each against its own reference image. Each fit runs from three starting poses (Eq. (A.6)) and
        keeps the one with the smallest segment residual.

        references : list with one reference image per segment.
        previous : (N_s, 6) smoothed poses of the previous round (zeros in Round 1).
        Returns the raw fitted poses (N_s, 6), zero at the anchor."""
        fitted = np.zeros((self.n_segments, 6))
        order = sorted((s for s in range(self.n_segments) if s != self.anchor), key=lambda s: (abs(s - self.anchor), s))
        for s in order:
            n = s + 1 if s < self.anchor else s - 1             # neighbour closer to the anchor, already fitted
            starts = (fitted[n],                                # the neighbour's pose
                      previous[s],                              # the previous-round pose
                      fitted[n] + previous[s] - previous[n])    # the neighbour plus the previous-round step
            best_loss, best_pose = None, None
            for init in starts:
                pose = self.fit_pose(references[s], init, y, s)
                loss = self.segment_loss(references[s], pose, y, s)
                if best_loss is None or loss < best_loss:
                    best_loss, best_pose = loss, pose
            ky_centre = abs(float(self.segments[s].mean()) - self.H // 2)
            period = self.H / max(ky_centre, 1e-6)
            if period < ALIAS_MAX_PERIOD:   # translation ambiguous modulo the period; never true for interleaving
                base = best_pose
                for k in (-2, -1, 1, 2):
                    pose = self.fit_pose(references[s], base + np.r_[0, 0, 0, 0, k * period, 0], y, s)
                    loss = self.segment_loss(references[s], pose, y, s)
                    if loss < best_loss:
                        best_loss, best_pose = loss, pose
            fitted[s] = best_pose
        return fitted

    # ------------------------------------------------------------------ objective
    def data_residual(self, x, poses, y):
        """Full-data residual sum_s ||A_s(theta_s) x - y_s||^2, used to rank the multi-anchor candidates."""
        with torch.no_grad():
            total = 0.
            for s, rows in enumerate(self.segments):
                moved = self.move(x, self._tensor(poses[s][:3]), self._tensor(poses[s][3:]))
                model = self.encode(moved, self.ky_operator(rows))
                total += (model - y[:, :, rows, :]).abs().square().sum()
            return float(total)

