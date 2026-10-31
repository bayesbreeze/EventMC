"""Installation check: run the full EventMC pipeline on a small synthetic phantom (12 coils, 13 interleaved segments,
two abrupt movements) and print the pipeline log, the selected candidate and the output shapes.

The phantom is far smaller and simpler than a brain (64 x 104 x 64 voxels), so the estimated motion here is NOT
representative of accuracy on real data; see the paper for the evaluation on CC359 and PMoC3D.

Run:  python examples/synthetic_demo.py              (about 3 min on a GPU, longer on CPU)
      python examples/synthetic_demo.py --single     (EventMC-S only, faster)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eventmc import EncodingModel, run_eventmc

D, H, W = 64, 104, 64       # kz, ky, kx
N_COILS = 12
N_SEGMENTS = 13             # 8 ky lines each
VOXEL_MM = 1.0


def ellipsoid(z, y, x, centre, radii):
    r2 = ((z - centre[0]) / radii[0]) ** 2 + ((y - centre[1]) / radii[1]) ** 2 + ((x - centre[2]) / radii[2]) ** 2
    return r2 <= 1


def phantom():
    """A head-like phantom: nested ellipsoids, small structures and a smooth texture."""
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, n) for n in (D, H, W)], indexing='ij')
    img = np.zeros((D, H, W))
    for centre, radii, value in [((0, 0, 0), (.8, .85, .75), 1.), ((0, 0, 0), (.7, .75, .65), -.6),
                                 ((.15, .25, .1), (.2, .25, .2), .5), ((-.2, -.2, -.1), (.25, .18, .22), .35),
                                 ((0, -.45, .2), (.12, .12, .15), .6)]:
        img += value * ellipsoid(z, y, x, centre, radii)
    rng = np.random.default_rng(1)
    for _ in range(150):                                   # small structures, like fine detail in a brain
        centre, radii = rng.uniform(-.6, .6, 3), rng.uniform(.04, .12, 3)
        img += rng.uniform(-.4, .4) * ellipsoid(z, y, x, centre, radii)
    f2 = np.add.outer(np.add.outer(np.fft.fftfreq(D) ** 2, np.fft.fftfreq(H) ** 2), np.fft.fftfreq(W) ** 2)
    texture = np.real(np.fft.ifftn(np.fft.fftn(rng.standard_normal((D, H, W))) * np.exp(-f2 / .03)))
    img = np.clip(img, 0, None) * (1 + .3 * texture / texture.std()) * (img > 0)
    return torch.tensor(img, dtype=torch.complex64)


def coil_maps():
    """Smooth sensitivities of coils on a ring around the head, normalised to unit root-sum-of-squares."""
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, n) for n in (D, H, W)], indexing='ij')
    maps = []
    for k in range(N_COILS):
        angle = 2 * np.pi * k / N_COILS
        cz, cy, cx = 1.3 * np.cos(angle), 1.3 * np.sin(angle), .5 * (-1) ** k
        maps.append(np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / 1.2) * np.exp(1j * (cz * x + cy * z)))
    maps = np.array(maps)
    maps /= np.sqrt((np.abs(maps) ** 2).sum(0, keepdims=True))
    return torch.tensor(maps, dtype=torch.complex64)


def simulate(device):
    """k-space of the phantom with one pose per segment: a nod before segment 4 and a shift after segment 9."""
    segments = [np.arange((5 * s) % N_SEGMENTS, H, N_SEGMENTS) for s in range(N_SEGMENTS)]   # interleaved
    truth = np.zeros((N_SEGMENTS, 6))                      # rotation vector (deg), translation (mm)
    truth[:4] = [4., 0, 0, 0, 3., 0]
    truth[10:] = [0, 3., 0, 0, 0, -4.]
    x_true, maps = phantom().to(device), coil_maps().to(device)
    sim = EncodingModel((N_COILS, D, H, W), maps, segments, np.ones((D, H), bool), voxel_mm=VOXEL_MM, device=device)
    y = torch.zeros((N_COILS, D, H, W), dtype=torch.complex64, device=device)
    with torch.no_grad():
        for s, rows in enumerate(segments):
            pose = torch.tensor(truth[s], dtype=torch.float32, device=device)
            y[:, :, rows, :] = sim.encode(sim.move(x_true, pose[:3], pose[3:]), sim.ky_operator(rows))
    g = torch.Generator(device='cpu').manual_seed(0)
    y += (1e-3 * torch.complex(torch.randn(y.shape, generator=g), torch.randn(y.shape, generator=g))).to(device)
    return y, segments, maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None)
    parser.add_argument('--single', action='store_true', help='run EventMC-S only')
    args = parser.parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    y, segments, maps = simulate(device)
    t0 = time.monotonic()
    out = run_eventmc(y, segments, maps=maps, voxel_mm=VOXEL_MM, multi_anchor=not args.single, device=device)
    print(f'\nanchor {out["anchor"]}, selected {out["selected"]}, residuals {out["residuals"]}')
    print(f'image {tuple(out["image"].shape)}, poses {out["poses"].shape}, {time.monotonic() - t0:.0f} s')


if __name__ == '__main__':
    main()
