"""Synthetic end-to-end example: a small ellipsoid phantom, 4 coils, 13 interleaved segments and two abrupt head movements.

Run:  python examples/synthetic_demo.py            (GPU if available, otherwise CPU; a few minutes on CPU)
      python examples/synthetic_demo.py --device cpu --single     (EventMC-S only, faster)
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eventmc import Core, run_eventmc

D, H, W, C, NS = 48, 78, 48, 4, 13          # kz, ky, kx, coils, segments (6 ky lines each)
VOX = 3.0                                     # mm; a 144 x 234 x 144 mm field of view


def phantom():
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, n) for n in (D, H, W)], indexing='ij')
    img = np.zeros((D, H, W))
    for (cz, cy, cx, az, ay, ax, v) in [(0, 0, 0, .8, .85, .75, 1.), (0, 0, 0, .7, .75, .65, -.6), (.15, .25, .1, .2, .25, .2, .5),
                                         (-.2, -.2, -.1, .25, .18, .22, .35), (0, -.45, .2, .12, .12, .15, .6)]:
        img += v * (((z - cz) / az) ** 2 + ((y - cy) / ay) ** 2 + ((x - cx) / ax) ** 2 <= 1)
    return torch.tensor(img, dtype=torch.complex64)


def coil_maps():
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, n) for n in (D, H, W)], indexing='ij'); maps = []
    for (cz, cy, cx) in [(1.2, 0, 0), (-1.2, 0, 0), (0, 1.3, 0), (0, -1.3, 0)]:
        maps.append(np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / 1.5) * np.exp(1j * (cz * x + cy * z)))
    maps = np.array(maps); maps /= np.sqrt((np.abs(maps) ** 2).sum(0, keepdims=True))
    return torch.tensor(maps, dtype=torch.complex64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default=None); ap.add_argument('--single', action='store_true'); a = ap.parse_args()
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    segments = [np.arange((5 * s) % NS, H, NS) for s in range(NS)]      # interleaved: every segment spans all of ky
    # true poses (rotation vector in degrees, translation in mm): a nod before segment 4 and a shift after segment 9
    truth = np.zeros((NS, 6)); truth[:4] = [4., 0, 0, 0, 3., 0]; truth[10:] = [0, 3., 0, 0, 0, -4.]
    x_true = phantom().to(dev); maps = coil_maps().to(dev)
    sim = Core((C, D, H, W), maps, segments, np.ones((D, H), bool), voxel_mm=VOX, device=dev)
    y = torch.zeros((C, D, H, W), dtype=torch.complex64, device=dev)
    with torch.no_grad():
        for s, rows in enumerate(segments):
            y[:, :, rows, :] = sim.rows_k(sim.warp(x_true, sim.tt(truth[s, :3]), sim.tt(truth[s, 3:])), sim.E(rows))
    g = torch.Generator(device='cpu').manual_seed(0)
    y += (1e-3 * torch.complex(torch.randn(y.shape, generator=g), torch.randn(y.shape, generator=g))).to(dev)
    t0 = time.monotonic()
    out = run_eventmc(y, segments, maps=maps, voxel_mm=VOX, multi_anchor=not a.single, device=dev)
    X = out['poses']; a_ = out['anchor']
    rel = truth - truth[a_]                     # the anchor is in the motion-free interval, so this is the true relative pose
    print(f'\nanchor {a_}, selected {out["selected"]}, {time.monotonic() - t0:.0f} s')
    print('segment | true (rx, ry, ty, tz)      | estimated')
    for s in range(NS):
        print(f'{s:7d} | {rel[s, 0]:5.2f} {rel[s, 1]:5.2f} {rel[s, 4]:5.2f} {rel[s, 5]:5.2f} | {X[s, 0]:5.2f} {X[s, 1]:5.2f} {X[s, 4]:5.2f} {X[s, 5]:5.2f}')
    err = np.abs(X - rel); print(f'max pose error: {err[:, :3].max():.2f} deg, {err[:, 3:].max():.2f} mm')
    xu = sim.comb(y).abs().cpu(); xe = out['image'].abs(); xt = x_true.abs().cpu()
    nrmse = lambda u: float((u - xt).norm() / xt.norm())
    print(f'NRMSE: uncorrected {nrmse(xu):.3f}, EventMC {nrmse(xe):.3f}')


if __name__ == '__main__':
    main()
