"""Centred FFTs and the smoothed isotropic total variation used by all image updates."""
import torch

from .params import TV_EPS


def fftc(x, dims=(-3, -2, -1)):
    """Centred, orthonormal FFT over `dims`."""
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm='ortho'), dim=dims)


def ifftc(x, dims=(-3, -2, -1)):
    """Centred, orthonormal inverse FFT over `dims`."""
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm='ortho'), dim=dims)


def tv3(u):
    """Smoothed isotropic 3D total variation, sum over voxels of sqrt(|grad u|^2 + eps) with forward differences
    (Eq. (A.2))."""
    grads = [u[1:, :-1, :-1] - u[:-1, :-1, :-1],
             u[:-1, 1:, :-1] - u[:-1, :-1, :-1],
             u[:-1, :-1, 1:] - u[:-1, :-1, :-1]]
    return torch.sqrt(sum(g.abs().square() for g in grads) + TV_EPS).sum()
