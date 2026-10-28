"""Centred FFTs and the smoothed isotropic total variation used by all TV reconstructions."""
import torch


def fftc(x, dims=(-3, -2, -1)):
    """Centred, orthonormal FFT over `dims`."""
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm='ortho'), dim=dims)


def ifftc(x, dims=(-3, -2, -1)):
    """Centred, orthonormal inverse FFT over `dims`."""
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm='ortho'), dim=dims)


def tv3(u):
    """Smoothed isotropic 3D total variation, sum over voxels of sqrt(|grad u|^2 + 1e-6)."""
    d = [u[1:, :-1, :-1] - u[:-1, :-1, :-1], u[:-1, 1:, :-1] - u[:-1, :-1, :-1], u[:-1, :-1, 1:] - u[:-1, :-1, :-1]]
    return torch.sqrt(sum(x.abs().square() for x in d) + 1e-6).sum()
