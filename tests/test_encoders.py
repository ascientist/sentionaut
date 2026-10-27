"""Stimulus encoders: vision and sound -> electrode amplitudes."""

from __future__ import annotations

import math

import torch

from sentionaut.encoders import (
    BandEnvelopeEncoder,
    ClosedLoopVisionEncoder,
    PhospheneMapEncoder,
    probe_centroids,
)


def _toy_render(amp: torch.Tensor) -> torch.Tensor:
    """Electrode e lights the single pixel (e // 4, e % 4) of a 4x4 grid."""
    img = torch.zeros(amp.shape[0], 4, 4)
    for e in range(amp.shape[1]):
        img[:, e // 4, e % 4] = amp[:, e]
    return img


def test_probe_centroids_and_silent_electrodes():
    def render(amp):
        img = _toy_render(amp)
        img[:, 0, 0] = 0.0  # electrode 0 produces nothing
        return img

    c = probe_centroids(render, 16, 1.0)
    assert torch.isnan(c[0]).all()
    assert torch.allclose(c[5], torch.tensor([1.0, 1.0]))
    assert torch.allclose(c[15], torch.tensor([3.0, 3.0]))


def test_phosphene_map_encoder_follows_the_image():
    cent = probe_centroids(_toy_render, 16, 1.0)
    enc = PhospheneMapEncoder(cent, (4, 4), a_max=3.0, a_min=0.5, on=0.3)
    img = torch.zeros(2, 4, 4)
    img[0, 2, 1] = 1.0  # electrode 9 at full level
    img[1, 2, 1] = 0.2  # below the on-level
    amp = enc(img)
    assert amp.shape == (2, 16)
    assert torch.isclose(amp[0, 9], torch.tensor(3.0))
    assert amp[0].count_nonzero() == 1
    assert amp[1].count_nonzero() == 0


def test_closed_loop_encoder_range():
    enc = ClosedLoopVisionEncoder((5, 6), n_electrodes=7, a_max=300.0)
    amp = enc(torch.rand(3, 5, 6))
    assert amp.shape == (3, 7)
    assert (amp >= 0).all() and (amp <= 300.0).all()


def test_band_envelope_encoder_routes_tones_to_bands():
    sr, n = 16000, 8
    enc = BandEnvelopeEncoder(n, a_max=1.0, sample_rate=sr, order=list(reversed(range(n))))
    t = torch.arange(sr // 5) / sr
    low = torch.sin(2 * math.pi * 300.0 * t)
    high = torch.sin(2 * math.pi * 5000.0 * t)
    amp = enc.encode_wave(torch.stack([low, high]))
    assert amp.shape == (2, 10, n)  # 200 ms in 20 ms frames
    # order is reversed: the lowest band drives the last electrode
    assert amp[0].mean(0).argmax() == n - 1
    assert amp[1].mean(0).argmax() < n // 2
    silent = enc.encode_wave(torch.zeros(1, sr // 50))
    assert silent.count_nonzero() == 0
