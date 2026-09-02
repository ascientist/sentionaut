"""Visualization utilities for prosthetic vision datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np


def _ensure_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def plot_percept(
    dataset_path: Path | str,
    sample_index: int = 0,
    frame_index: int = 0,
    *,
    cmap: str = "magma",
    colorbar: bool = True,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    path = _ensure_path(dataset_path)
    with h5py.File(path, "r") as h5:
        if "percepts" in h5:
            data = np.asarray(h5["percepts"][sample_index])
            meta = dict(h5["metadata"].attrs)
        else:
            data = np.asarray(h5["world"]["s_t"][sample_index])
            meta = dict(h5["metadata"].attrs)

    if data.ndim == 3:
        frame = max(0, min(frame_index, data.shape[0] - 1))
        image = data[frame] if data.shape[0] > 1 else data[0]
    elif data.ndim == 2:
        image = data
    else:
        raise ValueError("Unexpected percept shape")

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(image, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    display_title = title or f"sample {sample_index}, frame {frame_index}"
    if "implant" in meta:
        display_title = f"{display_title} ({meta['implant']})"
    ax.set_title(display_title)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax


def plot_transition(
    dataset_path: Path | str,
    index: int = 0,
    *,
    cmap: str = "magma",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot ``(s_t, action summary, s_{t+1})`` for flat world HDF5 schema."""
    path = _ensure_path(dataset_path)
    with h5py.File(path, "r") as h5:
        g = h5["world"]
        s_t = np.asarray(g["s_t"][index])
        s_tp1 = np.asarray(g["s_tp1"][index])
        cfg_idx = int(g["config_id"][index])
        table = json.loads(h5["metadata"].attrs["config_table"])
        cfg = table[cfg_idx]
        amp = g["amp"][index]
        n_active = int((amp != 0).sum())

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, img, lbl in zip(axes, [s_t, None, s_tp1], ["s_t", "action", "s_{t+1}"]):
        if img is None:
            ax.axis("off")
            ax.set_title(f"{cfg['model']}/{cfg['implant']}\n{n_active} active electrodes")
            continue
        ax.imshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lbl)
    plt.tight_layout()
    return fig, axes


def plot_world_sequence(
    dataset_path: Path | str,
    episode_index: int = 0,
    start_step: int = 0,
    length: int = 4,
    *,
    cols: int = 4,
    cmap: str = "magma",
    annotate: bool = True,
    colorbar: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    path = _ensure_path(dataset_path)
    with h5py.File(path, "r") as h5:
        if "states" in h5.get("world", {}):
            return _plot_legacy_world(
                h5, episode_index, start_step, length, cols, cmap, annotate, colorbar
            )
        g = h5["world"]
        ep_ids = g["episode_id"][:]
        mask = ep_ids == episode_index
        indices = np.where(mask)[0]
        if len(indices) == 0:
            indices = np.arange(min(length + 1, g["s_t"].shape[0]))
        sel = indices[start_step : start_step + length + 1]
        states = np.stack([g["s_t"][i] for i in sel] + [g["s_tp1"][sel[-1]]])

    frames = states.shape[0]
    rows = math.ceil(frames / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    last_im = None
    for i in range(rows * cols):
        ax = axes_arr.flat[i]
        if i >= frames:
            ax.axis("off")
            continue
        im = ax.imshow(states[i], cmap=cmap)
        last_im = im
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={start_step + i}")
    if colorbar and last_im is not None:
        fig.colorbar(last_im, ax=axes_arr.ravel().tolist(), fraction=0.02, pad=0.04)
    plt.tight_layout()
    return fig, axes_arr


def _plot_legacy_world(h5, episode_index, start_step, length, cols, cmap, annotate, colorbar):
    world = h5["world"]
    states = np.asarray(world["states"][episode_index, start_step : start_step + length + 1])
    frames = states.shape[0]
    rows = math.ceil(frames / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    last_im = None
    for i in range(rows * cols):
        ax = axes_arr.flat[i]
        if i >= frames:
            ax.axis("off")
            continue
        im = ax.imshow(states[i], cmap=cmap)
        last_im = im
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={start_step + i}")
    if colorbar and last_im is not None:
        fig.colorbar(last_im, ax=axes_arr.ravel().tolist(), fraction=0.02, pad=0.04)
    plt.tight_layout()
    return fig, axes_arr


@click.group()
def cli():
    """Visualization helpers for sentionaut datasets."""


@cli.command()
@click.argument("dataset_path", type=click.Path(path_type=Path))
@click.option("--index", "sample_index", type=int, default=0, show_default=True)
@click.option("--frame", "frame_index", type=int, default=0, show_default=True)
@click.option("--cmap", type=str, default="magma", show_default=True)
@click.option("--no-colorbar", is_flag=True)
def percept(dataset_path: Path, sample_index: int, frame_index: int, cmap: str, no_colorbar: bool):
    fig, _ = plot_percept(
        dataset_path,
        sample_index=sample_index,
        frame_index=frame_index,
        cmap=cmap,
        colorbar=not no_colorbar,
    )
    fig.show()


@cli.command(name="transition")
@click.argument("dataset_path", type=click.Path(path_type=Path))
@click.option("--index", type=int, default=0, show_default=True)
def transition_cmd(dataset_path: Path, index: int):
    fig, _ = plot_transition(dataset_path, index=index)
    fig.show()


@cli.command(name="world")
@click.argument("dataset_path", type=click.Path(path_type=Path))
@click.option("--episode", "episode_index", type=int, default=0, show_default=True)
@click.option("--start", "start_step", type=int, default=0, show_default=True)
@click.option("--length", type=int, default=4, show_default=True)
@click.option("--cols", type=int, default=4, show_default=True)
@click.option("--cmap", type=str, default="magma", show_default=True)
def world_command(
    dataset_path: Path,
    episode_index: int,
    start_step: int,
    length: int,
    cols: int,
    cmap: str,
):
    fig, _ = plot_world_sequence(
        dataset_path,
        episode_index=episode_index,
        start_step=start_step,
        length=length,
        cols=cols,
        cmap=cmap,
    )
    fig.show()


__all__ = ["plot_percept", "plot_transition", "plot_world_sequence", "cli"]
