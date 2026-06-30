"""Visualization utilities for prosthetic vision datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

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
        data = np.asarray(h5["percepts"][sample_index])
        meta = dict(h5["metadata"].attrs)

    if data.ndim == 3:
        frame = max(0, min(frame_index, data.shape[0] - 1))
        image = data[frame]
    elif data.ndim == 2:
        image = data
        frame = 0
    else:
        raise ValueError("Unexpected percept shape")

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(image, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    display_title = title or f"sample {sample_index}, frame {frame}"
    if "implant" in meta:
        display_title = f"{display_title} ({meta['implant']})"
    ax.set_title(display_title)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax


def _format_action_block(
    electrode_names: Sequence[str],
    electrode_indices: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
) -> str:
    active = mask.astype(bool)
    if not np.any(active):
        return "no stimulation"
    entries = []
    for idx, vector in zip(electrode_indices[active], actions[active]):
        name = electrode_names[int(idx)] if idx >= 0 else "?"
        amp, freq, phase, delay = vector
        entries.append(f"{name}: {amp:.2f}uA {freq:.0f}Hz {phase:.2f}ms d={delay:.2f}ms")
    return "\n".join(entries)


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
        world = h5["world"]
        states = np.asarray(world["states"][episode_index, start_step : start_step + length + 1])
        actions = np.asarray(world["actions"][episode_index, start_step : start_step + length])
        indices = np.asarray(
            world["electrode_indices"][episode_index, start_step : start_step + length]
        )
        masks = np.asarray(world["action_mask"][episode_index, start_step : start_step + length])
        electrode_names: Sequence[str] = json.loads(h5["metadata"].attrs["electrode_pool"])

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
        absolute_step = start_step + i
        ax.set_title(f"t={absolute_step}")
        if annotate and i > 0:
            text = _format_action_block(
                electrode_names,
                indices[i - 1],
                actions[i - 1],
                masks[i - 1],
            )
            ax.text(
                0.02,
                0.02,
                text,
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
            )

    if colorbar and last_im is not None:
        fig.colorbar(last_im, ax=axes_arr.ravel().tolist(), fraction=0.02, pad=0.04)
    plt.tight_layout()
    return fig, axes_arr


@click.group()
def cli():
    """Visualization helpers for retinawm datasets."""


@cli.command()
@click.argument("dataset_path", type=click.Path(path_type=Path))
@click.option("--index", "sample_index", type=int, default=0, show_default=True)
@click.option("--frame", "frame_index", type=int, default=0, show_default=True)
@click.option("--cmap", type=str, default="magma", show_default=True)
@click.option("--no-colorbar", is_flag=True, help="Hide the colorbar.")
def percept(dataset_path: Path, sample_index: int, frame_index: int, cmap: str, no_colorbar: bool):
    fig, _ = plot_percept(
        dataset_path,
        sample_index=sample_index,
        frame_index=frame_index,
        cmap=cmap,
        colorbar=not no_colorbar,
    )
    fig.show()


@cli.command(name="world")
@click.argument("dataset_path", type=click.Path(path_type=Path))
@click.option("--episode", "episode_index", type=int, default=0, show_default=True)
@click.option("--start", "start_step", type=int, default=0, show_default=True)
@click.option("--length", type=int, default=4, show_default=True)
@click.option("--cols", type=int, default=4, show_default=True)
@click.option("--cmap", type=str, default="magma", show_default=True)
@click.option("--annotate/--no-annotate", default=True, show_default=True)
@click.option("--colorbar/--no-colorbar", default=False, show_default=True)
def world_command(
    dataset_path: Path,
    episode_index: int,
    start_step: int,
    length: int,
    cols: int,
    cmap: str,
    annotate: bool,
    colorbar: bool,
):
    fig, _ = plot_world_sequence(
        dataset_path,
        episode_index=episode_index,
        start_step=start_step,
        length=length,
        cols=cols,
        cmap=cmap,
        annotate=annotate,
        colorbar=colorbar,
    )
    fig.show()


__all__ = ["plot_percept", "plot_world_sequence", "cli"]
