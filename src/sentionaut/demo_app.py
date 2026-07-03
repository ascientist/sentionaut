"""Streamlit demo: pick implant / topography / percept-model, sweep action params."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from sentionaut.calibrate import load_calibration
from sentionaut.core.base import Action, Pose
from sentionaut.core.config import CORTICAL_IMPLANTS, RETINAL_IMPLANTS, Config
from sentionaut.core.device import get_device
from sentionaut.core.registry import build_components

st.set_page_config(page_title="Sentionaut", layout="wide")
st.title("Sentionaut — modular prosthetic-vision world model")

MODELS = ["axonmap", "scoreboard", "dynaphos"]


@st.cache_resource
def _build(model, implant, xystep, calibration_path):
    kwargs = {}
    if calibration_path:
        kwargs["calibration_path"] = Path(calibration_path)
    cfg = Config(
        model=model, implant=implant, xrange=(-8, 8), yrange=(-8, 8), xystep=xystep, **kwargs
    )
    dev = get_device()
    return cfg, build_components(cfg, dev)


with st.sidebar:
    st.header("Components")
    model = st.selectbox("Percept model", MODELS)
    cortical = model in ("scoreboard", "dynaphos")
    options = sorted(CORTICAL_IMPLANTS if cortical else RETINAL_IMPLANTS)
    implant_name = st.selectbox("Implant", options)
    xystep = st.select_slider("Grid step (dva)", options=[0.25, 0.3, 0.5, 1.0], value=0.5)

    cal_file = st.file_uploader("Subject calibration JSON (optional)", type=["json"])
    cal_path = None
    if cal_file is not None:
        cal_path = Path(st.session_state.get("_cal_tmp", "/tmp/sentionaut_cal.json"))
        cal_path.write_bytes(cal_file.getvalue())

    st.header("Action")
    if cortical:
        amp = st.slider("Current (uA)", 50.0, 300.0, 200.0, 10.0)
        rho = None if model == "dynaphos" else st.slider("rho (microns)", 800.0, 1200.0, 1000.0, 50.0)
        freq = pdur = axl = None
    else:
        amp = st.slider("Amplitude (x threshold)", 0.0, 4.0, 2.0, 0.1)
        freq = st.slider("Frequency (Hz)", 10.0, 120.0, 30.0, 5.0)
        pdur = st.slider("Phase duration (ms)", 0.1, 0.6, 0.45, 0.05)
        rho = st.slider("rho (microns)", 100.0, 400.0, 200.0, 10.0)
        axl = st.slider("axlambda (microns)", 200.0, 800.0, 500.0, 25.0)
    n_active = st.slider("Active electrodes", 1, 5, 3)

cfg, (implant, topo, percept_model) = _build(model, implant_name, xystep, cal_path)
if cal_path and cal_path.exists():
    cal = load_calibration(cal_path)
    st.sidebar.caption(f"Calibration: rho={cal.rho:.0f}, axlambda={cal.axlambda:.0f}")

device = topo.grid_x.device if hasattr(topo, "grid_x") else topo.coords.device
N = implant.n_electrodes
idx = np.linspace(0, N - 1, n_active).astype(int)

amp_t = torch.zeros(N, device=device)
freq_t = torch.zeros(N, device=device)
pdur_t = torch.zeros(N, device=device)
for e in idx:
    amp_t[e] = amp
    if not cortical:
        freq_t[e] = freq
        pdur_t[e] = pdur

action = Action(
    amp=amp_t,
    freq=None if cortical else freq_t,
    phase_dur=None if cortical else pdur_t,
    rho=rho,
    axlambda=None if cortical else axl,
    pose=Pose(),
)

if model == "dynaphos":
    img = percept_model.predict_sequence(action, 10)[-1].detach().cpu().numpy()
else:
    img = percept_model.forward(action).detach().cpu().numpy()

col1, col2 = st.columns(2)
extent = [cfg.xrange[0], cfg.xrange[1], cfg.yrange[0], cfg.yrange[1]]
with col1:
    st.subheader("Percept")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(img, cmap="inferno", extent=extent, origin="lower")
    ax.set_xlabel("x (dva)")
    ax.set_ylabel("y (dva)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

with col2:
    st.subheader("Tissue geometry")
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
    elec = implant.electrode_coords().cpu().numpy()
    if cortical:
        cortex = topo.cortex_xy[cfg.regions[0]].cpu().numpy()
        cortex = cortex[np.isfinite(cortex).all(axis=1)]
        ax2.scatter(cortex[:, 0] / 1000, cortex[:, 1] / 1000, s=0.5, c="0.7", alpha=0.4)
        ax2.scatter(elec[:, 0] / 1000, elec[:, 1] / 1000, s=12, c="tab:blue")
        ax2.scatter(elec[idx, 0] / 1000, elec[idx, 1] / 1000, s=50, c="tab:red")
        ax2.set_xlabel("x (mm)")
        ax2.set_ylabel("y (mm)")
    else:
        pts = topo.coords.reshape(-1, 2).cpu().numpy()
        pts = pts[np.random.default_rng(0).choice(pts.shape[0], size=3000, replace=False)]
        ax2.scatter(pts[:, 0], pts[:, 1], s=0.5, c="0.7", alpha=0.4)
        ax2.scatter(elec[:, 0], elec[:, 1], s=12, c="tab:blue")
        ax2.scatter(elec[idx, 0], elec[idx, 1], s=50, c="tab:red")
        ax2.set_xlabel("x (microns)")
        ax2.set_ylabel("y (microns)")
    ax2.set_aspect("equal")
    st.pyplot(fig2)
