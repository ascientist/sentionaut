"""Patient data behind the pulse2percept models, and fitting the axon map to it.

pulse2percept ships the psychophysics its models were built from. The two
datasets that constrain the axon map's free parameters are:

- ``beyeler2019``: 400 phosphene drawings from single-electrode stimulation
  (2x threshold, 20 Hz, 0.45 ms) in four Argus I/II users, with each array's
  position and rotation on the retina. Beyeler et al. 2019 fitted ``rho`` and
  ``axlambda`` to these.
- ``horsager2009`` and ``greenwald2009``: detection thresholds and brightness
  ratings on Argus I, for the temporal and brightness models.

``fit_drawings`` fits ``rho``, ``axlambda`` and a drawing threshold to one
subject's drawings by gradient descent through ``BiphasicAxonMapTorch``, on
phosphene shape (area, orientation, elongation), as Beyeler et al. did. It
tests on electrodes held out of the fit.

Docs: https://ascientist.github.io/sentionaut/patient-data/
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

OSF = {
    "beyeler2019": (
        "https://osf.io/28uqg/download",
        "19817990a615d289cdc93b928c138f71977ea2cab54fd1b35d186f3b5a3c4ff5",
        "beyeler2019.h5",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, data_dir: str | Path = "data/p2p") -> Path:
    """Download an OSF dataset with TLS verification on and check its SHA-256.

    pulse2percept's own ``fetch_url`` turns certificate checking off; this
    does not. Set ``SSL_CERT_FILE`` if you sit behind a proxy with its own CA.
    """
    url, sha, fname = OSF[name]
    path = Path(data_dir) / fname
    if path.exists() and _sha256(path) == sha:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, context=ctx) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    if _sha256(tmp) != sha:
        tmp.unlink()
        raise OSError(f"{name}: checksum mismatch for {url}")
    tmp.rename(path)
    return path


def load_beyeler2019(data_dir: str | Path = "data/p2p"):
    """Phosphene drawings as a pandas ``DataFrame`` (one row per drawing)."""
    from pulse2percept.datasets import fetch_beyeler2019

    path = download("beyeler2019", data_dir)
    return fetch_beyeler2019(data_path=str(path.parent), download_if_missing=False)


# ------------------------------------------------------------- descriptors
def moments(weights: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> dict[str, torch.Tensor]:
    """Shape descriptors of non-negative images ``(B, H, W)`` on a grid in degrees.

    ``xs`` and ``ys`` are the pixel-centre coordinates ``(H, W)``. Returns area
    (deg²), centre, orientation (rad, of the long axis, in ``(-pi/2, pi/2]``)
    and elongation ``1 - minor/major`` from the second moments.
    """
    px = float((xs[0, 1] - xs[0, 0]).abs()) * float((ys[1, 0] - ys[0, 0]).abs())
    mass = weights.sum((1, 2)).clamp(min=1e-9)
    cx = (weights * xs).sum((1, 2)) / mass
    cy = (weights * ys).sum((1, 2)) / mass
    dx, dy = xs - cx[:, None, None], ys - cy[:, None, None]
    sxx = (weights * dx**2).sum((1, 2)) / mass
    syy = (weights * dy**2).sum((1, 2)) / mass
    sxy = (weights * dx * dy).sum((1, 2)) / mass
    theta = 0.5 * torch.atan2(2 * sxy, sxx - syy)
    tr, det = sxx + syy, (sxx * syy - sxy**2).clamp(min=0)
    disc = torch.sqrt(((tr / 2) ** 2 - det).clamp(min=1e-12))
    lmax, lmin = tr / 2 + disc, (tr / 2 - disc).clamp(min=0)
    elong = 1 - torch.sqrt(lmin / lmax.clamp(min=1e-12))
    return {"area": mass * px, "cx": cx, "cy": cy, "theta": theta, "elong": elong}


def drawing_grid(row) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-centre coordinates (deg) of one drawing; row 0 is the top (+y)."""
    h, w = row["image"].shape
    (x0, x1), (y0, y1) = row["xrange"], row["yrange"]
    xs = torch.linspace(x0, x1, w)
    ys = torch.linspace(y1, y0, h)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def electrode_targets(df) -> dict[str, dict[str, float]]:
    """Per electrode: mean log-area, mean elongation, circular-mean orientation."""
    out = {}
    for elec, rows in df.groupby("electrode"):
        d = []
        for _, r in rows.iterrows():
            xx, yy = drawing_grid(r)
            img = torch.from_numpy(np.asarray(r["image"], dtype=np.float32))[None]
            d.append(moments(img, xx, yy))
        log_area = torch.stack([torch.log(m["area"]) for m in d]).mean()
        elong = torch.stack([m["elong"] for m in d]).mean()
        th2 = torch.stack(
            [torch.stack([torch.cos(2 * m["theta"]), torch.sin(2 * m["theta"])]) for m in d]
        )
        th = 0.5 * torch.atan2(th2[:, 1].mean(), th2[:, 0].mean())
        out[str(elec)] = {
            "log_area": float(log_area),
            "elong": float(elong),
            "theta": float(th),
            "n": len(d),
        }
    return out


# -------------------------------------------------------------------- fit
@dataclass
class DrawingFit:
    subject: str
    rho: float
    axlambda: float
    threshold: float
    train_loss: float
    test_loss: float
    test_terms: dict
    default_test_loss: float
    default_test_terms: dict
    train_electrodes: list[str]
    test_electrodes: list[str]


def _loss(pred: dict, tgt: dict, names: list[str]):
    dev = pred["area"].device
    la = torch.tensor([tgt[n]["log_area"] for n in names], device=dev)
    el = torch.tensor([tgt[n]["elong"] for n in names], device=dev)
    th = torch.tensor([tgt[n]["theta"] for n in names], device=dev)
    area = (torch.log(pred["area"].clamp(min=1e-6)) - la) ** 2
    elong = (pred["elong"] - el) ** 2
    orient = el * (1 - torch.cos(2 * (pred["theta"] - th)))  # weighted: round blobs have no axis
    terms = {"log_area": area.mean(), "elong": elong.mean(), "orient": orient.mean()}
    return area.mean() + elong.mean() + orient.mean(), terms


def fit_drawings(
    df,
    subject: str,
    step: float = 1.0,
    iters: int = 150,
    lr: float = 0.05,
    axlambda_build: float = 2000.0,
    softness: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> DrawingFit:
    """Fit ``rho``, ``axlambda`` and the drawing threshold to one Argus II subject.

    Electrodes are split in two (alternating, sorted by name): the fit uses
    one half and the test reports the other. ``default_test_loss`` keeps
    ``rho = 200``, ``axlambda = 500`` and fits only the threshold, so the two
    test losses differ only by the physical parameters.
    """
    from pulse2percept.implants import ArgusII

    from .core.base import Pose
    from .core.config import Config
    from .core.registry import build_components

    rows = df[df.subject == subject]
    first = rows.iloc[0]
    if str(first["implant_type_str"]) != "ArgusII":
        raise ValueError(f"{subject}: only Argus II subjects are supported")
    xr, yr = tuple(first["xrange"]), tuple(first["yrange"])
    cfg = Config(
        model="axonmap",
        implant="argusii",
        xrange=xr,
        yrange=yr,
        xystep=step,
        axlambda=axlambda_build,
    )
    dev = torch.device(device)
    implant, topo, model = build_components(cfg, device=dev)
    pose = Pose(
        dx=float(first["implant_x"]),
        dy=float(first["implant_y"]),
        rot=math.radians(float(first["implant_rot"])),
    )
    p2p = ArgusII(
        x=float(first["implant_x"]), y=float(first["implant_y"]), rot=float(first["implant_rot"])
    )
    ours = implant.electrode_coords(pose).cpu()
    ref = torch.tensor([[p2p[n].x, p2p[n].y] for n in implant.names], dtype=ours.dtype)
    if not torch.allclose(ours, ref, atol=1.0):
        raise RuntimeError("electrode positions disagree with pulse2percept's ArgusII")

    tgt = electrode_targets(rows)
    names = sorted(tgt)
    train, test = names[0::2], names[1::2]
    index = {n: i for i, n in enumerate(implant.names)}
    xs, ys = topo.grid_x, topo.grid_y

    def predict(sel, log_rho, log_lam, thr_logit):
        amp = torch.zeros(len(sel), implant.n_electrodes, device=dev)
        for i, n in enumerate(sel):
            amp[i, index[n]] = 2.0
        rho = torch.exp(log_rho)
        p = model.spatial_forward_batch(
            amp, freq=20.0, phase_dur=0.45, rho=rho, axlambda=torch.exp(log_lam), pose=pose
        )
        p = p / p.amax((1, 2), keepdim=True).clamp(min=1e-9)
        w = torch.sigmoid((p - torch.sigmoid(thr_logit)) / softness)
        return moments(w, xs, ys)

    def optimise(free_physics: bool):
        torch.manual_seed(seed)
        log_rho = torch.tensor(math.log(200.0), device=dev, requires_grad=free_physics)
        log_lam = torch.tensor(math.log(500.0), device=dev, requires_grad=free_physics)
        thr = torch.tensor(0.0, device=dev, requires_grad=True)
        params = [thr] + ([log_rho, log_lam] if free_physics else [])
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(iters):
            loss, _ = _loss(predict(train, log_rho, log_lam, thr), tgt, train)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():  # keep inside the topography's valid range
                log_rho.clamp_(math.log(20.0), math.log(1500.0))
                log_lam.clamp_(math.log(50.0), math.log(axlambda_build))
        with torch.no_grad():
            tr, _ = _loss(predict(train, log_rho, log_lam, thr), tgt, train)
            te, terms = _loss(predict(test, log_rho, log_lam, thr), tgt, test)
        return log_rho, log_lam, thr, float(tr), float(te), {k: float(v) for k, v in terms.items()}

    lr_, ll_, th_, tr, te, terms = optimise(True)
    _, _, _, _, dte, dterms = optimise(False)
    return DrawingFit(
        subject=subject,
        rho=float(torch.exp(lr_)),
        axlambda=float(torch.exp(ll_)),
        threshold=float(torch.sigmoid(th_)),
        train_loss=tr,
        test_loss=te,
        test_terms=terms,
        default_test_loss=dte,
        default_test_terms=dterms,
        train_electrodes=train,
        test_electrodes=test,
    )


def cli():  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Fit the axon map to Beyeler et al. 2019 drawings.")
    p.add_argument("--data-dir", default="data/p2p")
    p.add_argument("--subjects", nargs="+", default=["S2", "S3", "S4"])
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--out", default="data/p2p/drawing_fits.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    df = load_beyeler2019(a.data_dir)
    fits = []
    for s in a.subjects:
        f = fit_drawings(df, s, step=a.step, iters=a.iters, device=a.device)
        print(json.dumps(asdict(f)))
        fits.append(asdict(f))
    Path(a.out).write_text(json.dumps(fits, indent=1))
