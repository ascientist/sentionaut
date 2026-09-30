import numpy as np
import torch

from sentionaut import animate
from sentionaut.core.base import Pose


def test_schedule_covers_patterns_ramp_and_train():
    zone = [7, 3, 9]
    steps = animate.sequence_schedule(zone, (1.0, 2.0, 3.0), on=2, rest=1, train_pulses=3)
    stages = {s.stage for s in steps}
    assert {"single", "pair", "ramp"} <= stages
    assert {f"pulse train {k}/3" for k in (1, 2, 3)} <= stages
    assert any(not s.electrodes for s in steps)
    assert all(set(s.electrodes) <= set(zone) for s in steps)
    assert all(zone[0] in s.electrodes for s in steps if s.stage == "pair" and s.electrodes)
    ramp = [s.amp for s in steps if s.stage == "ramp" and s.electrodes]
    assert ramp == sorted(ramp) and set(ramp) == {1.0, 2.0, 3.0}

    picks = animate.key_frames(steps)
    assert [steps[i].stage for i in picks[:5]] == ["single", "pair", "ramp", "ramp", "ramp"]
    assert [steps[i].amp for i in picks[2:5]] == [1.0, 2.0, 3.0]
    assert all(steps[i + 1] != steps[i] for i in picks[:5])
    assert not steps[picks[5]].electrodes and steps[picks[5] + 1].stage.startswith("pulse")


def test_stimulation_zone_is_centre_plus_neighbours():
    xy = np.array([[x, y] for y in range(3) for x in range(3)], dtype=float)
    zone = animate.stimulation_zone(xy, size=5)
    assert zone[0] == 4
    assert set(zone) == {1, 3, 4, 5, 7}


def test_sequence_render_keeps_pose_fixed(tmp_path, monkeypatch):
    poses = []
    build = animate.build_components

    def recording_build(cfg, device):
        implant, topo, model = build(cfg, device)
        step = model.step

        def recording_step(state, action):
            poses.append(action.pose)
            return step(state, action)

        model.step = recording_step
        return implant, topo, model

    monkeypatch.setattr(animate, "build_components", recording_build)
    paths = animate.animate_sequence(
        "scoreboard",
        tmp_path,
        torch.device("cpu"),
        zone_size=2,
        on=1,
        rest=1,
        train_pulses=1,
        train_on=1,
        train_rest=1,
    )
    assert [p.name for p in paths] == [
        "scoreboard_sequence.gif",
        "scoreboard_sequence_stages.png",
        "scoreboard_sequence_timeline.png",
    ]
    assert all(p.stat().st_size > 0 for p in paths)
    assert poses and all(p == Pose() for p in poses)


def test_dynaphos_axonmap_sequence_and_sweep(tmp_path):
    kw = dict(on=1, rest=1, train_pulses=1, train_on=1, train_rest=1)
    paths = animate.animate_sequence(
        "dynaphos_axonmap",
        tmp_path,
        torch.device("cpu"),
        zone=animate.PERIPHERAL_ZONE,
        axlambda=1500.0,
        suffix="_periphery",
        window=((-8.0, 0.0), (0.0, 8.0), 1.0),
        **kw,
    )
    assert [p.name for p in paths] == [
        "dynaphos_axonmap_sequence_periphery.gif",
        "dynaphos_axonmap_sequence_periphery_stages.png",
        "dynaphos_axonmap_sequence_periphery_timeline.png",
    ]
    sweep = animate.animate_dynaphos_axonmap(tmp_path, torch.device("cpu"), n_frames=8)
    assert all(p.stat().st_size > 0 for p in paths + sweep)


def test_percept_panel_is_not_flipped():
    """Row 0 of every percept grid is the top (y max); the panel must draw it on top."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(xrange=(-4, 4), yrange=(-4, 4))
    scene = SimpleNamespace(cfg=cfg, tissue=np.zeros((1, 2)), tissue_title="", unit="um", scale=1.0)
    img = np.zeros((9, 9))
    img[0, :] = 1.0  # top row bright
    frame = animate._draw_frame(scene, img, "", np.zeros((1, 2)), [], vmax=1.0)
    left = frame[:, : frame.shape[1] // 2].astype(int)
    # inferno at 1.0 is pale yellow: high red and green, lower blue
    bright = (left[..., 0] > 230) & (left[..., 1] > 230) & (left[..., 2] < 200)
    rows = np.nonzero(bright)[0]
    assert rows.size and rows.mean() < frame.shape[0] * 0.4
