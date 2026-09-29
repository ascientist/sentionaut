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
    assert [p.name for p in paths] == ["scoreboard_sequence.gif"]
    assert paths[0].stat().st_size > 0
    assert poses and all(p == Pose() for p in poses)
