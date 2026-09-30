"""NeuroStim: toy constrained POMDP for closed-loop neurostimulation control.

Docs: https://ascientist.github.io/sentionaut/neurostim/ (tutorial: ``examples/neurostim_tutorial.py``).
"""

from .env import PRESETS, NeuroStimConfig, NeuroStimEnv, make

__all__ = ["PRESETS", "NeuroStimConfig", "NeuroStimEnv", "make"]
