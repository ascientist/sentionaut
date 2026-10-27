"""Stimulus encoders: a signal in some modality -> electrode amplitudes.

The other half of the pipeline, stimulation -> percept, is the forward models
in ``sentionaut.models`` (analytical) and ``sentionaut.learned`` (learned).

Docs: https://ascientist.github.io/sentionaut/architecture/
"""

from .base import StimulusEncoder
from .sound import BandEnvelopeEncoder
from .vision import ClosedLoopVisionEncoder, PhospheneMapEncoder, probe_centroids

__all__ = [
    "StimulusEncoder",
    "PhospheneMapEncoder",
    "ClosedLoopVisionEncoder",
    "BandEnvelopeEncoder",
    "probe_centroids",
]
