"""Import all implemented machines here to make them accessible to scripts."""

from machines.full import FullWavefunction
from machines.full import FullWavefunctionNormalized
from machines.mps import SmallMPS
from machines.mps import SmallMPSNormalized
from optimization import sweeping


# Map from machines to compatible sweepers
machine_to_sweeper = {"FullWavefunction": sweeping.ExactGMResSweep,
                      "SmallMPS": sweeping.ExactGMResSweep}