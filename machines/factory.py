"""Import all implemented machines here to make them accessible to scripts."""

from machines.full import FullWavefunctionMachine
from machines.full import FullWavefunctionMachineNormalized
from machines.mps import SmallMPSMachine
from machines.mps import SmallMPSMachineNorm
from optimization import deterministic

# Maps each machine to the appropriate deterministic gradient
# calculation method
machine_to_gradfunc = {
    "FullWavefunctionMachine": deterministic.gradient,
    "FullWavefunctionMachineNormalized": deterministic.gradient,
    "SmallMPSMachine": deterministic.sampling_gradient,
    "SmallMPSMachineNorm": deterministic.sampling_gradient}