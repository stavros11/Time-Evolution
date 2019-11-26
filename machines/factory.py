"""Import all implemented machines here to make them accessible to scripts."""

from machines.full import FullWavefunction
from machines.mps import SmallMPS
from optimization import deterministic


# Map machines to compatible gradient calculation functions
machine_to_gradient_func = {"FullWavefunction": deterministic.gradient,
                            "SmallMPS": deterministic.sampling_gradient}