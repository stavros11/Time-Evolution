"""Import all implemented autograd machines."""

from machines.autograd import full
from machines.autograd import mps
from machines.autograd import neural
from machines.autograd import rbm

FullWavefunction = full.FullWavefunctionModel
FullPropagator = full.FullPropagatorModel
SmallMPS = mps.SmallMPSModel
MPSProductProp = mps.SmallMPSProductPropModel
StepConv = neural.StepConvModel
StepFeedForward = neural.StepFeedForwardModel
SmallRBM = rbm.SmallRBMModel
RBMProductProp = rbm.SmallRBMProductPropModel