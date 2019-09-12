# Time-Evolution

We propose a new computational method for the unitary time evolution of quantum many-body systems by mapping it to a ground state problem in a larger Hilbert space with a *Clock* degree of freedom. We can then use variational optimization techniques such as VMC to directly solve the ground state problem.

### Quantum system

Currently all numerical experiments are done on a transverse field Ising model (TFIM). We prepare the system in the `h=1.0` ground state and evolve with the `h=0.5` Hamiltonian. We only consider small systems and relatively short times (small number of time steps). Code is not optimized for large systems.

### Main scripts

TBA
