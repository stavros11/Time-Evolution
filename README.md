# Time-Evolution

We propose a new computational method for the unitary time evolution of quantum many-body systems by mapping it to a ground state problem in a larger Hilbert space with a *Clock* degree of freedom. We can then use variational optimization techniques such as VMC to directly solve the ground state problem.

### Quantum system

Currently all numerical experiments are done on a transverse field Ising model (TFIM). We prepare the system in the `h=1.0` ground state and evolve with the `h=0.5` Hamiltonian. We only consider small systems and relatively short times (small number of time steps). Code is not optimized for large systems.

### Notebooks

Notebooks are mostly used for visualization and plotting results.

* `ed.ipynb`: Exploration of exact diagonalization of the Clock Hamiltonian for small systems. 
  - We compare between diagonalization with a penalty term for the initial condition or projecting out and fixing the initial parameters.
  - We also check how the above methods compare with variational optimization.
  - We explore the gap of the Clock Hamiltonian with the penalty term.
  
* `mps_trotter.ipynb`: Compare Clock evolution of MPS to a simple TEBD implementation. TEBD shows better results in all cases.

* `plotter.ipynb`: Plot the results of variational optimization. Example of plots include training dynamics of the cost function (`Eloc`) as well as the overlap with the exact evolution solution.

* `plot_observables.ipynb`: Plot observables (typically `sigma_x`) and compare with their exact evolution.
