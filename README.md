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


### Main scripts

* `np_optimization.py`: Optimizes a `machine` using gradient descent (variational optimization). Expectation values for quantities and gradients are calculated exactly by summing over all bitstrings (this is tractable only for small systems).

* `np_optimization_sampling.py`: Optimizes a `machine` using gradient descent (variational optimization). Expectation values for quantities and gradients are calculated using Monte Carlo sampling. Different samplers are used (sample time vs force a uniform distribution in time).

* `tf_optimization.py`: Optimizes a `model` using gradient descent (variational optimization) where quantities are calculated exactly. Note that a `model` is written in TensorFlow, while `machine` uses pure NumPy. TensorFlow allows us to try various models (particularly based on NNs) without the need to hard-code the gradients.

* `sampler_convergence.py`: Tests the C++ samplers by calculating the various terms in `Eloc` of the Clock Hamiltonian and check whether they converge as the number of samples is increased.

* `rbm_ground_state.py`: Finds ground state of a Hamiltonian using an RBM ansatz. This is in order to initialize an RBM machine for time evolution, as (unlike MPS) it is not trivial how to go from dense wavefunction to RBM.