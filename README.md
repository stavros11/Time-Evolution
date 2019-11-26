# Time-Evolution

We propose a new computational method for the unitary time evolution of quantum many-body states by mapping the problem to a ground state optimization in a larger Hilbert space that has a *Clock* degree of freedom. We can then use variational optimization techniques such as VMC to directly solve the ground state problem.


## Main script

All optimizations can be run from `main.py`. This script optimizes a wavefunction with the given (ansatz & optimization) parameters and saves the following files:
* Optimization histories (`histories/{save-name}_{machine_type}_N{n_sites}M{time_steps}.h5`): .h5 dictionary that contains the Clock energy and overlaps with exact evolution at every step during optimization. When sampling is used, both the exact Clock energy and its sampled estimate is saved.
* Dense wavefunction (`final_dense/{save-name}_{machine_type}_N{n_sites}M{time_steps}.npy`): Final (fully optimized) dense wavefunction as an `npy` array. This can be used to calculate observables and compare methods. Not that the dense wavefunction (`(T + 1) * 2^N` parameters) is saved for concreteness even in the cases where we use a different ansatz (eg. MPS).

User can select the directory that `histories` and `final_dense` folders are with `--output-dir` and the save name with `--save-name`.

**Optimization**: Currently the are three optimization methods that all attempt to minimize the Clock energy:
* Global optimization: This is the default option where all time steps are updated simultaneously in every optimizaton step. Standard gradient descent is used (Adam) with default learning rate. User can specify `--learning-rate` and `--n-epochs`.
* Sweep optimization: Sweep optimization starts with growing in time and can be enabled by passing a positive integer for `n-sweeps`. Growing in time means adding an additional time step in the ansatz and optimizing only this, until we reach the total number of time steps `T`. `--sweep-epochs` is the number of epochs used for the *local* optimization of each time step. Note that when sweeping only one time step is updated at each step. There are two additional sweep options:
  *  `--binary-sweeps`: The Hamiltonian used at each *local in time* optimization consists of only two time steps. Otherwise the whole Clock Hamiltonian is used. Even when we use the whole Clock, we update a single time step when sweeping by masking the gradient.
  * `--sweep-both-directions` continues by sweeping back and forth (1 --> T followed by T --> 1 in DMRG like fashion) . If this is not enabled then only 1 --> T sweeps are performed.


**Machines**: Define the ansatz used for the wavefunction at each time. Currently implemented:
* Full wavefunction (`FullWavefunction`) with different (independent) parameters at each time step.
* MPS with different (`SmallMPS`) (independent) parameters at each time step. Note that even with MPS, some methods use the dense wavefunction to do calculations (transform MPS to dense and back).
Machines can be selected using `--machine-type`. Additional parameters such as `--d-phys` or `--d-bond` may be required.


**Quantum system**: Currently all numerical experiments are done on a transverse field Ising model (TFIM). We prepare the system in the ground state with `--h-init` and evolve under `--h-ev`. Other parameters such as `--n-sites`, `--time-steps` and `--t-final` may be selected.

**Sampling**: only implemented for global Clock optimization.
By default quantities (energies and gradients) are calculated exactly by summing over all possible `2^N` states (=infinite samples). Alternatively if `--n-samples` > 0 Metropolis MC is used to calculate energies and gradients. If `--sample-time` is used both spins and time are sampled, otherwise only spins.

## Files description

TBA
