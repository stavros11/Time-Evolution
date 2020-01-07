import os
import h5py
import numpy as np
import scipy
from utils import ed, calc, tfim


data_dir = "/home/stavros/DATA/MPQ/scalings"
n_sites = 6
t_final = 1.0
h_init, h_ev = 1.0, 0.5
filename = ["ed",
            "N{}".format(n_sites),
            "tf{}".format(t_final),
            "h{}_{}".format(h_init, h_ev),
            "run1"]
filename = "{}.h5".format("_".join(filename))
T_list = [20, 25, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]


ham = tfim.tfim_hamiltonian(n_sites, h=h_ev)
def find_ed_state(time_steps, t_final=1.0):
  dt = t_final / time_steps
  exact_state, _ = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                             h0=h_init, h=h_ev)

  clock = ed.construct_sparse_clock(ham, dt, time_steps, init_penalty=10.0,
                                    psi0=exact_state[0])

  #eigvals, eigstates = scipy.sparse.linalg.eigsh(clock, k=5, which="SM")
  eigvals, eigstates = np.linalg.eigh(np.array(clock.todense()))
  ed_state = eigstates[:, 0].reshape([time_steps + 1, 2**n_sites])

  return eigvals, ed_state, exact_state


if os.path.exists(os.path.join(data_dir, filename)):
  raise FileExistsError("{} already exists.".format(filename))

file = h5py.File(os.path.join(data_dir, filename), "w")
for time_steps in T_list:
  eigvals, ed_state, exact_state = find_ed_state(time_steps)
  file["T{}_eigvals".format(time_steps)] = np.copy(eigvals)
  file["T{}_ed_state".format(time_steps)] = np.copy(ed_state)
  print("T={} done.".format(time_steps))
file["T_list"] = T_list
file.close()
