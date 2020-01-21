import os
import h5py
import numpy as np
import scipy
from optimization import deterministic
from utils import calc, tfim

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = 26
cp = sns.color_palette()


data_dir = "/home/stavros/DATA/MPQ/scalings"
n_sites = 6
t_final = 1.0
h_init, h_ev = 1.0, 0.5
q = "clock_energy"
save = True


file_name = "ed_N{}_tf{}_h{}_{}_run2.h5".format(n_sites, t_final, h_init, h_ev)
print(file_name)
file = h5py.File(os.path.join(data_dir, file_name), "r")

T_list = list(file["T_list"][()])
T_list.remove(92)
T_list.remove(94)
T_list.remove(98)
ham = tfim.tfim_hamiltonian(n_sites, h=h_ev)

ed_clock_energy = []
exact_clock_energy = []
for time_steps in T_list:
  dt = t_final / time_steps
  exact_state, _ = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                             h0=h_init, h=h_ev)
  ed_state = np.copy(file["T{}_ed_state".format(time_steps)][()])

  heff_terms, _ = deterministic.energy(ed_state, ham=ham, dt=dt)
  ed_clock_energy.append(np.sum(heff_terms))
  heff_terms, _ = deterministic.energy(exact_state, ham=ham, dt=dt)
  exact_clock_energy.append(np.sum(heff_terms))


dt_list = t_final / np.array(T_list)
plt.figure(figsize=(7, 4))
plt.semilogy(dt_list, ed_clock_energy, color=cp[0], marker="^", markersize=8,
             label="Clock")
plt.semilogy(dt_list, exact_clock_energy, color="black", linestyle="--",
             marker="v", markersize=8, label="Exact")
plt.xlabel(r"$\delta t$")
plt.ylabel(r"$\left \langle H_\mathrm{eff}\right \rangle $")
plt.legend()
if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, "N{}.pdf".format(n_sites)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()
