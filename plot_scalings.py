import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import seaborn as sns
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = 20
cp = sns.color_palette()

data_dir = "/home/stavros/DATA/MPQ/scalings"
n_sites = 6
save = True
run_name = "run2"
t_final = 1.0
h_init, h_ev = 1.0, 0.5

filename = ["ed",
            "N{}".format(n_sites),
            "tf{}".format(t_final),
            "h{}_{}".format(h_init, h_ev),
            run_name]
filename = "{}.h5".format("_".join(filename))
file = h5py.File(os.path.join(data_dir, filename), "r")

dt_list = t_final / file["T_list"][()]
gs_energies = [file["T{}_eigvals".format(time_steps)][()][0]
               for time_steps in file["T_list"]]

print(scipy.stats.linregress(np.log(dt_list), np.log(gs_energies)))

# Plots
scalings = [3, 4, 5]
dt_list_exact = np.linspace(dt_list[-1], dt_list[0], 100)
plt.figure(figsize=(7, 4))
plt.loglog(dt_list, gs_energies, color=cp[0], linewidth=2.0, marker="o",
           markersize=6, label="Ground state")

for exponent in scalings:
  scaling = (dt_list_exact**exponent * gs_energies[-1] /
             dt_list_exact[0]**exponent)
  plt.semilogy(dt_list_exact, scaling, color=cp[exponent - 1],
               linewidth=2.0, linestyle="--",
               label=r"$\sim \delta t^{}$".format(exponent))

plt.xlabel(r"$\delta t$")
plt.ylabel(r"$\left \langle H_\mathrm{eff}\right \rangle $")
plt.legend()

if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, run_name, "N{}.pdf".format(n_sites)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()
