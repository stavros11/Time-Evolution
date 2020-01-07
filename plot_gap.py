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
save = True
run_name = "run2"
n_sites = 6
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
gs_energies = np.array([file["T{}_eigvals".format(time_steps)][()][0]
                        for time_steps in file["T_list"]])
excited_energies = np.array([file["T{}_eigvals".format(time_steps)][()][1]
                             for time_steps in file["T_list"]])
gap = excited_energies - gs_energies

# Plots
dt_list_exact = np.linspace(dt_list[-1], dt_list[0], 100)
plt.figure(figsize=(7, 4))
plt.semilogy(dt_list, gap, color=cp[0], linewidth=2.0, marker="o",
             markersize=6, label="Ground state")

scaling = dt_list_exact * gap[-1] / dt_list_exact[0]
plt.semilogy(dt_list_exact, scaling, color=cp[1],
             linewidth=2.0, linestyle="--",
             label=r"$\sim \delta t$")
scaling = dt_list_exact**2 * gap[-1] / dt_list_exact[0]**2
plt.semilogy(dt_list_exact, scaling, color=cp[3],
             linewidth=2.0, linestyle="--",
             label=r"$\sim \delta t^2$")
scaling = dt_list_exact**3 * gap[-1] / dt_list_exact[0]**3
plt.semilogy(dt_list_exact, scaling, color=cp[2],
             linewidth=2.0, linestyle="--",
             label=r"$\sim \delta t^3$")


plt.xlabel(r"$\delta t$")
plt.ylabel("Gap")
plt.legend()

if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, "N{}.pdf".format(n_sites)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()
