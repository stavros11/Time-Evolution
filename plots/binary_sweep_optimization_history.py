import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = 24
cp = sns.color_palette()
from mpl_toolkits.axes_grid import inset_locator

ylabels = {"sweeping_exact_Eloc": r"$\left \langle H_\mathrm{eff}\right \rangle $",
           "sweeping_avg_overlaps": r"$1 - \overline{\mathrm{Fid}(t)}$"}
global_quantities = {"sweeping_exact_Eloc": 0.0018196748798055369,
                     "sweeping_avg_overlaps": 1 - 0.9757716162996253}


data_dir = "D:/ClockV5/histories"
#data_dir = "/home/stavros/DATA/MPQ/ClockV5/histories"

machine = ["fullwv", "mpsD4"][1]
save = True
n_sites = 6
time_steps = 20
n_sweeps = 10
quantity = ["sweeping_exact_Eloc", "sweeping_avg_overlaps"][1]


filename = ["allstates1_binary", "nsweeps{}".format(n_sweeps),
            machine, "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")
if "overlaps" in quantity:
  sweep_data = 1 - data[quantity][()].ravel()
data.close()

cut_ind = 1000 * 20
x_values = np.linspace(0, n_sweeps, len(sweep_data))


fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(x_values, sweep_data, color=cp[0], linewidth=2.4)
plt.axhline(y=global_quantities[quantity], color=cp[1], linewidth=2.0)
#ax.semilogy(global_only_x, global_only_heff, color=cp[1], linewidth=2.4)
ax.axvline(x=1.0, linestyle="--", linewidth=2.0, color="black")
plt.xlabel("Sweeps")
plt.ylabel(ylabels[quantity])
plt.xticks(list(range(n_sweeps + 1)))
#plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=22)

inset_axes = inset_locator.inset_axes(ax, width="50%", height="50%", loc="upper right")
plt.plot(x_values[cut_ind:], sweep_data[cut_ind:], color=cp[0], linewidth=2.4)
if "overlaps" not in quantity:
  plt.axhline(y=global_quantities[quantity], color=cp[1], linewidth=2.0)
plt.xticks(list(range(1, n_sweeps + 1)), fontsize=18)
plt.yticks([])


#plt.semilogy(global_only_x, 1 - global_only_overlaps, color=cp[1], linewidth=2.4)
#plt.axvline(x=n_grow, linestyle="--", linewidth=2.0, color="black")
#inset_axes.xaxis.set_label_coords(0.8,0.185)
#inset_axes.yaxis.set_label_coords(0.16,0.4)
plt.xlabel("Sweeps", fontsize=20)
plt.ylabel(ylabels[quantity], fontsize=20)


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, quantity, machine, "N{}M{}.pdf".format(n_sites, time_steps)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()