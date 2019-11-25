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

ylabels = {"exact_Eloc": r"$\left \langle H_\mathrm{eff}\right \rangle $",
           "avg_overlaps": r"$1 - \overline{\mathrm{Fid}(t)}$"}
global_quantities = {"exact_Eloc": 0.001815,
                     "avg_overlaps": 1 - 0.9757716162996253}


data_dir = "D:/ClockV5/histories"
#data_dir = "/home/stavros/DATA/MPQ/ClockV5/histories"

machine = ["fullwv", "mpsD4"][1]
save = True
n_sites = 6
time_steps = 20
quantity = ["exact_Eloc", "avg_overlaps"][0]


filename = ["allstates1_growglobal_nsweeps1",
            machine, "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")
if "overlaps" in quantity:
  grow_data = 1 - data["sweeping_{}".format(quantity)][()].ravel()
  glob_data = 1 - data[quantity][()].ravel()
else:
  grow_data = data["sweeping_{}".format(quantity)][()].ravel()
  glob_data = data[quantity][()].ravel()
data.close()

glob_data = glob_data[:2000]


all_data = np.concatenate([grow_data, glob_data], axis=0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(all_data, color=cp[0], linewidth=2.4)
plt.axhline(y=global_quantities[quantity], color=cp[1], linewidth=2.0)
#ax.semilogy(global_only_x, global_only_heff, color=cp[1], linewidth=2.4)
ax.axvline(x=1000 * time_steps, linestyle="--", linewidth=2.0, color="black")
plt.xlabel("Time Steps")
plt.ylabel(ylabels[quantity])
ax.set_xticklabels(list(range(-5, time_steps + 1, 5)))
#plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=22)

inset_axes = inset_locator.inset_axes(ax, width="40%", height="40%", loc="center")
plt.plot(glob_data, color=cp[0], linewidth=2.4)
plt.axvline(x=0, linestyle="--", linewidth=2.0, color="black")
plt.axhline(y=global_quantities[quantity], color=cp[1], linewidth=2.0)
if "overlaps" not in quantity:
  plt.yticks([1.8e-3, 1.9e-3], fontsize=16)
else:
  plt.yticks(fontsize=16)
plt.xticks([0, 500, 1000, 1500, 2000], fontsize=16)
plt.xlabel("Epochs", fontsize=18)
plt.ylabel(ylabels[quantity], fontsize=18)


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, quantity, machine, "N{}M{}.pdf".format(n_sites, time_steps)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()