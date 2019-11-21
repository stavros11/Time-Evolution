import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = 26
cp = sns.color_palette()
from mpl_toolkits.axes_grid import inset_locator


#data_dir = "D:/ClockV5/histories"
data_dir = "/home/stavros/DATA/MPQ/ClockV5/histories"

machine = ["fullwv", "mpsD4"][1]
save = True
n_sites = 6
time_steps = 20


filename = ["allstates1_growglobal", "nsweeps1", machine, "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")

grow = data["_".join(["sweeping", "exact_Eloc"])][()].real.ravel()
n_grow = len(grow)
glob = data["exact_Eloc"][()].real
heff = np.concatenate([grow, glob], axis=0)

grow = data["_".join(["sweeping", "avg_overlaps"])][()].real.ravel()
glob = data["avg_overlaps"][()].real
overlaps = np.concatenate([grow, glob], axis=0)
data.close()


filename = ["allstates1_global", "nsweeps0", machine, "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")
global_only_heff = data["exact_Eloc"][()].real
global_only_overlaps = data["avg_overlaps"][()].real
data.close()

n_global = len(global_only_heff)
global_only_x = np.arange(n_grow, n_grow + n_global)


fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(heff, color=cp[0], linewidth=2.4)
ax.semilogy(global_only_x, global_only_heff, color=cp[1], linewidth=2.4)
ax.axvline(x=n_grow, linestyle="--", linewidth=2.0, color="black")
plt.xlabel("Epochs")
plt.ylabel(r"$\left \langle H_\mathrm{eff}\right \rangle $")
#plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=22)

inset_axes = inset_locator.inset_axes(ax, width="50%", height="50%", loc=3)
plt.semilogy(1 - overlaps, color=cp[0], linewidth=2.4)
plt.semilogy(global_only_x, 1 - global_only_overlaps, color=cp[1], linewidth=2.4)
plt.axvline(x=n_grow, linestyle="--", linewidth=2.0, color="black")
plt.xticks([])
plt.yticks([])
inset_axes.xaxis.set_label_coords(0.8,0.185)
inset_axes.yaxis.set_label_coords(0.16,0.4)
#plt.xlabel("Epochs")
plt.ylabel(r"$1 - \overline{\mathrm{Fid}(t)}$", fontsize=17)


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, machine, "N{}M{}.pdf".format(n_sites, time_steps)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()