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

name = "allstates1_growglobal"
machine = ["fullwv", "mpsD4"][1]
#time_steps_list = [30]
time_steps = 20
n_sweeps = 1
save = False

n_sites = 6


filename = [name, "nsweeps{}".format(n_sweeps), machine,
            "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")

grow = data["_".join(["sweeping", "exact_Eloc"])][()].real.ravel()
n_grow = len(grow)
glob = data["exact_Eloc"][()].real
h_eff = np.concatenate([grow, glob], axis=0)

grow = data["_".join(["sweeping", "avg_overlaps"])][()].real.ravel()
glob = data["avg_overlaps"][()].real
overlaps = np.concatenate([grow, glob], axis=0)
data.close()

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(h_eff, color=cp[0], linewidth=2.4)
ax.axvline(x=n_grow, linestyle="--", linewidth=2.0, color="black")
plt.xlabel("Epochs")
plt.ylabel(r"$\left \langle H_\mathrm{eff}\right \rangle $")
#plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=22)

inset_axes = inset_locator.inset_axes(ax, width="50%", height="50%", loc=3)
plt.semilogy(1 - overlaps, color=cp[0], linewidth=2.4)
plt.axvline(x=n_grow, linestyle="--", linewidth=2.0, color="black")
plt.xticks([])
plt.yticks([])
#plt.xlabel("Epochs")
#plt.ylabel(r"$1 - \overline{\mathrm{Fid}(t)}$")


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, name, machine,
               "n_sweeps{}".format(n_sweeps),
               "N{}.png".format(n_sites)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()