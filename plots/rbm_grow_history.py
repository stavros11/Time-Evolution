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


data_dir = "D:/ClockV5/histories"
#data_dir = "/home/stavros/DATA/MPQ/ClockV4/histories"

save = False
n_sites = 6
time_steps = 20


filename = ["allstates1_grow", "rbm_autograd", "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")
grow_heff = data["_".join(["growing", "exact_Eloc"])][()].real.ravel()
grow_overlaps = data["_".join(["growing", "avg_overlaps"])][()].real.ravel()
data.close()


filename = ["allstates1_global", "rbm_autograd", "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")
global_heff = data["exact_Eloc"][()].real
global_overlaps = data["avg_overlaps"][()].real
data.close()


fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(grow_heff, color=cp[0], linewidth=2.4, label="Grow")
ax.semilogy(global_heff, color=cp[1], linewidth=2.4, label="Global")
plt.xlabel("Epochs")
plt.ylabel(r"$\left \langle H_\mathrm{eff}\right \rangle $")
#plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=22)
plt.legend(fontsize=20, loc="upper left")

inset_axes = inset_locator.inset_axes(ax, width="50%", height="50%", loc="upper right")
plt.semilogy(1 - grow_overlaps, color=cp[0], linewidth=2.4)
plt.semilogy(1 - global_overlaps, color=cp[1], linewidth=2.4)
plt.xticks([])
plt.yticks([])
#inset_axes.xaxis.set_label_coords(0.8,0.185)
#inset_axes.yaxis.set_label_coords(0.16,0.4)
##plt.xlabel("Epochs")
plt.ylabel(r"$1 - \overline{\mathrm{Fid}(t)}$", fontsize=17)


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, "N{}M{}.pdf".format(n_sites, time_steps)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()