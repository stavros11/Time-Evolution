import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = 20
cp = sns.color_palette()
from mpl_toolkits.axes_grid import inset_locator


#data_dir = "D:/ClockV5/histories"
data_dir = "/home/stavros/DATA/MPQ/ClockAuto/histories"

d_bond = 4
machine = "smallmpsD{}".format(d_bond)
save = True
n_sites = 6
time_steps = 20

n_epochs = 5000

filename = ["allstates_global", "{}_autograd".format(machine), "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")
naive_heff = data["exact_Eloc"][()].real[:n_epochs]
naive_overlaps = data["avg_overlaps"][()].real[:n_epochs]
data.close()

filename = ["allstates_global", "{}_prodprop".format(machine), "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")
prop_heff = data["exact_Eloc"][()].real[:n_epochs]
prop_overlaps = data["avg_overlaps"][()].real[:n_epochs]
data.close()

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(naive_heff, color=cp[0], linewidth=2.4, label="Repeat")
ax.semilogy(prop_heff, color=cp[1], linewidth=2.4, label="Propagate")
plt.xlabel("Epochs", fontsize=22)
plt.ylabel(r"$\left \langle H_\mathrm{eff}\right \rangle $", fontsize=26)
#plt.legend(bbox_to_anchor=(0.071, 1.0), fontsize=20)
plt.text(0.92, 0.002, "$D={}$".format(d_bond), fontsize=20)
plt.legend(fontsize=20, loc="upper left")


inset_axes = inset_locator.inset_axes(ax, width="45%", height="50%", loc="upper right")
plt.semilogy(1 - naive_overlaps, color=cp[0], linewidth=2.4)
plt.semilogy(1 - prop_overlaps, color=cp[1], linewidth=2.4)
plt.xticks([0, 2000, 4000])
plt.yticks([])
#plt.xlabel("Epochs")
plt.ylabel(r"$1 - \overline{\mathrm{Fid}(t)}$", fontsize=18)


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, machine, "N{}M{}.pdf".format(n_sites, time_steps)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()
