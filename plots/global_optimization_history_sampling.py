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


data_dir = "D:/MPQ/ClockV2/histories/N=6"
#data_dir = "/home/stavros/DATA/MPQ/ClockV5/histories"

machine = "fullwv"
save = False
n_sites = 6
time_steps = 20
n_samples = 20000


filename = ["sampling{}".format(n_samples), machine,
            "N{}M{}.h5py".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")

exact_Eloc = data["exact_Eloc"][()]
sampled_Eloc = data["sampled_Eloc"][()]
overlaps = data["overlaps"][()]
data.close()


fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(sampled_Eloc, color=cp[2], linewidth=1.8, alpha=0.3)
ax.semilogy(exact_Eloc, color=cp[2], linewidth=2.4)
#ax.axvline(x=n_grow, linestyle="--", linewidth=2.0, color="black")
plt.xlabel("Epochs")
plt.ylabel(r"$\left \langle H_\mathrm{eff}\right \rangle $")
#plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=22)

inset_axes = inset_locator.inset_axes(ax, width="50%", height="50%", loc="upper right")
plt.semilogy(1 - overlaps, color=cp[2], linewidth=2.4)
plt.xticks([0, 5000, 10000], fontsize=20)
plt.yticks(fontsize=19)
#inset_axes.xaxis.set_label_coords(0.8,0.185)
#inset_axes.yaxis.set_label_coords(0.16,0.4)
#plt.xlabel("Epochs")
plt.ylabel(r"$1 - \overline{\mathrm{Fid}(t)}$", fontsize=22)


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, machine, "N{}M{}.pdf".format(n_sites, time_steps)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()