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
# Hard-coded quantities are for full wavefunction only
global_quantities = {"sweeping_exact_Eloc": 0.0018196748798055369,
                     "sweeping_avg_overlaps": 1 - 0.9757716162996253}
grow_quantities = {"sweeping_exact_Eloc": 0.0019174118255215061,
                   "sweeping_avg_overlaps": 1 - 0.9983866318977118}


data_dir = "D:/ClockV5/histories"

machine = "fullwv"
save = True
n_sites = 6
time_steps = 20
n_sweeps = 1000
quantity = ["sweeping_exact_Eloc", "sweeping_avg_overlaps"][0]


filename = ["fullsweeps", "nsweeps{}".format(n_sweeps),
            machine, "N{}M{}.h5".format(n_sites, time_steps)]
data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")
if "overlaps" in quantity:
  #sweep_data = 1 - data[quantity][()].ravel()
  sweep_data = 1 - data[quantity][()][:, 0]
else:
  #sweep_data = data[quantity][()].ravel()
  sweep_data = data[quantity][()][:, 0]
data.close()


sweep_starts = np.arange(0, n_sweeps * time_steps + 1, time_steps)


fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(sweep_data, color=cp[1], linewidth=2.4, label="Sweep")
ax.axhline(y=global_quantities[quantity], color=cp[0], linewidth=2.5, label="Global")
if "overlaps" not in quantity:
  ax.axhline(y=grow_quantities[quantity], color=cp[2], linewidth=2.5, label="Grow")
ax.set_xlabel("Sweeps")
ax.set_xticks(sweep_starts[::200])
ax.set_xticklabels(list(range(0, n_sweeps + 1, 200)))
ax.set_ylabel(ylabels[quantity])
if "overlaps" not in quantity:
  ax.legend(fontsize=20, loc="upper left")

inset_axes = inset_locator.inset_axes(ax, width="50%", height="50%", loc="upper right")
inset_axes.plot(sweep_data[sweep_starts[600]:], color=cp[1], linewidth=2.4)
inset_axes.axhline(y=global_quantities[quantity], color=cp[0], linewidth=2.5)
#if "overlaps" not in quantity:
inset_axes.axhline(y=grow_quantities[quantity], color=cp[2], linewidth=2.5)
inset_axes.set_yticks([])
inset_axes.set_xticks(sweep_starts[600::200] - sweep_starts[600])
inset_axes.set_xticklabels([600, 800, 1000], fontsize=20)
plt.ylabel(ylabels[quantity], fontsize=22)

if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, quantity, machine, "N{}M{}.pdf".format(n_sites, time_steps)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()