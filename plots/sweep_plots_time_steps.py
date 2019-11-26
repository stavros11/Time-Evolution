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


data_dir = "D:/ClockV5/histories"
#data_dir = "/home/stavros/DATA/MPQ/ClockV5/histories"
ylabels = {"exact_Eloc": r"$\left \langle H_\mathrm{eff}\right \rangle $",
           "avg_overlaps": r"$1 - \overline{\mathrm{Fid}(t)}$"}


quantity = ["exact_Eloc", "avg_overlaps"][1]
name = "allstates2"
machine = "fullwv"
n_sites = 6
time_steps_list = list(range(10, 31, 5)) + [12, 14, 16, 18]
time_steps_list.sort()
n_sweeps = 10

skip_grow = True
save = False


quantities = list(ylabels.keys())
glob = {q: [] for q in quantities}
grow = {q: [] for q in quantities}
sweep = {q: [] for q in quantities}
for time_step in time_steps_list:
  filename = ["allstates1_binary", "nsweeps{}".format(n_sweeps),
              machine, "N{}M{}.h5".format(n_sites, time_step)]
  data = h5py.File(os.path.join(data_dir, "_".join(filename)))
  for q in quantities:
    #grow[q].append(data["sweeping_{}".format(q)][time_step - 1, -1])
    sweep[q].append(data["sweeping_{}".format(q)][-1, -1])
  data.close()

  filename = ["allstates1_justgrow",
              machine, "N{}M{}.h5".format(n_sites, time_step)]
  data = h5py.File(os.path.join(data_dir, "_".join(filename)))
  for q in quantities:
    grow[q].append(data["sweeping_{}".format(q)][-1, -1])
  data.close()

  filename = ["allstates1_global",
              machine, "N{}M{}.h5".format(n_sites, time_step)]
  data = h5py.File(os.path.join(data_dir, "_".join(filename)))
  for q in quantities:
    glob[q].append(data[q][-1])
  data.close()

glob["avg_overlaps"] = 1 - np.array(glob["avg_overlaps"])
grow["avg_overlaps"] = 1 - np.array(grow["avg_overlaps"])
sweep["avg_overlaps"] = 1 - np.array(sweep["avg_overlaps"])


plt.figure(figsize=(7, 4))
plt.semilogy(time_steps_list, glob[quantity], color=cp[0], linewidth=2.6, marker="o",
             markersize=8, label="Global")
plt.semilogy(time_steps_list, grow[quantity], color=cp[2], linewidth=2.6, marker="^",
             markersize=8, label="Grow")
plt.semilogy(time_steps_list, sweep[quantity], color=cp[1], linewidth=2.6, marker="v",
             markersize=8, label="Sweep", linestyle=":")
plt.xlabel("$T$")
plt.ylabel(ylabels[quantity])
#plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=22)
plt.legend(fontsize=20)


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, quantity, name, machine,
               "n_sweeps{}".format(n_sweeps),
               "N{}.png".format(n_sites)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()