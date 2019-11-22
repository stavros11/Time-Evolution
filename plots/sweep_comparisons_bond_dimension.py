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

ylabels = {"exact_Eloc": r"$\left \langle H_\mathrm{eff}\right \rangle $",
           "avg_overlaps": r"$1 - \overline{\mathrm{Fid}(t)}$"}

data_dir = "D:/ClockV5/histories"
#data_dir = "/home/stavros/DATA/MPQ/ClockV5/histories"

save = True
n_sites = 6
time_steps = 20
d_bond_list = [2, 3, 4, 5, 6]
quantity = ["exact_Eloc", "avg_overlaps"][1]


def get_quantity(name, d_bond, sweeping, sweep_ind=-1):
  filename = [name, "mpsD{}".format(d_bond),
              "N{}M{}.h5".format(n_sites, time_steps)]
  filename = "_".join(filename)
  if not sweeping:
    filename += "py" # bug with filenames for old global files
  file = h5py.File(os.path.join(data_dir, filename), "r")

  if sweeping:
    x = file["sweeping_{}".format(quantity)][-1, sweep_ind]
  else:
    x = file[quantity][-1]
  file.close()

  if "overlaps" in quantity:
    return 1 - x
  return x


glob, grow, sweep = [], [], []
for d_bond in d_bond_list:
  grow.append(get_quantity("allstates1_growglobal_nsweeps1", d_bond, True))
  glob.append(get_quantity("allstates", d_bond, False))
  sweep.append(get_quantity("allstates1_binary_nsweeps10", d_bond, True, -1))

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(d_bond_list, glob, color=cp[1], linewidth=2.4,
            label="Global", marker="o", markersize=8)
ax.semilogy(d_bond_list, grow, color=cp[0], linewidth=2.4,
            label="Grow", marker="d", markersize=8)
ax.semilogy(d_bond_list, sweep, color=cp[2], linewidth=2.4,
            label="Sweep", marker="s", markersize=8)
plt.xlabel("$D$")
plt.ylabel(ylabels[quantity])
if "overlaps" not in quantity:
  plt.legend()


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, quantity, "N{}M{}.pdf".format(n_sites, time_steps)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()