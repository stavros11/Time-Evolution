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


#data_dir = "D:/ClockV5/histories"
data_dir = "/home/stavros/DATA/MPQ/ClockV5/histories"
ylabels = {"sweeping_exact_Eloc": "$E_\mathrm{loc}$",
           "sweeping_avg_overlaps": r"$1 - \overline{\mathrm{Fid}(t)}$"}


quantity = ["sweeping_exact_Eloc", "sweeping_avg_overlaps"][1]
name = "allstates2"
machine = ["fullwv", "mpsD4"][0]
#time_steps_list = [30]
time_steps_list = [10, 20, 30]
n_sweeps = 10

skip_grow = True
save = False

n_sites = 6


def get_filedir(time_steps: int) -> str:
  filename = [name,
              "nsweeps{}".format(n_sweeps),
              machine,
              "N{}M{}.h5".format(n_sites, time_steps)]
  return os.path.join(data_dir, "_".join(filename))

def transform_plot_data(x: np.ndarray) -> np.ndarray:
  time_steps = len(x) // n_sweeps
  if skip_grow:
    x = x[time_steps:]
  if "overlap" in quantity:
    return 1 - x
  return x

#data = h5py.File(get_filedir(n_sweeps, n_sites, time_steps), "r")
#for k, v in data.items():
#  print(k, v.shape)

plt.figure(figsize=(7, 4))
for time_steps, color in zip(time_steps_list, cp):
  data = h5py.File(get_filedir(time_steps), "r")

  x_values = np.linspace(0, n_sweeps, n_sweeps * time_steps)
  xticks = list(range(n_sweeps + 1))
  if skip_grow:
    xticks = xticks[1:]
    x_values = x_values[time_steps:]

  plt.semilogy(x_values, transform_plot_data(data[quantity][()][:, -1].real),
               label="$T={}$".format(time_steps), color=color, linewidth=2.4)
  data.close()

plt.axvline(x=1.0, color="black", linestyle="--", linewidth=2.0)
plt.xlabel("Sweep")
plt.xticks(xticks)
plt.ylabel(ylabels[quantity])
plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=22)


if save:
  script_name = __file__.split("/")[-1].split(".")[0]
  save_name = [script_name, quantity, name, machine,
               "n_sweeps{}".format(n_sweeps),
               "N{}.png".format(n_sites)]
  plt.savefig("_".join(save_name), bbox_inches="tight")
else:
  plt.show()