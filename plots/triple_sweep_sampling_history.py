import os
import argparse
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
"sweeping_sampled_Eloc": r"$\left \langle H_\mathrm{eff}\right \rangle _{T=3}$",
"sweeping_avg_overlaps": r"$1 - \overline{\mathrm{Fid}(t)}$"}
# Hard-coded quantities are for full wavefunction only
global_quantities = {"sweeping_exact_Eloc": 0.0018196748798055369,
                     "sweeping_avg_overlaps": 1 - 0.9757716162996253}
grow_quantities = {"sweeping_exact_Eloc": 0.0019174118255215061,
                   "sweeping_avg_overlaps": 1 - 0.9983866318977118}

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="/home/stavros/DATA/MPQ/ClockV5/histories", type=str)
parser.add_argument("--machine", default="fullwv", type=str)
parser.add_argument("--n-sites", default=6, type=int)
parser.add_argument("--time-steps", default=20, type=int)
parser.add_argument("--n-samples", default=500, type=int)
parser.add_argument("--n-sweeps", default=100, type=int)

parser.add_argument("--quantity", default="sweeping_exact_Eloc", type=str)
parser.add_argument("--save", action="store_true")

def sweep_start_x(n_sweeps: int, time_steps: int):
  counter = 0
  for i in range(n_sweeps):
    yield counter
    if i % 2:
      counter += time_steps - 1
    else:
      counter += time_steps

def main(data_dir: str, machine: str, n_sites: int, time_steps: int,
         n_samples: int, n_sweeps: int, quantity: str, save: bool = False):
  filename = ["sampling{}".format(n_samples), "triplesweeps{}".format(n_sweeps),
              machine, "N{}M{}.h5".format(n_sites, time_steps)]
  data = h5py.File(os.path.join(data_dir, "_".join(filename)), "r")
  if "overlaps" in quantity:
    #sweep_data = 1 - data[quantity][()].ravel()
    sweep_data = 1 - data[quantity][()][:, 0]
  else:
    #sweep_data = data[quantity][()].ravel()
    sweep_data = data[quantity][()][:, 0]
  sampled_exact_Eloc = data["sweeping_exact_sampled_Eloc"][()][:, 0]
  data.close()

  sweep_starts = list(sweep_start_x(n_sweeps, time_steps))[::100]
  sweep_starts.append(2 * sweep_starts[-1] - sweep_starts[-2])

  fig, ax = plt.subplots(figsize=(7, 4))
  if "sampled" in quantity:
    ax.semilogy(sweep_data, color=cp[1], linewidth=2.4, alpha=0.5)
    ax.semilogy(sampled_exact_Eloc, color=cp[3], linewidth=2.4)
  else:
    ax.semilogy(sweep_data, color=cp[1], linewidth=2.4, label="Sweep")
    ax.axhline(global_quantities[quantity], color=cp[0], linewidth=2.4,
               label="Global")
    ax.axhline(grow_quantities[quantity], color=cp[2], linewidth=2.4,
               label="Grow")

  ticks = list(range(0, n_sweeps + 1, n_sweeps // (len(sweep_starts) - 1)))

  ax.set_xticks(sweep_starts)
  ax.set_xticklabels(ticks)
  ax.set_xlabel("Sweeps")
  ax.set_ylabel(ylabels[quantity])
  if "overlaps" not in quantity and "sampled" not in quantity:
    ax.legend(fontsize=20, loc="upper right")

  # TODO: Add inset plot from `triple_sweep_optimization_history`

  if save:
    script_name = __file__.split("/")[-1].split(".")[0]
    save_name = [script_name, quantity, machine, "N{}M{}.pdf".format(n_sites, time_steps)]
    plt.savefig("_".join(save_name), bbox_inches="tight")
  else:
    plt.show()


if __name__ == '__main__':
  args = parser.parse_args()
  main(**vars(args))
