"""Script that generates the dynamic plot for the presentation."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import h5py

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = 26
cp = sns.color_palette()

data_dir = "/home/stavros/DATA"
#data_dir = "D:/"
figsize = (18, 14)
#figsize = (8, 6)

n_sites = 6
T_list = [100, 200]
t_grid = [np.linspace(0, 2.0, time_steps + 1) for time_steps in T_list]

file = [h5py.File("{}/ClockV3/histories/allstates_tf2_withoverlap_fullwv_N{}M{}.h5".format(data_dir, n_sites, time_steps), "r")
        for time_steps in T_list]
#n_epochs = len(file[-1]["time_overlaps"][()])
n_epochs = 20000
step = 1000

exact_file = h5py.File("{}/ClockV3/observables/fullwv_tf2_N{}.h5".format(data_dir, n_sites), "r")
exact_sigma_x = exact_file["metrics/exact/sigma_x/{}".format(3)][()]
exact_file.close()

for i, epoch in enumerate(range(0, n_epochs, step)):
  plt.figure(figsize=figsize)
  plt.subplot(221)
  plt.semilogy(file[0]["exact_Eloc"][()][:epoch].real, label="$T={}$".format(T_list[0]), color=cp[0], linewidth=2.5)
  plt.semilogy(file[1]["exact_Eloc"][()][:epoch].real, label="$T={}$".format(T_list[1]), color=cp[3], linewidth=2.5)
  plt.xlabel("Iterations")
  plt.ylabel(r"$\left \langle H_\mathrm{eff} \right \rangle$")
  plt.xlim([0, n_epochs])
  plt.ylim([1e-6, 2e-2])
  plt.text(3000, 0.01, "Iter: {}".format(epoch))
  plt.legend(loc="upper right")

  plt.subplot(222)
  plt.plot(t_grid[0], file[0]["time_sigma_x"][()][epoch], color=cp[0], linewidth=2.5)
  plt.plot(t_grid[1], file[1]["time_sigma_x"][()][epoch], color=cp[3], linewidth=2.5)
  plt.plot(t_grid[1], exact_sigma_x, linewidth=2.2, color="black", linestyle="--", label="Exact Evolution")
  plt.legend()
  plt.xlabel("$t$")
  plt.ylabel(r"$\left \langle \sigma ^x\right \rangle $")

  plt.subplot(223)
  plt.semilogy(t_grid[0], 1 - file[0]["time_overlaps"][()][epoch], color=cp[0], linewidth=2.5)
  plt.semilogy(t_grid[1], 1 - file[1]["time_overlaps"][()][epoch], color=cp[3], linewidth=2.5)
  plt.xlabel("$t$")
  plt.ylabel(r"$1 - \mathrm{Fid}(t)$")
  plt.ylim([1e-6, 1.0])

  plt.subplot(224)
  plt.semilogy(t_grid[0], file[0]["time_norm"][()][epoch] - 1, color=cp[0], linewidth=2.5)
  plt.semilogy(t_grid[1], file[1]["time_norm"][()][epoch] - 1, color=cp[3], linewidth=2.5)
  plt.xlabel("$t$")
  plt.ylabel(r"Norm(t) $- 1$")
  plt.ylim([1e-5, 10.0])


  #plt.show()
  plt.savefig("optimization_dynamic-{}.png".format(i))
  plt.close()
  #display.clear_output(wait=True)
  #display.display(plt.gcf())

file[0].close()
file[1].close()