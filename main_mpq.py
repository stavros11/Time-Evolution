"""Main script to run on MPQ/Windows.

Do not post on GitHub!
"""
import main


if __name__ == '__main__':
  args = main.parser.parse_args()
  args.data_dir = "D:/ClockV3"

  args.save_name = "sweep_oneway"
  args.n_sites = 6
  args.time_steps = 20
  args.h_ev = 0.5
  args.h_init = 1.0

  args.machine_type = "FullWavefunction"
  #args.d_bond = 4
  #args.d_phys = 2

  args.sweep_opt = True
  args.sweep_both_directions = False
  args.n_epochs = 1000
  args.learning_rate = None
  args.n_message = 200

  args.n_samples = 0
  args.n_corr = 1
  args.n_burn = 10
  args.sample_time = False

  main.main(**vars(args))