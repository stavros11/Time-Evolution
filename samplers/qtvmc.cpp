#include <complex>
#include <ctime>
#include <vector>
#include <cstdlib>

class QTVMC_FullWV {
  // Number of sites
  int N_;
  // Number of time steps
  int M_;
  // Current spin configuration
  std::vector<int> spin_conf_;
  // Current time
  int time_;
  // Full wavefunction
  std::vector<std::vector<std::complex<double> > > full_psi_;
  // Current probability
  double prob_;
  // Vector that takes you from binary to decimal
  std::vector<int> bin2dec_;

  double FindProb() {
    int conf_dec = 0;
    for (int i = 0; i < N_; i++) {
      conf_dec += bin2dec_[N_ - i - 1] * (1 - spin_conf_[i]) / 2;
    }
    double prob = std::abs(full_psi_[time_][conf_dec]);
    return prob * prob;
  }

 public:
  void initialize(const int N,
                  const std::vector<std::vector<std::complex<double> > >& psi) {
    // Set random seed
    srand(time(NULL));

    // Initialize wavefunction
    full_psi_ = psi;
    N_ = N;
    M_ = full_psi_.size();

    // Initialize configuration and bin2dec
    spin_conf_.push_back(2 * (rand() % 2) - 1);
    bin2dec_.push_back(1);
    for (int i = 1; i < N_; i++) {
      spin_conf_.push_back(2 * (rand() % 2) - 1);
      bin2dec_.push_back(2 * bin2dec_[i - 1]);
    }
    time_ = rand() % M_;

    // Find probability of current configuration
    prob_ = FindProb();
  }

  void spin_flip() {
    int site2flip = rand() % N_;
    spin_conf_[site2flip] *= -1;

    double new_prob = FindProb();

    float r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    if (new_prob / prob_ > r) {
      // Accept move
      prob_ = new_prob;
    } else {
      // Return to the old configuration
      spin_conf_[site2flip] *= -1;
    }
  }

  void time_flip() {
    int time_add = rand() % M_;
    time_ = (time_ + time_add) % M_;

    double new_prob = FindProb();

    float r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    if (new_prob / prob_ > r) {
      // Accept move
      prob_ = new_prob;
    } else {
      // Return to the old configuration
      time_ = (time_ - time_add + M_) % M_;
    }
  }

  int current_time() { return time_; }

  std::vector<int> current_config() { return spin_conf_; }
};

extern "C" void run(std::complex<double>* psi, int N, int M, int Nstates,
                    int Nsamples, int Ncorr, int Nburn, int* confs,
                    int* times) {
  // int Nsamples = 1000, Ncorr = 1, Nburn = 10;

  QTVMC_FullWV sampler;
  std::vector<std::complex<double> > psi_t(Nstates);
  std::vector<std::vector<std::complex<double> > > psiv;

  // std::cout << "Run variables created" << std::endl;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < Nstates; j++) {
      // psi_t[j] = (psi_re[Nstates * i + j], psi_im[Nstates * i + j]);
      psi_t[j] = psi[Nstates * i + j];
    }
    psiv.push_back(psi_t);
  }

  // std::cout << "Psi initialized" << std::endl;

  sampler.initialize(N, psiv);

  //  std::cout << "Sampler initialized" << std::endl;

  for (int i = 0; i < Nburn; i++) {
    sampler.spin_flip();
    sampler.time_flip();
  }

  // std::cout << "Burn in completed" << std::endl;

  for (int i = 0; i < Nsamples; i++) {
    for (int j = 0; j < Ncorr; j++) {
      sampler.spin_flip();
      sampler.time_flip();
    }
    times[i] = sampler.current_time();
    std::vector<int> current_conf = sampler.current_config();
    for (int j = 0; j < N; j++) {
      confs[N * i + j] = current_conf[j];
    }
  }

  // std::cout << "Samples calculated" << std::endl;
}
