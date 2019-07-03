#include <complex>
#include <ctime>
#include <vector>
#include <cstdlib>

class SpaceMC_FullWV {
  // Number of sites
  int N_;
  // Current spin configuration
  std::vector<int> spin_conf_;
  // Full wavefunction on a specific time (shape (2^N,))
  std::vector<std::complex<double> > full_psi_;
  // Current probability
  double prob_;
  // Vector that takes you from binary to decimal
  std::vector<int> bin2dec_;

  double FindProb() {
    int conf_dec = 0;
    for (int i = 0; i < N_; i++) {
      conf_dec += bin2dec_[N_ - i - 1] * (1 - spin_conf_[i]) / 2;
    }
    double prob = std::abs(full_psi_[conf_dec]);
    return prob * prob;
  }

 public:
  void initialize(const int N) {
    // Set random seed
    srand(time(NULL));
    N_ = N;

    // Initialize configuration and bin2dec
    spin_conf_.push_back(2 * (rand() % 2) - 1);
    bin2dec_.push_back(1);
    for (int i = 1; i < N_; i++) {
      spin_conf_.push_back(2 * (rand() % 2) - 1);
      bin2dec_.push_back(2 * bin2dec_[i - 1]);
    }
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

  void set_psi(const std::vector<std::complex<double> >& psi) {
   // To change psi when we change time
   full_psi_ = psi;
   prob_ = FindProb();
  }

  std::vector<int> current_config() { return spin_conf_; }
};

extern "C" void run(std::complex<double>* psi, int N, int M, int Nstates,
                    int Nsamples, int Ncorr, int Nburn, int* confs) {
  // Nsamples, Ncorr and Nburn are per time step!

  SpaceMC_FullWV sampler;
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

  sampler.initialize(N);

  //  std::cout << "Sampler initialized" << std::endl;

  for (int it = 0; it < M; it++) {
   sampler.set_psi(psiv[it]);

   // Burn in sweeps
   for (int i = 0; i < Nburn; i++) {
     sampler.spin_flip();
   }

   // Statistics sweeps
  for (int i = 0; i < Nsamples; i++) {
    for (int j = 0; j < Ncorr; j++) {
      sampler.spin_flip();
    }
    std::vector<int> current_conf = sampler.current_config();
    for (int j = 0; j < N; j++) {
      confs[N * Nsamples * it + N * i + j] = current_conf[j];
    }
  }
 }
// std::cout << "Samples calculated" << std::endl;
}
