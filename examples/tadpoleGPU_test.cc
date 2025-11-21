//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include "qcdloop/timer.h"
#include "qcdloop/tadpoleGPU.h"

using std::vector;
using std::cout;
using std::endl;
using complex = Kokkos::complex<double>;

/*!
* \brief Computes the TadPole integral defined as:
* \f[
* I_{1}^{D=4-2 \epsilon}(m^2)= m^2 \left( \frac{\mu^2}{m^2-i \epsilon}\right) \left[ \frac{1}{\epsilon} +1 \right] + O(\epsilon)
*   \f]
*
* Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
*
* \param mu2 is the squre of the scale mu
* \param m are the squares of the masses of the internal lines
* \param p are the four-momentum squared of the external lines
*/
template<typename TOutput, typename TMass, typename TScale>
std::vector<TOutput> TP(
    const TScale& mu2,
    vector<TMass> const& m,
    vector<TScale> const& p,
    int batch_size,
    int mode) {

    ql::Timer tt;

    // Initialize views
    Kokkos::View<TOutput* [3]> res_d("res", batch_size);
    auto res_h = Kokkos::create_mirror_view(res_d);
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,batch_size); 

    if (mode == 0) { // performance benchmark
        tt.start();
    }
    
    Kokkos::View<double*> p_d("p", 0);
    Kokkos::View<double*> m_d("m", 1); 

    auto p_h = Kokkos::create_mirror_view(p_d);
    auto m_h = Kokkos::create_mirror_view(m_d);
    
    // Populate views and copy to device
    const double mu2_d = mu2;
    m_h(0) = m[0];  
    Kokkos::deep_copy(m_d, m_h);

    if (mu2_d < 0) Kokkos::printf("TadPole integral mu2 is negative!");
    
    Kokkos::parallel_for("Tadpole Integral", policy, KOKKOS_LAMBDA(const int& i){       
        ql::TP0(res_d, mu2_d, m_d, p_d, i);                                      
    }); 

    Kokkos::deep_copy(res_h, res_d);

    if (mode == 0) { // performance benchmark
        tt.printTime(tt.stop());
        return std::vector<TOutput>();
    }

    // Return the results as a vector
    std::vector<TOutput> results;
    for (size_t i = 0; i < res_d.extent(1); i++) {
        results.push_back(res_h(batch_size - 1, i));
    }

    return results;
}




int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
      
      // Parse command line arguments
      int n_tests = 1000000; // default value
      int batch_size = 1000000; // default value
      
      #if MODE == 0
          // Performance benchmark mode: n_tests=1, batch_size from command line
          n_tests = 1;
          if (argc > 1) {
              try {
                  batch_size = std::stoi(argv[1]);
                  if (batch_size <= 0) {
                      std::cout << "Error: batch_size must be a positive integer. Using default value of 1000000." << std::endl;
                      batch_size = 1000000;
                  }
              } catch (const std::exception& e) {
                  std::cout << "Error: Invalid argument for batch_size. Using default value of 1000000." << std::endl;
                  batch_size = 1000000;
              }
          }
          
          if (argc > 2) {
              std::cout << "Usage: " << argv[0] << " [batch_size]" << std::endl;
              std::cout << "  batch_size: Number of batch iterations for performance benchmark (default: 1000000)" << std::endl;
              std::cout << "  MODE=0: Performance benchmark mode" << std::endl;
          }
      #elif MODE == 1
          // Accuracy test mode: batch_size=1, n_tests from command line
          batch_size = 1;
          if (argc > 1) {
              try {
                  n_tests = std::stoi(argv[1]);
                  if (n_tests <= 0) {
                      std::cout << "Error: n_tests must be a positive integer. Using default value of 1000000." << std::endl;
                      n_tests = 1000000;
                  }
              } catch (const std::exception& e) {
                  std::cout << "Error: Invalid argument for n_tests. Using default value of 1000000." << std::endl;
                  n_tests = 1000000;
              }
          }
          
          if (argc > 2) {
              std::cout << "Usage: " << argv[0] << " [n_tests]" << std::endl;
              std::cout << "  n_tests: Number of test iterations for accuracy testing (default: 1000000)" << std::endl;
              std::cout << "  MODE=1: Accuracy test mode" << std::endl;
          }
      #else
          // Fallback mode: both from command line
          if (argc > 1) {
              try {
                  n_tests = std::stoi(argv[1]);
                  if (n_tests <= 0) {
                      std::cout << "Error: n_tests must be a positive integer. Using default value of 1000000." << std::endl;
                      n_tests = 1000000;
                  }
              } catch (const std::exception& e) {
                  std::cout << "Error: Invalid argument for n_tests. Using default value of 1000000." << std::endl;
                  n_tests = 1000000;
              }
          }
          
          if (argc > 2) {
              try {
                  batch_size = std::stoi(argv[2]);
                  if (batch_size <= 0) {
                      std::cout << "Error: batch_size must be a positive integer. Using default value of 1000000." << std::endl;
                      batch_size = 1000000;
                  }
              } catch (const std::exception& e) {
                  std::cout << "Error: Invalid argument for batch_size. Using default value of 1000000." << std::endl;
                  batch_size = 1000000;
              }
          }
          
          if (argc > 3) {
              std::cout << "Usage: " << argv[0] << " [n_tests] [batch_size]" << std::endl;
              std::cout << "  n_tests: Number of test iterations (default: 1000000)" << std::endl;
              std::cout << "  batch_size: Number of batch iterations (default: 1000000)" << std::endl;
              std::cout << "  MODE not set: Both parameters configurable" << std::endl;
          }
      #endif

      std::cout << "Running with n_tests = " << n_tests << std::endl;
      std::cout << "Running with batch_size = " << batch_size << std::endl;

      #if MODE == 1
      // Print CSV header for accuracy test mode
      std::cout << "Target Integral,Test ID,mu2,ms,ps,Coeff 1,Coeff 2,Coeff 3" << std::endl;
      #endif
  }

  Kokkos::finalize();
  return 0;
}