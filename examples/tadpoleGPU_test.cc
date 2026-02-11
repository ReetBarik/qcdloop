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


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
      
      // Parse command line arguments
      int mode = 1; // default value
      int batch_size = 1000000; // default value
      
      if (argc < 2) {
          std::cout << "Usage: " << argv[0] << " <mode> [batch_size]" << std::endl;
          std::cout << "  mode: 0 for performance benchmark, 1 for accuracy test (required)" << std::endl;
          std::cout << "  batch_size: Number of batch iterations (default: 1000000)" << std::endl;
          Kokkos::finalize();
          return 1;
      }
      
      // Parse mode (required)
      try {
          mode = std::stoi(argv[1]);
          if (mode != 0 && mode != 1) {
              std::cout << "Error: mode must be 0 or 1. Using default value of 1." << std::endl;
              mode = 1;
          }
      } catch (const std::exception& e) {
          std::cout << "Error: Invalid argument for mode. Using default value of 1." << std::endl;
          mode = 1;
      }
      
      // Parse batch_size (optional)
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
          std::cout << "Usage: " << argv[0] << " <mode> [batch_size]" << std::endl;
          std::cout << "  mode: 0 for performance benchmark, 1 for accuracy test (required)" << std::endl;
          std::cout << "  batch_size: Number of batch iterations (default: 1000000)" << std::endl;
      }

      std::cout << "Running with mode = " << mode << std::endl;
      std::cout << "Running with batch_size = " << batch_size << std::endl;

      if (mode == 0) {
          // Print CSV header for performance benchmark mode
          std::cout << "Target Integral,Batch size,Time" << std::endl;
      } else if (mode == 1) {
          // Print CSV header for accuracy test mode
          std::cout << "Target Integral,Test ID,mu2,ms,ps,Coeff 1,Coeff 2,Coeff 3" << std::endl;
      }
      
      ql::Timer tt;
      
      // Create Kokkos Views for batch processing
      Kokkos::View<double*> mu2_d("mu2", batch_size);
      Kokkos::View<double* [1]> m_d("m", batch_size);
      Kokkos::View<double*> p_d("p", 0);  // Empty view for API consistency
      Kokkos::View<complex* [3]> res_d("res", batch_size);
      
      auto mu2_h = Kokkos::create_mirror_view(mu2_d);
      auto m_h = Kokkos::create_mirror_view(m_d);
      auto p_h = Kokkos::create_mirror_view(p_d);
      auto res_h = Kokkos::create_mirror_view(res_d);
      
      // Initialize mu2
      for (size_t i = 0; i < batch_size; ++i) {
          mu2_h(i) = 91.2*91.2;
      }
      
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, batch_size);
      
      // TODO: Add test calls here
      // Example pattern:
      // 1. Fill host mirrors with test data
      // 2. Copy to device
      // 3. Launch parallel_for with timing
      // 4. Copy results back
      // 5. Process results for output
  }

  Kokkos::finalize();
  return 0;
}