//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include "qcdloop/exceptions.h"
#include "qcdloop/timer.h"
#include "tadpoleGPU.cc"

using std::vector;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int batch_size = 2e8;
    ql::Timer tt;
    cout << scientific << setprecision(32);
    /*
    _________________________________________________________
    This is experimental for usage on GPUs 
    */

    // Initialize views
    
    using complex = Kokkos::complex<double>;
    const double mu2_d = std::pow(1.7,2.0);
    Kokkos::View<double*> p_d("p", 0);
    Kokkos::View<double*> m_d("m", 1); 
    Kokkos::View<complex*> cm_d("cm", 1); 
    Kokkos::View<complex* [3]> res_d("res", batch_size);
    auto p_h = Kokkos::create_mirror_view(p_d);
    auto m_h = Kokkos::create_mirror_view(m_d);
    auto cm_h = Kokkos::create_mirror_view(cm_d);

    // Populate views on host
    Kokkos::deep_copy(m_h, 5.0);  
    Kokkos::deep_copy(cm_h, Kokkos::complex<double>(5.0, 0.0));
    
    // Copy to device
    Kokkos::deep_copy(m_d, m_h);
    Kokkos::deep_copy(cm_d, cm_h);

    // Call the integral
    tt.start();
    if (mu2_d < 0) throw ql::RangeError("TadPole::integral","mu2 is negative!");
    ql::integral_gpu integral(mu2_d, p_d, cm_d, res_d);
    Kokkos::parallel_for("Tadpole Integral", batch_size, integral);
    
    // Copy result to host
    auto res_h = Kokkos::create_mirror_view(res_d);
    Kokkos::deep_copy(res_h, res_d);
    tt.printTime(tt.stop());
    // Print result and time
    for (size_t i = 0; i < res_d.extent(1); i++) 
      cout << "eps" << i << "\t" << res_h(batch_size - 1,i) << endl;
  }
  
  Kokkos::finalize();
  return 0;
}