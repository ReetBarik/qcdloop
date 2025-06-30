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
using std::setprecision;
using std::scientific;
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
void TP(
    const TScale& mu2,
    vector<TMass> const& m,
    vector<TScale> const& p,
    int batch_size,
    int mode) {

    ql::Timer tt;

    // Initialize views
    Kokkos::View<complex* [3]> res_d("res", batch_size);
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
    if (mode == 0) {std::cout << "Tadpole Integral TP0" << std::endl;}
    Kokkos::parallel_for("Tadpole Integral", policy, KOKKOS_LAMBDA(const int& i){       
      ql::TP0(res_d, mu2_d, m_d, p_d, i);                                      
    }); 

    Kokkos::deep_copy(res_h, res_d);

    if (mode == 0) { // performance benchmark
      tt.printTime(tt.stop());
      return;
    }

    // Print result and time
    for (size_t i = 0; i < res_d.extent(1); i++) {
      printf("%.15f",res_h(batch_size - 1,i).real()); cout << ", ";
      printf("%.15f",res_h(batch_size - 1,i).imag()); cout << endl;
    }
    std::cout << endl;
    
    return;
}




int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {

    /*
    _________________________________________________________
    This is experimental for usage on GPUs 
    */

    /*
    * Tadpole
    */
    double batch_size_d = std::strtod(argv[1], nullptr);
    int batch_size = static_cast<int>(batch_size_d);
    int mode = std::atoi(argv[2]);
    if (batch_size <= 0) {
        std::cerr << "Batch size must be a positive integer." << std::endl;
        return 1;
    }
    if (mode < 0 || mode > 1) {
        std::cerr << "Mode must be either 0 for performance benachmark or 1 for correctness test." << std::endl;
        return 1;
    }

    if (mode == 1) {
        batch_size = 1; // for correctness test, we only need one batch
    } else {
        std::cout << "Tadpole Integral Performance Benchmark with batch size: " << batch_size << std::endl;
    }

    vector<double> mu2s = { 
      std::pow(1.7,2.0) 
    };

    vector<vector<double>> ms = { 
      {5.0}
    };

    vector<vector<double>> ps = {
      {}
    };

    for (size_t i = 0; i < mu2s.size(); i++){
      TP<complex,double,double>(mu2s[i], ms[i], ps[i], batch_size, mode);
    }        
    
    
    
 
  }
  
  Kokkos::finalize();
  return 0;
}