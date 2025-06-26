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
#include "qcdloop/boxGPU.h"

using std::vector;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;
using complex = Kokkos::complex<double>;


/*!
* Computes the Box integral defined as:
* \f[
* I_{4}^{D}(p_1^2,p_2^2,p_3^2,p_4^2;s_{12},s_{23};m_1^2,m_2^2,m_3^2,m_4^2)= \frac{\mu^{4-D}}{i \pi^{D/2} r_{\Gamma}} \int d^Dl \frac{1}{(l^2-m_1^2+i \epsilon)((l+q_1)^2-m_2^2+i \epsilon)((l+q_2)^2-m_3^2+i\epsilon)((l+q_4)^2-m_4^2+i\epsilon)}
*   \f]
* where \f$ q_1=p_1, q_2=p_1+p_2, q_3=p_1+p_2+p_3\f$ and \f$q_0=q_4=0\f$.
*
* Implementation of the formulae of Denner et al. \cite Denner:1991qq,
* 't Hooft and Veltman \cite tHooft:1978xw, Bern et al. \cite Bern:1993kr.
*
* \param res output object res[0,1,2] the coefficients in the Laurent series
* \param mu2 is the square of the scale mu
* \param m are the squares of the masses of the internal lines
* \param p are the four-momentum squared of the external lines
*/
template<typename TOutput, typename TMass, typename TScale>
void BO(
    const TScale& mu2,
    vector<TMass> const& m,
    vector<TScale> const& p,
    int batch_size,
    int mode) {

    ql::Timer tt;
    Kokkos::View<complex* [3]> res_d("res", batch_size);
    auto res_h = Kokkos::create_mirror_view(res_d);
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,batch_size); 
    
    if (mode == 0) { // performance benchmark
        tt.start();
    }

    // LOGIC TO CALL B0m-B4m goes here


    Kokkos::deep_copy(res_h, res_d);

    if (mode == 0) { // performance benchmark
        tt.printTime(tt.stop());
        return;
    }
    
    for (size_t i = 0; i < res_d.extent(1); i++) {
        printf("%.15f",res_h(batch_size - 1,i).real()); std::cout << ", ";
        printf("%.15f",res_h(batch_size - 1,i).imag()); std::cout << endl;
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

        /**
        * Triangle
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
            std::cout << "Triangle Integral Performance Benchmark with batch size: " << batch_size << std::endl;
        }

        // Initialize params
        std::vector<double> mu2s = {
            1.0
        };
        
        std::vector<std::vector<double>> ms = {
            {5.0, 2.0, 3.0}
        };
        
        std::vector<std::vector<double>> ps = {
            {1.0, 2.0, 4.0}
        };

        // Call the integral
        for (size_t i = 0; i < mu2s.size(); i++){
            // BO<complex,double,double>(mu2s[i], ms[i], ps[i], batch_size, mode);
        } 

    }
    Kokkos::finalize();
    return 0;
}