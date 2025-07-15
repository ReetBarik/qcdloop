//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include "qcdloop/timer.h"
#include "qcdloop/boxGPU.h"

using std::vector;
using std::cout;
using std::endl;
using std::string;
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

    // Normalization
    const TScale scalefac = ql::Max(Kokkos::abs(p[4]),ql::Max(Kokkos::abs(p[5]),ql::Max(Kokkos::abs(p[0]),ql::Max(Kokkos::abs(p[1]),ql::Max(Kokkos::abs(p[2]),Kokkos::abs(p[3]))))));

    TMass xpi_temp[13];
    xpi_temp[0] = m[0] / scalefac;
    xpi_temp[1] = m[1] / scalefac;
    xpi_temp[2] = m[2] / scalefac;
    xpi_temp[3] = m[3] / scalefac;
    xpi_temp[4] = TMass(p[0] / scalefac);
    xpi_temp[5] = TMass(p[1] / scalefac);
    xpi_temp[6] = TMass(p[2] / scalefac);
    xpi_temp[7] = TMass(p[3] / scalefac);
    xpi_temp[8] = TMass(p[4] / scalefac);
    xpi_temp[9] = TMass(p[5] / scalefac);
    xpi_temp[10] = xpi_temp[4] + xpi_temp[5] + xpi_temp[6] + xpi_temp[7] - xpi_temp[8] - xpi_temp[9];
    xpi_temp[11] =-xpi_temp[4] + xpi_temp[5] - xpi_temp[6] + xpi_temp[7] + xpi_temp[8] + xpi_temp[9];
    xpi_temp[12] = xpi_temp[4] - xpi_temp[5] + xpi_temp[6] - xpi_temp[7] + xpi_temp[8] + xpi_temp[9];

    const Kokkos::Array<TMass, 13> xpi = {xpi_temp[0], xpi_temp[1], xpi_temp[2], xpi_temp[3],
                                          xpi_temp[4], xpi_temp[5], xpi_temp[6], xpi_temp[7],
                                          xpi_temp[8], xpi_temp[9], xpi_temp[10], xpi_temp[11],
                                          xpi_temp[12]};
    const TScale musq = mu2 / scalefac;

    // Count number of internal masses
    int massive = 0;
    for (size_t i = 0; i < 4; i++)
        if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(xpi[i]))) massive += 1;

    // check cayley elements
    const TMass y13 = xpi[0] + xpi[2] - xpi[8];
    const TMass y24 = xpi[1] + xpi[3] - xpi[9];
    if (ql::iszero<TOutput, TMass, TScale>(y13) || ql::iszero<TOutput, TMass, TScale>(y24)) {
        std::cout << "Box::integral: Modified Cayley elements y13 or y24=0" << std::endl;
        
        Kokkos::parallel_for("Box Integral 00", policy, KOKKOS_LAMBDA(const int& i) {     
            res_d(i,0) = TOutput(0.0); 
            res_d(i,1) = TOutput(0.0); 
            res_d(i,2) = TOutput(0.0);                                                 
        }); 
        
    } else {
        if (massive == 0) {
            Kokkos::parallel_for("Box Integral 0m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B0m<TOutput, TMass, TScale>(res_d, xpi, musq, i);                                                      
            });
        } else if (massive == 1) {
            Kokkos::parallel_for("Box Integral 1m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B1m<TOutput, TMass, TScale>(res_d, xpi, musq, i);                                                      
            });
        } else if (massive == 2) {
            Kokkos::parallel_for("Box Integral 2m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B2m<TOutput, TMass, TScale>(res_d, xpi, musq, i);                                                      
            });
        } else if (massive == 3) {
            Kokkos::parallel_for("Box Integral 3m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B3m<TOutput, TMass, TScale>(res_d, xpi, musq, i);                                                      
            });
        } else if (massive == 4) {
            Kokkos::parallel_for("Box Integral 4m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B4m<TOutput, TMass, TScale>(res_d, xpi, i);                                                      
            });
        }      

        Kokkos::parallel_for("Normalize Res", policy, KOKKOS_LAMBDA(const int& i) {
            res_d(i,0) /= (scalefac * scalefac);
            res_d(i,1) /= (scalefac * scalefac);
            res_d(i,2) /= (scalefac * scalefac);
        });
    }


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
        * Box
        */
        double batch_size_d = std::strtod(argv[1], nullptr);
        int batch_size = static_cast<int>(batch_size_d);
        int mode = std::atoi(argv[2]);
        if (batch_size <= 0) {
            std::cerr << "Batch size must be a positive integer." << std::endl;
            return 1;
        }
        if (!(mode == 0 || mode == 1)) {
            std::cerr << "Mode must be either 0 for performance benachmark or 1 for correctness test." << std::endl;
            return 1;
        }

        if (mode == 1) {
            batch_size = 1; // for correctness test, we only need one batch
        } else {
            std::cout << "Box Integral Performance Benchmark with batch size: " << batch_size << std::endl;
        }

        // Initialize params
        std::vector<double> mu2s = {
            1.0,
            1.0, 
            1.0,
            1.0, 
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            // 1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        };
        
        std::vector<std::vector<double>> ms = {
            {0.0, 0.0, 0.0, 0.0},  // B00
            {0.0, 0.0, 0.0, 0.0},  // B0m - BIN0
            {1.0, 0.0, 0.0, 0.0},  // B1m - BIN1
            {1.0, 1.0, 0.0, 0.0},  // B2m - BIN2
            {0.0, 1.0, 1.0, 1.0},  // B3m - BIN3
            {1.0, 1.0, 1.0, 1.0},  // B4m - BIN4
            {0.0, 0.0, 0.0, 0.0},  // B0m - B1
            {0.0, 0.0, 0.0, 0.0},  // B0m - B2
            {0.0, 0.0, 0.0, 0.0},  // B0m - B3
            {0.0, 0.0, 0.0, 0.0},  // B0m - B4
            {0.0, 0.0, 0.0, 0.0},  // B0m - B5
            {1.0, 0.0, 0.0, 0.0},  // B1m - B6
            {1.0, 0.0, 0.0, 0.0},  // B1m - B7 
            {1.0, 0.0, 0.0, 0.0},  // B1m - B8
            {0.0, 0.0, 1.0, 0.0},  // B1m - B9
            {0.0, 1.0, 0.0, 0.0},  // B1m - B10 
            // {0.0, 1.0, 0.0, 1.0},  // B2m - B11
            {1.0, 1.0, 0.0, 0.0},  // B2m - B12
            {1.0, 1.0, 0.0, 0.0},  // B2m - B13
            {0.0, 1.0, 0.0, 1.0},  // B2m - B14
            {0.0, 2.0, 0.0, 3.0},  // B2m - B15
            {0.0, 1.0, 1.0, 1.0},  // B3m - B16
        };
        
        std::vector<std::vector<double>> ps = {
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},  // B00
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},  // B0m - BIN0
            {1.0, 1.0, 1.0, 1.0, 2.0, 2.0},  // B1m - BIN1
            {1.0, 1.0, 1.0, 1.0, 2.0, 2.0},  // B2m - BIN2
            {2.0, 3.0, 4.0, 5.0, 6.0, 7.0},  // B3m - BIN3
            {3.0, 4.0, 5.0, 6.0, 7.0, 8.0},  // B4m - BIN4
            {0.0, 0.0, 0.0, 0.0, 3.0, 1.0},  // B0m - B1
            {1.0, 0.0, 0.0, 0.0, 3.0, 1.0},  // B0m - B2
            {0.0, 1.0, 0.0, 2.0, 3.0, 1.0},  // B0m - B3
            {1.0, 2.0, 0.0, 0.0, 3.0, 4.0},  // B0m - B4
            {1.0, 2.0, 3.0, 0.0, 4.0, 5.0},  // B0m - B5
            {1.0, 0.0, 0.0, 1.0, 2.0, 3.0},  // B1m - B6
            {1.0, 0.0, 0.0, 0.0, 2.0, 3.0},  // B1m - B7
            {0.0, 0.0, 0.0, 0.0, 2.0, 2.0},  // B1m - B8
            {1.0, 1.0, 1.0, 0.0, 2.0, 2.0},  // B1m - B9
            {0.0, 0.0, 1.0, 0.0, 2.0, 0.0},  // B1m - B10
            // {0.0, 0.0, 0.0, 1.0, 2.0, 1.0},  // B2m - B11
            {0.0, 0.0, 0.0, 1.0, 2.0, 0.0},  // B2m - B12
            {2.0, 0.0, 0.0, 0.0, 2.0, 0.0},  // B2m - B13
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0},  // B2m - B14
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},  // B2m - B15
            {1.0, 0.0, 0.0, 1.0, 0.0, 0.0},  // B3m - B16
        };

        vector<string> integrals = {
            "Box Integral B00",    // B00
            "Box Integral BIN0",   // BIN0
            "Box Integral BIN1",   // BIN1
            "Box Integral BIN2",   // BIN2
            "Box Integral BIN3",   // BIN3
            "Box Integral BIN4",   // BIN4
            "Box Integral B1",     // B1
            "Box Integral B2",     // B2
            "Box Integral B3",     // B3
            "Box Integral B4",     // B4
            "Box Integral B5",     // B5
            "Box Integral B6",     // B6
            "Box Integral B7",     // B7
            "Box Integral B8",     // B8
            "Box Integral B9",     // B9
            "Box Integral B10",    // B10
            // "Box Integral B11",    // B11
            "Box Integral B12",    // B12
            "Box Integral B13",    // B13
            "Box Integral B14",    // B14
            "Box Integral B15",    // B15
            "Box Integral B16",    // B16
        };

        // Call the integral
        for (size_t i = 0; i < mu2s.size(); i++){
            if (mode == 0) {std::cout << integrals[i] << std::endl;}
            BO<complex,double,double>(mu2s[i], ms[i], ps[i], batch_size, mode);
        } 

    }
    Kokkos::finalize();
    return 0;
}