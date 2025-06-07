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
#include "qcdloop/bubbleGPU.h"

using std::vector;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;
using complex = Kokkos::complex<double>;


/*!
* The integral is defined as:
* \f[
* I_{2}^{D=4-2 \epsilon}(p^2; m_1^2, m_2^2)= \mu^{2 \epsilon} \left[ \frac{1}{\epsilon} - \int_{0}^{1} da \ln (-a (1-a) p^2 + am_{2}^2 + (1-a) m_{1}^{2} - i \epsilon ) \right]  + O(\epsilon)
*   \f]
* Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
*
* \param mu2 is the squre of the scale mu
* \param m are the squares of the masses of the internal lines
* \param p are the four-momentum squared of the external lines
*/
template<typename TOutput, typename TMass, typename TScale>
void BB(
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
    const TScale scalefac = ql::Max(ql::Max(ql::Max(std::abs(p[0]), mu2), std::abs(m[0])), std::abs(m[1]));
    const TMass m0 = (ql::Min(m[0],m[1])) / scalefac;
    const TMass m1 = (ql::Max(m[0],m[1])) / scalefac;
    const TScale p0 = p[0] / scalefac;
    const TScale musq = mu2 / scalefac;
    if (ql::iszero<TOutput, TMass, TScale>(std::abs(p0)) && ql::iszero<TOutput, TMass, TScale>(std::abs(m0)) && ql::iszero<TOutput, TMass, TScale>(std::abs(m1))) {  // All zero result 
    
        if (mode == 0) {std::cout << "Bubble Integral BB0-1" << std::endl;}
        Kokkos::parallel_for("Bubble Integral 01", policy, KOKKOS_LAMBDA(const int& i){     // BB0-1  
            res_d(i,0) = TOutput(0.0); 
            res_d(i,1) = TOutput(0.0); 
            res_d(i,2) = TOutput(0.0);                                                 
        });
        
    }
    else if (ql::iszero<TOutput, TMass, TScale>(std::abs(p0 / musq)) && ql::iszero<TOutput, TMass, TScale>(std::abs(m0 / musq)) && ql::iszero<TOutput, TMass, TScale>(std::abs(m1 / musq))) { 
        
        if (mode == 0) {std::cout << "Bubble Integral BB0-2" << std::endl;}
        Kokkos::parallel_for("Bubble Integral 02", policy, KOKKOS_LAMBDA(const int& i){      // BB0-2 
            res_d(i,0) = TOutput(0.0); 
            res_d(i,1) = TOutput(1.0); 
            res_d(i,2) = TOutput(0.0);                                                  
        });
        
    }
    else if (ql::iszero<TOutput, TMass, TScale>(std::abs(m0 / musq))) {

        if (ql::iszero<TOutput, TMass, TScale>(std::abs((m1 - p0) / musq))) {

            if (mode == 0) {std::cout << "Bubble Integral BB1" << std::endl;}
            Kokkos::parallel_for("Bubble Integral 1", policy, KOKKOS_LAMBDA(const int& i){   // BB1     
                ql::BB1<TOutput, TMass, TScale>(res_d, musq, m1, i);                         // I(s;0,s) s = m1, DD(4.13)                             
            });
            
        }
                
        else if (ql::iszero<TOutput, TMass, TScale>(std::abs(p0 / musq))) {

            if (mode == 0) {std::cout << "Bubble Integral BB2" << std::endl;}
            Kokkos::parallel_for("Bubble Integral 2", policy, KOKKOS_LAMBDA(const int& i){  // BB2     
                ql::BB2<TOutput, TMass, TScale>(res_d, musq, m1, i);                        // I(0;0,m2)                            
            });
            
        }
                
        else if (ql::iszero<TOutput, TMass, TScale>(std::abs(m1 / musq))) {

            if (mode == 0) {std::cout << "Bubble Integral BB3" << std::endl;}
            Kokkos::parallel_for("Bubble Integral 3", policy, KOKKOS_LAMBDA(const int& i){  // BB3     
                ql::BB3<TOutput, TMass, TScale>(res_d, musq, m1 - TMass(p0), i);            // I(s;0,0)                            
            });
            
        }
            
        else  {

            if (mode == 0) {std::cout << "Bubble Integral BB4" << std::endl;}
            Kokkos::parallel_for("Bubble Integral 4", policy, KOKKOS_LAMBDA(const int& i){  // BB4    
                ql::BB4<TOutput, TMass, TScale>(res_d, musq, m1, p0, i);                    // I(s;0,m2)                  
            });
            
        }                               
        
    }
    else if (ql::iszero<TOutput, TMass, TScale>(std::abs(p0 / musq))) { // deal with special case, s = 0

        if (mode == 0) {std::cout << "Bubble Integral BB5" << std::endl;}
        Kokkos::parallel_for("Bubble Integral 5", policy, KOKKOS_LAMBDA(const int& i){  // BB5     
            ql::BB5<TOutput, TMass, TScale>(res_d, musq, m0, m1, i);                            
        });
        
    }
    else { 
        
        if (mode == 0) {std::cout << "Bubble Integral BB0" << std::endl;}
        Kokkos::parallel_for("Bubble Integral 0", policy, KOKKOS_LAMBDA(const int& i){  // BB0    
            ql::BB0<TOutput, TMass, TScale>(res_d, musq, m0, m1, p0, i);                                  
        });
        
    }

    Kokkos::deep_copy(res_h, res_d);
    
    if (mode == 0) { // performance benchmark
        tt.printTime(tt.stop());
        return;
    }
    
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

        /**
        * Bubble
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
            std::cout << "Bubble Integral Performance Benchmark with batch size: " << batch_size << std::endl;
        }

        // Initialize params
        vector<double> mu2s = {
            std::pow(1.0,2.0),
            std::pow(1.7,2.0),
            std::pow(1.0,2.0),
            std::pow(1.7,2.0),
            std::pow(1.7,2.0),
            std::pow(1.7,2.0),
            std::pow(1.7,2.0)
        };

        vector<vector<double>> ms = {
            {0.0, 0.0},                        // BB0-1
            {1.0, 0.0},                        // BB1
            {0.0, 1.0},                        // BB2
            {0.0, 0.0},                        // BB3
            {5.0, 2.0},                        // BB0
            {0.0, 5.0},                        // BB4
            {5.0, 3.0}                         // BB5
        };

        vector<vector<double>> ps = {
            {0.0},                        // BB0-1
            {1.0},                        // BB1
            {0.0},                        // BB2
            {1.0},                        // BB3
            {1.0},                        // BB0
            {1.0},                        // BB4
            {0.0}                         // BB5
        };

        // Call the integral
        for (size_t i = 0; i < mu2s.size(); i++){
            BB<complex,double,double>(mu2s[i], ms[i], ps[i], batch_size, mode);
        }  
    }
  
    Kokkos::finalize();
    return 0;
}