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

void printDoubleBits(double x)
{
    // We'll copy the double bits into a 64-bit integer.
    // A union is a common trick, or we can use memcpy.
    union {
        double d;
        uint64_t u;
    } conv;

    conv.d = x;

    // Use C99's PRIx64 for a portable 64-bit hex format.
    // %.16g prints up to 16 significant digits in decimal (just for reference).
    // std::printf("decimal=%.16g\n", x);
    std::printf("0x%016" PRIx64, conv.u);
}

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
    int batch_size) {

    ql::Timer tt;
    Kokkos::View<complex* [3]> res_d("res", batch_size);
    auto res_h = Kokkos::create_mirror_view(res_d);
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,batch_size); 

    tt.start();
    // Normalization
    auto t1 = std::abs(p[0]) > std::abs(mu2) ? std::abs(p[0]) : mu2;
    auto t2 = std::abs(t1) > std::abs(m[0]) ? t1 : std::abs(m[0]);
    auto t3 = std::abs(t2) > std::abs(m[1]) ? t2 : std::abs(m[1]);
    const TScale scalefac = t3;
    const TMass m0 = (std::abs(m[0]) > std::abs(m[1]) ? m[1] : m[0]) / scalefac;
    const TMass m1 = (std::abs(m[0]) > std::abs(m[1]) ? m[0] : m[1]) / scalefac;
    const TScale p0 = p[0] / scalefac;
    const TScale musq = mu2 / scalefac;
    if (std::abs(p0) < 1e-10 && std::abs(m0) < 1e-10 && std::abs(m1) < 1e-10) {  // All zero result
        Kokkos::parallel_for("Bubble Integral 01", policy, KOKKOS_LAMBDA(const int& i){       
            res_d(i,0) = TOutput(0.0); 
            res_d(i,1) = TOutput(0.0); 
            res_d(i,2) = TOutput(0.0);                                                 
        });
        Kokkos::deep_copy(res_h, res_d);
    }
    else if (std::abs(p0 / musq) < 1e-10 && std::abs(m0 / musq) < 1e-10 && std::abs(m1 / musq) < 1e-10) {
        Kokkos::parallel_for("Bubble Integral 02", policy, KOKKOS_LAMBDA(const int& i){       
            res_d(i,0) = TOutput(0.0); 
            res_d(i,1) = TOutput(1.0); 
            res_d(i,2) = TOutput(0.0);                                                  
        });
        Kokkos::deep_copy(res_h, res_d);
    }
    else if (std::abs(m0 / musq) < 1e-10) {

        if (std::abs((m1 - p0) / musq) < 1e-10) {
            Kokkos::parallel_for("Bubble Integral 1", policy, KOKKOS_LAMBDA(const int& i){       
                ql::BB1(res_d, musq, m1, i);                         // I(s;0,s) s = m1, DD(4.13)                             
            });
            Kokkos::deep_copy(res_h, res_d);
        }
                
        else if (std::abs(p0 / musq) < 1e-10) {
            Kokkos::parallel_for("Bubble Integral 2", policy, KOKKOS_LAMBDA(const int& i){       
                ql::BB2(res_d, musq, m1, i);                        // I(0;0,m2)                            
            });
            Kokkos::deep_copy(res_h, res_d);
        }
                
        else if (std::abs(m1 / musq) < 1e-10) {
            Kokkos::parallel_for("Bubble Integral 3", policy, KOKKOS_LAMBDA(const int& i){       
                ql::BB3(res_d, musq, m1 - TMass(p0), i);            // I(s;0,0)                            
            });
            Kokkos::deep_copy(res_h, res_d);
        }
            
        else  {
            Kokkos::parallel_for("Bubble Integral 4", policy, KOKKOS_LAMBDA(const int& i){       
                ql::BB4(res_d, musq, m1, p0, i);                    // I(s;0,m2)                  
            });
            Kokkos::deep_copy(res_h, res_d);
        }                               
        
    }
    else if (std::abs(p0 / musq) < 1e-10) { // deal with special case, s = 0
        Kokkos::parallel_for("Bubble Integral 5", policy, KOKKOS_LAMBDA(const int& i){       
            ql::BB5(res_d, musq, m0, m1, i);                            
        });
        Kokkos::deep_copy(res_h, res_d);
    }
    else { 
        Kokkos::parallel_for("Bubble Integral 0", policy, KOKKOS_LAMBDA(const int& i){       
            ql::BB0(res_d, musq, m0, m1, p0, i);                                  
        });
        Kokkos::deep_copy(res_h, res_d);
    }

    
    // tt.printTime(tt.stop());
    
    for (size_t i = 0; i < res_d.extent(1); i++) {
        printf("%.15f",res_h(batch_size - 1,i).real()); cout << ", ";
        printf("%.15f",res_h(batch_size - 1,i).imag()); cout << endl;
    }

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
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {0.0, 0.0},
            {5.0, 2.0},
            {0.0, 5.0},
            {5.0, 3.0}
        };

        vector<vector<double>> ps = {
            {0.0},
            {1.0},
            {0.0},
            {1.0},
            {1.0},
            {1.0},
            {0.0}
        };

        // Call the integral
        for (size_t i = 0; i < mu2s.size(); i++){
            BB<complex,double,double>(mu2s[i], ms[i], ps[i], 1);
        }        
        
    }
  
    Kokkos::finalize();
    return 0;
}