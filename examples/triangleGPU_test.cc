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
#include "qcdloop/triangleGPU.h"

using std::vector;
using std::cout;
using std::endl;
using std::string;
using complex = Kokkos::complex<double>;


/*!
* Computes the Triangle integral defined as:
* \f[
* I_{3}^{D}(p_1^2,p_2^2,p_3^2;m_1^2,m_2^2,m_3^2)= \frac{\mu^{4-D}}{i \pi^{D/2} r_{\Gamma}} \int d^Dl \frac{1}{(l^2-m_1^2+i \epsilon)((l+q_1)^2-m_2^2+i \epsilon)((l+q_2)^2-m_3^2+i\epsilon)}
*   \f]
*where \f$q_1=p_1,q_2=p_1+p_2\f$.
*
* Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn and
* 't Hooft and Veltman \cite tHooft:1978xw.
*
* \param mu2 is the square of the scale mu
* \param m are the squares of the masses of the internal lines
* \param p are the four-momentum squared of the external lines
*/
template<typename TOutput, typename TMass, typename TScale>
void TR(
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

    const TScale scalefac = ql::Max(Kokkos::abs(m[0]), ql::Max(Kokkos::abs(m[1]), ql::Max(Kokkos::abs(m[2]), ql::Max(Kokkos::abs(p[0]), ql::Max(Kokkos::abs(p[1]), Kokkos::abs(p[2]))))));
    const TScale musq = mu2 / scalefac;

    TMass msq[3];
    TScale psq[3];

    msq[0] = m[0] / scalefac;
    msq[1] = m[1] / scalefac;
    msq[2] = m[2] / scalefac;
    psq[0] = p[0] / scalefac;
    psq[1] = p[1] / scalefac;
    psq[2] = p[2] / scalefac;

    // Sort msq in ascending order
    ql::TriSort<TOutput, TMass, TScale>(psq, msq);


    // if internal masses all 0, reorder abs(psq) in ascending order
    const bool iszeros[3] = {ql::iszero<TOutput, TMass, TScale>(msq[0]), 
                             ql::iszero<TOutput, TMass, TScale>(msq[1]),
                             ql::iszero<TOutput, TMass, TScale>(msq[2])
                            };

    if (iszeros[0] && iszeros[1] && iszeros[2]) { 
        ql::SnglSort<TOutput, TMass, TScale>(psq);
    }   

    // calculate integral value
    const TMass Y01 = TMass(msq[0] + msq[1] - psq[0]) / TMass(2);
    const TMass Y02 = TMass(msq[0] + msq[2] - psq[2]) / TMass(2);
    const TMass Y12 = TMass(msq[1] + msq[2] - psq[1]) / TMass(2);

    int massive = 0;
    for (size_t i = 0; i < 3; i++)
        if (!iszeros[i]) massive += 1;

    // building xpi
    const Kokkos::Array<TMass, 6> xpi = { msq[0], msq[1], msq[2], TMass(psq[0]), TMass(psq[1]), TMass(psq[2]) };
    

    if (massive == 3) {  // three internal masses

        Kokkos::parallel_for("Triangle Integral 0-1", policy, KOKKOS_LAMBDA(const int& i) { // T0-1
            ql::T0<TOutput, TMass, TScale>(res_d, xpi, massive, i);
        });
        
    }
    else if (massive == 2) {  // two internal masses
        if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01)) && ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02))) {

            Kokkos::parallel_for("Triangle Integral 6", policy, KOKKOS_LAMBDA(const int& i) { // T6
                ql::T6<TOutput, TMass, TScale>(res_d, musq, msq[1], msq[2], psq[1], i);
            });
            
        }
        else {

            Kokkos::parallel_for("Triangle Integral 0-2", policy, KOKKOS_LAMBDA(const int& i) { // T0-2
                ql::T0<TOutput, TMass, TScale>(res_d, xpi, massive, i);
            });
            
        }
    } else if (massive == 1) { // one internal masses  
        if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01))) {

            Kokkos::parallel_for("Triangle Integral 0-3", policy, KOKKOS_LAMBDA(const int& i) { // T0-3
                ql::T0<TOutput, TMass, TScale>(res_d, xpi, massive, i);
            });
            
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02)) && ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {

            Kokkos::parallel_for("Triangle Integral 5", policy, KOKKOS_LAMBDA(const int& i) { // T5
                ql::T5<TOutput, TMass, TScale>(res_d, musq, msq[2], i);
            });
            
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02))) {

            Kokkos::parallel_for("Triangle Integral 4", policy, KOKKOS_LAMBDA(const int& i) { // T4-1
                ql::T4<TOutput, TMass, TScale>(res_d, musq, msq[2], psq[1], i);
            });
            
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {

            Kokkos::parallel_for("Triangle Integral 4", policy, KOKKOS_LAMBDA(const int& i) { // T4-2
                ql::T4<TOutput, TMass, TScale>(res_d, musq, msq[2], psq[2], i);
            });
            
        }
        else {

            Kokkos::parallel_for("Triangle Integral 3", policy, KOKKOS_LAMBDA(const int& i) { // T3
                ql::T3<TOutput, TMass, TScale>(res_d, musq, msq[2], psq[1], psq[2], i);
            });
            
        }
        
    } else {  // zero internal masses       
        if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01)) && ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {

            Kokkos::parallel_for("Triangle Integral 1", policy, KOKKOS_LAMBDA(const int& i) { // T1
                ql::T1<TOutput, TMass, TScale>(res_d, musq, psq[2], i);
            });
            
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01))) {

            Kokkos::parallel_for("Triangle Integral 2", policy, KOKKOS_LAMBDA(const int& i) { // T2
                ql::T2<TOutput, TMass, TScale>(res_d, musq, psq[1], psq[2], i);
            });
            
        }
        else {
            
            Kokkos::parallel_for("Triangle Integral 0", policy, KOKKOS_LAMBDA(const int& i) { // T0-4
                ql::T0<TOutput, TMass, TScale>(res_d, xpi, massive, i);
            });
            
        }
        
    }
    Kokkos::parallel_for("Normalize Res", policy, KOKKOS_LAMBDA(const int& i) {
        res_d(i,0) /= scalefac;
        res_d(i,1) /= scalefac;
        res_d(i,2) /= scalefac;
    });


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
        if (!(mode == 0 || mode == 1)) {
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
            1.0
            // 1.0
        };
        
        std::vector<std::vector<double>> ms = {
            {5.0, 2.0, 3.0},                        // T0-1
            {5.0, 2.0, 0.0},                        // T0-2
            {5.0, 0.0, 0.0},                        // T0-3
            {0.0, 0.0, 0.0},                        // T0-4
            {0.0, 0.0, 0.0},                        // T1
            {0.0, 0.0, 0.0},                        // T2
            {0.75, 0.0, 0.0},                       // T3
            {0.0, 1.0, 0.0},                        // T4-1
            {0.0, 0.0, 8.5},                        // T4-2
            {1.1, 0.0, 0.0},                        // T5
            {1.0, 1.0, 0.0}                         // T6
            // {0.0, 1.0, 0.0}                      // T0-3 denspence: argument on cut
        };
        
        std::vector<std::vector<double>> ps = {
            {1.0, 2.0, 4.0},                         // T0-1
            {7.0, 4.0, 10.0},                        // T0-2
            {1.0, 4.0, 4.0},                         // T0-3
            {4.0, 6.0, 3.0},                         // T0-4
            {0.0, 0.5, 0.0},                         // T1
            {0.5, 0.5, 0.0},                         // T2
            {0.0, 0.0, 1.5},                         // T3
            {0.0, 1.0, 0.0},                         // T4-1
            {0.0, 8.5, 0.0},                         // T4-2
            {1.1, 0.0, 1.1},                         // T5
            {1.0, 1.0, 1.0}                          // T6
            // {1.0, 1.0, 1.0}                       // T0-3 denspence: argument on cut
        };

        vector<string> integrals = {
            "Triangle Integral T0-1",   // T0-1
            "Triangle Integral T0-2",   // T0-2
            "Triangle Integral T0-3",   // T0-3
            "Triangle Integral T0-4",   // T0-4
            "Triangle Integral T1",     // T1
            "Triangle Integral T2",     // T2
            "Triangle Integral T3",     // T3
            "Triangle Integral T4-1",   // T4-1
            "Triangle Integral T4-2",   // T4-2
            "Triangle Integral T5",     // T5
            "Triangle Integral T6"      // T6
            // "Triangle Integral T0-3" // T0-3 denspence: argument on cut
        };

        // Call the integral
        for (size_t i = 0; i < mu2s.size(); i++){
            if (mode == 0) {std::cout << integrals[i] << std::endl;}
            TR<complex,double,double>(mu2s[i], ms[i], ps[i], batch_size, mode);
        } 

    }
    Kokkos::finalize();
    return 0;
}