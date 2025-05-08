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
#include "qcdloop/triangleGPU.h"

using std::vector;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;
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
    int batch_size) {

    ql::Timer tt;
    Kokkos::View<complex* [3]> res_d("res", batch_size);
    auto res_h = Kokkos::create_mirror_view(res_d);
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0,batch_size); 

    tt.start();

    const TScale scalefac = ql::Max(Kokkos::abs(m[0]), ql::Max(Kokkos::abs(m[1]), ql::Max(Kokkos::abs(m[2]), ql::Max(Kokkos::abs(p[0]), ql::Max(Kokkos::abs(p[1]), Kokkos::abs(p[2]))))));
    const TScale musq = mu2 / scalefac;

    Kokkos::View<TMass[3]> msq("msq");
    Kokkos::View<TScale[3]> psq("psq");

    auto h_msq = Kokkos::create_mirror(msq);
    auto h_psq = Kokkos::create_mirror(psq);

    h_msq(0) = m[0] / scalefac;
    h_msq(1) = m[1] / scalefac;
    h_msq(2) = m[2] / scalefac;
    h_psq(0) = p[0] / scalefac;
    h_psq(1) = p[1] / scalefac;
    h_psq(2) = p[2] / scalefac;

    Kokkos::deep_copy(msq, h_msq);
    Kokkos::deep_copy(psq, h_psq);

    // Sort msq in ascending order
    ql::TriSort<TOutput, TMass, TScale>(psq, msq);

    // if internal masses all 0, reorder abs(psq) in ascending order
    const bool iszeros[3] = {ql::iszero<TOutput, TMass, TScale>(msq(0)), 
                             ql::iszero<TOutput, TMass, TScale>(msq(1)),
                             ql::iszero<TOutput, TMass, TScale>(msq(2))
                            };

    if (iszeros[0] && iszeros[1] && iszeros[2])
        ql::SnglSort<TOutput, TMass, TScale>(psq);

    // calculate integral value
    const TMass Y01 = TMass(msq(0) + msq(1) - psq(0)) / TMass(2);
    const TMass Y02 = TMass(msq(0) + msq(2) - psq(2)) / TMass(2);
    const TMass Y12 = TMass(msq(1) + msq(2) - psq(1)) / TMass(2);

    int massive = 0;
    for (size_t i = 0; i < 3; i++)
        if (!iszeros[i]) massive += 1;

    // building xpi
    Kokkos::View<TMass[6]> xpi("xpi");
    auto h_xpi = Kokkos::create_mirror(xpi);

    const TMass xpi_values[6] = { msq(0), msq(1), msq(2), TMass(psq(0)), TMass(psq(1)), TMass(psq(2)) };

    for (size_t i = 0; i < 6; i++)
        h_xpi(i) = xpi_values[i];

    Kokkos::deep_copy(xpi, h_xpi);
    

    if (massive == 3) {  // three internal masses

        Kokkos::parallel_for("Triangle Integral 0", policy, KOKKOS_LAMBDA(const int& i) {
            ql::T0<TOutput, TMass, TScale>(res_d, xpi, massive, i);
        });
        
    }
    else if (massive == 2) {  // two internal masses
        if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01)) && ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02))) {

            Kokkos::parallel_for("Triangle Integral 6", policy, KOKKOS_LAMBDA(const int& i) {
                ql::T6<TOutput, TMass, TScale>(res_d, musq, msq(1), msq(2), psq(1), i);
            });
            
        }
        else {

            Kokkos::parallel_for("Triangle Integral 0", policy, KOKKOS_LAMBDA(const int& i) {
                ql::T0<TOutput, TMass, TScale>(res_d, xpi, massive, i);
            });
            
        }
    }
    else if (massive == 1) { // one internal masses  
        if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01))) {

            Kokkos::parallel_for("Triangle Integral 0", policy, KOKKOS_LAMBDA(const int& i) {
                ql::T0<TOutput, TMass, TScale>(res_d, xpi, massive, i);
            });
            
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02)) && ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {

            Kokkos::parallel_for("Triangle Integral 5", policy, KOKKOS_LAMBDA(const int& i) {
                ql::T5<TOutput, TMass, TScale>(res_d, musq, msq(2), i);
            });
            
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02))) {

            Kokkos::parallel_for("Triangle Integral 4", policy, KOKKOS_LAMBDA(const int& i) {
                ql::T4<TOutput, TMass, TScale>(res_d, musq, msq(2), psq(1), i);
            });
            
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {

            Kokkos::parallel_for("Triangle Integral 4", policy, KOKKOS_LAMBDA(const int& i) {
                ql::T4<TOutput, TMass, TScale>(res_d, musq, msq(2), psq(1), i);
            });
            
        }
        else {

            Kokkos::parallel_for("Triangle Integral 3", policy, KOKKOS_LAMBDA(const int& i) {
                ql::T3<TOutput, TMass, TScale>(res_d, musq, msq(2), psq(1), psq(2), i);
            });
            
        }
        
    }
    else {  // zero internal masses       
        if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01)) && ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {

            Kokkos::parallel_for("Triangle Integral 1", policy, KOKKOS_LAMBDA(const int& i) {
                ql::T1<TOutput, TMass, TScale>(res_d, musq, psq(2), i);
            });
            
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01))) {

            Kokkos::parallel_for("Triangle Integral 2", policy, KOKKOS_LAMBDA(const int& i) {
                ql::T2<TOutput, TMass, TScale>(res_d, musq, psq(1), psq(2), i);
            });
            
        }
        else {

            Kokkos::parallel_for("Triangle Integral 0", policy, KOKKOS_LAMBDA(const int& i) {
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
        * Triangle
        */

        // Initialize params
        vector<double> mu2s = {
            std::pow(1.0,2.0),
            std::pow(1.0,2.0)
        };

        vector<vector<double>> ms = {
            {5.0, 2.0, 3.0},
            {5.0, 2.0, 3.0}
        };

        vector<vector<double>> ps = {
            {1.0, 2.0, 4.0},
            {1.0, 2.0, 4.0}
        };

        // Call the integral
        for (size_t i = 0; i < mu2s.size(); i++){
            TR<complex,double,double>(mu2s[i], ms[i], ps[i], 1);
        } 

        
    }
    Kokkos::finalize();
    return 0;
}