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
* \param res output object res[0,1,2] the coefficients in the Laurent series
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

}




int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        ql::init_isort();

        
    }
    Kokkos::finalize();
    return 0;
}