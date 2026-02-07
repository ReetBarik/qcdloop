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
* \param res output object res[i,0,1,2] the coefficients in the Laurent series
* \param mu2 is the square of the scale mu (per element)
* \param m are the squares of the masses of the internal lines [batch][3]
* \param p are the four-momentum squared of the external lines [batch][3]
* \param i element index
*/
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void TR(
    const Kokkos::View<TOutput* [3]>& res,      // Output view
    const Kokkos::View<TScale*>& mu2,          // Scale parameter (per element)
    const Kokkos::View<TMass* [3]>& m,          // Masses view [batch][3]
    const Kokkos::View<TScale* [3]>& p,         // Momenta view [batch][3]
    const int i) {                              // Element index
    
    // Compute scalefac for this element
    const TScale scalefac = ql::Max(
        Kokkos::abs(m(i, 0)),
        ql::Max(Kokkos::abs(m(i, 1)),
        ql::Max(Kokkos::abs(m(i, 2)),
        ql::Max(Kokkos::abs(p(i, 0)),
        ql::Max(Kokkos::abs(p(i, 1)),
                 Kokkos::abs(p(i, 2)))))));
    
    // Compute musq for this element
    const TScale musq = mu2(i) / scalefac;
    
    // Normalize masses and momenta
    Kokkos::Array<TMass, 3> msq;
    Kokkos::Array<TScale, 3> psq;
    
    msq[0] = m(i, 0) / scalefac;
    msq[1] = m(i, 1) / scalefac;
    msq[2] = m(i, 2) / scalefac;
    psq[0] = p(i, 0) / scalefac;
    psq[1] = p(i, 1) / scalefac;
    psq[2] = p(i, 2) / scalefac;
    
    // Sort msq in ascending order
    ql::TriSort<TOutput, TMass, TScale>(psq, msq);
    
    // if internal masses all 0, reorder abs(psq) in ascending order
    const bool iszeros[3] = {
        ql::iszero<TOutput, TMass, TScale>(msq[0]), 
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
    for (size_t j = 0; j < 3; j++) {
        if (!iszeros[j]) massive += 1;
    }
    
    // building xpi
    const Kokkos::Array<TMass, 6> xpi = {
        msq[0], msq[1], msq[2],
        TMass(psq[0]), TMass(psq[1]), TMass(psq[2])
    };
    
    // Call appropriate T function based on conditions
    if (massive == 3) {  // three internal masses
        ql::T0<TOutput, TMass, TScale>(res, xpi, massive, i);
    }
    else if (massive == 2) {  // two internal masses
        if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01)) && 
            ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02))) {
            ql::T6<TOutput, TMass, TScale>(res, musq, msq[1], msq[2], psq[1], i);
        }
        else {
            ql::T0<TOutput, TMass, TScale>(res, xpi, massive, i);
        }
    }
    else if (massive == 1) { // one internal mass
        if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01))) {
            ql::T0<TOutput, TMass, TScale>(res, xpi, massive, i);
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02)) && 
                 ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {
            ql::T5<TOutput, TMass, TScale>(res, musq, msq[2], i);
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02))) {
            ql::T4<TOutput, TMass, TScale>(res, musq, msq[2], psq[1], i);
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {
            ql::T4<TOutput, TMass, TScale>(res, musq, msq[2], psq[2], i);
        }
        else {
            ql::T3<TOutput, TMass, TScale>(res, musq, msq[2], psq[1], psq[2], i);
        }
    }
    else {  // zero internal masses
        if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01)) && 
            ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {
            ql::T1<TOutput, TMass, TScale>(res, musq, psq[2], i);
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01))) {
            ql::T2<TOutput, TMass, TScale>(res, musq, psq[1], psq[2], i);
        }
        else {
            ql::T0<TOutput, TMass, TScale>(res, xpi, massive, i);
        }
    }
    
    // Normalize results
    res(i, 0) /= scalefac;
    res(i, 1) /= scalefac;
    res(i, 2) /= scalefac;
}




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
        Kokkos::View<double* [3]> m_d("m", batch_size);
        Kokkos::View<double* [3]> p_d("p", batch_size);
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