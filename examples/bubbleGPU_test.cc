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
#include "qcdloop/bubbleGPU.h"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using complex = Kokkos::complex<double>;


/*!
* The integral is defined as:
* \f[
* I_{2}^{D=4-2 \epsilon}(p^2; m_1^2, m_2^2)= \mu^{2 \epsilon} \left[ \frac{1}{\epsilon} - \int_{0}^{1} da \ln (-a (1-a) p^2 + am_{2}^2 + (1-a) m_{1}^{2} - i \epsilon ) \right]  + O(\epsilon)
*   \f]
* Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
*
* \param res output object res[i,0,1,2] the coefficients in the Laurent series
* \param mu2 is the square of the scale mu (per element)
* \param m are the squares of the masses of the internal lines [batch][2]
* \param p are the four-momentum squared of the external lines [batch][1]
* \param i element index
*/
template<typename TOutput, typename TMass, typename TScale>
KOKKOS_INLINE_FUNCTION
void BB(
    const Kokkos::View<TOutput* [3]>& res,      // Output view
    const Kokkos::View<TScale*>& mu2,          // Scale parameter (per element)
    const Kokkos::View<TMass* [2]>& m,          // Masses view [batch][2]
    const Kokkos::View<TScale* [1]>& p,         // Momenta view [batch][1]
    const int i) {                              // Element index
    
    // Normalization
    const TScale scalefac = ql::Max(
        ql::Max(ql::Max(Kokkos::abs(p(i, 0)), mu2(i)),
                Kokkos::abs(m(i, 0))),
        Kokkos::abs(m(i, 1)));
    
    const TMass m0 = (ql::Min(m(i, 0), m(i, 1))) / scalefac;
    const TMass m1 = (ql::Max(m(i, 0), m(i, 1))) / scalefac;
    const TScale p0 = p(i, 0) / scalefac;
    const TScale musq = mu2(i) / scalefac;
    
    // Call appropriate BB function based on conditions
    if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(p0)) && 
        ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(m0)) && 
        ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(m1))) {  // All zero result
        res(i, 0) = TOutput(0.0);
        res(i, 1) = TOutput(0.0);
        res(i, 2) = TOutput(0.0);
    }
    else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(p0 / musq)) && 
             ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(m0 / musq)) && 
             ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(m1 / musq))) {
        res(i, 0) = TOutput(0.0);
        res(i, 1) = TOutput(1.0);
        res(i, 2) = TOutput(0.0);
    }
    else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(m0 / musq))) {
        if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs((m1 - p0) / musq))) {
            ql::BB1<TOutput, TMass, TScale>(res, musq, m1, i);  // I(s;0,s) s = m1, DD(4.13)
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(p0 / musq))) {
            ql::BB2<TOutput, TMass, TScale>(res, musq, m1, i);  // I(0;0,m2)
        }
        else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(m1 / musq))) {
            ql::BB3<TOutput, TMass, TScale>(res, musq, m1 - TMass(p0), i);  // I(s;0,0)
        }
        else {
            ql::BB4<TOutput, TMass, TScale>(res, musq, m1, p0, i);  // I(s;0,m2)
        }
    }
    else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(p0 / musq))) {  // deal with special case, s = 0
        ql::BB5<TOutput, TMass, TScale>(res, musq, m0, m1, i);
    }
    else {
        ql::BB0<TOutput, TMass, TScale>(res, musq, m0, m1, p0, i);
    }
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
        Kokkos::View<double* [2]> m_d("m", batch_size);
        Kokkos::View<double* [1]> p_d("p", batch_size);
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