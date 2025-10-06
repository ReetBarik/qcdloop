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
* \param mu2 is the squre of the scale mu
* \param m are the squares of the masses of the internal lines
* \param p are the four-momentum squared of the external lines
*/
template<typename TOutput, typename TMass, typename TScale>
std::vector<TOutput> BB(
    const TScale& mu2,
    vector<TMass> const& m,
    vector<TScale> const& p,
    int batch_size,
    int mode) {

    ql::Timer tt;
    Kokkos::View<TOutput* [3]> res_d("res", batch_size);
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

        Kokkos::parallel_for("Bubble Integral 01", policy, KOKKOS_LAMBDA(const int& i){     // BB0-1  
            res_d(i,0) = TOutput(0.0); 
            res_d(i,1) = TOutput(0.0); 
            res_d(i,2) = TOutput(0.0);                                                 
        });
        
    }
    else if (ql::iszero<TOutput, TMass, TScale>(std::abs(p0 / musq)) && ql::iszero<TOutput, TMass, TScale>(std::abs(m0 / musq)) && ql::iszero<TOutput, TMass, TScale>(std::abs(m1 / musq))) { 

        Kokkos::parallel_for("Bubble Integral 02", policy, KOKKOS_LAMBDA(const int& i){      // BB0-2 
            res_d(i,0) = TOutput(0.0); 
            res_d(i,1) = TOutput(1.0); 
            res_d(i,2) = TOutput(0.0);                                                  
        });
        
    }
    else if (ql::iszero<TOutput, TMass, TScale>(std::abs(m0 / musq))) {

        if (ql::iszero<TOutput, TMass, TScale>(std::abs((m1 - p0) / musq))) {

            Kokkos::parallel_for("Bubble Integral 1", policy, KOKKOS_LAMBDA(const int& i){   // BB1     
                ql::BB1<TOutput, TMass, TScale>(res_d, musq, m1, i);                         // I(s;0,s) s = m1, DD(4.13)                             
            });
            
        }
                
        else if (ql::iszero<TOutput, TMass, TScale>(std::abs(p0 / musq))) {

            Kokkos::parallel_for("Bubble Integral 2", policy, KOKKOS_LAMBDA(const int& i){  // BB2     
                ql::BB2<TOutput, TMass, TScale>(res_d, musq, m1, i);                        // I(0;0,m2)                            
            });
            
        }
                
        else if (ql::iszero<TOutput, TMass, TScale>(std::abs(m1 / musq))) {

            Kokkos::parallel_for("Bubble Integral 3", policy, KOKKOS_LAMBDA(const int& i){  // BB3     
                ql::BB3<TOutput, TMass, TScale>(res_d, musq, m1 - TMass(p0), i);            // I(s;0,0)                            
            });
            
        }
            
        else  {

            Kokkos::parallel_for("Bubble Integral 4", policy, KOKKOS_LAMBDA(const int& i){  // BB4    
                ql::BB4<TOutput, TMass, TScale>(res_d, musq, m1, p0, i);                    // I(s;0,m2)                  
            });
            
        }                               
        
    }
    else if (ql::iszero<TOutput, TMass, TScale>(std::abs(p0 / musq))) { // deal with special case, s = 0

        Kokkos::parallel_for("Bubble Integral 5", policy, KOKKOS_LAMBDA(const int& i){  // BB5     
            ql::BB5<TOutput, TMass, TScale>(res_d, musq, m0, m1, i);                            
        });
        
    }
    else { 
        
        Kokkos::parallel_for("Bubble Integral 0", policy, KOKKOS_LAMBDA(const int& i){  // BB0    
            ql::BB0<TOutput, TMass, TScale>(res_d, musq, m0, m1, p0, i);                                  
        });
        
    }

    Kokkos::deep_copy(res_h, res_d);
    
    if (mode == 0) { // performance benchmark
        tt.printTime(tt.stop());
        return std::vector<TOutput>();
    }
    
    // Return the results as a vector
    std::vector<TOutput> results;
    for (size_t i = 0; i < res_d.extent(1); i++) {
        results.push_back(res_h(batch_size - 1, i));
    }

    return results;
}


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        
        // Parse command line arguments
        int n_tests = 1000000; // default value
        int batch_size = 1000000; // default value
        
        #if MODE == 0
            // Performance benchmark mode: n_tests=1, batch_size from command line
            n_tests = 1;
            if (argc > 1) {
                try {
                    batch_size = std::stoi(argv[1]);
                    if (batch_size <= 0) {
                        std::cout << "Error: batch_size must be a positive integer. Using default value of 1000000." << std::endl;
                        batch_size = 1000000;
                    }
                } catch (const std::exception& e) {
                    std::cout << "Error: Invalid argument for batch_size. Using default value of 1000000." << std::endl;
                    batch_size = 1000000;
                }
            }
            
            if (argc > 2) {
                std::cout << "Usage: " << argv[0] << " [batch_size]" << std::endl;
                std::cout << "  batch_size: Number of batch iterations for performance benchmark (default: 1000000)" << std::endl;
                std::cout << "  MODE=0: Performance benchmark mode" << std::endl;
            }
        #elif MODE == 1
            // Accuracy test mode: batch_size=1, n_tests from command line
            batch_size = 1;
            if (argc > 1) {
                try {
                    n_tests = std::stoi(argv[1]);
                    if (n_tests <= 0) {
                        std::cout << "Error: n_tests must be a positive integer. Using default value of 1000000." << std::endl;
                        n_tests = 1000000;
                    }
                } catch (const std::exception& e) {
                    std::cout << "Error: Invalid argument for n_tests. Using default value of 1000000." << std::endl;
                    n_tests = 1000000;
                }
            }
            
            if (argc > 2) {
                std::cout << "Usage: " << argv[0] << " [n_tests]" << std::endl;
                std::cout << "  n_tests: Number of test iterations for accuracy testing (default: 1000000)" << std::endl;
                std::cout << "  MODE=1: Accuracy test mode" << std::endl;
            }
        #else
            // Fallback mode: both from command line
            if (argc > 1) {
                try {
                    n_tests = std::stoi(argv[1]);
                    if (n_tests <= 0) {
                        std::cout << "Error: n_tests must be a positive integer. Using default value of 1000000." << std::endl;
                        n_tests = 1000000;
                    }
                } catch (const std::exception& e) {
                    std::cout << "Error: Invalid argument for n_tests. Using default value of 1000000." << std::endl;
                    n_tests = 1000000;
                }
            }
            
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
                std::cout << "Usage: " << argv[0] << " [n_tests] [batch_size]" << std::endl;
                std::cout << "  n_tests: Number of test iterations (default: 1000000)" << std::endl;
                std::cout << "  batch_size: Number of batch iterations (default: 1000000)" << std::endl;
                std::cout << "  MODE not set: Both parameters configurable" << std::endl;
            }
        #endif

        std::cout << "Running with n_tests = " << n_tests << std::endl;
        std::cout << "Running with batch_size = " << batch_size << std::endl;

        #if MODE == 1
        // Print CSV header for accuracy test mode
        std::cout << "Target Integral,Test ID,mu2,ms,ps,Coeff 1,Coeff 2,Coeff 3" << std::endl;
        #endif
    }
  
    Kokkos::finalize();
    return 0;
}