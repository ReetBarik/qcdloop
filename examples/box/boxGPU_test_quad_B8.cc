//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

// Define USE_QUAD_COMPLEX before including headers to use quad_complex instead of Kokkos::complex<double>
#define USE_QUAD_COMPLEX

extern "C" {
#include <quadmath.h>
}
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
// Quad precision headers must be included BEFORE boxGPU.h to ensure types are fully defined
// when boxGPU.h processes templates that reference them
#include "qcdloop/quad_math.hpp"      // Must come first (defines fp128_t)
#include "qcdloop/quad_complex.hpp"    // Depends on quad_math.hpp (defines quad_complex)
#include "qcdloop/timer.h"
#include "box/B1m.h"

using std::vector;
using std::cout;
using std::endl;
using std::string;
using quad_complex = ql::quad_complex;
using fp128_t = ql::quad::fp128_t;

// Helper function to format fp128_t for CSV output using quad decimal format
// Mark as host-only to prevent nvcc from trying to compile as device code
__host__ std::string fp128ToQuadString(fp128_t x) {
    char quad_str[128];
    quadmath_snprintf(quad_str, sizeof(quad_str), "%.30Qe", x.value);
    return std::string(quad_str);
}

// Helper function to format vector for CSV output
template<typename T>
std::string vectorToCSV(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) ss << ",";
        ss << vec[i];
    }
    ss << "]";
    return ss.str();
}

// Helper function to extract array from View for CSV output (for fp128_t)
std::string arrayToCSV(const fp128_t* arr, size_t size) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0) ss << ",";
        ss << fp128ToQuadString(arr[i]);
    }
    ss << "]";
    return ss.str();
}

// Helper function to format quad_complex number for CSV output using quad decimal format
std::string quadComplexToCSV(const quad_complex& c) {
    std::stringstream ss;
    ss << "(" << fp128ToQuadString(c.real()) << "," << fp128ToQuadString(c.imag()) << ")";
    return ss.str();
}

__float128 r(__float128 min, __float128 max) {
    return min+std::rand()*1.0/RAND_MAX * (max - min);
}
  
__float128 rs(__float128 min, __float128 max) {
    __float128 r1 = min+std::rand()*1.0/RAND_MAX * (max - min);
    __float128 rs = std::rand()*1.0/RAND_MAX;
    if (rs < 0.5)
        return -r1;
    else
        return r1;
}


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const unsigned int seed = 12345; // hardcoded seed for reproducible results
        std::srand(seed);
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

        // Call the integral
        __float128 low = 100.0q;
        __float128 up  = 1000000.0q;
		
        // Create Kokkos Views for batch processing (using fp128_t for quad precision)
        Kokkos::View<fp128_t*> mu2_d("mu2", batch_size);
        Kokkos::View<fp128_t* [4]> m_d("m", batch_size);
        Kokkos::View<fp128_t* [6]> p_d("p", batch_size);
        Kokkos::View<quad_complex* [3]> res_d("res", batch_size);
        
        auto mu2_h = Kokkos::create_mirror_view(mu2_d);
        auto m_h = Kokkos::create_mirror_view(m_d);
        auto p_h = Kokkos::create_mirror_view(p_d);
        auto res_h = Kokkos::create_mirror_view(res_d);
        
        // Initialize mu2 (as double, then cast to fp128_t)
        for (size_t i = 0; i < batch_size; ++i) {
                mu2_h(i) = static_cast<fp128_t>(91.2q*91.2q);
        }
        
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, batch_size);

        // single mass integrals
        __float128 m2 = 10.0q;

        // B8
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = static_cast<fp128_t>(0.); m_h(i, 1) = static_cast<fp128_t>(0.); m_h(i, 2) = static_cast<fp128_t>(0.); m_h(i, 3) = static_cast<fp128_t>(m2);
            p_h(i, 0) = static_cast<fp128_t>(0.); p_h(i, 1) = static_cast<fp128_t>(0.);
            p_h(i, 2) = static_cast<fp128_t>(rs(low,up)); p_h(i, 3) = static_cast<fp128_t>(rs(low,up));
            p_h(i, 4) = static_cast<fp128_t>(r(low,up)); p_h(i, 5) = static_cast<fp128_t>(r(low,up));
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Box Integral B8", policy, KOKKOS_LAMBDA(const int& i) {
            ql::BO<quad_complex, fp128_t, fp128_t>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        double elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "B8," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                fp128_t m_arr[4] = {m_h(i, 0), m_h(i, 1), m_h(i, 2), m_h(i, 3)};
                fp128_t p_arr[6] = {p_h(i, 0), p_h(i, 1), p_h(i, 2), p_h(i, 3), p_h(i, 4), p_h(i, 5)};
                std::cout << "B8," 
                          << (i+1) << "," 
                          << fp128ToQuadString(mu2_h(i)) << "," 
                          << arrayToCSV(m_arr, 4) << "," 
                          << arrayToCSV(p_arr, 6) << "," 
                          << quadComplexToCSV(res_h(i, 0)) << "," 
                          << quadComplexToCSV(res_h(i, 1)) << "," 
                          << quadComplexToCSV(res_h(i, 2)) << std::endl;
            }
        }

    }
    Kokkos::finalize();
    return 0;
}
