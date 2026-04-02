//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include "qcdloop/timer.h"
#include "qcdloop/bubbleGPU.h"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using complex = Kokkos::complex<double>;

std::string doubleToHex(double x)
{
    union {
        double d;
        uint64_t u;
    } conv;
    conv.d = x;
    char hex_str[19];
    std::sprintf(hex_str, "0x%016" PRIx64, conv.u);
    return std::string(hex_str);
}

template<typename T>
std::string arrayToCSV(const T* arr, size_t size) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0) ss << ",";
        ss << arr[i];
    }
    ss << "]";
    return ss.str();
}

std::string complexToCSV(const complex& c) {
    std::stringstream ss;
    ss << "(" << doubleToHex(c.real()) << "," << doubleToHex(c.imag()) << ")";
    return ss.str();
}

double r(double min, double max) {
    return min+std::rand()*1.0/RAND_MAX * (max - min);
}

double rs(double min, double max) {
    double r1 = min+std::rand()*1.0/RAND_MAX * (max - min);
    double rs = std::rand()*1.0/RAND_MAX;
    if (rs < 0.5)
        return -r1;
    else
        return r1;
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

        double low = 100;
        double up  = 1000000;

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

        // Trigger BIN0 - BIN2: sweep over number of internal masses
        // BIN0: both masses zero, random s  → BB3: I(s; 0, 0)
        // BIN1: one mass nonzero, random s  → BB4: I(s; 0, m^2)
        // BIN2: two masses nonzero, random s → BB0: general
        double m2 = 10;
        double m12 = 4.9*4.9;
        double m22 = 50.*50.;
        for (int n_masses(0); n_masses<3; n_masses++) {
            std::srand(12345);
            for (size_t i(0); i<batch_size; ++i) {
                m_h(i, 0) = 0.;
                m_h(i, 1) = 0.;
                if (n_masses >= 1) m_h(i, 1) = m2;
                if (n_masses >= 2) m_h(i, 0) = m12;
                p_h(i, 0) = rs(low, up);
            }
            Kokkos::deep_copy(mu2_d, mu2_h);
            Kokkos::deep_copy(m_d, m_h);
            Kokkos::deep_copy(p_d, p_h);
            tt.start();
            Kokkos::parallel_for("Bubble BIN", policy, KOKKOS_LAMBDA(const int& i) {
                ql::BB<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
            });
            Kokkos::fence();
            double elapsed = tt.stop();
            Kokkos::deep_copy(res_h, res_d);
            if (mode == 0) {
                std::cout << "BIN" << n_masses << "," << batch_size << "," << elapsed << std::endl;
            } else if (mode == 1) {
                for (size_t i = 0; i < batch_size; ++i) {
                    double m_arr[2] = {m_h(i, 0), m_h(i, 1)};
                    double p_arr[1] = {p_h(i, 0)};
                    std::cout << "BIN" << n_masses << ","
                              << (i+1) << ","
                              << mu2_h(i) << ","
                              << arrayToCSV(m_arr, 2) << ","
                              << arrayToCSV(p_arr, 1) << ","
                              << complexToCSV(res_h(i, 0)) << ","
                              << complexToCSV(res_h(i, 1)) << ","
                              << complexToCSV(res_h(i, 2)) << std::endl;
                }
            }
        }

        // BB1 - I(m^2; 0, m^2): one mass, s = m^2
        // Randomize m per element; set p = m to satisfy the BB1 kinematic condition.
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            double mi = r(low, up);
            m_h(i, 0) = 0.;
            m_h(i, 1) = mi;
            p_h(i, 0) = mi;  // p = m required for BB1: I(m^2; 0, m^2)
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Bubble BB1", policy, KOKKOS_LAMBDA(const int& i) {
            ql::BB<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        double elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "BB1," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[2] = {m_h(i, 0), m_h(i, 1)};
                double p_arr[1] = {p_h(i, 0)};
                std::cout << "BB1," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 2) << ","
                          << arrayToCSV(p_arr, 1) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // BB2 - I(0; 0, m^2): one mass, s = 0
        // Randomize m per element; p must remain 0 for this kinematic case.
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = 0.;
            m_h(i, 1) = r(low, up);  // random positive mass
            p_h(i, 0) = 0.;
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Bubble BB2", policy, KOKKOS_LAMBDA(const int& i) {
            ql::BB<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "BB2," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[2] = {m_h(i, 0), m_h(i, 1)};
                double p_arr[1] = {p_h(i, 0)};
                std::cout << "BB2," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 2) << ","
                          << arrayToCSV(p_arr, 1) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }

        // BB5 - I(0; m1^2, m2^2): two masses, s = 0
        // Randomize both masses per element; p must remain 0 for this kinematic case.
        std::srand(12345);
        for (size_t i(0); i<batch_size; ++i) {
            m_h(i, 0) = r(low, up);  // random positive mass m1
            m_h(i, 1) = r(low, up);  // random positive mass m2
            p_h(i, 0) = 0.;
        }
        Kokkos::deep_copy(mu2_d, mu2_h);
        Kokkos::deep_copy(m_d, m_h);
        Kokkos::deep_copy(p_d, p_h);
        tt.start();
        Kokkos::parallel_for("Bubble BB5", policy, KOKKOS_LAMBDA(const int& i) {
            ql::BB<complex, double, double>(res_d, mu2_d, m_d, p_d, i);
        });
        Kokkos::fence();
        elapsed = tt.stop();
        Kokkos::deep_copy(res_h, res_d);
        if (mode == 0) {
            std::cout << "BB5," << batch_size << "," << elapsed << std::endl;
        } else if (mode == 1) {
            for (size_t i = 0; i < batch_size; ++i) {
                double m_arr[2] = {m_h(i, 0), m_h(i, 1)};
                double p_arr[1] = {p_h(i, 0)};
                std::cout << "BB5," << (i+1) << "," << mu2_h(i) << ","
                          << arrayToCSV(m_arr, 2) << ","
                          << arrayToCSV(p_arr, 1) << ","
                          << complexToCSV(res_h(i, 0)) << ","
                          << complexToCSV(res_h(i, 1)) << ","
                          << complexToCSV(res_h(i, 2)) << std::endl;
            }
        }
    }

    Kokkos::finalize();
    return 0;
}
