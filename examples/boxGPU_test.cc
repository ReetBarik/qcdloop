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
#include <sstream>
#include <vector>
#include "qcdloop/timer.h"
#include "qcdloop/boxGPU.h"

using std::vector;
using std::cout;
using std::endl;
using std::string;
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

std::string doubleToHex(double x)
{
    // We'll copy the double bits into a 64-bit integer.
    // A union is a common trick, or we can use memcpy.
    union {
        double d;
        uint64_t u;
    } conv;

    conv.d = x;

    // Use C99's PRIx64 for a portable 64-bit hex format.
    char hex_str[19]; // "0x" + 16 hex digits + null terminator
    std::sprintf(hex_str, "0x%016" PRIx64, conv.u);
    return std::string(hex_str);
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

// Helper function to format complex number for CSV output using HEX format
std::string complexToCSV(const complex& c) {
    std::stringstream ss;
    ss << "(" << doubleToHex(c.real()) << "," << doubleToHex(c.imag()) << ")";
    return ss.str();
}

/*!
* Computes the Box integral defined as:
* \f[
* I_{4}^{D}(p_1^2,p_2^2,p_3^2,p_4^2;s_{12},s_{23};m_1^2,m_2^2,m_3^2,m_4^2)= \frac{\mu^{4-D}}{i \pi^{D/2} r_{\Gamma}} \int d^Dl \frac{1}{(l^2-m_1^2+i \epsilon)((l+q_1)^2-m_2^2+i \epsilon)((l+q_2)^2-m_3^2+i\epsilon)((l+q_4)^2-m_4^2+i\epsilon)}
*   \f]
* where \f$ q_1=p_1, q_2=p_1+p_2, q_3=p_1+p_2+p_3\f$ and \f$q_0=q_4=0\f$.
*
* Implementation of the formulae of Denner et al. \cite Denner:1991qq,
* 't Hooft and Veltman \cite tHooft:1978xw, Bern et al. \cite Bern:1993kr.
*
* \param res output object res[0,1,2] the coefficients in the Laurent series
* \param mu2 is the square of the scale mu
* \param m are the squares of the masses of the internal lines
* \param p are the four-momentum squared of the external lines
*/
template<typename TOutput, typename TMass, typename TScale>
std::vector<TOutput> BO(
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
    const TScale scalefac = ql::Max(Kokkos::abs(p[4]),ql::Max(Kokkos::abs(p[5]),ql::Max(Kokkos::abs(p[0]),ql::Max(Kokkos::abs(p[1]),ql::Max(Kokkos::abs(p[2]),Kokkos::abs(p[3]))))));

    TMass xpi_temp[13];
    xpi_temp[0] = m[0] / scalefac;
    xpi_temp[1] = m[1] / scalefac;
    xpi_temp[2] = m[2] / scalefac;
    xpi_temp[3] = m[3] / scalefac;
    xpi_temp[4] = TMass(p[0] / scalefac);
    xpi_temp[5] = TMass(p[1] / scalefac);
    xpi_temp[6] = TMass(p[2] / scalefac);
    xpi_temp[7] = TMass(p[3] / scalefac);
    xpi_temp[8] = TMass(p[4] / scalefac);
    xpi_temp[9] = TMass(p[5] / scalefac);
    xpi_temp[10] = xpi_temp[4] + xpi_temp[5] + xpi_temp[6] + xpi_temp[7] - xpi_temp[8] - xpi_temp[9];
    xpi_temp[11] =-xpi_temp[4] + xpi_temp[5] - xpi_temp[6] + xpi_temp[7] + xpi_temp[8] + xpi_temp[9];
    xpi_temp[12] = xpi_temp[4] - xpi_temp[5] + xpi_temp[6] - xpi_temp[7] + xpi_temp[8] + xpi_temp[9];

    const Kokkos::Array<TMass, 13> xpi = {xpi_temp[0], xpi_temp[1], xpi_temp[2], xpi_temp[3],
                                          xpi_temp[4], xpi_temp[5], xpi_temp[6], xpi_temp[7],
                                          xpi_temp[8], xpi_temp[9], xpi_temp[10], xpi_temp[11],
                                          xpi_temp[12]};
    const TScale musq = mu2 / scalefac;

    // Count number of internal masses
    int massive = 0;
    for (size_t i = 0; i < 4; i++)
        if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(xpi[i]))) massive += 1;

    // check cayley elements
    const TMass y13 = xpi[0] + xpi[2] - xpi[8];
    const TMass y24 = xpi[1] + xpi[3] - xpi[9];
    if (ql::iszero<TOutput, TMass, TScale>(y13) || ql::iszero<TOutput, TMass, TScale>(y24)) {
        std::cout << "Box::integral: Modified Cayley elements y13 or y24=0" << std::endl;
        
        Kokkos::parallel_for("Box Integral 00", policy, KOKKOS_LAMBDA(const int& i) {     
            res_d(i,0) = TOutput(0.0); 
            res_d(i,1) = TOutput(0.0); 
            res_d(i,2) = TOutput(0.0);                                                 
        }); 
        
    } else {
        if (massive == 0) {
            Kokkos::parallel_for("Box Integral 0m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B0m<TOutput, TMass, TScale>(res_d, xpi, musq, i);                                                      
            });
        } else if (massive == 1) {
            Kokkos::parallel_for("Box Integral 1m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B1m<TOutput, TMass, TScale>(res_d, xpi, musq, i);                                                      
            });
        } else if (massive == 2) {
            Kokkos::parallel_for("Box Integral 2m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B2m<TOutput, TMass, TScale>(res_d, xpi, musq, i);                                                      
            });
        } else if (massive == 3) {
            Kokkos::parallel_for("Box Integral 3m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B3m<TOutput, TMass, TScale>(res_d, xpi, musq, i);                                                      
            });
        } else if (massive == 4) {
            Kokkos::parallel_for("Box Integral 4m", policy, KOKKOS_LAMBDA(const int& i){        
                ql::B4m<TOutput, TMass, TScale>(res_d, xpi, i);                                                      
            });
        }      

        Kokkos::parallel_for("Normalize Res", policy, KOKKOS_LAMBDA(const int& i) {
            res_d(i,0) /= (scalefac * scalefac);
            res_d(i,1) /= (scalefac * scalefac);
            res_d(i,2) /= (scalefac * scalefac);
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
        int n_tests = 1000000; // default value
        int batch_size = 1000000; // default value
        const unsigned int seed = 12345; // hardcoded seed for reproducible results
        
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

        // Initialize random seed for reproducible results
        std::srand(seed);
        
        // std::cout << "Running with n_tests = " << n_tests << std::endl;
        // std::cout << "Running with batch_size = " << batch_size << std::endl;
        // std::cout << "Running with random seed = " << seed << std::endl;

        #if MODE == 1
        // Print CSV header for accuracy test mode
        std::cout << "Target Integral,Test ID,mu2,ms,ps,Coeff 1,Coeff 2,Coeff 3" << std::endl;
        #endif

        // Call the integral
        double low = 10;
        double up  = 1000;
		
        
        double mu2 = 91.2;
	
        // Trigger BIN0 - BIN4
        for (int n_masses(0); n_masses<5; n_masses++) {
            for (size_t i(0); i<n_tests; ++i) {
                #if MODE == 0
                std::cout << "BIN" << n_masses << " " ;
                #endif
                // should probably make this select randomly from {10., 50., 100., 200};
                std::vector<double> ms_expl {0.,0.,0.,0.};
                for(int j(0); j<n_masses; ++j) {
                    ms_expl[j] = 10;
                }
                std::vector<double> ps_expl {rs(low,up),rs(low,up),rs(low,up),rs(low,up),
                          r(low,up),r(low,up)};
                auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
                
                #if MODE == 1
                // Output CSV row
                std::cout << "BIN" << n_masses << "," 
                          << (i+1) << "," 
                          << mu2 << "," 
                          << vectorToCSV(ms_expl) << "," 
                          << vectorToCSV(ps_expl) << "," 
                          << complexToCSV(results[0]) << "," 
                          << complexToCSV(results[1]) << "," 
                          << complexToCSV(results[2]) << std::endl;
                #endif

            }
        }
	
        // Zero mass integrals
        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B1 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,0.};
            std::vector<double> ps_expl {0.,0.,0.,0.,r(low,up),r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B1," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B2 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,0.};
            std::vector<double> ps_expl {0.,0.,0.,rs(low,up),r(low,up),r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B2," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B3 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,0.};
            std::vector<double> ps_expl {0.,rs(low,up),0.,rs(low,up),r(low,up),r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B3," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B4 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,0.};
            std::vector<double> ps_expl {0.,0.,rs(low,up),rs(low,up),r(low,up),r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B4," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B5 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,0.};
            std::vector<double> ps_expl {0.,rs(low,up),rs(low,up),rs(low,up),r(low,up),r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B5," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        // single mass integrals
        double m2 = 10;
        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B6 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,m2};
            std::vector<double> ps_expl {0., 0., m2, m2, r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B6," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B7 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,m2};
            std::vector<double> ps_expl {0., 0., m2, rs(low,up), r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B7," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B8 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,m2};
            std::vector<double> ps_expl {0., 0., rs(low,up), rs(low,up), r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B8," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B9 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,m2};
            std::vector<double> ps_expl {0., rs(low,up), rs(low,up), m2, r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B9," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B10 ";
            #endif
            std::vector<double> ms_expl {0.,0.,0.,m2};
            std::vector<double> ps_expl {0., rs(low,up), rs(low,up), rs(low,up), r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B10," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        // two mass integrals
        double m22 = 4.9*4.9;
        double m32 = 10;
        double m42 = 50.*50.;
        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B11 ";
            #endif
            std::vector<double> ms_expl {0.,0.,m32,m42};
            std::vector<double> ps_expl {0., m32, rs(low,up), m42, r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B11," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }
	
		for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B12 ";
            #endif
            std::vector<double> ms_expl {0.,0.,m32,m42};
            std::vector<double> ps_expl {0., rs(low,up), rs(low,up), m42, r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B12," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }
	
		for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B13 ";
            #endif
            std::vector<double> ms_expl {0.,0.,m32,m42};
            std::vector<double> ps_expl {0., rs(low,up), rs(low,up), rs(low,up), r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B13," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B14 ";
            #endif
            std::vector<double> ms_expl {0.,m22,0.,m42};
            std::vector<double> ps_expl {m22, m22, m42, m42, r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B14," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B15 ";
            #endif
            std::vector<double> ms_expl {0.,m22,0.,m42};
            std::vector<double> ps_expl {m22, rs(low,up), rs(low,up), m42, r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B15," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }

        // three mass integrals
        for (size_t i(0); i<n_tests; ++i) {
            #if MODE == 0
                std::cout << "B16 ";
            #endif
            std::vector<double> ms_expl {0.,m22,m32,m42};
            std::vector<double> ps_expl {m22, rs(low,up), rs(low,up), m42, r(low,up), r(low,up)};
            auto results = BO<complex,double,double>(mu2, ms_expl, ps_expl, batch_size, MODE);
            
            #if MODE == 1
            std::cout << "B16," 
                      << (i+1) << "," 
                      << mu2 << "," 
                      << vectorToCSV(ms_expl) << "," 
                      << vectorToCSV(ps_expl) << "," 
                      << complexToCSV(results[0]) << "," 
                      << complexToCSV(results[1]) << "," 
                      << complexToCSV(results[2]) << std::endl;
            #endif
        }
        
        
    }
    Kokkos::finalize();
    return 0;
}