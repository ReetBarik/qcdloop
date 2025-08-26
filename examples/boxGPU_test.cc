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
#include "qcdloop/boxGPU.h"

using std::vector;
using std::cout;
using std::endl;
using std::string;
using complex = Kokkos::complex<double>;


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
void BO(
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
        return;
    }
    
    for (size_t i = 0; i < res_d.extent(1); i++) {
      printf("%.15f",res_h(batch_size - 1,i).real()); std::cout << ", ";
      printf("%.15f",res_h(batch_size - 1,i).imag()); std::cout << endl;
    }
    std::cout << endl;

    return;

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

    // Call the integral
    double low = 10;
    double up  = 1000;
		
    int batch_size = 1;
    double mu2 = 91.2;
    int n_tests = 20;
	
    // Trigger BIN0 - BIN4
    for (int n_masses(0); n_masses<5; n_masses++) {
      std::cout << "Target integral BIN" << n_masses << std::endl;
      for (size_t i(0); i<n_tests; ++i) {

	// should probably make this select randomly from {10., 50., 100., 200};
	std::vector<double> ms_expl {0.,0.,0.,0.};
	for(int j(0); j<n_masses; ++j) {
	  ms_expl[j] = 10;
	}
	std::vector<double> ps_expl {rs(low,up),rs(low,up),rs(low,up),rs(low,up),
				     r(low,up),r(low,up)};
	BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
      }
    }
	
    // Zero mass integrals
    std::cout << "Target integral B1"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,0.};
      std::vector<double> ps_expl {0.,0.,0.,0.,r(low,up),r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B2"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,0.};
      std::vector<double> ps_expl {0.,0.,0.,rs(low,up),r(low,up),r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B3"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,0.};
      std::vector<double> ps_expl {0.,rs(low,up),0.,rs(low,up),r(low,up),r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B4"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,0.};
      std::vector<double> ps_expl {0.,0.,rs(low,up),rs(low,up),r(low,up),r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B5"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,0.};
      std::vector<double> ps_expl {0.,rs(low,up),rs(low,up),rs(low,up),r(low,up),r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    // single mass integrals
    double m2 = 10;
    std::cout << "Target integral B6"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,m2};
      std::vector<double> ps_expl {0., 0., m2, m2, r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B7"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,m2};
      std::vector<double> ps_expl {0., 0., m2, rs(low,up), r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B8"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,m2};
      std::vector<double> ps_expl {0., 0., rs(low,up), rs(low,up), r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B9"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,m2};
      std::vector<double> ps_expl {0., rs(low,up), rs(low,up), m2, r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B10"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,0.,m2};
      std::vector<double> ps_expl {0., rs(low,up), rs(low,up), rs(low,up), r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    // two mass integrals
    double m22 = 4.9*4.9;
    double m32 = 10;
    double m42 = 50.*50.;
    std::cout << "Target integral B11"  << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,m32,m42};
      std::vector<double> ps_expl {0., m32, rs(low,up), m42, r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }
	
    std::cout << "Target integral B12" << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,m32,m42};
      std::vector<double> ps_expl {0., rs(low,up), rs(low,up), m42, r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }
	
    std::cout << "Target integral B13" << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,0.,m32,m42};
      std::vector<double> ps_expl {0., rs(low,up), rs(low,up), rs(low,up), r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B14" << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,m22,0.,m42};
      std::vector<double> ps_expl {m22, m22, m42, m42, r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    std::cout << "Target integral B15" << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,m22,0.,m42};
      std::vector<double> ps_expl {m22, rs(low,up), rs(low,up), m42, r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }

    // three mass integrals
    std::cout << "Target integral B16" << std::endl;
    for (size_t i(0); i<n_tests; ++i) {
      std::vector<double> ms_expl {0.,m22,m32,m42};
      std::vector<double> ps_expl {m22, rs(low,up), rs(low,up), m42, r(low,up), r(low,up)};
      BO<complex,double,double>(mu2, ms_expl, ps_expl, 1, 1);
    }
	
	
  }
  Kokkos::finalize();
  return 0;
}
