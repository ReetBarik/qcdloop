//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <qcdloop/qcdloop.h>
#include "qcdloop/tadpoleGPU.h"
#include "qcdloop/tools.h"
#include "qcdloop/maths.h"
#include "qcdloop/exceptions.h"

namespace ql
{
  template<typename TOutput, typename TMass, typename TScale>
  TadPoleGPU<TOutput, TMass, TScale>::TadPoleGPU()
      : TadPole<TOutput, TMass, TScale>() // Call base constructor
  {
  }

  template<typename TOutput, typename TMass, typename TScale>
  TadPoleGPU<TOutput, TMass, TScale>::~TadPoleGPU()
  {
  }


  /*!
   * \brief Computes the TadPole integral defined as:
   * \f[
   * I_{1}^{D=4-2 \epsilon}(m^2)= m^2 \left( \frac{\mu^2}{m^2-i \epsilon}\right) \left[ \frac{1}{\epsilon} +1 \right] + O(\epsilon)
   *   \f]
   *
   * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
   *
   * \param res output object res[0,1,2] the coefficients in the Laurent series
   * \param mu2 is the squre of the scale mu
   * \param m are the squares of the masses of the internal lines
   * \param p are the four-momentum squared of the external lines
   */
  template<typename TOutput, typename TMass, typename TScale>
  void TadPoleGPU<TOutput,TMass,TScale>::integral(vector<TOutput> &res,
                                               const TScale& mu2,
                                               vector<TMass> const& m,
                                               vector<TScale> const& p)
  {
    if (mu2 < 0) throw RangeError("TadPole::integral","mu2 is negative!");

    const double mu2_d = std::pow(1.7, 2.0);
    Kokkos::View<double*> p_d("p", 0); // Empty vector (0-length)
    Kokkos::View<double*> m_d("m", 1); // Vector with 1 element (5.0)
    m_d(0) = 5.0;
    // For complex numbers, using Kokkos::complex
    Kokkos::View<Kokkos::complex<double>*> cm_d("cm", 1); // Vector with 1 complex element (5.0, 0.0)
    cm_d(0) = Kokkos::complex<double>(5.0, 0.0);
    Kokkos::View<Kokkos::complex<double>*> res_d("res", 3);

    // Fill res with _czero
    Kokkos::parallel_for("fill_res", 3, KOKKOS_LAMBDA(const int i) {
        res_d(i) = this->_czero;
    });
    Kokkos::fence();

    if (!iszero(m(0))) {
        // Assuming TOutput is std::complex<double>
        TOutput res1(m(0)); // Convert m[0] to TOutput type
        res_d(1) = res1;
        res_d(0) = res1 * TOutput(std::log(mu2 / m_d(0)) + this->_cone);
    }

    
    // if (res.size() != 3) { res.reserve(3); }
    // std::fill(res.begin(), res.end(), this->_czero);
    // if (!this->iszero(m[0]))
    //   {
    //     res[1] = TOutput(m[0]);
    //     res[0] = res[1] * TOutput(Log(mu2 / m[0]) + this->_cone);
    //   }

    return;
  }

  // explicity template declaration
  template class TadPoleGPU<complex,double,double>;
  template class TadPoleGPU<complex,complex,double>;
}

using std::vector;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;
using ql::complex;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  const double mu2 = ql::Pow(1.7,2.0);
  vector<double> p   = {};
  vector<double>   m = {5.0};
  vector<complex> cm = {{5.0,0.0}};
  vector<complex> res(3);

  ql::Timer tt;
  ql::TadPoleGPU<complex,double> tp;
  cout << scientific << setprecision(32);

  tt.start();
  for (int i = 0; i < 1e7; i++) tp.integral(res, mu2, m, p);
//   Kokkos::parallel_for("Loop1", 1e7, KOKKOS_LAMBDA (const int i) {
//         tp.integral(res, mu2, m, p);
//     });
  tt.printTime(tt.stop());

  for (size_t i = 0; i < res.size(); i++)
  cout << "eps" << i << "\t" << res[i] << endl;
  

  Kokkos::finalize();
  return 0;
}