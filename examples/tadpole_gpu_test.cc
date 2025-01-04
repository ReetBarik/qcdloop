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

using std::vector;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;
using ql::complex;


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
    if (!this->checkCache(mu2,m,p))
      {
        if (mu2 < 0) throw RangeError("TadPole::integral","mu2 is negative!");

        std::fill(this->_val.begin(), this->_val.end(), this->_czero);
        if (!this->iszero(m[0]))
          {
            this->_val[1] = TOutput(m[0]);
            this->_val[0] = this->_val[1]*TOutput(Log(mu2/m[0])+this->_cone);
          }          
        this->storeCache(mu2,m,p);
      }

    if (res.size() != 3) { res.reserve(3); }
    std::copy(this->_val.begin(), this->_val.end(), res.begin());

    return;
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
  void TadPoleGPU<TOutput,TMass,TScale>::integral_gpu(vector<TOutput> &res,
                                               const TScale& mu2,
                                               vector<TMass> const& m,
                                               vector<TScale> const& p)
  {
    
    //TODO::possibly have to use this to take advantage of the topology parent class
    return;
  }

  template<typename TOutput, typename TMass, typename TScale>
  struct integral_gpu {

    using complex = Kokkos::complex<double>;
    const TScale mu2;
    Kokkos::View<TScale*> p;
    Kokkos::View<TMass*> m;
    Kokkos::View<TOutput*> res;

    integral_gpu( 
      const TScale mu2_,
      const Kokkos::View<TScale*>& p_,
      const Kokkos::View<TMass*>& m_,
      const Kokkos::View<TOutput*>& res_
      ): mu2(mu2_), p(p_), m(m_), res(res_) {};


    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
      Kokkos::printf("Hello from i = %i\n", i);
  }
};

  // explicity template declaration
  template class TadPoleGPU<complex,double,double>;
  template class TadPoleGPU<complex,complex,double>;
}


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);


  /*
  This is CPU only used for benchmarking 
  */
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
  tt.printTime(tt.stop());

  for (size_t i = 0; i < res.size(); i++)
  cout << "eps" << i << "\t" << res[i] << endl;

  /*
  This is experimental for usage on GPUs 
  */

  // Initialize views
  using complex = Kokkos::complex<double>;
  Kokkos::View<double*> p_d("p", 0);
  Kokkos::View<double*> m_d("m", 1); 
  Kokkos::View<complex*> cm_d("cm", 1); 
  Kokkos::View<complex*> res_d("res", 3);
  auto res_h = Kokkos::create_mirror_view(res_d);

  // Populate views


  // Call the integral
  // Kokkos::parallel_for("HelloWorld", 15, ql::integral_gpu());

  // Copy result to host
  Kokkos::deep_copy(res_h, res_d);

  // Print result and time
  
  Kokkos::finalize();
  return 0;
}