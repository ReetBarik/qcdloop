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

  template<typename TOutput, typename TMass, typename TScale>
  struct integral_gpu {

    using complex = Kokkos::complex<double>;
    const TScale mu2;
    Kokkos::View<TScale*> p;
    Kokkos::View<TMass*> m;
    Kokkos::View<TOutput* [3]> res;

    integral_gpu( 
      const TScale mu2_,
      const Kokkos::View<TScale*>& p_,
      const Kokkos::View<TMass*>& m_,
      Kokkos::View<TOutput* [3]>& res_
      ): mu2(mu2_), p(p_), m(m_), res(res_) {};


    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
      if (m(0) != TMass(0.0)) {
        res(i,1) = TOutput(m(0));
        res(i,0) = res(i,1) * TOutput(Kokkos::log(mu2 / m(0)) + TOutput(1.0));
      }     
    }
  };

  // explicity template declaration
  template class TadPoleGPU<complex,double,double>;
  template class TadPoleGPU<complex,complex,double>;
}