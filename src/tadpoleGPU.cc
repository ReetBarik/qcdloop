//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include "qcdloop/tadpoleGPU.h"
#include "qcdloop/exceptions.h"

using std::vector;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;


namespace ql
{
  template<typename TOutput, typename TMass, typename TScale>
  TadPoleGPU<TOutput, TMass, TScale>::TadPoleGPU()   
  {
  }

  template<typename TOutput, typename TMass, typename TScale>
  TadPoleGPU<TOutput, TMass, TScale>::~TadPoleGPU()
  {
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