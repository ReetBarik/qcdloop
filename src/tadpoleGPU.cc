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
  TadPoleGPU<TOutput, TMass, TScale>::TadPoleGPU(
    Kokkos::View<TOutput* [3]>& res_
  ): res(res_)   
  {
  }

  template<typename TOutput, typename TMass, typename TScale>
  TadPoleGPU<TOutput, TMass, TScale>::~TadPoleGPU()
  {
  }

  template<typename TOutput, typename TMass, typename TScale>
  KOKKOS_INLINE_FUNCTION
  void TadPoleGPU<TOutput, TMass, TScale>::integral(
    const TScale& mu2,
    const Kokkos::View<TMass*>& m,
    const Kokkos::View<TScale*>& p,
    const int i) const {
    if (m(0) != TMass(0.0)) {
      res(i,1) = TOutput(m(0));
      res(i,0) = res(i,1) * TOutput(Kokkos::log(mu2 / m(0)) + TOutput(1.0));
    }     
  } 

  // explicity template declaration
  template class TadPoleGPU<complex,double,double>;
  template class TadPoleGPU<complex,complex,double>;
}