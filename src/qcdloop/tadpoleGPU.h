//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#pragma once
#include "kokkosMaths.h"


namespace ql
{
  using complex = Kokkos::complex<double>;
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
   * \param i corresponds to the thread ID that calls this kernel
   */
  template<typename TOutput, typename TMass, typename TScale>
  KOKKOS_INLINE_FUNCTION
  void TP0(
    const Kokkos::View<TOutput* [3]>& res,
    const TScale& mu2,
    const Kokkos::View<TMass*>& m,
    const Kokkos::View<TScale*>& p,
    const int i) {
    if (!ql::iszero(Kokkos::abs(m(0)))) { // replaceing iszero() TODO::revisit for quad
      res(i,1) = TOutput(m(0));
      res(i,0) = res(i,1) * TOutput(Kokkos::log(mu2 / m(0)) + TOutput(1.0));
    }     
  }
}
