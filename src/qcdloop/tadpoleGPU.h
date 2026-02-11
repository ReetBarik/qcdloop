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
   * \param m is the square of the mass of the internal line
   * \param p are the four-momentum squared of the external lines
   * \param i corresponds to the thread ID that calls this kernel
   */
  template<typename TOutput, typename TMass, typename TScale>
  KOKKOS_INLINE_FUNCTION
  void TP0(
    const Kokkos::View<TOutput* [3]>& res,
    const TScale& mu2,
    const TMass& m,
    const Kokkos::View<TScale*>& p,
    const int i) {
    res(i,1) = TOutput(m);
    res(i,0) = res(i,1) * TOutput(Kokkos::log(mu2 / m) + TOutput(1.0));
    res(i,2) = TOutput(0.0);
  }

  /*!
  * \brief Computes the TadPole integral defined as:
  * \f[
  * I_{1}^{D=4-2 \epsilon}(m^2)= m^2 \left( \frac{\mu^2}{m^2-i \epsilon}\right) \left[ \frac{1}{\epsilon} +1 \right] + O(\epsilon)
  *   \f]
  *
  * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
  *
  * \param res output object res[i,0,1,2] the coefficients in the Laurent series
  * \param mu2 is the square of the scale mu (per element)
  * \param m are the squares of the masses of the internal lines [batch][1]
  * \param p are the four-momentum squared of the external lines [batch][0] (empty, kept for API consistency)
  * \param i element index
  */
  template<typename TOutput, typename TMass, typename TScale>
  KOKKOS_INLINE_FUNCTION
  void TP(
    const Kokkos::View<TOutput* [3]>& res,      // Output view
    const Kokkos::View<TScale*>& mu2,          // Scale parameter (per element)
    const Kokkos::View<TMass* [1]>& m,         // Masses view [batch][1]
    const Kokkos::View<TScale*>& p,            // Momenta view (empty, size 0, kept for API consistency)
    const int i) {                              // Element index
    
    // Handle zero-mass case
    if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(m(i, 0)))) {
        res(i, 0) = TOutput(0.0);
        res(i, 1) = TOutput(0.0);
        res(i, 2) = TOutput(0.0);
    } else {
        ql::TP0<TOutput, TMass, TScale>(res, mu2(i), m(i, 0), p, i);
    }
  }
}
