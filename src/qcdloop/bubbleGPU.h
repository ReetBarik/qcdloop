//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#pragma once

#include "qcdloop/kokkosUtils.h"

namespace ql
{
    using complex = Kokkos::complex<double>;
    
    /*!
    * The integral is defined as in the general case but with the first term:
    * \f[
    * I_{2}(s; m_0^2, m_1^2) = \Delta + 2 - \ln \left( \frac{m_0 m_1}{m_0^2} \right) + \frac{m_0^2-m_1^2}{s} \ln \left(\frac{m_1}{m_0} \right) - \frac{m_0 m_1}{s} \left(\frac{1}{r}-r \right) \ln r
    *   \f]
    * with
    * \f[
    * x^2 + \frac{m_0^2 + m_1^2 - s - i \epsilon}{m_0 m_1} +1 = (x+r)\left(x+\frac{1}{r} \right)
    * \f]
    * Implementation of the formulae of Denner and Dittmaier \cite Denner:1991kt.
    *
    * \return the output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param m0 is the square of the first mass of the internal line
    * \param m1 is the square of the second mass of the internal line
    * \param s is the square of the four-momentum external line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BB0(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2, 
        const TMass& m0, 
        const TMass& m1, 
        const TScale& s, 
        const int i) {

        // general case from Denner 0709.1075, Eq. (4.23)
        const TMass sqm0 = Kokkos::sqrt(m0);
        const TMass sqm1 = Kokkos::sqrt(m1);
        const TOutput bb = TOutput(m0 + m1 - s);
        const TOutput rtt= Kokkos::sqrt(bb * bb - TOutput(4.0) * TOutput(m1 * m0));
        const TOutput x1 = TOutput(0.5) * (bb + rtt) / (sqm0 * sqm1);
        const TOutput x2 = TOutput(1.0) / x1;
        double real = (x1 - x2).real(); // TOutput will always be a complex
        auto sign = (double(0) == real) ? 0 : ((double(0) < real) ? 1 : -1); // 'real' will always be a real double
        res(i,0) = TOutput(2.0) - Kokkos::log(sqm0 * sqm1 / mu2) + (m0 - m1) / s * Kokkos::log(sqm1 / sqm0) - sqm0 * sqm1 / s * (x2 - x1) * ql::cLn<TOutput, TMass, TScale>(x1, sign);
        res(i,1) = TOutput(1.0);
        res(i,2) = TOutput(0.0);

    }

    /*!
    * The integral is defined as:
    * \f[
    * I_{2}^{D=4-2 \epsilon}(m^2; 0, m^2)=\left( \frac{\mu^{2}}{m^{2}} \right)^{\epsilon} \left[ \frac{1}{\epsilon} + 2 \right]  + O(\epsilon)
    *   \f]
    * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
    *
    * \return the output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param m is the squares of the mass of the internal line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BB1(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2, 
        const TMass& m, 
        const int i) {

        res(i,0) = TOutput(Kokkos::log(mu2 / m)) + TOutput(2.0);
        res(i,1) = TOutput(1.0);
        res(i,2) = TOutput(0.0);

    }

    /*!
    * The integral is defined as:
    * \f[
    * I_{2}^{D=4-2 \epsilon}(0; 0, m^2)= \left( \frac{\mu^{2}}{m^{2}} \right)^{\epsilon} \left[ \frac{1}{\epsilon} + 1 \right]  + O(\epsilon)
    *   \f]
    * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
    *
    * \return the output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param m is the squares of the mass of the internal line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BB2(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2, 
        const TMass& m, 
        const int i) {

        res(i,0) = TOutput(Kokkos::log(mu2 / m)) + TOutput(1.0);
        res(i,1) = TOutput(1.0);
        res(i,2) = TOutput(0.0);

    }

    /*!
    * The integral is defined as:
    * \f[
    * I_{2}^{D=4-2 \epsilon}(s; 0, 0)= \left( \frac{\mu^{2}}{-s-i \epsilon} \right)^{\epsilon} \left[ \frac{1}{\epsilon} + 2 \right]  + O(\epsilon)
    *   \f]
    * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
    *
    * \return the output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param s is the four-momentum squared of the external line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BB3(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2, 
        const TMass& s, 
        const int i) {

        res(i,0) = - ql::cLn<TOutput, TMass, TScale>(s / mu2, -1) + TOutput(2.0);
        res(i,1) = TOutput(1.0);
        res(i,2) = TOutput(0.0);

    }

    /*!
    * The integral is defined as:
    * \f[
    * I_{2}^{D=4-2 \epsilon}(s; 0, m^2)= \left( \frac{\mu^{2}}{m^2} \right)^{\epsilon} \left[ \frac{1}{\epsilon} + 2 + \frac{m^2-s}{s} \ln \left( \frac{m^2-s-i \epsilon}{m^2} \right) \right]  + O(\epsilon)
    *   \f]
    * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
    *
    * \return the output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param m is the squares of the mass of the internal line
    * \param s is the four-momentum squared of the external line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BB4(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2, 
        const TMass& m, 
        const TScale& s, 
        const int i) {

        res(i,0) = - ql::cLn<TOutput, TMass, TScale>((m - s) / mu2, -1) + TOutput(1.0) - ql::fndd<TOutput, TMass, TScale>(0, TOutput(1.0 - m / s), 1); // TODO::quad
        res(i,1) = TOutput(1.0);
        res(i,2) = TOutput(0.0);

    }

    /*!
    * The integral is defined as in the general case but with the first term:
    * \f[
    * I_{2}^{\epsilon=0}(0; m_0^2, m_1^2) = \ln \left( \frac{\mu}{m_0^2} \right) - f_0 \left( \frac{m0}{m0-m1} \right)
    *   \f]
    * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
    *
    * \return the output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param m0 is the square of the first mass of the internal line
    * \param m1 is the square of the second mass of the internal line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BB5(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2, 
        const TMass& m0, 
        const TMass& m1, 
        const int i) {

        res(i,0) = TOutput(Kokkos::log(mu2 / m0));
        if (Kokkos::abs((m1-m0)/mu2) >= 1e-10) { // replaceing !iszero() TODO::revisit for quad
            res(i,0) = ql::fndd<TOutput, TMass, TScale>(0, TOutput(m0 / (m0 - m1)), 1);
        }
        res(i,1) = TOutput(1.0);
        res(i,2) = TOutput(0.0);

    }
}