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
    * Computes the finite triangle with all internal masses zero.
    * Formulae from 't Hooft and Veltman \cite tHooft:1978xw following the LoopTools implementation \cite Hahn:2006qw.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TIN0(
        const Kokkos::View<TOutput* [3]>& res,
        const TMass (&xpi)[6]
        const int i) {


    }


    /*!
    * Computes the finite triangle with all internal masses zero.
    * Formulae from 't Hooft and Veltman \cite tHooft:1978xw following the LoopTools and OneLoop implementation \cite Hahn:2006qw, \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TIN1(
        const Kokkos::View<TOutput* [3]>& res,
        const TMass (&xpi)[6], 
        const TMass (&sxpi)[6],
        const int &massive
        const int i) {

    }


    /*!
    * Computes the finite triangle with all internal masses zero.
    * Formulae from 't Hooft and Veltman \cite tHooft:1978xw following the LoopTools and OneLoop implementation \cite Hahn:2006qw, \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TIN2(
        const Kokkos::View<TOutput* [3]>& res,
        const TMass (&xpi)[6], 
        const TMass (&sxpi)[6],
        const int &massive
        const int i) {

    }


    /*!
    * Computes the finite triangle with all internal masses zero.
    * Formulae from 't Hooft and Veltman \cite tHooft:1978xw following the LoopTools and OneLoop implementation \cite Hahn:2006qw, \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TIN3(
        const Kokkos::View<TOutput* [3]>& res,
        const TMass (&xpi)[6], 
        const TMass (&sxpi)[6],
        const int &massive
        const int i) {

    }    


    /*!
    * Computes the finite triangle, when Kallen2 > 0 and 3 massive particles.
    * Formulae from Denner, Nierste and Scharf \cite Denner:1991qq following the OneLoop implementation \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TINDNS(
        const Kokkos::View<TOutput* [3]>& res,
        const TMass (&xpi)[6]
        const int i) {

    }


    /*!
    * Computes the finite triangle, with 2 massive particles
    * Formulae from Denner, Nierste and Scharf \cite Denner:1991qq following the OneLoop implementation \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TINDNS2(
        const Kokkos::View<TOutput* [3]>& res,
        const TMass (&xpi)[6]
        const int i) {

    }


    /*!
    * Computes the finite triangle, with 1 massive particles.
    * Formulae from Denner, Nierste and Scharf \cite Denner:1991qq following the OneLoop implementation \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TINDNS1(
        const Kokkos::View<TOutput* [3]>& res,
        const TMass (&xpi)[6]
        const int i) {

    }


    //TODO:: Put Kallen and Kallen2 in KokkosUtils


    /*!
    * The integral is defined as:
    * \f[
    * I_{3}^{D=4-2 \epsilon}(0,0,p^2;0,0,0)= \frac{1}{p^2} \left( \frac{1}{\epsilon^2} + \frac{1}{\epsilon} \ln \left( \frac{\mu^2}{-p^2-i \epsilon} \right) + \frac{1}{2} \ln^2 \left( \frac{\mu^2}{-p^2-i \epsilon} \right) \right) + O(\epsilon)
    *   \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:2002nc.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param p is the four-momentum squared of the external line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void T1(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2,
        const TScale& p
        const int i) {

    }


    /*!
    * The integral is defined as:
    * \f[
    * I_{3}^{D=4-2 \epsilon}(0,p_1^2,p_2^2;0,0,0)= \frac{1}{p_1^2-p_2^2} \left\{ \frac{1}{\epsilon} \left[ \ln \left( \frac{\mu^2}{-p_1^2-i \epsilon} \right) - \ln \left( \frac{\mu^2}{-p_2^2-i \epsilon} \right) \right] + \frac{1}{2} \left[ \ln^2 \left( \frac{\mu^2}{-p_1^2-i \epsilon} \right) - \ln^2 \left( \frac{\mu^2}{-p_2^2-i \epsilon} \right) \right] \right\} + O(\epsilon)
    *   \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:2002nc.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param p1 is the four-momentum squared of the external line
    * \param p2 is the four-momentum squared of the external line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void T2(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2,
        const TScale& p1,
        const TScale& p2
        const int i) {

    }


    /*!
    * The integral is defined as:
    * \f[
    * I_{3}^{D=4-2 \epsilon}(0,p_1^2,p_2^2;0,0,m^2)= \frac{1}{p_1^2-p_2^2} \left( \frac{\mu^2}{m^2} \right)^\epsilon \left\{ \frac{1}{\epsilon} \ln \left( \frac{m^2-p_2^2}{m^2-p_1^2} \right) + {\rm Li}_2 \left( \frac{p_1^2}{m^2} \right) - {\rm Li}_2 \left( \frac{p_2^2}{m^2} \right) + \ln^2 \left( \frac{m^2-p_1^2}{m^2} \right) - \ln^2 \left( \frac{m^2-p_2^2}{m^2} \right) \right\} + O(\epsilon)
    *   \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:2002nc.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param m is the square of the mass of the internal line
    * \param p1 is the four-momentum squared of the external line
    * \param p2 is the four-momentum squared of the external line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void T3(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2,
        const TMass& m,
        const TScale& p2,
        const TScale& p3
        const int i) {

    }


    /*!
    * The integral is defined as:
    * \f[
    * I_{3}^{D=4-2 \epsilon}(0,m^2,p_2^2;0,0,m^2)= \left( \frac{\mu^2}{m^2} \right)^\epsilon \frac{1}{p_2^2-m^2} \left[ \frac{1}{2 \epsilon^2} + \frac{1}{\epsilon} \ln \left( \frac{m^2}{m^2-p_2^2} \right) + \frac{\pi^2}{12} + \frac{1}{2} \ln^2 \left( \frac{m^2}{m^2-p_2^2} \right) - {\rm Li}_2 \left( \frac{-p_2^2}{m^2-p_2^2} \right) \right] + O(\epsilon)
    *   \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:2002nc.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param m is the square of the mass of the internal line
    * \param p is the four-momentum squared of the external line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void T4(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2,
        const TMass& m,
        const TScale& p2
        const int i) {

    }


    /*!
    * The integral is defined as:
    * \f[
    * I_{3}^{D=4-2 \epsilon}(0,m^2,m^2;0,0,m^2)= \left( \frac{\mu^2}{m^2} \right)^\epsilon \frac{1}{m^2} \left( -\frac{1}{2 \epsilon} + 1 \right) + O(\epsilon)
    *   \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:2002nc.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param m is the square of the mass of the internal line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void T5(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2,
        const TMass& m
        const int i) {

    }


    /*!
    * The integral is defined as:
    * \f[
    * I_{3}^{D=4-2 \epsilon}(m_2^2,s,m_3^2;0,m_2^2,m_3^2)= \frac{\Gamma(1+\epsilon)\mu^\epsilon}{2\epsilon r_\Gamma} \int_0^1 d\gamma \frac{1}{\left[ \gamma m_2^2 + (1-\gamma) m_3^2 - \gamma (1-\gamma)s - i\epsilon \right]^{1+\epsilon}} + O(\epsilon)
    *   \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:2002nc.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 is the squre of the scale mu
    * \param m2 is the square of the mass of the internal line
    * \param m3 is the square of the mass of the internal line
    * \param p2 is the four-momentum squared of the external line
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void T6(
        const Kokkos::View<TOutput* [3]>& res,
        const TScale& mu2,
        const TMass& m2sq,
        const TMass& m3sq,
        const TScale& p2
        const int i) {

    }


}