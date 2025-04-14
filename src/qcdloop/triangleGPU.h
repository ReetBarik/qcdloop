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
    * Computes the Kallen function defined as:
    * \f[
    *  K(p_1,p_2,p_3) = p_1^2+p_2^2+p_3^2-2 (p_1 \cdot p_2 + p_2 \cdot p_3 +p_3 \cdot p_1)
    * \f]
    * \param p1 four-momentum squared
    * \param p2 four-momentum squared
    * \param p3 four-momentum squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Kallen2(TOutput const& p1, TOutput const& p2, TOutput const& p3) {
        return TOutput(p1 * p1 + p2 * p2 + p3 * p3 - TOutput(2.0) * (p1 * p2 + p2 * p3 + p3 * p1));
    }


    /*!
    * Computes the Kallen function defined as:
    * \f[
    *  K(p_1,p_2,p_3) = \sqrt{p_1^2+p_2^2+p_3^2-2 (p_1 \cdot p_2 + p_2 \cdot p_3 +p_3 \cdot p_1)}
    * \f]
    * \param p1 four-momentum squared
    * \param p2 four-momentum squared
    * \param p3 four-momentum squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Kallen(TOutput const& p1, TOutput const& p2, TOutput const& p3) {
        return Kokkos::sqrt(ql::Kallen2<TOutput, TMass, TScale>(p1,p2,p3));
    }


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
        const TMass (&xpi)[6],
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
        const int &massive,
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
        const int &massive,
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
        const int &massive,
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
    void TINDNS(const TOutput& res, const Kokkos::View<TMass[6]>& xpi) {

        TOutput m1 = xpi(0);
        TOutput m2 = xpi(1);
        TOutput m3 = xpi(2);
        TOutput p1 = TOutput(xpi(3));
        TOutput p2 = TOutput(xpi(4));
        TOutput p3 = TOutput(xpi(5));
    
        const TOutput sm1 = Kokkos::sqrt(m1);
        const TOutput sm2 = Kokkos::sqrt(m2);
        const TOutput sm3 = Kokkos::sqrt(m3);
    
        TOutput k12 = TOutput(0.0), k13 = TOutput(0.0), k23 = TOutput(0.0);
        if (m1 + m2 != p1) k12 = (m1 + m2 - p1 - p1 * ql::Constants::_ieps2<TOutput, TMass, TScale>()) / (sm1 * sm2);
        if (m1 + m3 != p3) k13 = (m1 + m3 - p3 - p3 * ql::Constants::_ieps2<TOutput, TMass, TScale>()) / (sm1 * sm3);
        if (m2 + m3 != p2) k23 = (m2 + m3 - p2 - p2 * ql::Constants::_ieps2<TOutput, TMass, TScale>()) / (sm2 * sm3);
    
        TOutput r12, r13, r23, d12, d13, d23;
        ql::R<TOutput, TMass, TScale>(r12, d12, k12);
        ql::R<TOutput, TMass, TScale>(r13, d13, k13);
        ql::R<TOutput, TMass, TScale>(r23, d23, k23);
    
        const TOutput a = sm2 / sm3 - k23 + r13 * (k12 - sm2 / sm1);
        if (a == TOutput(0.0)) {
            Kokkos::printf("Triangle::TINDNS: threshold singularity, return 0\n");
            res = TOutput(0.0);
            return;
        }
        const TOutput b = d13 / sm2 + k12 / sm3 - k23 / sm1;
        const TOutput c = (sm1 / sm3 - 1.0 / r13) / (sm1 * sm2);
    
        TOutput x[2];
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, x);
        x[0] = -x[0];
        x[1] = -x[1];
    
        const TOutput qz[2]  = { x[0] * sm2, x[1] * sm2 };
        const TOutput qz2[2] = { x[0] * r13 * sm2, x[1] * r13 * sm2};
        const TOutput qz3[2] = { x[0] * r13, x[1] * r13};
        const TOutput oneOr12 = TOutput(1.0) / r12;
        const TOutput oneOr23 = TOutput(1.0) / r23;
        const TOutput dqz = qz[0] - qz[1];
        const TOutput dqz2 = qz2[0] - qz2[1];
        const TOutput dqz3 = qz3[0] - qz3[1];
        const TScale siqz[2] = { (TScale) ql::Sign(ql::Imag(qz[0])), (TScale) ql::Sign(ql::Imag(qz[1]))};
        const TScale siqz2[2]= { (TScale) ql::Sign(ql::Imag(qz2[0])), (TScale) ql::Sign(ql::Imag(qz2[1]))};
        const TScale siqz3[2]= { (TScale) ql::Sign(ql::Imag(qz3[0])), (TScale) ql::Sign(ql::Imag(qz3[1]))};
        const TScale sigx[2]= { (TScale) ql::Sign(ql::Imag(x[0])), (TScale) ql::Sign(ql::Imag(x[1]))};
    
        res = (-ql::xspence<TOutput, TMass, TScale>(qz, siqz, r12, ql::Sign(ql::Imag(r12))) / dqz
               -ql::xspence<TOutput, TMass, TScale>(qz, siqz, oneOr12, ql::Sign(ql::Imag(oneOr12))) / dqz) * sm2
              +(ql::xspence<TOutput, TMass, TScale>(qz2, siqz2, r23, ql::Sign(ql::Imag(r23))) / dqz2
              + ql::xspence<TOutput, TMass, TScale>(qz2, siqz2, oneOr23, ql::Sign(ql::Imag(oneOr23))) / dqz2) * r13 * sm2
              - ql::xspence<TOutput, TMass, TScale>(qz3, siqz3, sm3, ql::Sign(ql::Imag(sm3))) / dqz3 * r13
              + ql::xspence<TOutput, TMass, TScale>(x, sigx, sm1, ql::Sign(ql::Imag(sm1))) / (x[0] - x[1]);
    
        if (x[1] != TOutput(0.0)) {
            const TOutput arg1 = qz3[0] / qz3[1];
            const TOutput arg2 = qz3[0] * qz3[1] / (sm3 * sm3);
            const TOutput arg3 = x[0] / x[1];
            const TOutput arg4 = x[0] * x[1] / (sm1 * sm1);
    
            TOutput log1 = ql::cLn<TOutput, TMass, TScale>(arg2, ql::Sign(ql::Imag(arg2)));
            TOutput log2 = ql::cLn<TOutput, TMass, TScale>(arg4, ql::Sign(ql::Imag(arg4)));
            if (ql::Real(arg2) < 0.0 && ql::Imag(arg2) < 0.0) log1 += ql::Constants::_2ipi<TOutput, TMass, TScale>();
            if (ql::Real(arg4) < 0.0 && ql::Imag(arg4) < 0.0) log2 += ql::Constants::_2ipi<TOutput, TMass, TScale>();
    
            res += (ql::cLn<TOutput, TMass, TScale>(arg1, ql::Sign(ql::Imag(arg1))) / (TOutput(1.0) - arg1) * log1
                  - ql::cLn<TOutput, TMass, TScale>(arg3, ql::Sign(ql::Imag(arg3))) / (TOutput(1.0) - arg3) * log2) / (TOutput(2.0) * x[1]);
        }
        res /= (a * sm1 * sm2 * sm3);

    }


    /*!
    * Computes the finite triangle, with 2 massive particles
    * Formulae from Denner, Nierste and Scharf \cite Denner:1991qq following the OneLoop implementation \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TINDNS2(const TOutput& res, const Kokkos::View<TMass[6]>& xpi) {

        TOutput m2 = xpi(1);
        TOutput m4 = xpi(2);
        TOutput p2 = TOutput(xpi(3));
        TOutput p3 = TOutput(xpi(5));
        TOutput p23 = TOutput(xpi(4));
    
        const TOutput sm2 = Kokkos::sqrt(m2);
        const TOutput sm3 = Kokkos::abs(sm2);
        const TOutput sm4 = Kokkos::sqrt(m4);
    
        TOutput r23 = TOutput(0.0); 
        TOutput k24 = TOutput(0.0);
        TOutput r34 = TOutput(0.0);

        r23 = (m2 - p2 - p2 * ql::Constants::_ieps2<TOutput, TMass, TScale>()) / (sm2 * sm3);
        k24 = (m2 + m4 - p23 - p23 * ql::Constants::_ieps2<TOutput, TMass, TScale>()) / (sm2 * sm4);
        r34 = (m4 - p3 - p3 * ql::Constants::_ieps2<TOutput, TMass, TScale>()) / (sm3 * sm4);
    
        TOutput r24, d24;
        ql::R<TOutput, TMass, TScale>(r24, d24, k24);
    
        const TOutput a = r34 / r24 - r23;
        if (a == TOutput(0.0)) {
            Kokkos::printf("Triangle::TINDNS2: threshold singularity, return 0\n");
            res = TOutput(0.0);
            return;
        }
    
        const TOutput b = -d24 / sm3 + r34 / sm2 - r23 / sm4;
        const TOutput c = (sm4 / sm2 - r24) / (sm3 * sm4);
    
        TOutput x[2];
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, x);
        x[0] = -x[0];
        x[1] = -x[1];
    
        const TOutput qz[2]  = { x[0] / r24, x[1] / r24};
        const TScale siqz[2] = { (TScale) ql::Sign(ql::Imag(qz[0])), (TScale) ql::Sign(ql::Imag(qz[1]))};
    
        res = -ql::xspence<TOutput, TMass, TScale>(qz, siqz, sm2, ql::Sign(ql::Imag(sm2))) / (qz[0] - qz[1]) / r24;
    
        if (x[1] != TOutput(0.0)) {
            const TOutput arg1 = qz[0] / qz[1];
            const TOutput arg2 = qz[0] * qz[1] / (sm2 * sm2);
            const TOutput arg3 = x[0] / x[1];
            const TOutput arg4 = x[0] * x[1] / (sm4 * sm4);
    
            TOutput log1 = ql::cLn<TOutput, TMass, TScale>(arg2, ql::Sign(ql::Imag(arg2)));
            TOutput log2 = ql::cLn<TOutput, TMass, TScale>(arg4, ql::Sign(ql::Imag(arg4)));
            if (ql::Real(arg2) < 0.0 && ql::Imag(arg2) < 0.0) log1 += ql::Constants::_2ipi<TOutput, TMass, TScale>();
            if (ql::Real(arg4) < 0.0 && ql::Imag(arg4) < 0.0) log2 += ql::Constants::_2ipi<TOutput, TMass, TScale>();
    
            res += (ql::cLn<TOutput, TMass, TScale>(arg1, ql::Sign(ql::Imag(arg1))) / (TOutput(1.0) - arg1) * log1
                  - ql::cLn<TOutput, TMass, TScale>(arg3, ql::Sign(ql::Imag(arg3))) / (TOutput(1.0) - arg3) * log2) / (TOutput(2.0) * x[1]);
        }
    
        const TScale siqx[2] = { (TScale) ql::Sign(ql::Imag(x[0])), (TScale) ql::Sign(ql::Imag(x[1])) };
        res += ql::xspence<TOutput, TMass, TScale>(x, siqx, sm4, ql::Sign(ql::Imag(sm4))) / (x[0] - x[1]);
    
        if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(r23))) {
            const TOutput arg = r23 * sm3 / r24;
            res += ql::xspence<TOutput, TMass, TScale>(x, siqx, arg, ql::Sign(ql::Imag(arg))) / (x[0] - x[1]);
        }
    
        if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(r34))) {
            const TOutput arg = r34 * sm3;
            res -= ql::xspence<TOutput, TMass, TScale>(x, siqx, arg, ql::Sign(ql::Imag(arg))) / (x[0] - x[1]);
        }
    
        res /= (a * sm2 * sm3 * sm4);
    }


    /*!
    * Computes the finite triangle, with 1 massive particles.
    * Formulae from Denner, Nierste and Scharf \cite Denner:1991qq following the OneLoop implementation \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TINDNS1(const TOutput& res, const Kokkos::View<TMass[6]>& xpi) {

        TOutput m4 = xpi(2);
        TOutput p2 = TOutput(xpi(3));
        TOutput p3 = TOutput(xpi(4));
        TOutput p4 = TOutput(xpi(5));
        TOutput p23 = p4;
    
        const TOutput sm4 = Kokkos::sqrt(m4);
        const TOutput sm3 = Kokkos::abs(sm4);
        const TOutput sm2 = sm3;
    
        TOutput r23 = TOutput(0.0);
        TOutput r24 = TOutput(0.0);
        TOutput r34 = TOutput(0.0);

        r23 = (-p2 - p2 * ql::Constants::_ieps2<TOutput, TMass, TScale>()) / (sm2 * sm3);
        r24 = (m4 - p23 - p23 * ql::Constants::_ieps2<TOutput, TMass, TScale>())/(sm2 * sm4);
        r34 = (m4 - p3 - p3 * ql::Constants::_ieps2<TOutput, TMass, TScale>())/(sm3 * sm4);
    
        const TOutput a = r34 * r24 - r23;
        if (a == TOutput(0.0)) {
            Kokkos::printf("Triangle::TINDNS1: threshold singularity, return 0\n");
            res = TOutput(0.0);
            return;
        }
    
        const TOutput b = r24 / sm3 + r34 / sm2 - r23 / sm4;
        const TOutput c = TOutput(1.0) / (sm2 * sm3);
    
        TOutput x[2];
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, x);
        x[0] = -x[0];
        x[1] = -x[1];
    
        const TScale siqx[2] = { (TScale) ql::Sign(ql::Imag(x[0])), (TScale) ql::Sign(ql::Imag(x[1])) };
    
        const TOutput arg1 = x[0] / x[1];
        const TOutput arg2 = x[0] * x[1] / (sm4 * sm4);
        const TOutput arg3 = r23 / (sm3 * sm3);
    
        const TOutput log0 = ql::cLn<TOutput, TMass, TScale>(arg1, ql::Sign(ql::Imag(arg1))) / (TOutput(1.0) - arg1);
    
        TOutput log1 = ql::cLn<TOutput, TMass, TScale>(arg2, ql::Sign(ql::Imag(arg2)));
        TOutput log2 = ql::cLn<TOutput, TMass, TScale>(arg3, ql::Sign(ql::Imag(arg3)));
        if (ql::Real(arg2) < 0.0 && ql::Imag(arg2) < 0.0) log1 += ql::Constants::_2ipi<TOutput, TMass, TScale>();
        if (Real(arg3) < 0.0 && Imag(arg3) < 0.0) log2 += ql::Constants::_2ipi<TOutput, TMass, TScale>();
    
        res = ql::xspence<TOutput, TMass, TScale>(x, siqx, sm4, ql::Sign(ql::Imag(sm4))) / (x[0] - x[1])
             -log0 * log1 / (TOutput(2.0) * x[1])
             -log0 * log2 / x[1];
    
        if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(r24))) {
            const TOutput arg = r24 * sm3;
            res += ql::xspence<TOutput, TMass, TScale>(x, siqx, arg, ql::Sign(ql::Imag(arg))) / (x[0] - x[1]);
        }
    
        if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(r34))) {
            const TOutput arg = r34 * sm3;
            res -= ql::xspence<TOutput, TMass, TScale>(x, siqx, arg, ql::Sign(ql::Imag(arg))) / (x[0] - x[1]);
        }
    
        res /= (a * sm2 * sm3 * sm4);

    }


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
        const TScale& p,
        const int i) {

        const TOutput wlogm = ql::Lnrat<TOutput, TMass, TScale>(mu2, -p);
        res(i,2) = TOutput(1.0) / TOutput(p);
        res(i,1) = res(i,2) * wlogm;
        res(i,0) = TOutput(0.5) * res(i,2) * wlogm * wlogm;

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
        const TScale& p2,
        const int i) {

        const TOutput wlog1 = ql::Lnrat<TOutput, TMass, TScale>(mu2, -p1);
        const TOutput wlog2 = ql::Lnrat<TOutput, TMass, TScale>(mu2, -p2);
        const TScale r = (p2 - p1) / p1;
        res(i,2) = TOutput(0.0);
        if (Kokkos::abs(r) < ql::Constants::_eps()) {
            const TOutput ro2 = r / TOutput(2.0);
            res(i,1) = -TOutput(1.0) / p1 * (TOutput(1.0) - ro2);
            res(i,0) = res(i,1) * wlog1 + ro2 / p1;
        }
        else {
            res(i,1) = (wlog1 - wlog2) / TOutput(p1 - p2);
            res(i,0) = TOutput(0.5) * res(i,1) * (wlog1 + wlog2);
        }

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
        const TScale& p3,
        const int i) {

        const TMass m2sqb = m - TMass(p2);
        const TMass m3sqb = m - TMass(p3);
        const TOutput dilog2 = ql::Li2omrat<TOutput, TMass, TScale>(m2sqb, m);
        const TOutput dilog3 = ql::Li2omrat<TOutput, TMass, TScale>(m3sqb, m);
    
        const TOutput wlog2  = ql::Lnrat<TOutput, TMass, TScale>(m2sqb,m);
        const TOutput wlog3  = ql::Lnrat<TOutput, TMass, TScale>(m3sqb,m);
        const TOutput wlogm  = ql::Lnrat<TOutput, TMass, TScale>(mu2,m);
        const TMass r = (m3sqb - m2sqb) / m2sqb;
    
        res(i,2) = TOutput(0.0);    
        if (Kokkos::abs(r) < ql::Constants::_eps()) {        
            res(i,1) = (TOutput(1.0) - TOutput(0.5) * r) / m2sqb;
            res(i,0) = (wlogm - (m + p2) / p2 * wlog2);
            res(i,0) += -TOutput(0.5) * (r * ((m * m - TOutput(2.0) * p2 * m - p2 * p2) * wlog2 + p2 * (m + p2 + p2 * wlogm)) / (p2 * p2));
            res(i,0) /= m2sqb;        
        }
        else {        
            const TOutput fac = TOutput(1.0) / (p2 - p3);
            res(i,1) = fac * (wlog3 - wlog2);
            res(i,0) = res(i,1) * wlogm + fac * (dilog2 - dilog3 + (wlog2 * wlog2 - wlog3 * wlog3));
        }

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
        const TScale& p2,
        const int i) {

        const TOutput wlog  = ql::Lnrat<TOutput, TMass, TScale>(m, m - p2);
        const TOutput wlogm = ql::Lnrat<TOutput, TMass, TScale>(mu2, m);
        const TOutput fac   = TOutput(0.5) / (p2 - m);
        const TMass arg2    = -p2 / (m - p2);
        const TMass omarg2  = TMass(1.0) - arg2;
        const TOutput ct = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>());
    
        TOutput dilog2;
        if (ql::Real(omarg2) < 0.0)
            dilog2 = ct - TOutput(ql::ddilog<TOutput, TMass, TScale>(omarg2)) - Kokkos::log(arg2) * wlog;
        else
            dilog2 = TOutput(ql::ddilog<TOutput, TMass, TScale>(arg2));
    
        res(i,2) = fac;
        res(i,1) = res(i,2) * wlogm + fac * TOutput(2.0) * wlog;
        res(i,0) = -res(i,2) * TOutput(0.5) * wlogm * wlogm + res(i,1) * wlogm + fac * (wlog * wlog + ct - TOutput(2.0) * dilog2);

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
        const TMass& m,
        const int i) {

        const TOutput fac = TOutput(1.0) / m;
        const TOutput wlogm = ql::Lnrat<TOutput, TMass, TScale>(mu2, m);
        res(i,2) = TOutput(0.0);
        res(i,1) = -TOutput(0.5) * fac;
        res(i,0) = res(i,1) * wlogm + fac;

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
        const TScale& p2,
        const int i) {


        const TMass m2 = Kokkos::sqrt(m2sq);
        const TMass m3 = Kokkos::sqrt(m3sq);
    
        TScale iepsd = 0;
        Kokkos::View<TOutput[3]> cxs;
        ql::kfn<TOutput, TMass, TScale>(cxs, iepsd, p2, m2, m3);
    
        const TOutput xlog= ql::cLn<TOutput, TMass, TScale>(cxs(0), iepsd);
        const TScale resx = ql::Real(cxs[0]);
        const TScale imxs = ql::Imag(cxs[0]);
    
        if (ql::iszero<TOutput, TMass, TScale>(resx - 1.0) && ql::iszero<TOutput, TMass, TScale>(imxs)) {
            const TMass arg = mu2 / (m2 * m3);
            const TOutput fac = TOutput(0.5) / (m2 * m3);
    
            res(i,1) = fac;
            if (ql::iszero<TOutput, TMass, TScale>(m2-m3))
                res(i,0) = fac * Kokkos::log(arg);
            else
                res(i,0) = fac * (Kokkos::log(arg) - TOutput(2.0) - (m3 + m2) / (m3 - m2) * Kokkos::log(m2 / m3));
        }
        else {
            const TMass arg = m2 / m3;
            const TMass arg2 = m2 * m3;
            const TOutput logarg = TOutput(Kokkos::log(arg));
            const TOutput fac = TOutput(1.0) / arg2 * cxs(0) / (cxs(1) * cxs(2));
            res(i,1) = -fac * xlog;
            res(i,0) = fac * (xlog * (-TOutput(0.5) * xlog + Kokkos::log(arg2 / mu2))
                            - ql::cLi2omx2<TOutput, TMass, TScale>(cxs(0), cxs(0), iepsd, iepsd)
                            + TOutput(0.5) * logarg * logarg
                            + ql::cLi2omx2<TOutput, TMass, TScale>(cxs[0], arg, iepsd, 0.0)
                            + ql::cLi2omx2<TOutput, TMass, TScale>(cxs[0], TOutput(1.0) / arg, iepsd, 0.0));
        }
        res(i,2) = TOutput(0.0);

    }


}