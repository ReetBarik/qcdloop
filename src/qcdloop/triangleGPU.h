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

    KOKKOS_INLINE_FUNCTION
    constexpr int isort(int i, int j) {
        constexpr int val[6][6] = {
            {0,1,2,3,4,5},
            {1,2,0,4,5,3},
            {2,0,1,5,3,4},
            {1,0,2,3,5,4},
            {0,2,1,5,4,3},
            {2,1,0,4,3,5}
        };
        return val[i][j];
    }



    /*!
    * Sort arguments of triangle so that they are ordered in mass.
    * \param psq are the four-momentum squared of the external lines
    * \param msq are the squares of the masses of the internal lines
    */
    template<typename TOutput, typename TMass, typename TScale> 
    KOKKOS_INLINE_FUNCTION
    void TriSort(Kokkos::Array<TScale, 3>& psq, Kokkos::Array<TMass, 3>& msq) {    
        const int x1[3] = {2,0,1};
        const int x2[3] = {1,2,0};
        Kokkos::Array<TScale, 3> psqtmp;
        Kokkos::Array<TMass, 3> msqtmp;

        for (int i = 0; i < 3; i++) {
            psqtmp[i] = psq[i];
            msqtmp[i] = msq[i];
        }

        const TMass mmax = ql::Max(msqtmp[0], ql::Max(msqtmp[1], msqtmp[2]));
        if (mmax == msqtmp[0]) {
            for (int i = 0; i < 3; i++) {
                msq[x1[i]] = msqtmp[i];
                psq[x1[i]] = psqtmp[i];
            }
        }
        
        else if (mmax == msqtmp[1]) {
            for (int i = 0; i < 3; i++) {
                msq[x2[i]] = msqtmp[i];
                psq[x2[i]] = psqtmp[i];
            }
        }
        

        if (Kokkos::abs(msq[0]) > Kokkos::abs(msq[1])) {
            for (int i = 0; i < 2; i++) {
                msqtmp[i] = msq[i];
                psqtmp[i+1] = psq[i + 1];
            }
            msq[0] = msqtmp[1];
            msq[1] = msqtmp[0];
            psq[1] = psqtmp[2];
            psq[2] = psqtmp[1];
        }      
    }


    /*!
    * Sort arguments of triangle so that |p3sq| > |p2sq| > |p1sq| and permute masses accordingly.
    * \param xpi i=0,2: mass^2, i=3,5: p^2
    * \param ypi abs(p3) < abs(p4) < abs(p5)
    */
    template<typename TOutput, typename TMass, typename TScale> 
    KOKKOS_INLINE_FUNCTION void TriSort2(const Kokkos::Array<TMass, 6>& xpi, Kokkos::Array<TMass, 6>& ypi) {
        
        // Kokkos::View<int**> isort = ql::isort();
        
        const TScale p1sq = Kokkos::abs(xpi[3]);
        const TScale p2sq = Kokkos::abs(xpi[4]);
        const TScale p3sq = Kokkos::abs(xpi[5]);

        int j = 0;
        if      ( (p3sq >= p2sq) && (p2sq >= p1sq) ) j = 0;
        else if ( (p1sq >= p3sq) && (p3sq >= p2sq) ) j = 1;
        else if ( (p2sq >= p1sq) && (p1sq >= p3sq) ) j = 2;
        else if ( (p2sq >= p3sq) && (p3sq >= p1sq) ) j = 3;
        else if ( (p1sq >= p2sq) && (p2sq >= p3sq) ) j = 4;
        else if ( (p3sq >= p1sq) && (p1sq >= p2sq) ) j = 5;
        else j = 0;
        
        for (size_t k = 0; k < 6; k++)
            ypi[k] = xpi[ql::isort(j,k)];

    }


    /*!
    * Sort an input vector of TScale objets based on its abs.
    * \param psq parameter to sort in ascending order
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void SnglSort(Kokkos::Array<TScale, 3>& psq) {
        Kokkos::Array<TScale, 3> absp = { Kokkos::abs(psq[0]), Kokkos::abs(psq[1]), Kokkos::abs(psq[2])};
        if (absp[0] > absp[1]) {
            const TScale ptmp = psq[0], atmp = absp[0];
            psq[0] = psq[1]; absp[0] = absp[1];
            psq[1] = ptmp;   absp[1] = atmp;
        }

        if (absp[0] > absp[2]) {
            const TScale ptmp = psq[0], atmp = absp[0];
            psq[0] = psq[2]; absp[0] = absp[2];
            psq[2] = ptmp;   absp[2] = atmp;
        }

        if (absp[1] > absp[2]) {
            const TScale ptmp = psq[1];
            psq[1] = psq[2];
            psq[2] = ptmp;
        }
    }

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
    * Computes the finite triangle, when Kallen2 > 0 and 3 massive particles.
    * Formulae from Denner, Nierste and Scharf \cite Denner:1991qq following the OneLoop implementation \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TINDNS(TOutput& res, const Kokkos::Array<TMass, 6>& xpi) {

        TOutput m1 = xpi[0];
        TOutput m2 = xpi[1];
        TOutput m3 = xpi[2];
        TOutput p1 = TOutput(xpi[3]);
        TOutput p2 = TOutput(xpi[4]);
        TOutput p3 = TOutput(xpi[5]);
    
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
    
        Kokkos::Array<TOutput, 2> x;
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, x);
        x[0] = -x[0];
        x[1] = -x[1];
    
        const Kokkos::Array<TOutput, 2> qz = { x[0] * sm2, x[1] * sm2 };
        const Kokkos::Array<TOutput, 2> qz2 = { x[0] * r13 * sm2, x[1] * r13 * sm2};
        const Kokkos::Array<TOutput, 2> qz3 = { x[0] * r13, x[1] * r13};
        const TOutput oneOr12 = TOutput(1.0) / r12;
        const TOutput oneOr23 = TOutput(1.0) / r23;
        const TOutput dqz = qz[0] - qz[1];
        const TOutput dqz2 = qz2[0] - qz2[1];
        const TOutput dqz3 = qz3[0] - qz3[1];
        const Kokkos::Array<TScale, 2> siqz = { (TScale) ql::Sign(ql::Imag(qz[0])), (TScale) ql::Sign(ql::Imag(qz[1]))};
        const Kokkos::Array<TScale, 2> siqz2 = { (TScale) ql::Sign(ql::Imag(qz2[0])), (TScale) ql::Sign(ql::Imag(qz2[1]))};
        const Kokkos::Array<TScale, 2> siqz3 = { (TScale) ql::Sign(ql::Imag(qz3[0])), (TScale) ql::Sign(ql::Imag(qz3[1]))};
        const Kokkos::Array<TScale, 2> sigx = { (TScale) ql::Sign(ql::Imag(x[0])), (TScale) ql::Sign(ql::Imag(x[1]))};
    
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
    void TINDNS2(TOutput& res, const Kokkos::Array<TMass, 6>& xpi) {

        TOutput m2 = xpi[1];
        TOutput m4 = xpi[2];
        TOutput p2 = TOutput(xpi[3]);
        TOutput p3 = TOutput(xpi[5]);
        TOutput p23 = TOutput(xpi[4]);
    
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
    
        Kokkos::Array<TOutput, 2> x;
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, x);
        x[0] = -x[0];
        x[1] = -x[1];
    
        const Kokkos::Array<TOutput, 2> qz = { x[0] / r24, x[1] / r24};
        const Kokkos::Array<TScale, 2> siqz = { (TScale) ql::Sign(ql::Imag(qz[0])), (TScale) ql::Sign(ql::Imag(qz[1]))};
    
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
    
        const Kokkos::Array<TScale, 2> siqx = { (TScale) ql::Sign(ql::Imag(x[0])), (TScale) ql::Sign(ql::Imag(x[1])) };
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
    void TINDNS1(TOutput& res, const Kokkos::Array<TMass, 6>& xpi) {

        TOutput m4 = xpi[2];
        TOutput p2 = TOutput(xpi[3]);
        TOutput p3 = TOutput(xpi[4]);
        TOutput p4 = TOutput(xpi[5]);
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
    
        Kokkos::Array<TOutput, 2> x;
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, x);
        x[0] = -x[0];
        x[1] = -x[1];
    
        const Kokkos::Array<TScale, 2> siqx = { (TScale) ql::Sign(ql::Imag(x[0])), (TScale) ql::Sign(ql::Imag(x[1])) };
    
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
    * Computes the finite triangle with all internal masses zero.
    * Formulae from 't Hooft and Veltman \cite tHooft:1978xw following the LoopTools implementation \cite Hahn:2006qw.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    TOutput TIN0(const Kokkos::Array<TMass, 6>& xpi) {
        TOutput res;
        const TMass m1sq = xpi[0];
        const TMass m2sq = xpi[1];
        const TMass m3sq = xpi[2];

        if (ql::iszero<TOutput, TMass, TScale>(m1sq - m2sq) && ql::iszero<TOutput, TMass, TScale>(m2sq - m3sq))
            res = -TOutput(0.5) / m1sq;
        else if (ql::iszero<TOutput, TMass, TScale>(m1sq - m2sq))
            res = TOutput((m3sq * Kokkos::log(m2sq / m3sq) + m3sq - m2sq) / ql::kPow<TOutput, TMass, TScale>(m3sq - m2sq, 2));
        else if (ql::iszero<TOutput, TMass, TScale>(m2sq - m3sq))
            res = TOutput((m1sq * Kokkos::log(m3sq / m1sq) + m1sq - m3sq) / ql::kPow<TOutput, TMass, TScale>(m1sq - m3sq, 2));
        else if (ql::iszero<TOutput, TMass, TScale>(m3sq - m1sq))
            res = TOutput((m2sq * Kokkos::log(m1sq / m2sq) + m2sq - m1sq) / ql::kPow<TOutput, TMass, TScale>(m2sq - m1sq, 2));
        else
            res = TOutput( m3sq * Kokkos::log(m3sq / m1sq) / ((m1sq - m3sq) * (m3sq - m2sq)) - m2sq * Kokkos::log(m2sq / m1sq) / ((m1sq - m2sq) * (m3sq - m2sq)));
        return res;
    }


    /*!
    * Computes the finite triangle with all internal masses zero.
    * Formulae from 't Hooft and Veltman \cite tHooft:1978xw following the LoopTools and OneLoop implementation \cite Hahn:2006qw, \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    TOutput TIN1(const Kokkos::Array<TMass, 6>& xpi, const Kokkos::Array<TMass, 6>& sxpi, const int &massive) {
        TOutput res;
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(xpi[0])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(xpi[1])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(xpi[2])))
        {
            const TMass s1Ds1 = xpi[0];
            const TMass s2Ds2 = xpi[1];
            const TMass s3Ds3 = xpi[2];
            const TMass p3Dp3 = xpi[5];

            const TOutput x0 = TOutput((s1Ds1 - s2Ds2) / p3Dp3);
            const TMass D23 = s2Ds2 - s3Ds3;

            Kokkos::Array<TOutput, 2> l;
            ql::solveabc<TOutput, TMass, TScale>(p3Dp3, s3Ds3 - s1Ds1 - p3Dp3, s1Ds1, l);

            if (ql::iszero<TOutput, TMass, TScale>(D23))
                res = TOutput(-(-ql::Rint<TOutput, TMass, TScale>(x0, l[0], -1) - ql::Rint<TOutput, TMass, TScale>(x0, l[1], 1)) / p3Dp3);
            else {
                const TOutput u0 = TOutput(s2Ds2 / D23);
                const TScale ieps = ql::Sign(ql::Real(-D23));
                res = -(ql::Rint<TOutput, TMass, TScale>(x0, u0, -ieps) - ql::Rint<TOutput, TMass, TScale>(x0, l[0], -1) - ql::Rint<TOutput, TMass, TScale>(x0, l[1], 1)) / p3Dp3;
            }
        }
        else {
            if (massive == 2)
                ql::TINDNS2<TOutput, TMass, TScale>(res, sxpi);
            else if (massive == 1)
                ql::TINDNS1<TOutput, TMass, TScale>(res, sxpi);
            else {
              
                if (ql::Real(ql::Kallen2<TOutput, TMass, TScale>(xpi[3], xpi[4], xpi[5])) < 0.0) {  // never happens with real momenta (but just in case..)
                    
                    const TOutput p2 = TOutput(xpi[5]);
                    TOutput m[3] = {TOutput(xpi[0]), TOutput(xpi[1]), TOutput(xpi[2])};

                    m[0] -= ql::Constants::_ieps2<TOutput, TMass, TScale>() * TOutput(Kokkos::abs(ql::Real(m[0])));
                    m[1] -= ql::Constants::_ieps2<TOutput, TMass, TScale>() * TOutput(Kokkos::abs(ql::Real(m[1])));
                    m[2] -= ql::Constants::_ieps2<TOutput, TMass, TScale>() * TOutput(Kokkos::abs(ql::Real(m[2])));

                    const TOutput sm0 = Kokkos::sqrt(m[0]) - ql::Constants::_ieps2<TOutput, TMass, TScale>();
                    const TOutput sm2 = Kokkos::sqrt(m[2]) - ql::Constants::_ieps2<TOutput, TMass, TScale>();

                    const TOutput yy = -((m[0] - m[1]) - p2)/p2;

                    res = -(ql::R3int<TOutput, TMass, TScale>(p2, sm0, sm2, yy) - ql::R2int<TOutput, TMass, TScale>(m[1] - m[2], m[2], yy));
                    res /= p2;
                }
                else
                    ql::TINDNS<TOutput, TMass, TScale>(res, xpi);
            }
        }
        return res;
    }


    /*!
    * Computes the finite triangle with all internal masses zero.
    * Formulae from 't Hooft and Veltman \cite tHooft:1978xw following the LoopTools and OneLoop implementation \cite Hahn:2006qw, \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    TOutput TIN2(const Kokkos::Array<TMass, 6>& xpi, const Kokkos::Array<TMass, 6>& sxpi, const int &massive) {
        TOutput res;
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(xpi[0])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(xpi[1])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(xpi[2]))) {
            
            const TMass s1Ds1 = xpi[0];
            const TMass s2Ds2 = xpi[1];
            const TMass s3Ds3 = xpi[2];
            const TMass p2Dp2 = xpi[4];
            const TMass p3Dp3 = xpi[5];

            const TOutput z0 = TOutput((s1Ds1 - s2Ds2) / (p3Dp3 - p2Dp2));
            Kokkos::Array<TOutput, 2> zu;
            Kokkos::Array<TOutput, 2> zl;
            ql::solveabc<TOutput, TMass, TScale>(p2Dp2, s3Ds3 - s2Ds2 - p2Dp2, s2Ds2, zu);
            ql::solveabc<TOutput, TMass, TScale>(p3Dp3, s3Ds3 - s1Ds1 - p3Dp3, s1Ds1, zl);

            if (ql::iszero<TOutput, TMass, TScale>(p2Dp2 - p3Dp3)) {
                res = TOutput(-(ql::Zlogint<TOutput, TMass, TScale>(zu[0], -1) + ql::Zlogint<TOutput, TMass, TScale>(zu[1], 1) - ql::Zlogint<TOutput, TMass, TScale>(zl[0], -1) + ql::Zlogint<TOutput, TMass, TScale>(zl[1], 1)) / (s2Ds2 - s1Ds1));
            }
            else {
                res = TOutput(-(ql::Rint<TOutput, TMass, TScale>(z0, zu[0], -1) + ql::Rint<TOutput, TMass, TScale>(z0, zu[1], 1) - ql::Rint<TOutput, TMass, TScale>(z0, zl[0], -1)-ql::Rint<TOutput, TMass, TScale>(z0, zl[1], 1)) / (p3Dp3 - p2Dp2));
            }
        }
        else {
            if (massive == 2)
                ql::TINDNS2<TOutput, TMass, TScale>(res, sxpi);
            else if (massive == 1)
                ql::TINDNS1<TOutput, TMass, TScale>(res, sxpi);
            else {

                TOutput K2 = ql::Kallen2<TOutput, TMass, TScale>(xpi[3], xpi[4], xpi[5]);
                if (ql::Real(K2) < 0.0)
                {
                    const TOutput p[2] = {TOutput(xpi[4]), TOutput(xpi[5])};
                    TOutput m[3] = {TOutput(xpi[0]), TOutput(xpi[1]), TOutput(xpi[2])};

                    if (p[0] == p[1]) {
                        Kokkos::printf("Triangle::TIN2 threshold singularity\n");
                    }

                    m[0] -= ql::Constants::_ieps2<TOutput, TMass, TScale>() * TOutput(Kokkos::abs(ql::Real(m[0])));
                    m[1] -= ql::Constants::_ieps2<TOutput, TMass, TScale>() * TOutput(Kokkos::abs(ql::Real(m[1])));
                    m[2] -= ql::Constants::_ieps2<TOutput, TMass, TScale>() * TOutput(Kokkos::abs(ql::Real(m[2])));

                    const TOutput sm0 = Kokkos::sqrt(m[0]) - ql::Constants::_ieps2<TOutput, TMass, TScale>();
                    const TOutput sm1 = Kokkos::sqrt(m[1]) - ql::Constants::_ieps2<TOutput, TMass, TScale>();
                    const TOutput sm2 = Kokkos::sqrt(m[2]) - ql::Constants::_ieps2<TOutput, TMass, TScale>();

                    const TOutput yy = ((m[0] - m[1]) - p[1] + p[0]) / (p[0] - p[1]);

                    res = ql::R3int<TOutput, TMass, TScale>(p[1], sm0, sm2, yy) - ql::R3int<TOutput, TMass, TScale>(p[0], sm1, sm2, yy);
                    res /= (p[0] - p[1]);
                }
                else
                    ql::TINDNS<TOutput, TMass, TScale>(res, xpi);
            }
        }
        return res;
    }


    /*!
    * Computes the finite triangle with all internal masses zero.
    * Formulae from 't Hooft and Veltman \cite tHooft:1978xw following the LoopTools and OneLoop implementation \cite Hahn:2006qw, \cite vanHameren:2010cp.
    * \param res the output object for the finite part.
    * \param xpi an array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    TOutput TIN3(const Kokkos::Array<TMass, 6>& xpi, const Kokkos::Array<TMass, 6>& sxpi, const int &massive) {
        TOutput res; 
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(xpi[0])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(xpi[1])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(xpi[2]))) {
            
            TOutput Del2[3], y[3];
            Kokkos::Array<TOutput, 2> z;
            TMass siDsi[3], piDpj[3][3], siDpj[3][3], kdel[3];
            int jp1[3] = {1,2,0}, jm1[3] = {2,0,1};

            for (int j = 0; j < 3; j++) {
                siDsi[j] = xpi[j];
                piDpj[j][j] = xpi[j + 3];
                siDpj[j][j] = 0.5 * (xpi[jp1[j]] - xpi[j] - piDpj[j][j]);
            }

            for (int j = 0; j < 3; j++) {
                piDpj[j][jp1[j]] = 0.5 * (piDpj[jm1[j]][jm1[j]] - piDpj[j][j] - piDpj[jp1[j]][jp1[j]]);
                piDpj[jp1[j]][j] = piDpj[j][jp1[j]];
                siDpj[j][jm1[j]] = piDpj[jm1[j]][jm1[j]] + siDpj[jm1[j]][jm1[j]];
                siDpj[j][jp1[j]] = -piDpj[j][jp1[j]] + siDpj[jp1[j]][jp1[j]];
            }

            for (int j = 0; j < 3; j++) {
                Del2[j] = TOutput(piDpj[j][jp1[j]] * piDpj[j][jp1[j]] - piDpj[j][j] * piDpj[jp1[j]][jp1[j]]);
                Del2[j] = Kokkos::sqrt(Del2[j]);
                kdel[j] = piDpj[j][j] * siDpj[jp1[j]][jp1[j]] - piDpj[j][jp1[j]] * siDpj[jp1[j]][j];
                y[j] = TOutput((siDpj[jp1[j]][j] + kdel[j] / Del2[j]) / piDpj[j][j]);
            }
            
            res = TOutput(0.0);
            
            for (int j = 0; j < 3; j++) {
                const TMass a = piDpj[j][j], b = -2.0 * siDpj[jp1[j]][j], c = siDsi[jp1[j]];
                ql::solveabc<TOutput, TMass, TScale>(a, b, c, z);
                res += ql::Rint<TOutput, TMass, TScale>(y[j], z[0], -1) + ql::Rint<TOutput, TMass, TScale>(y[j], z[1], 1);
            }

            res = -res / (TOutput(2.0) * Del2[0]);
        } 
        else { 
            if (massive == 2)
                ql::TINDNS2<TOutput, TMass, TScale>(res, sxpi);
            else if (massive == 1)
                ql::TINDNS1<TOutput, TMass, TScale>(res, sxpi);
            else {
                
                const TOutput K2 = ql::Kallen2<TOutput, TMass, TScale>(xpi[3], xpi[4], xpi[5]);
                if (ql::Real(K2) < 0.0) {
                    
                    const TOutput p[3] = {TOutput(xpi[3]), TOutput(xpi[4]), TOutput(xpi[5])};
                    TOutput m[3] = {TOutput(xpi[0]), TOutput(xpi[1]), TOutput(xpi[2])};

                    m[0] -= ql::Constants::_ieps2<TOutput, TMass, TScale>() * TOutput(Kokkos::abs(ql::Real(m[0])));
                    m[1] -= ql::Constants::_ieps2<TOutput, TMass, TScale>() * TOutput(Kokkos::abs(ql::Real(m[1])));
                    m[2] -= ql::Constants::_ieps2<TOutput, TMass, TScale>() * TOutput(Kokkos::abs(ql::Real(m[2])));

                    const TOutput alpha = Kokkos::sqrt(K2) + ql::Constants::_ieps2<TOutput, TMass, TScale>();
                    const TOutput sm0 = Kokkos::sqrt(m[0]) - ql::Constants::_ieps2<TOutput, TMass, TScale>();
                    const TOutput sm1 = Kokkos::sqrt(m[1]) - ql::Constants::_ieps2<TOutput, TMass, TScale>();
                    const TOutput sm2 = Kokkos::sqrt(m[2]) - ql::Constants::_ieps2<TOutput, TMass, TScale>();

                    res = -(ql::R3int<TOutput, TMass, TScale>(p[0], sm0, sm1, m[1] - m[2] + p[1], p[2] - p[0] - p[1], p[1], alpha)
                          - ql::R3int<TOutput, TMass, TScale>(p[2], sm0, sm2, -(m[0] - m[1]) + p[2] - p[1], p[1] - p[0] - p[2], p[0], alpha)
                          + ql::R3int<TOutput, TMass, TScale>(p[1], sm1, sm2, -(m[0] - m[1]) + p[2] - p[1], p[0] + p[1] - p[2], p[0], alpha));

                    res /= alpha;
                }
                else
                    ql::TINDNS<TOutput, TMass, TScale>(res, xpi);
            }
        }
        return res;
    }    


    /*!
    * Parses finite triangle integrals. Formulae from 't Hooft and Veltman \cite tHooft:1978xw
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi an array with masses and momenta squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void T0(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<TMass, 6>& xpi,
        const int &massive,
        const int i) {
            
        // Set poles to zero
        res(i,1) = TOutput(0.0);
        res(i,2) = TOutput(0.0);
        
        // Sort
        Kokkos::Array<TMass, 6> ypi;
        ql::TriSort2<TOutput, TMass, TScale>(xpi, ypi); 
        
        const bool zypi3 = ql::iszero<TOutput, TMass, TScale>(ypi[3]);
        const bool zypi4 = ql::iszero<TOutput, TMass, TScale>(ypi[4]);
        
        // Trigger the finite topology
        if (zypi3 && zypi4 && ql::iszero<TOutput, TMass, TScale>(ypi[5])) 
            res(i,0) = ql::TIN0<TOutput, TMass, TScale>(ypi);
        else if (zypi3 && zypi4) 
            res(i,0) = ql::TIN1<TOutput, TMass, TScale>(ypi, xpi, massive);
        else if (zypi3) 
            res(i,0) = ql::TIN2<TOutput, TMass, TScale>(ypi, xpi, massive);
        else 
            res(i,0) = ql::TIN3<TOutput, TMass, TScale>(ypi, xpi, massive);
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
        Kokkos::Array<TOutput, 3> cxs;
        ql::kfn<TOutput, TMass, TScale>(cxs, iepsd, p2, m2, m3);
    
        const TOutput xlog= ql::cLn<TOutput, TMass, TScale>(cxs[0], iepsd);
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
            const TOutput fac = TOutput(1.0) / arg2 * cxs[0] / (cxs[1] * cxs[2]);
            res(i,1) = -fac * xlog;
            res(i,0) = fac * (xlog * (-TOutput(0.5) * xlog + Kokkos::log(arg2 / mu2))
                            - ql::cLi2omx2<TOutput, TMass, TScale>(cxs[0], cxs[0], iepsd, iepsd)
                            + TOutput(0.5) * logarg * logarg
                            + ql::cLi2omx2<TOutput, TMass, TScale>(cxs[0], arg, iepsd, 0.0)
                            + ql::cLi2omx2<TOutput, TMass, TScale>(cxs[0], TOutput(1.0) / arg, iepsd, 0.0));
        }
        res(i,2) = TOutput(0.0);

    }

    /*!
    * Computes the Triangle integral defined as:
    * \f[
    * I_{3}^{D}(p_1^2,p_2^2,p_3^2;m_1^2,m_2^2,m_3^2)= \frac{\mu^{4-D}}{i \pi^{D/2} r_{\Gamma}} \int d^Dl \frac{1}{(l^2-m_1^2+i \epsilon)((l+q_1)^2-m_2^2+i \epsilon)((l+q_2)^2-m_3^2+i\epsilon)}
    *   \f]
    *where \f$q_1=p_1,q_2=p_1+p_2\f$.
    *
    * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn and
    * 't Hooft and Veltman \cite tHooft:1978xw.
    *
    * \param res output object res[i,0,1,2] the coefficients in the Laurent series
    * \param mu2 is the square of the scale mu (per element)
    * \param m are the squares of the masses of the internal lines [batch][3]
    * \param p are the four-momentum squared of the external lines [batch][3]
    * \param i element index
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void TR(
        const Kokkos::View<TOutput* [3]>& res,      // Output view
        const Kokkos::View<TScale*>& mu2,          // Scale parameter (per element)
        const Kokkos::View<TMass* [3]>& m,          // Masses view [batch][3]
        const Kokkos::View<TScale* [3]>& p,         // Momenta view [batch][3]
        const int i) {                              // Element index
        
        // Compute scalefac for this element
        const TScale scalefac = ql::Max(
            Kokkos::abs(m(i, 0)),
            ql::Max(Kokkos::abs(m(i, 1)),
            ql::Max(Kokkos::abs(m(i, 2)),
            ql::Max(Kokkos::abs(p(i, 0)),
            ql::Max(Kokkos::abs(p(i, 1)),
                     Kokkos::abs(p(i, 2)))))));
        
        // Compute musq for this element
        const TScale musq = mu2(i) / scalefac;
        
        // Normalize masses and momenta
        Kokkos::Array<TMass, 3> msq;
        Kokkos::Array<TScale, 3> psq;
        
        msq[0] = m(i, 0) / scalefac;
        msq[1] = m(i, 1) / scalefac;
        msq[2] = m(i, 2) / scalefac;
        psq[0] = p(i, 0) / scalefac;
        psq[1] = p(i, 1) / scalefac;
        psq[2] = p(i, 2) / scalefac;
        
        // Sort msq in ascending order
        ql::TriSort<TOutput, TMass, TScale>(psq, msq);
        
        // if internal masses all 0, reorder abs(psq) in ascending order
        const bool iszeros[3] = {
            ql::iszero<TOutput, TMass, TScale>(msq[0]), 
            ql::iszero<TOutput, TMass, TScale>(msq[1]),
            ql::iszero<TOutput, TMass, TScale>(msq[2])
        };
        
        if (iszeros[0] && iszeros[1] && iszeros[2]) { 
            ql::SnglSort<TOutput, TMass, TScale>(psq);
        }
        
        // calculate integral value
        const TMass Y01 = TMass(msq[0] + msq[1] - psq[0]) / TMass(2);
        const TMass Y02 = TMass(msq[0] + msq[2] - psq[2]) / TMass(2);
        const TMass Y12 = TMass(msq[1] + msq[2] - psq[1]) / TMass(2);
        
        int massive = 0;
        for (size_t j = 0; j < 3; j++) {
            if (!iszeros[j]) massive += 1;
        }
        
        // building xpi
        const Kokkos::Array<TMass, 6> xpi = {
            msq[0], msq[1], msq[2],
            TMass(psq[0]), TMass(psq[1]), TMass(psq[2])
        };
        
        // Call appropriate T function based on conditions
        if (massive == 3) {  // three internal masses
            ql::T0<TOutput, TMass, TScale>(res, xpi, massive, i);
        }
        else if (massive == 2) {  // two internal masses
            if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01)) && 
                ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02))) {
                ql::T6<TOutput, TMass, TScale>(res, musq, msq[1], msq[2], psq[1], i);
            }
            else {
                ql::T0<TOutput, TMass, TScale>(res, xpi, massive, i);
            }
        }
        else if (massive == 1) { // one internal mass
            if (!ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01))) {
                ql::T0<TOutput, TMass, TScale>(res, xpi, massive, i);
            }
            else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02)) && 
                     ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {
                ql::T5<TOutput, TMass, TScale>(res, musq, msq[2], i);
            }
            else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y02))) {
                ql::T4<TOutput, TMass, TScale>(res, musq, msq[2], psq[1], i);
            }
            else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {
                ql::T4<TOutput, TMass, TScale>(res, musq, msq[2], psq[2], i);
            }
            else {
                ql::T3<TOutput, TMass, TScale>(res, musq, msq[2], psq[1], psq[2], i);
            }
        }
        else {  // zero internal masses
            if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01)) && 
                ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y12))) {
                ql::T1<TOutput, TMass, TScale>(res, musq, psq[2], i);
            }
            else if (ql::iszero<TOutput, TMass, TScale>(Kokkos::abs(Y01))) {
                ql::T2<TOutput, TMass, TScale>(res, musq, psq[1], psq[2], i);
            }
            else {
                ql::T0<TOutput, TMass, TScale>(res, xpi, massive, i);
            }
        }
        
        // Normalize results
        res(i, 0) /= scalefac;
        res(i, 1) /= scalefac;
        res(i, 2) /= scalefac;
    }


}