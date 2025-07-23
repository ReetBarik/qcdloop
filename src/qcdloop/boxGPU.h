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
    constexpr int swap_b0m(int i, int j) {
        constexpr int val[13][4] = {
            {3, 2, 1, 0},
            {0, 3, 2, 1},
            {1, 0, 3, 2},
            {2, 1, 0, 3},
            {7, 6, 5, 4},
            {4, 7, 6, 5},
            {5, 4, 7, 6},
            {6, 5, 4, 7},
            {9, 8, 9, 8},
            {8, 9, 8, 9},
            {10, 10, 10, 10},
            {12, 11, 12, 11},
            {11, 12, 11, 12},
        };
        return val[i][j];
    }

    KOKKOS_INLINE_FUNCTION
    constexpr int jsort_b0m(int i) {
        constexpr int val[4] = {3,0,1,2};
        return val[i];
    }

    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b1m(int i, int j) {
        constexpr int val[13][5] = { 
            {3, 2, 1, 0, 2},
            {0, 3, 2, 1, 1},
            {1, 0, 3, 2, 0},
            {2, 1, 0, 3, 3},
            {7, 6, 5, 4, 5},
            {4, 7, 6, 5, 4},
            {5, 4, 7, 6, 7},
            {6, 5, 4, 7, 6},
            {9, 8, 9, 8, 8},
            {8, 9, 8, 9, 9},
            {10, 10, 10, 10, 10},
            {12, 11, 12, 11, 11},
            {11, 12, 11, 12, 12},
        };
        return val[i][j];
    }

    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b2m(int i, int j) {
        constexpr int val[13][5] = { 
            {3, 2, 1, 0, 2},
            {0, 3, 2, 1, 1},
            {1, 0, 3, 2, 0},
            {2, 1, 0, 3, 3},
            {7, 6, 5, 4, 5},
            {4, 7, 6, 5, 4},
            {5, 4, 7, 6, 7},
            {6, 5, 4, 7, 6},
            {9, 8, 9, 8, 8},
            {8, 9, 8, 9, 9},
            {10, 10, 10, 10, 10},
            {11, 11, 11, 11, 11},
            {12, 12, 12, 12, 12},
        };
        return val[i][j];
    }

    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b3m(int i, int j) {
        constexpr int val[13][4] = { 
            {0, 3, 2, 1},
            {1, 0, 3, 2},
            {2, 1, 0, 3},
            {3, 2, 1, 0},
            {4, 7, 6, 5},
            {5, 4, 7, 6},
            {6, 5, 4, 7},
            {7, 6, 5, 4},
            {8, 9, 8, 9},
            {9, 8, 9, 8},
            {10, 10, 10, 10},
            {11, 12, 11, 12},
            {12, 11, 12, 11},
        };
        return val[i][j];
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void Ycalc(
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Yalt,
        int const& massive, 
        bool const& opposite= false) {
            
        if (massive == 1) {
            //C---exchange (1<-->3)
            Yalt[0][0] = Y[2][2];
            Yalt[1][1] = Y[1][1];
            Yalt[2][2] = Y[0][0];
            Yalt[3][3] = Y[3][3];
            Yalt[0][1] = Yalt[1][0] = Y[1][2];
            Yalt[0][2] = Yalt[2][0] = Y[0][2];
            Yalt[0][3] = Yalt[3][0] = Y[2][3];
            Yalt[1][2] = Yalt[2][1] = Y[0][1];
            Yalt[1][3] = Yalt[3][1] = Y[1][3];
            Yalt[2][3] = Yalt[3][2] = Y[0][3];
        } else if (massive == 2 && opposite) {
            //C---exchange (2<-->4) .and (1<-->3)
            Yalt[0][0] = Y[2][2];
            Yalt[1][1] = Y[3][3];
            Yalt[2][2] = Y[0][0];
            Yalt[3][3] = Y[1][1];
            Yalt[0][1] = Yalt[1][0] = Y[2][3];
            Yalt[0][2] = Yalt[2][0] = Y[0][2];
            Yalt[0][3] = Yalt[3][0] = Y[1][2];
            Yalt[1][2] = Yalt[2][1] = Y[0][3];
            Yalt[1][3] = Yalt[3][1] = Y[1][3];
            Yalt[2][3] = Yalt[3][2] = Y[0][1];
        } else if (massive == 2 && !opposite) {
            //C---exchange (1<-->2)and(3<-->4)
            Yalt[0][0] = Y[1][1];
            Yalt[1][1] = Y[0][0];
            Yalt[2][2] = Y[3][3];
            Yalt[3][3] = Y[2][2];
            Yalt[0][1] = Yalt[1][0] = Y[0][1];
            Yalt[2][3] = Yalt[3][2] = Y[2][3];
            Yalt[0][2] = Yalt[2][0] = Y[1][3];
            Yalt[0][3] = Yalt[3][0] = Y[1][2];
            Yalt[1][2] = Yalt[2][1] = Y[0][3];
            Yalt[1][3] = Yalt[3][1] = Y[0][2];
        } else
            Kokkos::printf("Box::Ycalc - massive value not implemented");
    }

    /*!
    * Finite box with zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN0(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m2 = Kokkos::abs(Y[1][3]);
        const TMass k12 = 2.0 * Y[0][1] / m2;
        const TMass k13 = 2.0 * Y[0][2] / m2;
        const TMass k14 = 2.0 * Y[0][3] / m2;
        const TMass k23 = 2.0 * Y[1][2] / m2;
        const TMass k24 = 2.0 * Y[1][3] / m2;
        const TMass k34 = 2.0 * Y[2][3] / m2;

        const TOutput k12c = TOutput(k12 - ql::Max(Kokkos::abs(k12), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k13c = TOutput(k13 - ql::Max(Kokkos::abs(k13), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k14c = TOutput(k14 - ql::Max(Kokkos::abs(k14), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k23c = TOutput(k23 - ql::Max(Kokkos::abs(k23), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k24c = TOutput(k24 - ql::Max(Kokkos::abs(k24), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>()) / k12c;
        const TOutput k34c = TOutput(k34 - ql::Max(Kokkos::abs(k34), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>()) / k13c;

        const TMass a = k34 * k24;
        const TMass b = k13 * k24 + k12 * k34 - k14 * k23;
        const TOutput c = TOutput(k13 * k12) + TOutput(k23) * ql::Constants::_ieps50<TOutput, TMass, TScale>();
        const TOutput disc = Kokkos::sqrt(TOutput(b * b) - TOutput(4.0 * a) * c);

        Kokkos::Array<TOutput, 2> x4;
        x4[0] = TOutput(0.5) * (TOutput(b) - disc) / TOutput(a);
        x4[1] = TOutput(0.5) * (TOutput(b) + disc) / TOutput(a);
        if (Kokkos::abs(x4[0]) > Kokkos::abs(x4[1]))
            x4[1] = c / (TOutput(a) * x4[0]);
        else
            x4[0] = c / (TOutput(a) * x4[1]);

        const Kokkos::Array<TScale, 2> imzero = {0.0, 0.0};
        const TOutput log_x40 = Kokkos::log(x4[0]);
        const TOutput log_x41 = Kokkos::log(x4[1]);

        res(i, 0) = (
            (log_x41 - log_x40) * (-TOutput(0.5) * (log_x41 + log_x40) +
            Kokkos::log(k12c) + Kokkos::log(k13c) - Kokkos::log(k23c) - Kokkos::log(k14c)) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k34c, 0.0) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k24c, 0.0)
            ) / (m2 * m2 * disc);

        res(i, 1) = res(i, 2) = TOutput(0.0);
    }

    /*!
    * Finite box with 1 non-zero mass. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN1(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m4 = Y[3][3];
        const TMass k34 = 2.0 * Y[0][1] / m4;
        const TMass k23 = 2.0 * Y[0][2] / m4;
        const TMass k13 = 2.0 * Y[0][3] / m4;
        const TMass k24 = 2.0 * Y[1][2] / m4;
        const TMass k14 = 2.0 * Y[1][3] / m4;
        const TMass k12 = 2.0 * Y[2][3] / m4;

        const TOutput k12c = TOutput(k12 - ql::Max(Kokkos::abs(k12), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k13c = TOutput(k13 - ql::Max(Kokkos::abs(k13), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k14c = TOutput(k14 - ql::Max(Kokkos::abs(k14), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k23c = TOutput(k23 - ql::Max(Kokkos::abs(k23), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k24c = TOutput(k24 - ql::Max(Kokkos::abs(k24), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>()) / k12c;
        const TOutput k34c = TOutput(k34 - ql::Max(Kokkos::abs(k34), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>()) / k13c;

        const TMass a = k34 * k24;
        const TMass b = k13 * k24 + k12 * k34 - k14 * k23;
        const TOutput c = TOutput(k13 * k12) - TOutput(k23) * (TOutput(1.0) - ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput disc = Kokkos::sqrt(TOutput(b * b) - TOutput(4.0) * TOutput(a) * c);

        Kokkos::Array<TOutput, 2> x4;
        x4[0] = TOutput(0.5) * (TOutput(b) - disc) / TOutput(a);
        x4[1] = TOutput(0.5) * (TOutput(b) + disc) / TOutput(a);
        if (Kokkos::abs(x4[0]) > Kokkos::abs(x4[1]))
            x4[1] = c / (TOutput(a) * x4[0]);
        else
            x4[0] = c / (TOutput(a) * x4[1]);

        const Kokkos::Array<TScale, 2> imzero = {0.0, 0.0};
        res(i, 0) = (
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k14c, 0.0) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k34c, 0.0) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k24c, 0.0) +
            (Kokkos::log(x4[1]) - Kokkos::log(x4[0])) * (Kokkos::log(k12c) + Kokkos::log(k13c) - Kokkos::log(k23c))
            ) / (TOutput(ql::kPow<TOutput, TMass, TScale>(m4, 2)) * disc);

        res(i, 1) = res(i, 2) = TOutput(0.0);
    }


    /*!
    * Finite box with 2 non-zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN2(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m3 = Y[2][2];
        const TMass m4 = Y[3][3];
        const TMass m = Kokkos::sqrt(m3 * m4);

        const TMass k12 = 2.0 * Y[1][2] / m3;
        const TMass k13 = 2.0 * Y[0][2] / m3;
        const TMass k14 = 2.0 * Y[2][3] / m;
        const TMass k23 = 2.0 * Y[0][1] / m3;
        const TMass k24 = 2.0 * Y[1][3] / m;
        const TMass k34 = 2.0 * Y[0][3] / m;

        const TOutput k12c = TOutput(k12 - ql::Max(Kokkos::abs(k12), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k13c = TOutput(k13 - ql::Max(Kokkos::abs(k13), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k23c = TOutput(k23 - ql::Max(Kokkos::abs(k23), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput k24c = TOutput(k24 - ql::Max(Kokkos::abs(k24), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>()) / k12c;
        const TOutput k34c = TOutput(k34 - ql::Max(Kokkos::abs(k34), 1.0) * ql::Constants::_ieps50<TOutput, TMass, TScale>()) / k13c;

        TOutput r14 = TOutput(0.5) * (TOutput(k14) + TOutput(ql::Sign(ql::Real(k14))) * Kokkos::sqrt(TOutput((k14 - 2.0) * (k14 + 2.0))));
        r14 *= (TOutput(1.0) + ql::Constants::_ieps50<TOutput, TMass, TScale>() * TOutput(ql::Sign(ql::Real(TOutput(1.0) / r14 - r14))));

        const TMass a = k34 * k24 - k23;
        const TMass b = k13 * k24 + k12 * k34 - k14 * k23;
        const TOutput c = TOutput(k13 * k12) - TOutput(k23) * (TOutput(1.0) - ql::Constants::_ieps50<TOutput, TMass, TScale>());
        const TOutput disc = Kokkos::sqrt(TOutput(b * b) - TOutput(4.0 * a) * c);

        Kokkos::Array<TOutput, 2> x4;
        x4[0] = TOutput(0.5) * (TOutput(b) - disc) / TOutput(a);
        x4[1] = TOutput(0.5) * (TOutput(b) + disc) / TOutput(a);
        if (Kokkos::abs(x4[0]) > Kokkos::abs(x4[1]))
            x4[1] = c / (TOutput(a) * x4[0]);
        else
            x4[0] = c / (TOutput(a) * x4[1]); 

        const Kokkos::Array<TScale, 2> imzero = {0.0, 0.0};
        res(i, 0) = (
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, r14, 0.0) +
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, TOutput(1.0) / r14, 0.0) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k34c, 0.0) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k24c, 0.0) +
            (Kokkos::log(x4[1]) - Kokkos::log(x4[0])) * (Kokkos::log(k12c) + Kokkos::log(k13c) - Kokkos::log(k23c))
        ) / (m3 * m * disc);

        res(i, 1) = res(i, 2) = TOutput(0.0);
    }


    /*!
    * Finite box with 3 non-zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN3(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m2 = Y[1][1];
        const TMass m3 = Y[2][2];
        const TMass m4 = Y[3][3];
        const TMass m_0 = Kokkos::sqrt(m3 * m4);
        const TMass m_1 = Kokkos::sqrt(m2 * m3);
        const TMass m_2 = Kokkos::sqrt(m2 * m4);

        const TMass k12 = 2.0 * Y[2][3] / m_0;
        const TMass k13 = 2.0 * Y[0][2] / m3;
        const TMass k14 = 2.0 * Y[1][2] / m_1;
        const TMass k23 = 2.0 * Y[0][3] / m_0;
        const TMass k24 = 2.0 * Y[1][3] / m_2;
        const TMass k34 = 2.0 * Y[0][1] / m_1;

        int ir12 = 0, ir14 = 0, ir24 = 0;
        const TOutput r12 = TOutput(0.5) * (TOutput(k12) + TOutput(ql::Sign(ql::Real(k12))) * Kokkos::sqrt(TOutput((k12 - 2.0) * (k12 + 2.0))));
        const TOutput r14 = TOutput(0.5) * (TOutput(k14) + TOutput(ql::Sign(ql::Real(k14))) * Kokkos::sqrt(TOutput((k14 - 2.0) * (k14 + 2.0))));
        const TOutput r24 = TOutput(0.5) * (TOutput(k24) + TOutput(ql::Sign(ql::Real(k24))) * Kokkos::sqrt(TOutput((k24 - 2.0) * (k24 + 2.0))));
        if (ql::Real(k12) < -2.0) ir12 = 10.0 * ql::Sign(1.0 - Kokkos::abs(r12));
        if (ql::Real(k14) < -2.0) ir14 = 10.0 * ql::Sign(1.0 - Kokkos::abs(r14));
        if (ql::Real(k24) < -2.0) ir24 = 10.0 * ql::Sign(1.0 - Kokkos::abs(r24));

        const TOutput q24 = r24 - TOutput(1.0) / r24;
        const TOutput q12 = TOutput(k12) - r24 * TOutput(k14);

        const TOutput a = TOutput(k34) / r24 - TOutput(k23);
        const TOutput b = TOutput(k12 * k34) - TOutput(k13) * q24 - TOutput(k14 * k23);
        const TOutput c = TOutput(k13) * q12 + r24 * TOutput(k34) - TOutput(k23);
        const TOutput d = TOutput((k12 * k34 - k13 * k24 - k14 * k23) * (k12 * k34 - k13 * k24 - k14 * k23)) -
                        TOutput(4.0) * TOutput(k13 * (k13 - k23 * (k12 - k14 * k24)) +
                                        k23 * (k23 - k24 * k34) + k34 * (k34 - k13 * k14));
        const TOutput discr = Kokkos::sqrt(d);

        Kokkos::Array<TOutput, 2> x4;
        Kokkos::Array<TOutput, 2> x1;
        Kokkos::Array<TOutput, 2> l4;
        x4[0] = TOutput(0.5) * (b - discr) / a;
        x4[1] = TOutput(0.5) * (b + discr) / a;
        if (Kokkos::abs(x4[0]) > Kokkos::abs(x4[1]))
            x4[1] = c / (a * x4[0]);
        else
            x4[0] = c / (a * x4[1]);

        const TOutput dd = -TOutput(k34) * r24 + TOutput(k23);
        Kokkos::Array<TScale, 2> ix4;
        Kokkos::Array<TScale, 2> ix1;
        ix4[0] = ql::Sign(ql::Real(dd));
        ix4[1] = -ix4[0];

        x1[0] = x4[0] / r24;
        x1[1] = x4[1] / r24;
        ix1[0] = ql::Sign(ix4[0] * ql::Real(r24));
        ix1[1] = -ix1[0];

        const TOutput cc = ql::cLn<TOutput, TMass, TScale>(ql::Real(k13), -1.0);
        l4[0] = cc + ql::cLn<TOutput, TMass, TScale>((q12 + q24 * x4[0]) / dd, ql::Real(q24 * ix4[0] / dd));
        l4[1] = cc + ql::cLn<TOutput, TMass, TScale>((q12 + q24 * x4[1]) / dd, ql::Real(q24 * ix4[1] / dd));

        res(i, 2) = res(i, 1) = TOutput(0.0);
        res(i, 0) = (
            ql::xspence<TOutput, TMass, TScale>(x4, ix4, r14, ir14) +
            ql::xspence<TOutput, TMass, TScale>(x4, ix4, TOutput(1.0) / r14, -ir14) -
            ql::xspence<TOutput, TMass, TScale>(x4, ix4, TOutput(k34 / k13), -ql::Real(k13)) -
            ql::xspence<TOutput, TMass, TScale>(x1, ix1, r12, ir12) -
            ql::xspence<TOutput, TMass, TScale>(x1, ix1, TOutput(1.0) / r12, -ir12) +
            ql::xspence<TOutput, TMass, TScale>(x1, ix1, TOutput(k23 / k13), -ql::Real(k13)) -
            TOutput{0.0, 2.0 * ql::Constants::_pi()} *
                ql::xetatilde<TOutput, TMass, TScale>(x4, ix4, TOutput(1.0) / r24, -ir24, l4)
        ) / (TOutput(m3 * m_2) * discr);
    }

    /*! 
    * Finite box with 4 non-zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN4(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {
    
        TMass tmp;
        TMass M[4];
        for (int j = 0; j < 4; j++)
            M[j] = Y[j][j];

        TMass k12 = 2.0 * Y[0][1] / Kokkos::sqrt(M[0] * M[1]);
        TMass k23 = 2.0 * Y[1][2] / Kokkos::sqrt(M[1] * M[2]);
        TMass k34 = 2.0 * Y[2][3] / Kokkos::sqrt(M[2] * M[3]);
        TMass k14 = 2.0 * Y[0][3] / Kokkos::sqrt(M[0] * M[3]);
        TMass k13 = 2.0 * Y[0][2] / Kokkos::sqrt(M[0] * M[2]);
        TMass k24 = 2.0 * Y[1][3] / Kokkos::sqrt(M[1] * M[3]);

        if (Kokkos::abs(k13) >= 2.0) { /*do nothing*/ 
        } else if (Kokkos::abs(k12) >= 2.0) {
            // 2 <-> 3
            tmp = k12;
            k12 = k13;
            k13 = tmp;
            tmp = k24;
            k24 = k34;
            k34 = tmp;
        } else if (Kokkos::abs(k14) >= 2.0) {
            tmp = k13;
            k13 = k14;
            k14 = tmp;
            tmp = k23;
            k23 = k24;
            k24 = tmp;
        } else if (Kokkos::abs(k23) >= 2.0) {
            tmp = k13;
            k13 = k23;
            k23 = tmp;
            tmp = k14;
            k14 = k24;
            k24 = tmp;
        } else if (Kokkos::abs(k24) >= 2.0) {
            tmp = k12;
            k12 = k23;
            k23 = k34;
            k34 = k14;
            k14 = tmp;
            tmp = k13;
            k13 = k24;
            k24 = tmp;
        } else if (Kokkos::abs(k34) >= 2.0) {
            tmp = k12;
            k12 = k24;
            k24 = tmp;
            tmp = k13;
            k13 = k34;
            k34 = tmp;
        }
        // else nothing found, all r_ij on the complex unit circle

        TMass kij[6];
        kij[0] = k12;
        kij[1] = k23;
        kij[2] = k34;
        kij[3] = k14;
        kij[4] = k13;
        kij[5] = k24;

        TOutput rij[6];
        TOutput r12 = rij[0] = TOutput(0.5) * (TOutput(k12) + TOutput(ql::Sign(ql::Real(k12))) * Kokkos::sqrt(TOutput((k12 - 2.0) * (k12 + 2.0))));
        TOutput r23 = rij[1] = TOutput(0.5) * (TOutput(k23) + TOutput(ql::Sign(ql::Real(k23))) * Kokkos::sqrt(TOutput((k23 - 2.0) * (k23 + 2.0))));
        TOutput r34 = rij[2] = TOutput(0.5) * (TOutput(k34) + TOutput(ql::Sign(ql::Real(k34))) * Kokkos::sqrt(TOutput((k34 - 2.0) * (k34 + 2.0))));
        TOutput r14 = rij[3] = TOutput(0.5) * (TOutput(k14) + TOutput(ql::Sign(ql::Real(k14))) * Kokkos::sqrt(TOutput((k14 - 2.0) * (k14+2.0))));
        TOutput r13 = rij[4] = TOutput(1.0) / (TOutput(0.5) * (TOutput(k13) + TOutput(ql::Sign(ql::Real(k13))) * Kokkos::sqrt(TOutput((k13 - 2.0) * (k13 + 2.0)))));
        TOutput r24 = rij[5] = TOutput(1.0) / (TOutput(0.5) * (TOutput(k24) + TOutput(ql::Sign(ql::Real(k24))) * Kokkos::sqrt(TOutput((k24 - 2.0) * (k24 + 2.0)))));

        TScale irij[6];
        for (int j = 0; j < 6; j++) {
            if (ql::Imag(rij[j]) == 0.0) {
                const TOutput ki = TOutput(kij[j]) - ql::Constants::_ieps50<TOutput, TMass, TScale>();
                const TOutput kk = TOutput(0.5) * (ki + TOutput(ql::Sign(ql::Real(ki))) * Kokkos::sqrt((ki - TOutput(2.0)) * (ki + TOutput(2.0))));
                irij[j] = ql::Sign(Kokkos::abs(rij[j]) - 1.0) * ql::Imag(kk);
            } else
                irij[j] = 0.0;
        }

        TScale ir13 = irij[4], ir24 = irij[5];
        const TScale ir1324 = ql::Sign(ql::Real(r24)) * ir13 - ql::Sign(ql::Real(r13)) * ir24;

        const TOutput a = TOutput(k34) / r24 - TOutput(k23) + (TOutput(k12) - TOutput(k14) / r24) * r13;
        const TOutput b = (TOutput(1.0) / r13 - r13) * (TOutput(1.0) / r24 - r24) + TOutput(k12 * k34) - TOutput(k14 * k23);
        const TOutput c = TOutput(k34) * r24 - TOutput(k23) + (TOutput(k12) - TOutput(k14) * r24) / r13;
        const TOutput d = TOutput(k23) + (r24 * TOutput(k14) - TOutput(k12)) * r13 - r24 * TOutput(k34);
        TOutput disc = Kokkos::sqrt(b * b - TOutput(4.0) * a * (c + d * ql::Constants::_ieps50<TOutput, TMass, TScale>()));

        TScale ix[2][4];
        ix[0][3] = ql::Imag(TOutput(0.5) / a * (b - disc));
        ix[1][3] = ql::Imag(TOutput(0.5) / a * (b + disc));

        disc = Kokkos::sqrt(b * b - TOutput(4.0) * a * c);
        TOutput x[2][4];
        x[0][3] = TOutput(0.5) / a * (b - disc);
        x[1][3] = TOutput(0.5) / a * (b + disc);
        if (Kokkos::abs(x[0][3]) > Kokkos::abs(x[1][3]))
            x[1][3] = c / (a * x[0][3]);
        else
            x[0][3] = c / (a * x[1][3]);

        x[0][0] = x[0][3] / r24;
        x[1][0] = x[1][3] / r24;
        x[0][1] = x[0][3] * r13 / r24;
        x[1][1] = x[1][3] * r13 / r24;
        x[0][2] = x[0][3] * r13;
        x[1][2] = x[1][3] * r13;

        const TScale s1 = ql::Sign(ql::Real(x[0][3]));
        const TScale s2 = ql::Sign(ql::Real(x[1][3]));
        ix[0][0] = ix[0][3] * ql::Real(x[0][0]) * s1;
        ix[1][0] = ix[1][3] * ql::Real(x[1][0]) * s2;
        ix[0][1] = ix[0][3] * ql::Real(x[0][1]) * s1;
        ix[1][1] = ix[1][3] * ql::Real(x[1][1]) * s2;
        ix[0][2] = ix[0][3] * ql::Real(x[0][2]) * s1;
        ix[1][2] = ix[1][3] * ql::Real(x[1][2]) * s2;

        res(i,0) = TOutput(0.0);
        for (int j = 0; j < 4; j++) {
            const Kokkos::Array<TOutput, 2> x_in = { x[0][j], x[1][j] }; 
            const Kokkos::Array<TScale, 2> ix_in = { ix[0][j], ix[1][j] };
            res(i,0) += ql::kPow<TOutput, TMass, TScale>(-TOutput(1.0) ,j+1) * (
                    ql::xspence<TOutput, TMass, TScale>(x_in, ix_in, rij[j], irij[j]) +
                    ql::xspence<TOutput, TMass, TScale>(x_in, ix_in, TOutput(1.0) / rij[j], -irij[j])
                    );
        }

        const TScale gamma = ql::Sign(ql::Real(a*(x[1][3] - x[0][3])) + ql::Constants::_reps());
        TOutput l[2][4];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                l[i][j] = TOutput(0.0);
        l[0][3] = ql::Constants::_2ipi<TOutput, TMass, TScale>() * TOutput(ql::eta<TOutput, TMass, TScale>(r13, ir13, TOutput(1.0) / r24, -ir24, ir1324));
        l[1][3] = l[0][3];

        TOutput etas = TOutput(0.0);
        if (ql::Imag(r13) == 0) {
            r12 = TOutput(k12) - r24 * TOutput(k14);
            r23 = TOutput(k23) - r24 * TOutput(k34);
            r34 = TOutput(k34) - r13 * TOutput(k14);
            r14 = TOutput(k23) - r13 * TOutput(k12);
            const TOutput q13 = TOutput(k13) - TOutput(2.0) * r13;
            const TOutput q24 = TOutput(k24) - TOutput(2.0) * r24;

            TScale cc = gamma * ql::Sign(ql::Imag(r24) + ir24);
            l[0][0] = ql::cLn<TOutput, TMass, TScale>(-x[0][0], -ix[0][0]) +
                ql::cLn<TOutput, TMass, TScale>(r14 - q13/x[0][0], -1.0) +
                ql::cLn<TOutput, TMass, TScale>((r12 - q24 * x[0][3])/d, cc);
            l[1][0] = ql::cLn<TOutput, TMass, TScale>(-x[1][0], -ix[1][0]) +
                ql::cLn<TOutput, TMass, TScale>(r14 - q13/x[1][0], -1.0) +
                ql::cLn<TOutput, TMass, TScale>((r12 - q24 * x[1][3])/d, -cc);

            cc = gamma * ql::Sign(ql::Real(r13) * (ql::Imag(r24) + ir24));
            l[0][1] = ql::cLn<TOutput, TMass, TScale>(-x[0][1], -ix[0][1]) +
                ql::cLn<TOutput, TMass, TScale>(r14 - q13/x[0][0], -1.0) +
                ql::cLn<TOutput, TMass, TScale>((r23 - q24 * x[0][2]) / d, cc);
            l[1][1] = ql::cLn<TOutput, TMass, TScale>(-x[1][1], -ix[1][1]) +
                ql::cLn<TOutput, TMass, TScale>(r14 - q13 / x[1][0], -1.0) +
                ql::cLn<TOutput, TMass, TScale>((r23 - q24 * x[1][2]) / d, -cc);
            l[0][2] = ql::cLn<TOutput, TMass, TScale>(-x[0][2], -ix[0][2]) +
                ql::cLn<TOutput, TMass, TScale>(r34 - q13 / x[0][3], -1.0) +
                ql::cLn<TOutput, TMass, TScale>((r23 - q24 * x[0][2]) / d, cc);
            l[1][2] = ql::cLn<TOutput, TMass, TScale>(-x[1][2], -ix[1][2]) +
                ql::cLn<TOutput, TMass, TScale>(r34 - q13 / x[1][3], -1.0) +
                ql::cLn<TOutput, TMass, TScale>((r23 - q24 * x[1][2]) / d, -cc);

            const Kokkos::Array<TOutput, 2> x_in = {x[0][3], x[1][3]}; 
            const Kokkos::Array<TScale, 2> ix_in = {ix[0][3], ix[1][3]};
            const Kokkos::Array<TOutput, 2> l_in_a = {l[0][2], l[1][2]};
            const Kokkos::Array<TOutput, 2> l_in_b = {l[0][0], l[1][0]};
            const Kokkos::Array<TOutput, 2> l_in_c = {l[0][1], l[1][1]};
            const Kokkos::Array<TOutput, 2> l_in_d = {l[0][3], l[1][3]};

            etas = ql::xetatilde<TOutput, TMass, TScale>(x_in, ix_in, r13, ir13, l_in_a) +
                ql::xetatilde<TOutput, TMass, TScale>(x_in, ix_in, TOutput(1.0) / r24, -ir24, l_in_b) -
                ql::xetatilde<TOutput, TMass, TScale>(x_in, ix_in, r13 / r24, ir1324, l_in_c) +
                ql::xetatilde<TOutput, TMass, TScale>(x_in, ix_in, -r13 / r24, -ir1324, l_in_d);
        } else {
            for (int j = 0; j < 3; j++) {
                l[0][j] = Kokkos::log(-x[0][j]) + ql::cLn<TOutput, TMass, TScale>(TOutput(kij[j]) - TOutput(1.0) / x[0][j]-x[0][j], -ql::Real(x[0][j] * b * TOutput(gamma)));
                l[1][j] = Kokkos::log(-x[1][j]) + ql::cLn<TOutput, TMass, TScale>(TOutput(kij[j]) - TOutput(1.0) / x[1][j]-x[1][j], -ql::Real(x[1][j] * b * TOutput(gamma)));
            }

            const Kokkos::Array<TOutput, 2> x_in = {x[0][3], x[1][3]}; 
            const Kokkos::Array<TScale, 2> ix_in = {ix[0][3], ix[1][3]};
            const Kokkos::Array<TOutput, 2> l_in_a = {l[0][2], l[1][2]};
            const Kokkos::Array<TOutput, 2> l_in_b = {l[0][0], l[1][0]};
            const Kokkos::Array<TOutput, 2> l_in_c = {l[0][1], l[1][1]};
            const Kokkos::Array<TOutput, 2> l_in_d = {l[0][3], l[1][3]};

            etas = ql::xeta<TOutput, TMass, TScale>(x_in, ix_in, r13, ir13, ix[0][2], l_in_a) +
                ql::xeta<TOutput, TMass, TScale>(x_in, ix_in, TOutput(1.0) / r24, -ir24, ix[0][0], l_in_b) -
                ql::xeta<TOutput, TMass, TScale>(x_in, ix_in, r13 / r24, ir1324, ix[0][1], l_in_c) +
                ql::xeta<TOutput, TMass, TScale>(x_in, ix_in, -r13 / r24, -ir1324, ix[0][3], l_in_d) *
                (TOutput(1.0) - TOutput(ql::Sign(ql::Real(b)) * gamma));
            }

        res(i,0) = (res(i,0) - ql::Constants::_2ipi<TOutput, TMass, TScale>() * TOutput(etas) +
            (l[1][1] - l[0][1]) * l[0][3]) / (TOutput(Kokkos::sqrt(M[0] * M[1] * M[2] * M[3])) * disc) ;

        res(i,2) = res(i,1) = TOutput(0.0);
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,0,0;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}}\left[ \frac{2}{\epsilon^2}\left( (-s_{12}-i\epsilon)^{-\epsilon} + (-s_{23}-i \epsilon)^{-\epsilon} \right) - \ln^2 \left( \frac{-s_{12}-i\epsilon}{-s_{23}-i\epsilon} \right) - \pi^2 \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B1(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = 2.0 * Y[0][2];
        const TMass ta = 2.0 * Y[1][3];
        const TOutput fac = TOutput(1.0) / (si * ta);

        const TOutput lnrat_tamu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
        const TOutput lnrat_simu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput lnrat_tasi = ql::Lnrat<TOutput, TMass, TScale>(ta, si);

        res(i,2) = fac * TOutput(2.0) * TOutput(2.0);  
        res(i,1) = fac * TOutput(2.0) * (-lnrat_tamu2 - lnrat_simu2);
        res(i,0) = fac * (lnrat_tamu2 * lnrat_tamu2 + lnrat_simu2 * lnrat_simu2 - lnrat_tasi * lnrat_tasi - TOutput(ql::Constants::_pi2()));
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,0,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}}\left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon} -(-p_4^2)^{-\epsilon} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) - \frac{\pi^2}{3} \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = 2.0 * Y[0][2];
        const TMass ta = 2.0 * Y[1][3];
        const TMass mp4sq = 2.0 * Y[0][3];
        const TOutput fac = TOutput(1.0) / (si * ta);

        const TOutput ln_mp4_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, mu2);
        const TOutput ln_ta_mu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
        const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput ln_mp4_ta = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, ta);
        const TOutput ln_mp4_si = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, si);
        const TOutput ln_ta_si = ql::Lnrat<TOutput, TMass, TScale>(ta, si);

        res(i,2) = fac * TOutput(2.0);
        res(i,1) = res(i,2) * (ln_mp4_mu2 - ln_ta_mu2 - ln_si_mu2);
        res(i,0) = fac * (-ln_mp4_mu2 * ln_mp4_mu2 + ln_ta_mu2 * ln_ta_mu2 + ln_si_mu2 * ln_si_mu2
                    + TOutput(2.0) * (ql::Li2omrat<TOutput, TMass, TScale>(ta, mp4sq) + ql::Li2omrat<TOutput, TMass, TScale>(si, mp4sq) - TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()))
                    + ln_mp4_ta * ln_mp4_ta + ln_mp4_si * ln_mp4_si - ln_ta_si * ln_ta_si);
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,p_2^2,0,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}-p_4^2 p_2^2} \\
    *    \left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon}-(-p_2^2)^{-\epsilon}-(-p_4^2)^{-\epsilon} \right) \\
    * - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{23}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) \\
    * + 2 {\rm Li}_2 \left( 1-\frac{p_4^2 p_2^2}{s_{23}s_{12}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B3(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = 2.0 * Y[0][2];
        const TMass ta = 2.0 * Y[1][3];
        const TMass mp4sq = 2.0 * Y[0][3];
        const TMass mp2sq = 2.0 * Y[1][2];
        const TMass r = 1.0 - mp2sq * mp4sq / (si * ta);
        const TScale SignRealsi = ql::Sign(ql::Real(si));
        const TScale SignRealmp2sq = ql::Sign(ql::Real(mp2sq));

        // Use expansion only in cases where signs are not ++-- or --++
        const bool landau = ((SignRealsi == ql::Sign(ql::Real(ta))) &&
                            (SignRealmp2sq == ql::Sign(ql::Real(mp4sq))) &&
                            (SignRealsi != SignRealmp2sq));

        if (Kokkos::abs(r) < ql::Constants::_eps() && landau == false) {
            // Expanded case
            const TOutput fac = TOutput(1.0) / (si * ta);
            const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
            const TOutput ln_ta_mp4 = ql::Lnrat<TOutput, TMass, TScale>(ta, mp4sq);
            const TOutput l0_mp4_ta = ql::L0<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput l0_mp4_si = ql::L0<TOutput, TMass, TScale>(mp4sq, si);
            const TOutput l1_mp4_ta = ql::L1<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput l1_mp4_si = ql::L1<TOutput, TMass, TScale>(mp4sq, si);

            res(i,2) = TOutput(0.0);
            res(i,1) = -(TOutput(2.0) + r) * fac;
            res(i,0) = fac * ( TOutput(2.0) - TOutput(0.5) * r + (TOutput(2.0) + r) * (ln_si_mu2 + ln_ta_mp4)
                        + TOutput(2.0) * (l0_mp4_ta + l0_mp4_si) + r * (l1_mp4_ta + l1_mp4_si));
        } else {
            // General case
            const TOutput fac = TOutput(1.0) / (si * ta - mp2sq * mp4sq);
            const TOutput ln_mp2_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp2sq, mu2);
            const TOutput ln_mp4_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, mu2);
            const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
            const TOutput ln_ta_mu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
            const TOutput ln_s_t = ql::Lnrat<TOutput, TMass, TScale>(si, ta);
            const TOutput li2_1 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, si);
            const TOutput li2_2 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, ta);
            const TOutput li2_3 = ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, si);
            const TOutput li2_4 = ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput li2_5 = ql::Li2omx2<TOutput, TMass, TScale>(mp2sq, mp4sq, si, ta);

            res(i,2) = TOutput(0.0);
            res(i,1) = fac * TOutput(2.0) * (ql::Lnrat<TOutput, TMass, TScale>(mp2sq, si) + ql::Lnrat<TOutput, TMass, TScale>(mp4sq, ta));
            res(i,0) = fac * (ln_si_mu2 * ln_si_mu2 + ln_ta_mu2 * ln_ta_mu2
                        - ln_mp2_mu2 * ln_mp2_mu2 - ln_mp4_mu2 * ln_mp4_mu2
                        + TOutput(2.0) * (li2_5 - li2_1 - li2_2 - li2_3 - li2_4 - TOutput(0.5) * ln_s_t * ln_s_t));
        }
    }


    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,p_3^2,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}} \\
    *     \left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon}-(-p_3^2)^{-\epsilon}-(-p_4^2)^{-\epsilon} \right) + \frac{1}{\epsilon^2} \left( (-p_3^2)^{-\epsilon}(-p_4)^{-\epsilon} \right) / (-s_{12})^{-\epsilon} \\
    * - 2 {\rm Li}_2 \left( 1-\frac{p_3^2}{s_{23}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B4(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = 2.0 * Y[0][2];
        const TMass ta = 2.0 * Y[1][3];
        const TMass mp4sq = 2.0 * Y[0][3];
        const TMass mp3sq = 2.0 * Y[2][3];
        const TOutput fac = TOutput(1.0) / (si * ta);

        const TOutput ln_ta_mu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
        const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput ln_mp3_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp3sq, mu2);
        const TOutput ln_mp4_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, mu2);
        const TOutput ln_si_ta = ql::Lnrat<TOutput, TMass, TScale>(si, ta);
        const TOutput ln_si_mp3 = ql::Lnrat<TOutput, TMass, TScale>(si, mp3sq);

        res(i,2) = fac;
        res(i,1) = -fac * (ln_si_mp3 + ql::Lnrat<TOutput, TMass, TScale>(ta, mp4sq) + ln_ta_mu2);
        res(i,0) = fac * (ln_ta_mu2 * ln_ta_mu2
                    + TOutput(0.5) * ln_si_mu2 * ln_si_mu2
                    - TOutput(0.5) * ln_mp3_mu2 * ln_mp3_mu2
                    - TOutput(0.5) * ln_mp4_mu2 * ln_mp4_mu2
                    + TOutput(2.0) * (-ql::Li2omrat<TOutput, TMass, TScale>(mp3sq, ta) - ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, ta) +
                                    TOutput(0.5) * (ln_si_mp3 * ql::Lnrat<TOutput, TMass, TScale>(si, mp4sq) - ln_si_ta * ln_si_ta)));
    }


    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,p_2^2,p_3^2,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}-p_2^2 p_4^2} \\
    *     \left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon}-(-p_2^2)^{-\epsilon}-(-p_3^2)^{-\epsilon} -(-p_4^2)^{-\epsilon} \right) \\
    *  + \frac{1}{\epsilon^2} \left( (-p_2^2)^{-\epsilon}(-p_3)^{-\epsilon} \right) / (-s_{23})^{-\epsilon}+ \frac{1}{\epsilon^2} \left( (-p_3^2)^{-\epsilon}(-p_4)^{-\epsilon} \right) / (-s_{12})^{-\epsilon} \\
    * - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) + 2 {\rm Li}_2 \left( 1-\frac{p_2^2 p_4^2}{s_{12} s_{23}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B5(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = 2.0 * Y[0][2];
        const TMass ta = 2.0 * Y[1][3];
        const TMass mp2sq = 2.0 * Y[1][2];
        const TMass mp3sq = 2.0 * Y[2][3];
        const TMass mp4sq = 2.0 * Y[0][3];
        const TMass r = 1.0 - mp2sq * mp4sq / (si * ta);
        const TScale SignRealsi = ql::Sign(ql::Real(si));
        const TScale SignRealmp2sq = ql::Sign(ql::Real(mp2sq));

        // Use expansion only in cases where signs are not ++-- or --++
        const bool landau = ((SignRealsi == ql::Sign(ql::Real(ta))) &&
                            (SignRealmp2sq == ql::Sign(ql::Real(mp4sq))) &&
                            (SignRealsi != SignRealmp2sq));

        if (Kokkos::abs(r) < ql::Constants::_eps() && landau == false) {
            const TOutput l0_mp4_ta = ql::L0<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput l1_mp4_ta = ql::L1<TOutput, TMass, TScale>(mp4sq, ta);

            // Expanded case
            res(i,2) = TOutput(0.0);
            res(i,1) = -(TOutput(1.0) + TOutput(0.5) * r) / (si * ta);
            res(i,0) = res(i,1) * (ql::Lnrat<TOutput, TMass, TScale>(mu2, si) +
                                ql::Lnrat<TOutput, TMass, TScale>(mp3sq, ta) - TOutput(2.0) -
                                (TOutput(1.0) + mp4sq / ta) * l0_mp4_ta) +
                    (r / (si * ta)) * (l1_mp4_ta - l0_mp4_ta - TOutput(1.0));
        } else {
            // General case
            const TOutput fac = TOutput(1.0) / (si * ta - mp2sq * mp4sq);
            const TOutput li2_1 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, si);
            const TOutput li2_2 = ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput li2_3 = ql::Li2omx2<TOutput, TMass, TScale>(mp2sq, mp4sq, si, ta);
            const TOutput ln_ta_mp2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mp2sq);
            const TOutput ln_si_mp4 = ql::Lnrat<TOutput, TMass, TScale>(si, mp4sq);
            const TOutput ln_si_ta = ql::Lnrat<TOutput, TMass, TScale>(si, ta);

            res(i,2) = TOutput(0.0);
            res(i,1) = -ln_ta_mp2 - ln_si_mp4;
            res(i,0) = -TOutput(0.5) * (ln_ta_mp2 * ln_ta_mp2 + ln_si_mp4 * ln_si_mp4) -
                    (ql::Lnrat<TOutput, TMass, TScale>(mp3sq, ta) +
                        ql::Lnrat<TOutput, TMass, TScale>(mu2, ta)) * ln_ta_mp2 -
                    (ql::Lnrat<TOutput, TMass, TScale>(mp3sq, si) +
                        ql::Lnrat<TOutput, TMass, TScale>(mu2, si)) * ln_si_mp4 -
                    TOutput(2.0) * (li2_1 + li2_2 - li2_3) - ln_si_ta * ln_si_ta;

            res(i,1) *= fac;
            res(i,0) *= fac;
        }
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,m^2,m^2;s_{12},s_{23};0,0,0,m^2) = -\frac{1}{s_{12} (m^2-s_{23})} \left( \frac{\mu^2}{m^2} \right)^\epsilon \\
    *     \left[ \frac{2}{\epsilon^2} - \frac{1}{\epsilon} \left( 2 \ln \left( \frac{m^2-s_{23}}{m^2} \right) + \ln \left( \frac{-s_{12}}{m^2} \right) \right) + 2 \ln \left( \frac{m^2-s_{23}}{m^2} \right) \ln \left( \frac{-s_{12}}{m^2} \right) - \frac{\pi^2}{2} \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988bq.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B6(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = 2.0 * Y[0][2];
        const TMass tabar = 2.0 * Y[1][3];
        const TMass msq = Y[3][3];
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(si, msq);
        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, msq);
        const TOutput wlogm = ql::Lnrat<TOutput, TMass, TScale>(mu2, msq);

        res(i,2) = TOutput(2.0);  
        res(i,1) = TOutput(2.0) * (wlogm - wlogt) - wlogs;
        res(i,0) = wlogm * wlogm - wlogm * (TOutput(2.0) * wlogt + wlogs) 
                + TOutput(2.0) * wlogt * wlogs - TOutput(0.5) * ql::Constants::_pi2();

        const TOutput d = TOutput(si * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,m^2,p_2^2;s_{12},s_{23};0,0,0,m^2) = \left( \frac{\mu^2}{m^2} \right)^\epsilon \frac{1}{s_{12} (s_{23}-m^2)} \\
    *     \left[ \frac{3}{2 \epsilon^2} - \frac{1}{\epsilon} \left\{ 2 \ln \left( 1-\frac{s_{23}}{m^2} \right) + \ln \left( \frac{-s_{12}}{m^2} \right) - \ln \left( 1-\frac{p_4^2}{m^2} \right) \right\} \\
    *   -2 {\rm Li}_2 \left( 1 - \frac{m^2-p_4^2}{m^2-s_{23}} \right) + 2 \ln \left( \frac{-s_{12}}{m^2} \right) \ln \left( 1-\frac{s_{23}}{m^2} \right) - \ln^2 \left( 1 - \frac{p_4^2}{m^2} \right) -\frac{5\pi^2}{12} \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988bq.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B7(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {
    
        const TMass tabar = 2.0 * Y[1][3];
        const TMass p4sqbar = 2.0 * Y[0][3];
        const TMass si = 2.0 * Y[0][2];
        const TMass msq = Y[3][3];
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(si, msq);
        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, msq);
        const TOutput wlogp = ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, msq);
        const TOutput wlogm = ql::Lnrat<TOutput, TMass, TScale>(mu2, msq);
    
        res(i,2) = TOutput(1.5);  // Equivalent to this->_three/2.0
        res(i,1) = TOutput(1.5) * wlogm - TOutput(2.0) * wlogt - wlogs + wlogp;
        res(i,0) = TOutput(2.0) * wlogs * wlogt - wlogp * wlogp - TOutput(ql::Constants::_pi2o12<TOutput, TMass, TScale>() * 5.0)
                 + TOutput(0.75) * wlogm * wlogm + wlogm * (-TOutput(2.0) * wlogt - wlogs + wlogp)
                 - 2.0 * ql::Li2omrat<TOutput, TMass, TScale>(p4sqbar, tabar);
    
        const TOutput d = TOutput(si * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }



    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,p_3^2,p_4^2;s_{12},s_{23};0,0,0,m^2) = \frac{1}{s_{12} (s_{23}-m^2)} \left[ \frac{1}{\epsilon^2} - \frac{1}{\epsilon} \left[ \ln \frac{-s_{12}}{\mu^2} + \ln \frac{(m^2-s_{23}^2)}{(m^2-p_3^2)(m^2-p_4^2)} \right] \\
    *   - 2 {\rm Li}_2 \left( 1 - \frac{m^2-p_3^2}{m^2-s_{23}} \right) - 2 {\rm Li}_2 \left( 1-\frac{m^2-p_4^2}{m^2-s_{23}} \right) - {\rm Li}_2 \left( 1 + \frac{(m^2-p_3^2)(m^2-p_4^2)}{s_{12} m^2} \right) \\
    *   - \frac{\pi^2}{6} + \frac{1}{2} \ln^2 \left( \frac{-s_{12}}{\mu^2} \right) - \frac{1}{2} \ln^2 \left( \frac{-s_{12}}{m^2} \right) + 2 \ln \left( \frac{-s_{12}}{\mu^2} \right) \ln \left( \frac{m^2-s_{23}}{m^2} \right) \\
    *   - \ln \left( \frac{m^2-p_3^2}{\mu^2} \right) \ln \left( \frac{m^2-p_3^2}{m^2} \right) - \ln \left( \frac{m^2-p_4^2}{\mu^2} \right) \ln \left( \frac{m^2-p_4^2}{m^2} \right)  \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988bq.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B8(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass msq = Y[3][3];
        const TMass tabar = 2.0 * Y[1][3];
        const TMass si = 2.0 * Y[0][2];
        const TMass p3sqbar = 2.0 * Y[2][3];
        const TMass p4sqbar = 2.0 * Y[0][3];
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput wlogp3 = ql::Lnrat<TOutput, TMass, TScale>(p3sqbar, tabar);
        const TOutput wlogp4 = ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, tabar);

        const TOutput dilog3 = ql::Li2omrat<TOutput, TMass, TScale>(p3sqbar, tabar);
        const TOutput dilog4 = ql::Li2omrat<TOutput, TMass, TScale>(p4sqbar, tabar);
        const TOutput dilog34 = ql::Li2omx2<TOutput, TMass, TScale>(p3sqbar, p4sqbar, si, msq);
        const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);

        res(i,2) = TOutput(1.0);
        res(i,1) = wlogp3 + wlogp4 - wlogs;
        res(i,0) = -TOutput(2.0) * dilog3 - TOutput(2.0) * dilog4 - dilog34
                - TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) + TOutput(0.5) * (ln_si_mu2 * ln_si_mu2 - ql::kPow<TOutput, TMass, TScale>(ql::Lnrat<TOutput, TMass, TScale>(si, msq), 2))
                + TOutput(2.0) * ln_si_mu2 * ql::Lnrat<TOutput, TMass, TScale>(tabar, msq)
                - ql::Lnrat<TOutput, TMass, TScale>(p3sqbar, mu2) * ql::Lnrat<TOutput, TMass, TScale>(p3sqbar, msq)
                - ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, mu2) * ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, msq);

        const TOutput d = TOutput(si * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,p_2^2,p_3^2,m^2;s_{12},s_{23};0,0,0,m^2) = \frac{1}{s_{12} (s_{23}-m^2)} \left[ \frac{1}{2 \epsilon^2} - \frac{1}{\epsilon} \left( \frac{s_{12} (m^2-s_{23})}{p_2^2 \mu m} \right) \\
    *  + {\rm Li}_2 \left(1+\frac{(m^2-p_3^2)(m^2-s_{23})}{m^2 p_2^2} \right) + 2 {\rm Li}_2 \left( 1-\frac{s_{12}}{p_2^2} \right) + \frac{\pi^2}{12} + \ln^2 \left( \frac{s_{12}(m^2-s_{23})}{p_2^2 \mu m} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B9(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass msq = Y[3][3];
        const TMass mean = Kokkos::sqrt(TMass(mu2 * msq));
        const TMass tabar = 2.0 * Y[1][3];
        const TMass si = 2.0 * Y[0][2];
        const TMass m3sqbar = 2.0 * Y[2][3];
        const TMass mp2sq = 2.0 * Y[1][2];
        const TOutput fac = TOutput(si * tabar);

        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, mean);
        const TOutput wlog2 = ql::Lnrat<TOutput, TMass, TScale>(si, mp2sq);

        const TOutput dilog1 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, tabar, mp2sq, msq);
        const TOutput dilog2 = ql::Li2omrat<TOutput, TMass, TScale>(si, mp2sq);

        res(i,2) = TOutput(0.5);
        res(i,1) = -wlogt - wlog2;
        res(i,0) = dilog1 + 2.0 * dilog2 + TOutput(ql::Constants::_pi2o12<TOutput, TMass, TScale>()) + (wlogt + wlog2) * (wlogt + wlog2);

        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= fac;
        }
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,p_2^2,p_3^2,p_4^2;s_{12},s_{23};0,0,0,m^2) = \frac{1}{(s_{12}s_{23}-m^2 s_{12} - p_2^2 p_4^2 + m^2 p_2^2)} \\
    *  \left[ \frac{1}{\epsilon} \ln \left( \frac{(m^2-p_4^2) p_2^2}{(m^2-s_{23})s_{12})} \right) + {\rm Li}_2 \left( 1 + \frac{(m^2-p_3^2)(m^2-s_{23})}{p_2^2 m^2} \right) - {\rm Li}_2 \left( 1 + \frac{(m^2-p_3^2)(m^2-p_4^2)}{s_{12} m^2} \right)  \\
    *   +2 {\rm Li}_2 \left( \right) - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{12}} \right) + 2 {\rm Li}_2 \left( 1-\frac{p_2 (m^2-p_4^2)}{s_{12}(m^2-s_{23})} \right) \\
    *   +2 \ln \left( \frac{\mu m}{m^2-s_{23}} \right) \ln \left( \frac{(m^2-p_4^2) p_2^2}{(m^2-s_{23}) s_{12}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B10(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass msq = Y[3][3];
        const TMass si = 2.0 * Y[0][2];
        const TMass tabar = 2.0 * Y[1][3];
        const TMass m4sqbar = 2.0 * Y[0][3];
        const TMass m3sqbar = 2.0 * Y[2][3];
        const TMass mp2sq = 2.0 * Y[1][2];
        const TMass mean = Kokkos::sqrt(mu2 * msq);

        const TOutput fac = TOutput(si * tabar - mp2sq * m4sqbar);
        const TOutput wlogsmu = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(tabar, mu2);
        const TOutput wlog2mu = ql::Lnrat<TOutput, TMass, TScale>(mp2sq, mu2);
        const TOutput wlog4mu = ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, mu2);

        const TOutput dilog1 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, si);
        const TOutput dilog2 = ql::Li2omrat<TOutput, TMass, TScale>(tabar, m4sqbar);
        const TOutput dilog3 = ql::Li2omx2<TOutput, TMass, TScale>(mp2sq, m4sqbar, si, tabar);
        const TOutput dilog4 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, tabar, mp2sq, msq);
        const TOutput dilog5 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, m4sqbar, si, msq);

        res(i,2) = TOutput(0.0);
        res(i,1) = wlog2mu + wlog4mu - wlogsmu - wlogtmu;
        res(i,0) = dilog4 - dilog5
            - 2.0 * dilog1 + 2.0 * dilog2 + 2.0 * dilog3
            + 2.0 * res(i,1) * ql::Lnrat<TOutput, TMass, TScale>(mean, tabar);
           
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= fac;
        }
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,m_3^2,p_3^2,m_4^2;s_{12},s_{23};0,0,m_3^2,m_4^2) = \frac{1}{(m_3^2-s_{12})(m_4^2 - s_{23})} \\
    *  \left[ \frac{1}{\epsilon^2} - \frac{1}{\epsilon} \ln \left( \frac{(m^2-s_{23})(m_3^2-s_{12})}{m_3 m_4 \mu^2} \right) + 2 \ln \left( \frac{m_3^2-s_{12}}{m_3 \mu} \right) \ln \left( \frac{m_4^2-s_{23}}{m_4 \mu} \right)  \\
    *  - \frac{\pi^2}{2} + \ln^2 \frac{m_3}{m_4} - \frac{1}{2} \ln^2 \left( \frac{\gamma^+_{34}}{\gamma^+_{34} - 1} \right) - \frac{1}{2} \ln \left( \frac{\gamma^-_{34}}{\gamma^-_{34} - 1} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B11(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass sibar = 2.0 * Y[0][2];
        const TMass tabar = 2.0 * Y[1][3];
        const TMass p3sq = -(2.0 * Y[2][3] - Y[2][2] - Y[3][3]);
        const TMass m3mu = Kokkos::sqrt(m3sq * mu2);
        const TMass m4mu = Kokkos::sqrt(m4sq * mu2);

        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, m4mu);
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(sibar, m3mu);

        TOutput root;
        TMass x43p, x43pm1, x43m, x43mm1;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            root = TOutput(1.0);
            x43p = -1.0;
            x43pm1 = -1.0;
            x43m = m3sq;
            x43mm1 = m4sq;
        } else {
            root = Kokkos::sqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq + m3sq - m4sq, 2) - 4.0 * m3sq * p3sq));
            const TOutput ga43p   = TOutput(+p3sq + m3sq - m4sq) + root;
            const TOutput ga43pm1 = TOutput(-p3sq + m3sq - m4sq) + root;
            const TOutput ga43m   = TOutput(+p3sq + m3sq - m4sq) - root;
            const TOutput ga43mm1 = TOutput(-p3sq + m3sq - m4sq) - root;

            x43p = -ql::Real(ga43p);
            x43pm1 = -ql::Real(ga43pm1);
            x43m = ql::Real(ga43m);
            x43mm1 = ql::Real(ga43mm1);
        }

        // deal with real roots
        TOutput ln43p, ln43m, rat2p, rat2m;
        TScale ieps2;
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(root))) {
            ln43p = ql::Lnrat<TOutput, TMass, TScale>(x43p, x43pm1);
            ln43m = ql::Lnrat<TOutput, TMass, TScale>(x43m, x43mm1);
        } else {
            ql::ratgam<TOutput, TMass, TScale>(rat2p, rat2m, ieps2, p3sq, m4sq, m3sq);
            ln43p = ql::cLn<TOutput, TMass, TScale>(rat2p, ieps2);
            ln43m = ql::cLn<TOutput, TMass, TScale>(rat2m, ieps2);
        }

        TOutput intbit;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            intbit = -TOutput(0.5) * TOutput(ql::kPow<TOutput, TMass, TScale>(Kokkos::log(m3sq / m4sq), 2));
        } else {
            intbit = -TOutput(0.5) * (ql::kPow<TOutput, TMass, TScale>(ln43p, 2) + ql::kPow<TOutput, TMass, TScale>(ln43m, 2));
        }

        res(i,2) = TOutput(1.0);
        res(i,1) = -wlogt - wlogs;
        res(i,0) = intbit
            + TOutput(2.0) * wlogt * wlogs - TOutput(0.5 * ql::Constants::_pi2())
            + TOutput(ql::kPow<TOutput, TMass, TScale>(Kokkos::log(m3sq / m4sq), 2) / 4.0);

        const TOutput d = TOutput(sibar * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,m_3^2,p_3^2,p_4^2;s_{12},s_{23};0,0,m_3^2,m_4^2) = \frac{1}{(s_{12}-m_3^2)(s_{23}-m_4^2)} \\
    *  \left[ \frac{1}{2 \epsilon^2} - \frac{1}{\epsilon} \ln \left( \frac{(m_4^2-s_{23})(m_3^2-s_{12})}{(m_4-p_4^2) m_3 \mu} \right) + 2 \ln \left( \frac{m_4^2-s_{23}}{m_3 \mu} \right) \ln \left( \frac{m_3^2-s_{12}}{m_3 \mu} \right)  \\
    *  - \ln^2 \left( \frac{m_4^2-p_4^2}{m_3 \mu}\right) -\frac{\pi^2}{12} + \ln \left( \frac{m_4^2-p_4^2}{m_3^2-s_{12}} \right) \ln \left( \frac{m_4^2}{m_3^2} \right) - \frac{1}{2} \ln^2 \left( \frac{\gamma^+_{34}}{ \gamma^+_{34}-1 }\right) - \frac{1}{2} \ln^2 \left( \frac{\gamma^-_{34}}{\gamma^-_{34}-1} \right) \\
    *  - 2 {\rm Li}_2 \left( 1 - \frac{(m_4^2-p_4^2)}{(m_4^2-s_{23})} \right) - {\rm Li}_2 \left( 1 - \frac{(m_4-p_4^2) \gamma^+_{43}}{(m_3^2-s_{12})(\gamma^+_{43}-1)} \right)- {\rm Li}_2 \left( 1 - \frac{(m_4-p_4^2) \gamma^-_{43}}{(m_3^2-s_{12})(\gamma^-_{43}-1)} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B12(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass sibar = 2.0 * Y[0][2];
        const TMass tabar = 2.0 * Y[1][3];
        const TMass m4sqbar = 2.0 * Y[0][3];
        const TMass p3sq = -(2.0 * Y[2][3] - Y[2][2] - Y[3][3]);

        const TMass mean = Kokkos::sqrt(mu2 * ql::Real(m3sq));
        const TOutput fac = TOutput(sibar * tabar);

        const TOutput wlogsmu = ql::Lnrat<TOutput, TMass, TScale>(sibar, mean);
        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(tabar, mean);
        const TOutput wlog4mu = ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, mean);
        const TOutput wlog = wlogsmu + wlogtmu - wlog4mu;

        TOutput root;
        TMass x43p, x43pm1, x43m, x43mm1;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            root = TOutput(1.0);
            x43p = -1.0;
            x43pm1 = -1.0;
            x43m = m3sq;
            x43mm1 = m4sq;
        } else {
            root = Kokkos::sqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq + m3sq - m4sq, 2) - 4.0 * m3sq * p3sq));
            const TOutput ga43p   = TOutput(+p3sq + m3sq - m4sq) + root;
            const TOutput ga43pm1 = TOutput(-p3sq + m3sq - m4sq) + root;
            const TOutput ga43m   = TOutput(+p3sq + m3sq - m4sq) - root;
            const TOutput ga43mm1 = TOutput(-p3sq + m3sq - m4sq) - root;

            x43p = -ql::Real(ga43p);
            x43pm1 = -ql::Real(ga43pm1);
            x43m = ql::Real(ga43m);
            x43mm1 = ql::Real(ga43mm1);
        }

        const TOutput dilog1 = ql::Li2omrat<TOutput, TMass, TScale>(m4sqbar, tabar);
        TOutput dilog2, dilog3;

        // deal with real roots
        TScale ieps2, ieps1;
        TMass rat1;
        TOutput ln43p, ln43m, rat2p, rat2m, zrat1;
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(root))) {
            ln43p = ql::Lnrat<TOutput, TMass, TScale>(x43p, x43pm1);
            ln43m = ql::Lnrat<TOutput, TMass, TScale>(x43m, x43mm1);
            dilog2 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43p, sibar, x43pm1);
            dilog3 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43m, sibar, x43mm1);
        } else {
            ql::ratreal<TOutput, TMass, TScale>(m4sqbar, sibar, rat1, ieps1);
            ql::ratgam<TOutput, TMass, TScale>(rat2p, rat2m, ieps2, p3sq, m4sq, m3sq);
            zrat1 = TOutput(rat1);
            ln43p = ql::cLn<TOutput, TMass, TScale>(rat2p, ieps2);
            ln43m = ql::cLn<TOutput, TMass, TScale>(rat2m, ieps2);

            dilog2 = ql::spencer<TOutput, TMass, TScale>(zrat1, rat2p, ieps1, ieps2);
            dilog3 = ql::spencer<TOutput, TMass, TScale>(zrat1, rat2m, ieps1, ieps2);
        }

        res(i,2) = TOutput(0.5);
        res(i,1) = -wlog;
        res(i,0) = -TOutput(ql::Constants::_pi2o12<TOutput, TMass, TScale>())
            + TOutput(2.0) * wlogsmu * wlogtmu - wlog4mu * wlog4mu
            + (wlog4mu - wlogsmu) * Kokkos::log(m4sq / m3sq) - TOutput(0.5) * (ln43p * ln43p + ln43m * ln43m)
            - TOutput(2.0) * dilog1 - dilog2 - dilog3;

        for (size_t j = 0; j < 3; j++)
            res(i,j) /= fac;
    }


    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,p_2^2,p_3^2,p_4^2;s_{12},s_{23};0,0,m_3^2,m_4^2) = \frac{1}{\Delta} \left[ \frac{1}{\epsilon} \ln \left( \frac{(m_3^2-p_2^2)(m_4^2-p_4^2)}{(m_3^2-s_{12})(m_4^2-s_{23})} \right) \\
    * - 2 {\rm Li}_2 \left(1-\frac{(m_3^2-p_2^2)}{(m_3^2-s_{12})} \right) - {\rm Li}_2 \left( 1 - \frac{(m_3^2-p_2^2)\gamma^+_{34}}{(m_4^2-s_{23})(\gamma_{34}^+ - 1)} \right) - {\rm Li}_2 \left( 1 - \frac{(m_3^2-p_2^2)\gamma^-_{34}}{(m_4^2-s_{23})(\gamma_{34}^- - 1)} \right) \\
    * - 2 {\rm Li}_2 \left(1-\frac{(m_4^2-p_4^2)}{(m_4^2-s_{23})} \right) - {\rm Li}_2 \left( 1 - \frac{(m_4^2-p_4^2)\gamma^+_{43}}{(m_3^2-s_{12})(\gamma_{43}^+ - 1)} \right) - {\rm Li}_2 \left( 1 - \frac{(m_4^2-p_4^2)\gamma^-_{43}}{(m_2^2-s_{12})(\gamma_{43}^- - 1)} \right) \\
    * + 2 {\rm Li}_2 \left(1-\frac{(m_3^2-p_2^2)(m_4^2-p_4^2)}{(m_3^2-s_{12})(m_4^2-s_{23})} \right) + 2 \ln \left( \frac{m_3^2-s_{12}}{\mu^2} \right) \ln \left( \frac{m_4^2-s_{23}}{\mu^2} \right) \\
    * - \ln^2 \left( \frac{m_3^2-p_2^2}{\mu^2} \right) -ln^2 \left( \frac{m_4^2-p_4^2}{\mu^2} \right) + \ln \left( \frac{m_3^2-p_2^2}{m_4^2-s_{23}} \right) \ln \left( \frac{m_3^2}{\mu^2} \right) + \ln \left( \frac{m_4^2-p_4^2}{m_3^2 - s_{12}} \right) \ln \left( \frac{m_4^2}{\mu^2} \right) \\
    * -\frac{1}{2} \ln^2 \left( \frac{\gamma_{34}^+}{\gamma_{34}^+-1} \right) -\frac{1}{2} \ln^2 \left( \frac{\gamma_{34}^-}{\gamma_{34}^--1} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B13(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {
            
        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass sibar = 2.0 * Y[0][2];
        const TMass tabar = 2.0 * Y[1][3];
        const TMass m4sqbar = 2.0 * Y[0][3];
        const TMass m3sqbar = 2.0 * Y[1][2];
        const TMass p3sq = -(2.0 * Y[2][3] - Y[2][2] - Y[3][3]);

        const TOutput fac = TOutput(sibar * tabar - m3sqbar * m4sqbar);
        const TOutput wlogsmu = ql::Lnrat<TOutput, TMass, TScale>(sibar, TMass(mu2));
        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(tabar, TMass(mu2));
        const TOutput wlog3mu = ql::Lnrat<TOutput, TMass, TScale>(m3sqbar, TMass(mu2));
        const TOutput wlog4mu = ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, TMass(mu2));
        const TOutput dilog1 = ql::Li2omrat<TOutput, TMass, TScale>(m3sqbar, sibar);
        const TOutput dilog4 = ql::Li2omrat<TOutput, TMass, TScale>(m4sqbar, tabar);
        const TOutput dilog7 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, m4sqbar, sibar, tabar);

        TOutput root, ga34p, ga34pm1, ga34m, ga34mm1;
        TOutput ga43p, ga43pm1, ga43m, ga43mm1;
        TMass x34p, x34pm1, x34m, x34mm1;
        TMass x43p, x43pm1, x43m, x43mm1;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            root = TOutput(1.0);
            x34p = -1.0;
            x34pm1 = -1.0;
            x34m = m4sq;
            x34mm1 = m3sq;

            x43p = m3sq;
            x43pm1 = m4sq;
            x43m = -1.0;
            x43mm1 = -1.0;
        } else {
            root = Kokkos::sqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq-m3sq+m4sq, 2) - 4.0 * m4sq * p3sq));
            ga34p   = TOutput(+p3sq + m4sq - m3sq) + root;
            ga34pm1 = TOutput(-p3sq + m4sq - m3sq) + root;
            ga34m   = TOutput(+p3sq + m4sq - m3sq) - root;
            ga34mm1 = TOutput(-p3sq + m4sq - m3sq) - root;

            ga43p   = TOutput(+p3sq + m3sq - m4sq) + root;
            ga43pm1 = TOutput(-p3sq + m3sq - m4sq) + root;
            ga43m   = TOutput(+p3sq + m3sq - m4sq) - root;
            ga43mm1 = TOutput(-p3sq + m3sq - m4sq) - root;

            x34p = -ql::Real(ga34p);
            x34pm1 = -ql::Real(ga34pm1);
            x34m = ql::Real(ga34m);
            x34mm1 = ql::Real(ga34mm1);

            x43p = -ql::Real(ga43p);
            x43pm1 = -ql::Real(ga43pm1);
            x43m = ql::Real(ga43m);
            x43mm1 = ql::Real(ga43mm1);
        }

        TMass rat3t, rat4s;
        TOutput ln43p, ln43m, dilog2, dilog3, dilog5, dilog6;
        TOutput zrat3t, zrat4s, rat34p, rat34m, rat43p, rat43m;
        TScale ieps3t, ieps4s, ieps34, ieps43;

        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(root))) {
            ln43p = ql::Lnrat<TOutput, TMass, TScale>(x43p, x43pm1);
            ln43m = ql::Lnrat<TOutput, TMass, TScale>(x43m, x43mm1);

            dilog2 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, x34p, tabar, x34pm1);
            dilog3 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, x34m, tabar, x34mm1);
            dilog5 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43p, sibar, x43pm1);
            dilog6 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43m, sibar, x43mm1);
        } else {
            ql::ratreal<TOutput, TMass, TScale>(m3sqbar, tabar, rat3t, ieps3t);
            ql::ratreal<TOutput, TMass, TScale>(m4sqbar, sibar, rat4s, ieps4s);

            ql::ratgam<TOutput, TMass, TScale>(rat34p, rat34m, ieps34, p3sq, m3sq, m4sq);
            ql::ratgam<TOutput, TMass, TScale>(rat43p, rat43m, ieps43, p3sq, m4sq, m3sq);

            zrat3t = TOutput(rat3t);
            zrat4s = TOutput(rat4s);

            dilog2 = ql::spencer<TOutput, TMass, TScale>(zrat3t, rat34p, ieps3t, ieps34);
            dilog3 = ql::spencer<TOutput, TMass, TScale>(zrat3t, rat34m, ieps3t, ieps34);
            dilog5 = ql::spencer<TOutput, TMass, TScale>(zrat4s, rat43p, ieps4s, ieps43);
            dilog6 = ql::spencer<TOutput, TMass, TScale>(zrat4s, rat43m, ieps4s, ieps43);

            ln43p = ql::cLn<TOutput, TMass, TScale>(rat43p, 0.0);
            ln43m = ql::cLn<TOutput, TMass, TScale>(rat43m, 0.0);
        }


        res(i,2) = TOutput(0.0);
        res(i,1) = wlog3mu + wlog4mu - wlogsmu - wlogtmu;
        res(i,0) = -TOutput(2.0) * dilog1 - dilog2 - dilog3
                 -TOutput(2.0) * dilog4 - dilog5 - dilog6
                 +TOutput(2.0) * dilog7
                 +TOutput(2.0) * wlogsmu * wlogtmu - wlog3mu * wlog3mu - wlog4mu * wlog4mu
                 +(wlog3mu - wlogtmu) * Kokkos::log(m3sq / mu2)
                 +(wlog4mu - wlogsmu) * Kokkos::log(m4sq / mu2)
                 -TOutput(0.5) * (ln43p * ln43p + ln43m * ln43m);

        for (size_t j = 0; j < 3; j++)
            res(i,j) /= fac;
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(m_2^2,m_2^2,m_4^2,m_4^2;t,s;0,m_2^2,0,m_4^2) = \frac{-2}{m_2 m_4 t} \frac{x_s \ln x_s}{1-x_s^2} \left[ \frac{1}{\epsilon} + \ln \left( \frac{\mu^2}{-t} \right) \right],\, s-(m_2-m_4)^2 \neq 0 \\
    *  = \frac{1}{m_2 m_4 t} \left[ \frac{1}{\epsilon} + \ln \left( \frac{\mu^2}{-t} \right) \right], s-(m_2-m_4)^2 = 0.
    * \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988jr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B14(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m2sq = Y[1][1];
        const TMass m4sq = Y[3][3];
        const TMass ta = 2.0 * Y[0][2];
        const TMass si = 2.0 * Y[1][3] - Y[1][1] - Y[3][3];
        const TMass m2 = Kokkos::sqrt(m2sq);
        const TMass m4 = Kokkos::sqrt(m4sq);

        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(mu2, ta);

        TScale ieps = 0;
        Kokkos::Array<TOutput, 3> cxs;
        ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
        const TScale xs = ql::Real(cxs[0]);
        const TScale imxs = ql::Imag(cxs[0]);

        TOutput fac;
        if ( ql::iszero<TOutput, TMass, TScale>(xs - 1.0) && ql::iszero<TOutput, TMass, TScale>(imxs))
            fac = TOutput(-xs / (m2 * m4 * ta));
        else {
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(cxs[0], ieps);
            fac = TOutput(2.0 / (m2 * m4 * ta)) * cxs[0] / (cxs[1] * cxs[2]) * xlog;
        }
        res(i,2) = TOutput(0.0);
        res(i,1) = fac;
        res(i,0) = fac * wlogtmu;
    }



    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(m_2^2,p_2^2,p_3^2,m_4^2;t,s;0,m_2^2,0,m_4^2) = \\
    *  \frac{x_s}{m_2 m_4 t (1-x_s^2)} \left\{ \ln x_s \left[ -\frac{1}{\epsilon} - \frac{1}{2} \ln x_s - \ln \left( \frac{\mu^2}{m_2 m_4} \right) - \ln \left( \frac{m_2^2-p_2^2}{-t} \right) - \ln \left( \frac{m_4^2-p_3^2}{-t} \right) \right] \\
    *  - {\rm Li}_2 (1-x_s^2) + \frac{1}{2} ln^2 y + \sum_{\rho=\pm1} {\rm Li}_2 (1-x_s y^\rho) \right\}
    * \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988jr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B15(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m2sq = Y[1][1];
        const TMass m4sq = Y[3][3];
        const TMass m2sqbar = 2.0 * Y[1][2];
        const TMass m4sqbar = 2.0 * Y[2][3];
        const TMass si = 2.0 * Y[1][3] - Y[1][1] - Y[3][3];
        const TMass ta = 2.0 * Y[0][2];
        const TMass m2 = Kokkos::sqrt(m2sq);
        const TMass m4 = Kokkos::sqrt(m4sq);

        TScale ieps = 0;
        Kokkos::Array<TOutput, 3> cxs;
        ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
        const TOutput xs = cxs[0];

        TOutput fac;
        if (ql::iszero<TOutput, TMass, TScale>(m2sqbar) && !ql::iszero<TOutput, TMass, TScale>(m4sqbar)) {
            TMass yi;
            TScale iepyi;
            ql::ratreal<TOutput, TMass, TScale>(m4 * m2sqbar, m2 * m4sqbar, yi, iepyi);
            TOutput cyi = TOutput(yi);
            fac = xs / (TOutput(1.0) - xs * xs) / TOutput(-m2 * m4 * ta);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            res(i,2) = TOutput(0.0);
            res(i,1) = -xlog;
            res(i,0) = xlog * (-xlog - TOutput(Kokkos::log(mu2 / m4sq))
                -TOutput(2.0) * ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, ta))
                -ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                +ql::cLi2omx2<TOutput, TMass, TScale>(xs, cyi, ieps, iepyi)
                -ql::cLi2omx2<TOutput, TMass, TScale>(TOutput(1.0) / xs, cyi, -ieps, iepyi);

            for (size_t j = 0; j < 3; j++)
                res(i,j) *= TOutput(fac);

            return;
        } else if (ql::iszero<TOutput, TMass, TScale>(m4sqbar) && !ql::iszero<TOutput, TMass, TScale>(m2sqbar)) {
            TMass yy;
            TScale iepsyy;
            ql::ratreal<TOutput, TMass, TScale>(m2 * m4sqbar, m4 * m2sqbar, yy, iepsyy);
            TOutput cyy = TOutput(yy);
            fac = xs / (TOutput(1.0) - xs * xs) / TOutput(-m2 * m4 * ta);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            res(i,2) = TOutput(0.0);
            res(i,1) = -xlog;
            res(i,0) = xlog * (-xlog - TOutput(Kokkos::log(mu2 / m2sq))
                      -TOutput(2.0) * ql::Lnrat<TOutput, TMass, TScale>(m2sqbar, ta))
                -ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                +ql::cLi2omx2<TOutput, TMass, TScale>(xs, cyy, ieps, iepsyy)
                -ql::cLi2omx2<TOutput, TMass, TScale>(TOutput(1.0) / xs, cyy, -ieps, iepsyy);

            for (size_t j = 0; j < 3; j++)
                res(i,j) *= TOutput(fac);

            return;
        } else if (ql::iszero<TOutput, TMass, TScale>(m4sqbar) && ql::iszero<TOutput, TMass, TScale>(m2sqbar))
            Kokkos::printf("Box::B15 wrong kinematics, this is really B14.");

        TMass yy;
        TScale iepsyy;
        ql::ratreal<TOutput, TMass, TScale>(m2*m4sqbar, m4*m2sqbar, yy, iepsyy);
        const TScale rexs = ql::Real(xs);
        const TScale imxs = ql::Imag(xs);

        if (ql::iszero<TOutput, TMass, TScale>(rexs-1.0) && ql::iszero<TOutput, TMass, TScale>(imxs)) {
            fac = TOutput(TOutput(0.5) / (m2 * m4 * ta));
            res(i,2) = TOutput(0.0);
            res(i,1) = TOutput(1.0);
            res(i,0) = TOutput(Kokkos::log(mu2 / (m2 * m4)))
                      -ql::Lnrat<TOutput, TMass, TScale>(m2sqbar, ta) - ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, ta) - TOutput(2.0)
                     -TOutput((1.0 + yy) / (1.0 - yy)) * ql::Lnrat<TOutput, TMass, TScale>(m2 * m4sqbar, m4 * m2sqbar);
        } else {
            fac = xs / (TOutput(1.0) - xs * xs) / TOutput(-m2 * m4 * ta);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            res(i,2) = TOutput(0.0);
            res(i,1) = -xlog;
            res(i,0) = xlog * (-TOutput(0.5) * xlog - TOutput(Kokkos::log(mu2 / (m2 * m4)))
                    -ql::Lnrat<TOutput, TMass, TScale>(m2sqbar, ta) - ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, ta))
                    -ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                    +TOutput(0.5) * ql::kPow<TOutput, TMass, TScale>(ql::Lnrat<TOutput, TMass, TScale>(m2 * m4sqbar, m4 * m2sqbar), 2)
                    +ql::cLi2omx2<TOutput, TMass, TScale>(xs, TOutput(yy), ieps, iepsyy)
                    +ql::cLi2omx2<TOutput, TMass, TScale>(xs, TOutput(1.0 / yy), ieps, -iepsyy);
        }

        for (size_t j = 0; j < 3; j++)
            res(i,j) *= TOutput(fac);
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(m_2^2,p_2^2,p_3^2,m_4^2;t,s;0,m_2^2,m_3^2,m_4^2) = frac{x_s}{m_2 m_4 (t-m_3^2)(t-x_s^2)} \\
    *  \left\{ - \frac{\ln x_s}{ \epsilon} - 2 \ln x_s \ln \left( \frac{m_3 \mu}{m_3^2-t} \right) + \ln^2 x_2 + \ln^2 x_3 - {\rm Li}_2 (1-x_s^2) \\
    *  {\rm Li}_2 (1-x_s x_2 x_3) + {\rm Li}_2 \left( 1- \frac{x_s}{x_2 x_3} \right) + {\rm Li}_2 \left( 1- \frac{x_s x_2}{x_3} \right) + {\rm Li}_2 \left( 1- \frac{x_s x_3}{x_2} \right) \right\}
    * \f]
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988jr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B16(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m2sq = Y[1][1];
        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass tabar = 2.0 * Y[0][2];
        const TMass si = 2.0 * Y[1][3] - Y[1][1] - Y[3][3];
        const TMass mp2sq = 2.0 * Y[1][2] - m3sq - m2sq;
        const TMass mp3sq = 2.0 * Y[2][3] - m3sq - m4sq;
        const TMass m2 = Kokkos::sqrt(m2sq);
        const TMass m3 = Kokkos::sqrt(m3sq);
        const TMass m4 = Kokkos::sqrt(m4sq);
        const TMass mean = Kokkos::sqrt(ql::Real(m3sq) * mu2);

        TScale ieps = 0, iep2 = 0, iep3 = 0;
        Kokkos::Array<TOutput, 3> cxs;
        Kokkos::Array<TOutput, 3> cx2;
        Kokkos::Array<TOutput, 3> cx3;
        ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
        ql::kfn<TOutput, TMass, TScale>(cx2, iep2, -mp2sq, m2, m3);
        ql::kfn<TOutput, TMass, TScale>(cx3, iep3, -mp3sq, m3, m4);

        const TOutput xs = cxs[0];
        const TScale rexs = ql::Real(xs);
        const TScale imxs = ql::Imag(xs);

        TOutput fac;
        if (ql::iszero<TOutput, TMass, TScale>(rexs - 1.0) && ql::iszero<TOutput, TMass, TScale>(imxs)) {
            fac = TOutput(-0.5 / (m2 * m4 * tabar));
            res(i,2) = TOutput(0.0);
            res(i,1) = TOutput(1.0);
            res(i,0) = 2.0 * ql::Lnrat<TOutput, TMass, TScale>(mean, tabar) - TOutput(2.0);

            if (ql::iszero<TOutput, TMass, TScale>(ql::Real(cx2[0] - cx3[0])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(cx2[0] - cx3[0]))
                && ql::iszero<TOutput, TMass, TScale>(ql::Real(cx2[0] - 1.0)) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(cx2[0])))
                res(i,0) += TOutput(4.0);
            else if (ql::iszero<TOutput, TMass, TScale>(ql::Real(cx2[0] - cx3[0])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(cx2[0] - cx3[0])))
                res(i,0) += TOutput(2.0) + TOutput(2.0) * (cx2[0] * cx2[0] + TOutput(1.0)) * ql::cLn<TOutput, TMass, TScale>(cx2[0], iep2) / (cx2[0] * cx2[0] - TOutput(1.0));
            else {
                const TOutput cln_cx2_iep2 = ql::cLn<TOutput, TMass, TScale>(cx2[0], iep2);
                const TOutput cln_cx3_iep3 = ql::cLn<TOutput, TMass, TScale>(cx3[0], iep3);
                res(i,0) += -(TOutput(1.0) + cx2[0] * cx3[0]) / (TOutput(1.0) - cx2[0] * cx3[0])
                    * (cln_cx2_iep2 + cln_cx3_iep3)
                    -(TOutput(1.0) + cx2[0] / cx3[0]) / (TOutput(1.0) - cx2[0] / cx3[0])
                    *(cln_cx2_iep2 - cln_cx3_iep3);
            }
        } else {
            fac = TOutput(-1.0 / (m2 * m4 * tabar)) * cxs[0] / (TOutput(1.0) - cxs[0] * cxs[0]);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            const TOutput cln_cx2_iep2 = ql::cLn<TOutput, TMass, TScale>(cx2[0], iep2);
            const TOutput cln_cx3_iep3 = ql::cLn<TOutput, TMass, TScale>(cx3[0], iep3);
            res(i,2) = TOutput(0.0);
            res(i,1) = -xlog;
            res(i,0) = -TOutput(2.0) * xlog * ql::Lnrat<TOutput, TMass, TScale>(mean, tabar)
                + cln_cx2_iep2 * cln_cx2_iep2 + cln_cx3_iep3 * cln_cx3_iep3
                - ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], cx2[0], cx3[0], ieps, iep2, iep3)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], TOutput(1.0) / cx2[0], TOutput(1.0) / cx3[0], ieps, -iep2, -iep3)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], cx2[0], TOutput(1.0) / cx3[0], ieps, iep2, -iep3)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], TOutput(1.0) / cx2[0], cx3[0], ieps, -iep2, iep3);            
        }

        for (size_t j = 0; j < 3; j++)
            res(i,j) *= TOutput(fac);

    }

        /*!
    * This function trigger the topologies with 2-offshell external lines.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2ma(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;

        Y[0][0] = xpi[0];
        Y[1][1] = xpi[1];
        Y[2][2] = xpi[2];
        Y[3][3] = xpi[3];
        Y[0][1] = Y[1][0] = 0.5 * (xpi[0] + xpi[1] - xpi[4]);
        Y[0][2] = Y[2][0] = 0.5 * (xpi[0] + xpi[2] - xpi[8]);
        Y[0][3] = Y[3][0] = 0.5 * (xpi[0] + xpi[3] - xpi[7]);
        Y[1][2] = Y[2][1] = 0.5 * (xpi[1] + xpi[2] - xpi[5]);
        Y[1][3] = Y[3][1] = 0.5 * (xpi[1] + xpi[3] - xpi[9]);
        Y[2][3] = Y[3][2] = 0.5 * (xpi[2] + xpi[3] - xpi[6]);

        ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 2, ql::iszero<TOutput, TMass, TScale>(xpi[2]));

        const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
        const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
        const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);

        if (zY01 && zY12 && zY03) 
            ql::B11<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY01 && zY12 && !zY03) 
            ql::B12<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (ql::iszero<TOutput, TMass, TScale>(Yalt[0][1]) && ql::iszero<TOutput, TMass, TScale>(Yalt[1][2]) && !ql::iszero<TOutput, TMass, TScale>(Yalt[0][3]))
            ql::B12<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        else if (zY01 && !zY12 && !zY03)
            ql::B13<TOutput, TMass, TScale>(res, Y, mu2, i);
        else
            ql::BIN2<TOutput, TMass, TScale>(res, Y, i);
    }

    /*!
    * This function trigger the topologies with 2-offshell external lines.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2mo(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;

        Y[0][0] = xpi[0];
        Y[1][1] = xpi[1];
        Y[2][2] = xpi[2];
        Y[3][3] = xpi[3];
        Y[0][1] = Y[1][0] = 0.5 * (xpi[0] + xpi[1] - xpi[4]);
        Y[0][2] = Y[2][0] = 0.5 * (xpi[0] + xpi[2] - xpi[8]);
        Y[0][3] = Y[3][0] = 0.5 * (xpi[0] + xpi[3] - xpi[7]);
        Y[1][2] = Y[2][1] = 0.5 * (xpi[1] + xpi[2] - xpi[5]);
        Y[1][3] = Y[3][1] = 0.5 * (xpi[1] + xpi[3] - xpi[9]);
        Y[2][3] = Y[3][2] = 0.5 * (xpi[2] + xpi[3] - xpi[6]);

        const bool zY00 = ql::iszero<TOutput, TMass, TScale>(Y[0][0]);
        const bool zY22 = ql::iszero<TOutput, TMass, TScale>(Y[2][2]);
        const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
        const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);
        const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
        const bool zY23 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);

        if (zY00 && zY22 && zY01 && zY03 && zY12 && zY23) 
            ql::B14<TOutput, TMass, TScale>(res, Y, mu2, i); 
        else if (zY00 && zY22 && zY01 && zY03) 
            ql::B15<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY00 && zY22 && zY12 && zY23) {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
            ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 2, ql::iszero<TOutput, TMass, TScale>(xpi[2]));
            ql::B15<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        } else {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yadj;
            const int swap23[4] = {0, 2, 1, 3};
            for (int j = 0; j < 4; j++) 
                for (int k = 0; k < 4; k++) 
                    Yadj[j][k] = Y[swap23[j]][swap23[k]];

            ql::BIN2<TOutput, TMass, TScale>(res, Yadj, i);
        }
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B0m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        int offshell = 0, jsort0 = 0, jsort1 = 0, jsort2 = 0;
        for (int j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi[j + 4])) {
                offshell += 1;
                if (jsort1 == 0) jsort1 = j + 1;
                else jsort2 = j + 1;
            } else 
                jsort0 = j;
        }

        Kokkos::Array<TMass, 13> xpiout;
        bool swapped = true;
        const int jdiff = jsort2 - jsort1;

        if (offshell == 1) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, jsort1 - 1)] = xpi[j];
            }
        } else if (offshell == 2 && (jdiff == 2 || jdiff == 1)) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, jsort2 - 1)] = xpi[j];
            }
        } else if (offshell == 2 && jdiff == 3) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, 0)] = xpi[j];
            }
        } else if (offshell == 3) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, ql::jsort_b0m(jsort0))] = xpi[j];
            }
        } else {
            swapped = false;
        }

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Kokkos::Array<TMass, 13> xpi_in;
        if (swapped) 
            xpi_in = xpiout;
        else 
            xpi_in = xpi;

        
        Y[0][0] = xpi_in[0];
        Y[1][1] = xpi_in[1];
        Y[2][2] = xpi_in[2];
        Y[3][3] = xpi_in[3];
        Y[0][1] = Y[1][0] = 0.5 * (xpi_in[0] + xpi_in[1] - xpi_in[4]);
        Y[0][2] = Y[2][0] = 0.5 * (xpi_in[0] + xpi_in[2] - xpi_in[8]);
        Y[0][3] = Y[3][0] = 0.5 * (xpi_in[0] + xpi_in[3] - xpi_in[7]);
        Y[1][2] = Y[2][1] = 0.5 * (xpi_in[1] + xpi_in[2] - xpi_in[5]);
        Y[1][3] = Y[3][1] = 0.5 * (xpi_in[1] + xpi_in[3] - xpi_in[9]);
        Y[2][3] = Y[3][2] = 0.5 * (xpi_in[2] + xpi_in[3] - xpi_in[6]);

        if (offshell == 4) 
            ql::BIN0<TOutput, TMass, TScale>(res, Y, i);
        else if (offshell == 3) 
            ql::B5<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (offshell == 2) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi_in[5])) 
                ql::B3<TOutput, TMass, TScale>(res, Y, mu2, i);
            else if (!ql::iszero<TOutput, TMass, TScale>(xpi_in[6]) && !ql::iszero<TOutput, TMass, TScale>(xpi_in[7])) 
                ql::B4<TOutput, TMass, TScale>(res, Y, mu2, i);
        } else if (offshell == 1)
            ql::B2<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (offshell == 0) 
            ql::B1<TOutput, TMass, TScale>(res, Y, mu2, i);
    }           


    /*!
    * This function trigger the topologies with 3-offshell external lines.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B1m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        int jsort = 0;
        for (int j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi[j])) {
                jsort = j;
            }
        }

        Kokkos::Array<TMass, 13> xpi_in;
        for (size_t j = 0; j < 13; j++) {
            xpi_in[ql::swap_b1m(j, jsort)] = xpi[j];
        }

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Y[0][0] = xpi_in[0];
        Y[1][1] = xpi_in[1];
        Y[2][2] = xpi_in[2];
        Y[3][3] = xpi_in[3];
        Y[0][1] = Y[1][0] = 0.5 * (xpi_in[0] + xpi_in[1] - xpi_in[4]);
        Y[0][2] = Y[2][0] = 0.5 * (xpi_in[0] + xpi_in[2] - xpi_in[8]);
        Y[0][3] = Y[3][0] = 0.5 * (xpi_in[0] + xpi_in[3] - xpi_in[7]);
        Y[1][2] = Y[2][1] = 0.5 * (xpi_in[1] + xpi_in[2] - xpi_in[5]);
        Y[1][3] = Y[3][1] = 0.5 * (xpi_in[1] + xpi_in[3] - xpi_in[9]);
        Y[2][3] = Y[3][2] = 0.5 * (xpi_in[2] + xpi_in[3] - xpi_in[6]);

        if (!ql::iszero<TOutput, TMass, TScale>(Y[0][0]) || 
            !ql::iszero<TOutput, TMass, TScale>(Y[1][1]) || 
            !ql::iszero<TOutput, TMass, TScale>(Y[2][2])) {
            Kokkos::printf("Box::B1m - Wrong ordering.");
        }

        const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
        const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
        const bool zY23 = ql::iszero<TOutput, TMass, TScale>(Y[2][3]);
        const bool zY30 = ql::iszero<TOutput, TMass, TScale>(Y[3][0]);

        if (zY01 && zY12 && zY23 && zY30) 
            ql::B6<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY01 && zY12 && zY23) 
            ql::B7<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY01 && zY12 && zY30) {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
            ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 1);
            ql::B7<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        } else if (zY01 && zY12) 
            ql::B8<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY01 && zY30) 
            ql::B9<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY12 && zY23) {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
            ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 1);
            ql::B9<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        } else if (zY01)
            ql::B10<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY12) {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
            ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 1);
            ql::B10<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        } else 
            ql::BIN1<TOutput, TMass, TScale>(res, Y, i);
    }           

    /*!
    * This function trigger the topologies with 2-offshell external lines.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        int jsort1 = -1, jsort2 = -1;
        for (int j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi[j])) {
                if (jsort1 == -1) {
                    jsort1 = j;
                } else {
                    jsort2 = j;
                }
            }
        }

        int jdiff = jsort2 - jsort1;
        Kokkos::Array<TMass, 13> xpiout;

        if (jdiff == 1 || jdiff == 2) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b2m(j, jsort2)] = xpi[j];
            }
        } else if (jdiff == 3) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b2m(j, 0)] = xpi[j];
            }
        }

        if (jdiff == 2) {
            ql::B2mo<TOutput, TMass, TScale>(res, xpiout, mu2, i);
        } else {
            ql::B2ma<TOutput, TMass, TScale>(res, xpiout, mu2, i);
        }
    }

    /*!
    * This function trigger the topologies with 1-offshell external lines.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B3m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        int jsort = 0;
        for (int j = 0; j < 4; j++) 
            if (ql::iszero<TOutput, TMass, TScale>(xpi[j])) jsort = j;
            

        Kokkos::Array<TMass, 13> xpo;
        for (size_t j = 0; j < 13; j++) 
            xpo[ql::swap_b3m(j, jsort)] = xpi[j];
        

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Y[0][0] = xpo[0];
        Y[1][1] = xpo[1];
        Y[2][2] = xpo[2];
        Y[3][3] = xpo[3];
        Y[0][1] = Y[1][0] = 0.5 * (xpo[0] + xpo[1] - xpo[4]);
        Y[0][2] = Y[2][0] = 0.5 * (xpo[0] + xpo[2] - xpo[8]);
        Y[0][3] = Y[3][0] = 0.5 * (xpo[0] + xpo[3] - xpo[7]);
        Y[1][2] = Y[2][1] = 0.5 * (xpo[1] + xpo[2] - xpo[5]);
        Y[1][3] = Y[3][1] = 0.5 * (xpo[1] + xpo[3] - xpo[9]);
        Y[2][3] = Y[3][2] = 0.5 * (xpo[2] + xpo[3] - xpo[6]);

        if (ql::iszero<TOutput, TMass, TScale>(Y[0][0]) && ql::iszero<TOutput, TMass, TScale>(Y[0][1]) && ql::iszero<TOutput, TMass, TScale>(Y[0][3])) 
            ql::B16<TOutput, TMass, TScale>(res, Y, mu2, i);
        else 
            ql::BIN3<TOutput, TMass, TScale>(res, Y, i);
    }

    /*!
    * This function trigger the topologies with 0-offshell external lines.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B4m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const int i) {

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;

        Y[0][0] = xpi[0];
        Y[1][1] = xpi[1];
        Y[2][2] = xpi[2];
        Y[3][3] = xpi[3];
        Y[0][1] = Y[1][0] = 0.5 * (xpi[0] + xpi[1] - xpi[4]);
        Y[0][2] = Y[2][0] = 0.5 * (xpi[0] + xpi[2] - xpi[8]);
        Y[0][3] = Y[3][0] = 0.5 * (xpi[0] + xpi[3] - xpi[7]);
        Y[1][2] = Y[2][1] = 0.5 * (xpi[1] + xpi[2] - xpi[5]);
        Y[1][3] = Y[3][1] = 0.5 * (xpi[1] + xpi[3] - xpi[9]);
        Y[2][3] = Y[3][2] = 0.5 * (xpi[2] + xpi[3] - xpi[6]);

        ql::BIN4<TOutput, TMass, TScale>(res, Y, i);
    }


}