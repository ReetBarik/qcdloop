//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov


#pragma once

#include <math.h>

namespace ql
{
    using complex = Kokkos::complex<double>;

    struct Constants {
        KOKKOS_INLINE_FUNCTION static double _qlonshellcutoff() { return 1e-10; }
        KOKKOS_INLINE_FUNCTION static double _pi() { return M_PI; }
        KOKKOS_INLINE_FUNCTION static complex _ipio2() { complex temp(0.0, 0.5 * M_PI); return temp; }
        
    };

    template<typename TOutput>
    KOKKOS_INLINE_FUNCTION TOutput kPow(TOutput const& base, int const& exponent) {
        TOutput temp = TOutput(1.0);

        for (int i = 0; i < exponent; i++)
            temp *= base;

        return temp;
    }

    template<typename TOutput>
    KOKKOS_INLINE_FUNCTION bool iszero(TOutput const& x) {
        return (x < ql::Constants::_qlonshellcutoff()) ? true : false;
    }

    KOKKOS_INLINE_FUNCTION double Imag(double const& x) {
        return 0.0;
    }
        
    KOKKOS_INLINE_FUNCTION double Imag(complex const& x) {
        return x.imag();
    }

    KOKKOS_INLINE_FUNCTION double Real(double const& x) {
        return x;   
    }

    KOKKOS_INLINE_FUNCTION double Real(complex const& x) {
        return x.real();
    }

    KOKKOS_INLINE_FUNCTION int Sign(double const& x) {
        return (double(0) < x) - (x < double(0));
    }

    KOKKOS_INLINE_FUNCTION complex Sign(complex const& x) {
        return x / Kokkos::abs(x);
    }

    template<typename TOutput>
    KOKKOS_INLINE_FUNCTION TOutput Max(TOutput const& a, TOutput const& b) {
        if (Kokkos::abs(a) > Kokkos::abs(b)) 
            return a;
        else 
            return b;
    }

    template<typename TOutput>
    KOKKOS_INLINE_FUNCTION TOutput Min(TOutput const& a, TOutput const& b) {
        if (Kokkos::abs(a) > Kokkos::abs(b)) 
            return b;
        else 
            return a;
    }

}