//
// QCDLoop + Kokkos 2025
//
// Overloads of the ql:: math helpers for one xpmath-kokkos backend.
// Included only when exactly one XPMATH_BACKEND_{dd,ff,qf,tf} macro is set.
// The double overloads in kokkosMaths.h stay in place.

#pragma once

// xpmath's Kokkos::abs / log / sqrt overloads must be visible before
// kokkosMaths.h defines ql::kAbs and the other wrappers. Those calls are
// qualified, so later declarations are not part of the overload set.
#if defined(XPMATH_BACKEND_dd)
#include <Kokkos_xpmath/dd_math.hpp>
#include <Kokkos_xpmath/dd_complex.hpp>
#define QL_XP_REAL Kokkos::Experimental::DoubleDouble
#define QL_XP_COMPLEX Kokkos::Experimental::DoubleDoubleComplex
#define QL_XP_PI() Kokkos::Experimental::DoubleDouble_pi()
#elif defined(XPMATH_BACKEND_ff)
#include <Kokkos_xpmath/ff_math.hpp>
#include <Kokkos_xpmath/ff_complex.hpp>
#define QL_XP_REAL Kokkos::Experimental::FloatFloat
#define QL_XP_COMPLEX Kokkos::Experimental::FloatFloatComplex
#define QL_XP_PI() Kokkos::Experimental::FloatFloat_pi()
#elif defined(XPMATH_BACKEND_qf)
#include <Kokkos_xpmath/qf_math.hpp>
#include <Kokkos_xpmath/qf_complex.hpp>
#define QL_XP_REAL Kokkos::Experimental::QuadFloat
#define QL_XP_COMPLEX Kokkos::Experimental::QuadFloatComplex
#define QL_XP_PI() Kokkos::Experimental::QuadFloat_pi()
#elif defined(XPMATH_BACKEND_tf)
#include <Kokkos_xpmath/tf_math.hpp>
#include <Kokkos_xpmath/tf_complex.hpp>
#define QL_XP_REAL Kokkos::Experimental::TripleFloat
#define QL_XP_COMPLEX Kokkos::Experimental::TripleFloatComplex
#define QL_XP_PI() Kokkos::Experimental::TripleFloat_pi()
#else
#error "kokkosMaths_xp.h requires one of XPMATH_BACKEND_dd, _ff, _qf, _tf"
#endif

#include "kokkosMaths.h"

namespace ql {

    using xp_real = QL_XP_REAL;
    using xp_complex = QL_XP_COMPLEX;

    // Library pi fills both limbs. The primary Constants<T>::_pi() builds T(M_PI).
    template<>
    KOKKOS_INLINE_FUNCTION
    xp_real Constants<xp_real>::_pi() {
        return QL_XP_PI();
    }

    // Dilogarithm tables and branch cutoffs. FloatFloat keeps the double
    // values from kokkosMaths.h. The other three backends replace them.
#include "kokkosMaths_xp_constants.h"

    KOKKOS_INLINE_FUNCTION xp_real Real(xp_real const& x) { return x; }

    KOKKOS_INLINE_FUNCTION xp_real Real(xp_complex const& x) { return x.real(); }

    KOKKOS_INLINE_FUNCTION xp_real Imag(xp_real const&) { return xp_real(0.0); }

    KOKKOS_INLINE_FUNCTION xp_real Imag(xp_complex const& x) { return x.imag(); }

    KOKKOS_INLINE_FUNCTION xp_real Sign(xp_real const& x) {
        if (xp_real(0.0) < x) return xp_real(1.0);
        if (x < xp_real(0.0)) return xp_real(-1.0);
        return xp_real(0.0);
    }

    KOKKOS_INLINE_FUNCTION xp_real kAbs(xp_complex const& x) {
        return Kokkos::abs(x);
    }

    KOKKOS_INLINE_FUNCTION xp_complex Sign(xp_complex const& x) {
        return x / kAbs(x);
    }

    KOKKOS_INLINE_FUNCTION xp_complex kLog1p(xp_complex const& z) {
        const xp_real re = z.real();
        const xp_real im = z.imag();
        return xp_complex(
            xp_real(0.5) * Kokkos::log1p(xp_real(2.0) * re + (re * re + im * im)),
            Kokkos::atan2(im, xp_real(1.0) + re));
    }

    KOKKOS_INLINE_FUNCTION xp_real Max(xp_real const& a, xp_real const& b) {
        if (kAbs(a) > kAbs(b))
            return a;
        return b;
    }

    KOKKOS_INLINE_FUNCTION xp_complex Max(xp_complex const& a, xp_complex const& b) {
        if (kAbs(a) > kAbs(b))
            return a;
        return b;
    }

    KOKKOS_INLINE_FUNCTION xp_real Min(xp_real const& a, xp_real const& b) {
        if (kAbs(a) > kAbs(b))
            return b;
        return a;
    }

    KOKKOS_INLINE_FUNCTION xp_complex Min(xp_complex const& a, xp_complex const& b) {
        if (kAbs(a) > kAbs(b))
            return b;
        return a;
    }

    KOKKOS_INLINE_FUNCTION xp_real Htheta(xp_real const& x) {
        return xp_real(0.5) * (xp_real(1.0) + Sign(x));
    }

}

#undef QL_XP_REAL
#undef QL_XP_COMPLEX
#undef QL_XP_PI
