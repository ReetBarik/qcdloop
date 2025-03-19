//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#pragma once
#include <math.h>
#include <inttypes.h>

namespace ql
{
    using complex = Kokkos::complex<double>;

    void printDoubleBits(double x)
    {
        // We'll copy the double bits into a 64-bit integer.
        // A union is a common trick, or we can use memcpy.
        union {
            double d;
            uint64_t u;
        } conv;

        conv.d = x;

        // Use C99's PRIx64 for a portable 64-bit hex format.
        // %.16g prints up to 16 significant digits in decimal (just for reference).
        // std::printf("decimal=%.16g\n", x);
        std::printf("0x%016" PRIx64, conv.u);
    }

    template<typename TOutput>
    KOKKOS_INLINE_FUNCTION TOutput kPow(TOutput const& base, int const& exponent) {
        TOutput temp = TOutput(1.0);

        for (int i = 0; i < exponent; i++)
            temp *= base;

        return temp;
    }

    template<typename TMass>
    KOKKOS_INLINE_FUNCTION void printDoubleBits(TMass x) {
        union {
            TMass d;
            uint64_t u;
        } conv;

        conv.d = x;

        Kokkos::printf("0x%016" PRIx64, conv.u);
    }

    /*!
    * Computes the log of a complex number z.
    * If the imag(z)=0 and real(z)<0 and extra ipi term is included.
    * \param z the complex argument of the logarithm
    * \param isig the sign of the imaginary part
    * \return the complex log(z)
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cLn(TOutput const& z, TScale const& isig) {
        TOutput cln;
        auto sign = (std::is_same<TScale, double>::value) ? (double(0) < isig) - (isig < double(0)) : isig / Kokkos::abs(isig);
        auto imag = z.imag();
        auto real = z.real();
        
        if (imag == 0.0 && real <= 0.0) {
            complex temp(0.0, M_PI * sign);
            cln = Kokkos::log(-z) + temp;
        }
        else
            cln = Kokkos::log(z);
        return cln;
    }

    /*!
    * Computes the log of a real number x.
    * If the x<0 and extra ipi term is included.
    * \param x the real argument of the logarithm
    * \param isig the sign of the imaginary part
    * \return the complex log(x)
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cLn(TScale const& x, TScale const& isig) {
        TOutput ln;
        auto sign = (std::is_same<TScale, double>::value) ? (double(0) < isig) - (isig < double(0)) : isig / Kokkos::abs(isig);
        if (x > 0)
            ln = TOutput(Kokkos::log(x));
        else {
            complex temp(0.0, M_PI * sign);
            ln = TOutput(Kokkos::log(-x)) + temp;
        }
        return ln;
    }

    /*!
    * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
    * \f[
    * f_{n}(x) = \ln \left( 1 - \frac{1}{x} \right) + \sum_{l=n+1}^{\infty} \frac{x^{n-l}}{l+1}
    *   \f]
    *
    * \param n the lower index of
    * \param x the argument of the function
    * \param iep the epsilon value
    * \return function DD from Eq. 4.11 of \cite Denner:2005nn.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput fndd(int const& n, TOutput const& x, TScale const& iep) {
        const int infty = 16;
        TOutput res = TOutput(0.0);
        
        if (Kokkos::abs(x) < 10.0) {
            
            if (Kokkos::abs(x - TOutput(1.0)) >= 1e-10) {
                res = (TOutput(1.0) - ql::kPow<TOutput>(x, n + 1)) * (ql::cLn<TOutput, TMass, TScale>(x - TOutput(1.0), iep) - ql::cLn<TOutput, TMass, TScale>(x, iep)); 
            }
            for (int j = 0; j <= n; j++) {
                res -= ql::kPow<TOutput>(x, n - j) / (j + 1.0);
            }
        } else {

            res = ql::cLn<TOutput, TMass, TScale>(TOutput(1.0) - TOutput(1.0) / x, iep); 
            for (int j = n + 1; j <= n + infty; j++)
                res += ql::kPow<TOutput>(x, n - j) / (j + 1.0);
        }
        
        return res;
    }



}
