//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#pragma once
#include <math.h>
#include <inttypes.h>
#include "kokkosMaths.h"

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
        
        if (ql::Imag(z) == 0.0 && ql::Real(z) <= 0.0) {
            complex temp(0.0, ql::Constants::_pi() * ql::Sign(isig));
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
        if (x > 0)
            ln = TOutput(Kokkos::log(x));
        else {
            complex temp(0.0, ql::Constants::_pi() * ql::Sign(isig));
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
                res = (TOutput(1.0) - ql::kPow<TOutput, TMass, TScale>(x, n + 1)) * (ql::cLn<TOutput, TMass, TScale>(x - TOutput(1.0), iep) - ql::cLn<TOutput, TMass, TScale>(x, iep)); 
            }
            for (int j = 0; j <= n; j++) {
                res -= ql::kPow<TOutput, TMass, TScale>(x, n - j) / (j + 1.0);
            }
        } else {

            res = ql::cLn<TOutput, TMass, TScale>(TOutput(1.0) - TOutput(1.0) / x, iep); 
            for (int j = n + 1; j <= n + infty; j++)
                res += ql::kPow<TOutput, TMass, TScale>(x, n - j) / (j + 1.0);
        }
        
        return res;
    }



    /*!
    * Computes \f${\rm Lnrat}(x,y) = \log(x-i \epsilon)-\log(y-i \epsilon)\f$
    * \param x TMass object for the numerator
    * \param y TMass object for the denumerator
    * \return returns the ratio of logs
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Lnrat(TOutput const& x, TOutput const& y) {
        
        const TOutput r = x / y;
        auto imag_r = (std::is_same<TOutput, double>::value) ? 0 : r.imag();
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(r))) {
            return TOutput(Kokkos::log(Kokkos::abs(r))) - ql::Constants::_ipio2() * TOutput(ql::Sign(-ql::Real(x)) - ql::Sign(-ql::Real(y)));
        }  
        else
            return Kokkos::log(r);
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Lnrat(TScale const& x, TScale const& y) {
        return TOutput(Kokkos::log(Kokkos::abs(x / y))) - (ql::Constants::_ipio2() * TOutput(ql::Sign(-x) - ql::Sign(-y)));
    }

    /*!
    * The dilog function for real argument
    * \param x the argument of the dilog
    * \return the dilog
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TMass ddilog(TMass const& x) {

        if (x == 1.0)
            return ql::Constants::_pi2o6<TOutput, TMass, TScale>();
        else if (x == -1.0)
            return - 0.5 * ql::Constants::_pi2o6<TOutput, TMass, TScale>();

        const TMass T = -x;
        TMass Y, S, A;

        if (ql::Real(T) <= -2.0) {

            Y = -1.0 / (1.0 + T);
            S = 1.0;
            A = TMass(-ql::Constants::_pi2o6<TOutput, TMass, TScale>() + 0.5 * ql::Real(ql::kPow<TOutput, TMass, TScale>(Kokkos::log(-T), 2) - ql::kPow<TOutput, TMass, TScale>(Kokkos::log(1.0 + 1.0 / T), 2)));

        } else if (ql::Real(T) < -1.0) {

            Y = -1.0 - T;
            S = -1.0;
            A = Kokkos::log(-T);
            A = TMass(-ql::Constants::_pi2o6<TOutput, TMass, TScale>() + A * (A + Kokkos::log(1.0 + 1.0 / T)));            

        } else if (ql::Real(T) <= -0.5) {

            Y = (-1.0 - T) / T;
            S = 1.0;
            A = Kokkos::log(-T);
            A = TMass(-ql::Constants::_pi2o6<TOutput, TMass, TScale>() + A * (-0.5 * A + Kokkos::log(1.0 + T)));

        } else if (ql::Real(T) < 0.0) {

            Y = -T / (1.0 + T);
            S = -1.0;
            A = TMass(0.5 * ql::Real(ql::kPow<TOutput, TMass, TScale>(Kokkos::log(1.0 + T),2)));

        } else if (ql::Real(T) <= 1.0) {

            Y = T;
            S = 1.0;
            A = TMass(0.0);

        } else {

            Y = 1.0 / T;
            S = -1.0;
            A = TMass(ql::Constants::_pi2o6<TOutput, TMass, TScale>() + 0.5 * ql::Real(ql::kPow<TOutput, TMass, TScale>(Kokkos::log(T),2)));

        }
        
        const TMass H = Y + Y - 1.0;
        const TMass ALFA = H + H;
        TMass B1 = 0.0, B2 = 0.0, B0 = 0.0;
        Kokkos::View<double*> _C = ql::Constants::_C();

        
        for (size_t i = _C.extent(0) - 1; i >= 0; --i) {
            
            B0 = _C(i) + ALFA * B1 - B2;
            B2 = B1;
            B1 = B0;

        }

        return -(S * (B0 - H * B2) + A);

    }


    /*!
    * \param z
    * \param isig
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput li2series(TOutput const& z, TScale const& isig) {
        
        TOutput xm = -ql::cLn<TOutput, TMass, TScale>(TOutput(1.0) - z, -isig);
        const TOutput x2 = xm * xm;
        TOutput res = xm - x2/ TOutput(4.0);
        Kokkos::View<double*> _B = ql::Constants::_B();

        for (int j = 0; j < _B.extent(0); j++)
        {
            xm *= x2;
            const TOutput n = res + xm * _B(j);
            if (n == res) return res;
            else res = n;
        }
        Kokkos::printf("li2series: bad convergence\n");
        return TOutput(0.0);
    }


    /*!
    * \param z1
    * \param s
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput ltli2series(TOutput const& z1, TScale const& s) {
        TOutput xm = -ql::cLn<TOutput, TMass, TScale>(z1, -s);
        const TOutput x2 = xm * xm;
        TOutput res = xm - x2 / TOutput(4.0);
        Kokkos::View<double*> _B = ql::Constants::_B();

        for (int i = 0; i < _B.extent(0); i++) {
            xm *= x2;
            const TOutput n = res + xm * _B(i);
            if (n == res) return res;
            else res = n;
        }
        Kokkos::printf("ltli2series: bad convergence\n");
        return TOutput(0.0);
    }


    /*!
    * \param z the argument
    * \param isig the sign of z
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput denspence(TOutput const& z, TScale const& isig) {
        const TOutput z1 = TOutput(1.0) - z;
        const TScale az1 = Kokkos::abs(z1);

        if (isig == 0.0 && ql::Imag(z) == 0.0 && Kokkos::abs(ql::Real(z1)) < ql::Constants::_qlonshellcutoff<TOutput, TMass, TScale>())
            Kokkos::printf("denspence: argument on cut\n");

        if (az1 < ql::Constants::_eps15())
            return TOutput{ql::Constants::_pi2o6<TOutput, TMass, TScale>(), 0.0};

   
        else if (ql::Real(z) < 0.5)
        {
            if (Kokkos::abs(z) < 1.0)
                return ql::li2series<TOutput, TMass, TScale>(z, isig);
            else
                return -ql::Constants::_pi2o6<TOutput, TMass, TScale>() - 0.5 * ql::kPow<TOutput, TMass, TScale>(ql::cLn<TOutput, TMass, TScale>(-z, -isig),2) - ql::li2series<TOutput, TMass, TScale>(1.0 / z, -isig);
        }
        else
        {
            if (az1 < 1.0)
                return ql::Constants::_pi2o6<TOutput, TMass, TScale>() - ql::cLn<TOutput, TMass, TScale>(z, isig) * ql::cLn<TOutput, TMass, TScale>(z1, -isig) - ql::li2series<TOutput, TMass, TScale>(z1, -isig);
            else
                return TOutput(2.0) * ql::Constants::_pi2o6<TOutput, TMass, TScale>() + 0.5 * ql::kPow<TOutput, TMass, TScale>(ql::cLn<TOutput, TMass, TScale>(-z1, -isig),2) - ql::cLn<TOutput, TMass, TScale>(z, isig) * ql::cLn<TOutput, TMass, TScale>(z1, -isig) + ql::li2series<TOutput, TMass, TScale>(1.0 / z1, isig);
        }
    }


    /*!
    * Calculate Li[2](1-(x1+ieps1)*(x2+ieps2)) for real x1,x2
    * Using +Li2(1-x1*x2)                           for x1*x2<1
    * and   -Li2(1-1/(x1*x2))-1/2*(ln(x1)+ln(x2))^2 for x1*x2>1
    * \param x1 numerator
    * \param x2 denominator
    * \param ieps1 sign of x1
    * \param ieps2 sign of x2
    * \return the ratio Li2
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omx(TMass const& x1, TMass const& x2, TScale const& ieps1, TScale const& ieps2) {
        
        TOutput prod, Li2omx;

        TMass arg = x1 * x2;
        const TScale ieps = ql::Sign(ql::Real(x2) * ieps1 + ql::Real(x1) * ieps2);
        if (ql::Real(arg) <= 1.0) {
            if (ql::Real(arg) == 1.0 || ql::Real(arg) == 0.0)
                prod = TOutput(0.0);
            else {
                const TOutput lnarg = ql::cLn<TOutput, TMass, TScale>(ql::Real(x1), ieps) + ql::cLn<TOutput, TMass, TScale>(ql::Real(x2), ieps2);
                const TOutput lnomarg = TOutput(Kokkos::log(1.0 - arg));
                prod = lnarg * lnomarg;
            }
            Li2omx = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(TOutput(arg), ieps) - prod;
        }
        else if (ql::Real(arg) > 1.0) {
            arg = 1.0 / (x1 * x2);
            const TOutput lnarg = - ql::cLn<TOutput, TMass, TScale>(ql::Real(x1), ieps1) - ql::cLn<TOutput, TMass, TScale>(ql::Real(x2), ieps2);
            const TOutput lnomarg = TOutput(Kokkos::log(1.0 - arg));
            Li2omx = -TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) + ql::denspence<TOutput, TMass, TScale>(TOutput(arg), ieps) + lnarg * lnomarg - TOutput(0.5) * lnarg * lnarg;
        }
        
        return Li2omx;
    }


    /*!
    * \param zrat1 first argument
    * \param zrat2 second argument
    * \param ieps1 sign for zrat1
    * \param ieps2 sign for zrat2
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput spencer(TOutput const& zrat1, TOutput const& zrat2, TScale const&ieps1, TScale const&ieps2) {
        
        const TScale x1 = ql::Real(zrat1);
        const TScale x2 = ql::Real(zrat2);
        const TScale y1 = ql::Imag(zrat1);
        const TScale y2 = ql::Imag(zrat2);

        TOutput res, prod;
        if (ql::iszero<TOutput, TMass, TScale>(y1) && ql::iszero<TOutput, TMass, TScale>(y2))
            res = ql::Li2omx<TOutput, TMass, TScale>(x1, x2, ieps1, ieps2);
        else {
            TOutput arg = zrat1 * zrat2;
            const TScale ieps = 0.0;
            if (Kokkos::abs(arg) <= 1.0) {
                if (arg == 0.0 || arg == 1.0)
                    prod = TOutput(0.0);
                else{
                    const TOutput lnarg = ql::cLn<TOutput, TMass, TScale>(zrat1, ieps1) + ql::cLn<TOutput, TMass, TScale>(zrat2, ieps2);
                    const TOutput lnomarg = Kokkos::log(TOutput(1.0) - arg);
                    prod = lnarg * lnomarg;
                }
                res = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(arg, ieps) - prod;
            }
            else if (Kokkos::abs(arg) > 1.0) {
                arg = TOutput(1.0) / (zrat1 * zrat2);
                const TOutput lnarg = -ql::cLn<TOutput, TMass, TScale>(zrat1, ieps1) - ql::cLn<TOutput, TMass, TScale>(zrat2, ieps2);
                const TOutput lnomarg = Kokkos::log(TOutput(1.0) - arg);
                res = -TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) + ql::denspence<TOutput, TMass, TScale>(arg, ieps) + lnarg * lnomarg - TOutput(0.5) * ql::kPow<TOutput, TMass, TScale>(lnarg,2);
            }
        }
        return res;
    }

    /*!
    * \param z1
    * \param s1
    * \param z2
    * \param s2
    * \param s12
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION int eta(TOutput const& z1, TScale const& s1, TOutput const& z2, TScale const& s2, TScale const& s12) {
        
        TScale im1 = ql::Imag(z1), im2 = ql::Imag(z2), im12 = ql::Imag(z1 * z2);
        if (im1 == 0.0) im1 = s1;
        if (im2 == 0.0) im2 = s2;
        if (im12 == 0.0) im12 = s12;

        int eta;
        if (im1 < 0.0 && im2 < 0.0 && im12 > 0.0)
            eta = 1;
        else if (im1 > 0.0 && im2 > 0.0 && im12 < 0.0)
            eta = -1;
        else
            eta = 0;

        return eta;
    }


    /*!
    * \param z2
    * \param im2
    * \param im12
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput xeta(Kokkos::View<const TOutput[2]> &z1, Kokkos::View<const TScale[2]> &im1, TOutput const& z2, TScale const& im2, TScale const& im12, Kokkos::View<const TOutput[2]> &l1) {  
        
            return l1(1) * TOutput(ql::eta(z1(1), im1(1), z2, im2, im12)) - l1(0) * TOutput(ql::eta(z1(0), im1(0), z2, im2, im12));
    }
    

    /*!
    * \param c1
    * \param im1x
    * \param c2
    * \param im2x
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION int etatilde(TOutput const& c1, TScale const& im1x, TOutput const& c2, TScale const& im2x) {
        int etatilde;
        TScale im1 = ql::Imag(c1);
        TScale im2 = ql::Imag(c2);

        if (im1 == 0.0)
        im1 = im1x;

        if (im2 != 0.0)
            etatilde = ql::eta(c1, im1x, c2, 0.0, 0.0);
        else if ( ql::Real(c2) > 0.0)
            etatilde = 0;
        else if (im1 > 0.0 && im2x > 0.0)
            etatilde = -1;
        else if (im1 < 0.0 && im2x < 0.0)
            etatilde = 1;
        else
            etatilde = 0;

        return etatilde;
    }


    /*!
    * \param z2
    * \param im2
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput xetatilde(Kokkos::View<const TOutput[2]> &z1, Kokkos::View<const TScale[2]> &im1, TOutput const& z2, TScale const& im2, Kokkos::View<const TOutput[2]> &l1) { 
    
            return l1(1) * TOutput(ql::etatilde(z1(1), im1(1), z2, im2)) - l1(0) * TOutput(ql::etatilde(z1(0), im1(0), z2, im2));
    }


    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput eta5(TOutput const& a, TOutput const& b, TOutput const& c, TOutput const& d, TOutput const& e) {
        TOutput res = TOutput(0.0);
        const TScale ima = ql::Sign(ql::Imag(a));
        const TScale imb = ql::Sign(ql::Imag(b));
        const TScale imc = ql::Sign(ql::Imag(c));
        const TScale imd = ql::Sign(ql::Imag(d));
        const TScale ime = ql::Sign(ql::Imag(e));

        if (ima == imb)
        {
            if (ima == imd)
            {
                if (imc == ime)      res = TOutput(0.0); 
                else if (ima != imc) res = ql::Constants::_2ipi<TOutput, TMass, TScale>() * TOutput(imc);
                else                 res = ql::Constants::_2ipi<TOutput, TMass, TScale>() * TOutput(-ime);
            }
            else if (ima != imc) res = ql::Constants::_2ipi<TOutput, TMass, TScale>() * TOutput(imc);
            else res = TOutput(0.0);
        }
        else if (ima == imd && ima != ime) res =  ql::Constants::_2ipi<TOutput, TMass, TScale>() * TOutput(-ime);
        else res = TOutput(0.0);

        return res;
    }


    /*!
    * \param a
    * \param b
    * \param c
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput eta3(TOutput const& a,TOutput const& b, TOutput const& c) {
        TOutput res = TOutput(0.0);
        const TScale ima = ql::Sign(ql::Imag(a));
        const TScale imb = ql::Sign(ql::Imag(b));
        const TScale imc = ql::Sign(ql::Imag(c));
        if (ima == imb && ima != imc) res = ql::Constants::_2ipi<TOutput, TMass, TScale>() * TOutput(imc);
        return res;
    }

    /*!
    * \param a
    * \param b
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput eta2(TOutput const& a,TOutput const& b) {
        const TScale ima = ql::Imag(a);
        const TScale imb = ql::Imag(b);
        const TScale imab = ql::Imag(a * b);
        return ql::Constants::_2ipi<TOutput, TMass, TScale>() * (ql::Htheta(-ima) * ql::Htheta(-imb) * ql::Htheta(imab) - ql::Htheta(ima) * ql::Htheta(imb) * ql::Htheta(-imab));
    }


    /*!
    * \param i_in
    * \param z_in
    * \param s
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput ltspence(int const& i_in, TOutput const& z_in, TScale const& s) {
        
        TOutput z[2]; //TODO:: change to hardcoded array of size 2
        z[i_in] = z_in;
        z[1 - i_in] = TOutput(1.0) - z_in;

        TOutput ltspence;
        if (ql::Real(z[0]) < 0.5){
            if (Kokkos::abs(z[0]) < 1.0)
                ltspence = ql::ltli2series<TOutput, TMass, TScale>(z[1],s);
            else {
                const TOutput clnz = ql::cLn<TOutput, TMass, TScale>(-z[0], -s);
                ltspence = TOutput(-ql::Constants::_pi2o6<TOutput, TMass, TScale>()) - TOutput(0.5) * clnz * clnz - ql::ltli2series<TOutput, TMass, TScale>(-z[1] / z[0], -s);
            }
        }
        else {
            const TScale az1 = Kokkos::abs(z[1]);
            if (az1 < ql::Constants::_eps15())
                ltspence = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>());
            else if (az1 < 1.0)
                ltspence = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) - ql::cLn<TOutput, TMass, TScale>(z[0],s) * ql::cLn<TOutput, TMass, TScale>(z[1], -s) - ql::ltli2series<TOutput, TMass, TScale>(z[0],-s);
            else {
                const TOutput clnz = ql::cLn<TOutput, TMass, TScale>(-z[0], -s);
                ltspence = TOutput(2.0 * ql::Constants::_pi2o6<TOutput, TMass, TScale>()) + TOutput(0.5) * clnz * clnz - ql::cLn<TOutput, TMass, TScale>(z[0], s) * ql::cLn<TOutput, TMass, TScale>(z[1], -s) + ql::ltli2series<TOutput, TMass, TScale>(-z[0] / z[1], s);
            }
        }

        return ltspence;
    }


    /*!
    * \param z1 input argument
    * \param im1 sign of z1
    * \param z2 input argument
    * \param im2 sign of z2
    * \return the complex Spence's function
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cspence(TOutput const& z1, TScale const& im1, TOutput const& z2, TScale const& im2) {
        TOutput cspence = TOutput(0.0);
        const TOutput z12 = z1 * z2;
        const TScale im12 = im2 * ql::Sign(ql::Real(z1));

        if (ql::Real(z12) > 0.5) {
            cspence = ql::ltspence<TOutput, TMass, TScale>(1, z12, 0.0);
            const int etas = ql::eta<TOutput, TMass, TScale>(z1, im1, z2, im2, im12);
            if (etas != 0) 
                cspence += TOutput(etas) * ql::cLn<TOutput, TMass, TScale>(TOutput(1.0) - z12, -im12) * ql::Constants::_2ipi<TOutput, TMass, TScale>();
        }
        else if (Kokkos::abs(z12) < ql::Constants::_eps4()) {
            cspence = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>());
            if (Kokkos::abs(z12) >  ql::Constants::_eps14())
                cspence += -ql::ltspence<TOutput, TMass, TScale>(0, z12, 0.0) + (ql::cLn<TOutput, TMass, TScale>(z1,im1) + ql::cLn<TOutput, TMass, TScale>(z2,im2)) * z12 * (TOutput(1.0) + z12 * (TOutput(0.5) + z12 * (TOutput(1.0) / TOutput(3.0) + z12 / TOutput(4.0))));
        }
        else
            cspence = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) - ql::ltspence<TOutput, TMass, TScale>(0, z12, 0.0) - (ql::cLn<TOutput, TMass, TScale>(z1, im1) + ql::cLn<TOutput, TMass, TScale>(z2, im2)) * ql::cLn<TOutput, TMass, TScale>(TOutput(1.0) - z12, 0.0);

        return cspence;
    }


    /*!
    * \param z2 input arguments
    * \param im2 input signs.
    * \return the difference of cspence functions 
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput xspence(TOutput const (&z1)[2], TScale const (&im1)[2], TOutput const& z2, TScale const& im2) { 
        
        return ql::cspence<TOutput, TMass, TScale>(z1[1], im1[1], z2, im2) - ql::cspence<TOutput, TMass, TScale>(z1[0], im1[0], z2, im2);

    }
    

    /*!
    * \param x numerator
    * \param y denominator
    * \return the Li2 ratio
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omrat(TScale const& x, TScale const& y) {
        const TScale omarg = x / y;
        const TScale arg = 1.0 - omarg;
        if (arg > 1.0)
            return TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>() - ql::ddilog<TOutput, TMass, TScale>(omarg)) - Kokkos::log(arg) * ql::Lnrat<TOutput, TMass, TScale>(x, y);
        else
            return TOutput(ql::ddilog<TOutput, TMass, TScale>(arg));
    }


    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omrat(TOutput const& x, TOutput const& y, TScale const& ieps1, TScale const& ieps2) {
        const TOutput omarg = x / y;
        const TOutput arg = TOutput(1.0) - omarg;
        const TScale isarg = ql::Sign(ql::Real(x) * ieps2 - ql::Real(y) * ieps1);

        if (Abs(arg) > 1.0) {
            const TScale isomarg = -isarg;
            return TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(omarg, isomarg) - ql::cLn<TOutput, TMass, TScale>(omarg, isomarg) * ql::cLn<TOutput, TMass, TScale>(arg, isarg);
        }
        else
            return ql::denspence<TOutput, TMass, TScale>(arg, isarg);
    }


    /*!
    * Calculates Li[2](1-(z1+ieps1)*(z2+ieps2)) for complex z1,z2
    * Using +Li2(1-z1*z2)                           for z1*z2<1
    * and   -Li2(1-1/(z1*z2))-1/2*(ln(z1)+ln(z2))^2 for z1*z2>1
    * \param z1 input argument
    * \param z2 input argument
    * \param ieps1 sign of z1
    * \param ieps2 sign of z2
    * \return Li2 of the 1-product
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cLi2omx2(TOutput const& z1, TOutput const& z2, TScale const& ieps1, TScale const& ieps2) {
        const TOutput arg = z1 * z2;
        const TScale ieps = ql::Sign(ql::Real(z2) * ieps1 + ql::Real(z1) * ieps2);

        TOutput prod, res;
        if (Kokkos::abs(arg) <= 1.0)
        {
            if (arg == TOutput(0.0) || arg == TOutput(1.0))
                prod = TOutput(0.0);
            else
                prod = (ql::cLn<TOutput, TMass, TScale>(z1, ieps1)+ ql::cLn<TOutput, TMass, TScale>(z2, ieps2)) * ql::cLn<TOutput, TMass, TScale>(TOutput(1.0) - arg, -ieps);
            res = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(arg, ieps) - prod;
        }
        else if (Kokkos::abs(arg) > 1.0)
        {
            const TOutput arg2 = TOutput(1.0) / (z1 * z2);
            const TOutput lnomarg = ql::cLn<TOutput, TMass, TScale>(TOutput(1.0) - arg2, -ieps);
            const TOutput lnarg = -ql::cLn<TOutput, TMass, TScale>(z1, ieps1) - ql::cLn<TOutput, TMass, TScale>(z2, ieps2);
            res = TOutput(-ql::Constants::_pi2o6<TOutput, TMass, TScale>()) + ql::denspence<TOutput, TMass, TScale>(arg2, ieps) + lnarg * lnomarg - TOutput(0.5) * lnarg * lnarg;
        }
        return res;
    }


    /*!
    * expression for dilog(1-(v-i*ep)*(w-i*ep)/(x-i*ep)/(y-i*ep)) for real v,w,x and y
    * \param v numerator
    * \param w numerator
    * \param x denominator
    * \param y denominator
    * \return the dilog ratio
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omx2(TScale const& v, TScale const& w, TScale const& x, TScale const& y) {
        const TScale arg = (v * w) / (x * y);
        const TScale omarg = 1.0 - arg;
        TOutput prod, Li2omx2;

        if (Kokkos::abs(arg) <= 1.0) {
            if (Kokkos::abs(arg) == 0.0 || Kokkos::abs(arg) == 1.0)
                prod = 0.0;
            else
                prod = (ql::Lnrat<TOutput, TMass, TScale>(v, x) + ql::Lnrat<TOutput, TMass, TScale>(w, y)) * TOutput(Kokkos::log(omarg));
            Li2omx2 = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>() - ql::ddilog<TOutput, TMass, TScale>(arg)) - prod;
        }
        else if (Kokkos::abs(arg) > 1.0) {
            const TScale arg2 = (x * y) / (v * w);
            const TOutput lnarg = TOutput(-ql::Lnrat<TOutput, TMass, TScale>(v, x) - ql::Lnrat<TOutput, TMass, TScale>(w, y));
            const TOutput lnomarg = TOutput(Log(1.0 - arg2));
            Li2omx2 = -TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>() - ql::ddilog<TOutput, TMass, TScale>(arg2)) + lnarg * lnomarg - TOutput(0.5) * lnarg * lnarg;
        }

        return Li2omx2;
    }


    /*!
    * Calculates Li[2](1-(z1+ieps1)*(z2+ieps2)) for complex z1,z2
    * Using +Li2(1-z1*z2)                           for z1*z2<1
    * and   -Li2(1-1/(z1*z2))-1/2*(ln(z1)+ln(z2))^2 for z1*z2>1
    * \param z1 input argument
    * \param z2 input argument
    * \param ieps1 sign of z1
    * \param ieps2 sign of z2
    * \return Li2 of the 1-product
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omx2(TOutput const& v, TOutput const& w, TOutput const& x, TOutput const& y, TScale const& ieps1, TScale const& ieps2) {
        return ql::cLi2omx2<TOutput, TMass, TScale>(v / x, w / y, ieps1, ieps2);
    }


    /*!
    * Generalization of cLi2omx2 for 3 arguments.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cLi2omx3(TOutput const& z1, TOutput const& z2, TOutput const& z3, TScale const& ieps1, TScale const& ieps2, TScale const& ieps3) {
        const TOutput arg = z1 * z2 * z3;

        TScale ieps;
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(arg)))
        ieps = ql::Sign(ql::Real(z2 * z3) * ieps1 + ql::Real(z1 * z3) * ieps2 + ql::Real(z1 * z2) * ieps3);

        TOutput res = TOutput(0.0);
        if (Kokkos::abs(arg) <= 1.0) {
            TOutput prod;
            if (arg == TOutput(0.0) || arg == TOutput(1.0))
                prod = 0.0;
            else {
                const TOutput lnarg = ql::cLn<TOutput, TMass, TScale>(z1, ieps1) + ql::cLn<TOutput, TMass, TScale>(z2, ieps2) + ql::cLn<TOutput, TMass, TScale>(z3, ieps3);
                const TOutput lnomarg = ql::cLn<TOutput, TMass, TScale>(TOutput(1.0) - arg, 0.0);
                prod = lnarg * lnomarg;
            }
            res = TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(arg, ieps) - prod;
        }
        else {
            const TOutput arg2 = TOutput(1.0) / (z1 * z2 * z3);
            const TOutput lnarg = -ql::cLn<TOutput, TMass, TScale>(z1, ieps1) - ql::cLn<TOutput, TMass, TScale>(z2, ieps2) - ql::cLn<TOutput, TMass, TScale>(z3, ieps3);
            const TOutput lnomarg = ql::cLn<TOutput, TMass, TScale>(TOutput(1.0) - arg2, 0.0);
            res = - TOutput(ql::Constants::_pi2o6<TOutput, TMass, TScale>()) + ql::denspence<TOutput, TMass, TScale>(arg2, ieps) + lnarg * lnomarg - TOutput(0.5) * lnarg * lnarg;
        }

        return res;
    }


    /*!
    * \param x input argument
    * \param y input argument
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput L0(TMass const& x, TMass const& y) {
        TOutput L0;
        const TMass denom = 1.0 - x / y;
        if (Kokkos::abs(denom) < ql::Constants::_eps7())
            L0 = -TOutput(1.0) - TOutput(denom * (0.5 + denom /3.0));
        else
            L0 = ql::Lnrat<TOutput, TMass, TScale>(x, y) / TOutput(denom);

        return L0;
    }


    /*!
    * \param x input argument
    * \param y input argument
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput L1(TMass const& x, TMass const& y) {
        TOutput L1;
        const TMass denom = 1.0 - x / y;
        if (Kokkos::abs(denom) < ql::Constants::_eps7())
            L1 = -TOutput(1.0) * TOutput(0.5) - TOutput(denom / 3.0 * (1.0 + 3.0 * denom / 4.0));
        else
            L1 = (ql::L0<TOutput, TMass, TScale>(x, y) + TOutput(1.0)) / TOutput(denom);

        return L1;
    }


    /*!
    * Solution of a quadratic equation a*z^2+b*z+c=0.
    * \param a coefficient
    * \param b coefficient
    * \param c coefficient
    * \param l result [-im,+im]
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void solveabc(TMass const& a, TMass const&b, TMass const& c, TOutput (&z)[2]) {
        const TMass discr = b * b - TMass(4.0) * a * c;

        if (ql::iszero<TOutput, TMass, TScale>(a)) Kokkos::printf("solveabc -- equation is not quadratic");

        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(discr))) {
            
            const TMass sgnb = ql::Sign(ql::Real(b));
            if (ql::iszero<TOutput, TMass, TScale>(ql::Real(b))) {
                z[0] = -(TOutput(b) - Kokkos::sqrt(TOutput(discr))) / (TOutput(2.0) * a);
                z[1] = -(TOutput(b) + Kokkos::sqrt(TOutput(discr))) / (TOutput(2.0) * a);
            }
            else {
                if (ql::Real(discr) > 0) {
                    const TMass q = -0.5 * (b + sgnb * Kokkos::sqrt(discr));
                    if (ql::Real(b) > 0) {
                        z[0] = TOutput(c / q);
                        z[1] = TOutput(q / a);
                    }
                    else {
                        z[0] = TOutput(q / a);
                        z[1] = TOutput(c / q);
                    }
                }
                else {
                    z[1] = -(TOutput(b) + sgnb * Kokkos::sqrt(TOutput(discr))) / (TOutput(2.0) * a);
                    z[0] = Kokkos::conj(z[1]);
                    if (ql::Real(b) < 0) {
                        z[0] = z[1];
                        z[1] = Kokkos::conj(z[0]);
                    }
                }
            }
        }
        else {
            TOutput qq = -TOutput(b) + Kokkos::sqrt(TOutput(discr));
            TOutput hh = -TOutput(b) - Kokkos::sqrt(TOutput(discr));

            z[0] = qq * TOutput(0.5) / TOutput(a);
            z[1] = (TOutput(2.0) * TOutput(c)) / qq;

            if (ql::Imag(z[0]) > 0.0) {
                z[0] = hh * TOutput(0.5) / TOutput(a);
                z[1] = (TOutput(2.0) * TOutput(c)) / hh;
            }
        }
    }


    /*!
    * Solution of a quadratic equation a*z^2+b*z+c=0, with a give input discriminant.
    * \param a coefficient
    * \param b coefficient
    * \param c coefficient
    * \param d discriminant
    * \param l result [-im,+im]
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void solveabcd(TOutput const& a, TOutput const&b, TOutput const& c, TOutput const& d, TOutput (&z)[2]) {
        if (a == TOutput(0.0)) {
            if (b == TOutput(0.0)) Kokkos::printf("solveabcd - no possible solution\n");
            z[0] = - c / b; z[1] = z[0];
        }
        else if (c == TOutput(0.0)) {
            z[0] = d / a; z[1] = TOutput(0.0);
        }
        else {
            const TOutput up = - b + d;
            const TOutput dn = - b - d;
            if (Kokkos::abs(up) >= Kokkos::abs(dn)) {
                z[0] = TOutput(0.5) * up / a;
                z[1] = TOutput(2.0) * c / up;
            }
            else {
                z[1] = TOutput(0.5) * dn / a;
                z[0] = TOutput(2.0) * c / dn;
            }
        }
    }


    /*!
    * Solution of a quadratic equation a*z^2+b*z+c=0, with a give input discriminant.
    * \param a coefficient
    * \param b coefficient
    * \param c coefficient
    * \param d discriminant
    * \param l result [-im,+im]
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void solveabcd(TOutput const& a, TOutput const&b, TOutput const& c, TOutput (&z)[2]) {
        if (a == TOutput(0.0)) {
            if (b == TOutput(0.0)) Kokkos::printf("solveabcd - no possible solution\n");
            z[0] = -c / b; z[1] = z[0];
        }
        else if (c == TOutput(0.0)) {
            z[0] = TOutput(0.0); z[1] = TOutput(0.0);
        }
        else {
            const TOutput d = Kokkos::sqrt(b * b - 4.0 * a * c);
            const TOutput up = - b + d;
            const TOutput dn = - b - d;
            if (Kokkos::abs(up) >= Kokkos::abs(dn)) {
                z[0] = TOutput(0.5) * up / a;
                z[1] = TOutput(2.0) * c / up;
            }
            else {
                z[1] = TOutput(0.5) * dn / a;
                z[0] = TOutput(2.0) * c / dn;
            }
        }
    }


    /*!
    * Calculate the function
    * \f[
    * R(p^2,m,m_p) = \frac{p_3^2+m_4^2-m_3^2+\sqrt{(p_3^2+m_4^2-m_3^2)^2-4 p_3^2 m_4^2}}{-p_3^2+m_4^2-m_3^2+\sqrt{(p_3^2+m_4^2-m_3^2)^2-4 p_3^2 m_4^2}}
    * \f]
    * where the roots are allowed to be imaginary.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void ratgam(TOutput &ratp, TOutput &ratm, TScale &ieps, TMass const& p3sq, TMass const& m3sq, TMass const& m4sq) {
        
        const TOutput root = Kokkos::sqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq - m3sq + m4sq, 2) - 4.0 * m4sq * p3sq));
        ratp = (TOutput(p3sq + m4sq - m3sq) + root) / (TOutput(-p3sq + m4sq - m3sq) + root);
        ratm = (TOutput(p3sq + m4sq - m3sq) - root) / (TOutput(-p3sq + m4sq - m3sq) - root);
        ieps = 0.0;
    }



    /*!
    * Calculate the function
    * \f[
    * R = \frac{\sigma-i \epsilon}{\tau-i \epsilon}
    * \f]
    * where sigma and tau are real and ieps give the sign if i*pi.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void ratreal(TMass const& si, TMass const& ta, TMass &rat, TScale &ieps) {
        rat = si / ta;
        if (ql::Real(rat) > 0.0)
            ieps = 0.0;
        else if (ql::Real(si) < 0.0)
            ieps = -1.0;
        else if (ql::Real(ta) < 0.0)
            ieps = 1.0;
        else if (ql::Real(ta) == 0.0)
            Kokkos::printf("error in ratreal\n");
    }
    

    /*!
    * \brief Tools<TOutput, TMass, TScale>::qlZlogint
    * \param z
    * \param ieps
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Zlogint(TOutput const& z, TScale const& ieps) {
        const TOutput omz = TOutput(1.0) - z;
        return omz * (ql::cLn<TOutput, TMass, TScale>(omz, ieps) - TOutput(1.0)) - (-z) * (ql::cLn<TOutput, TMass, TScale>(-z, ieps) - TOutput(1.0));
    }


    /*!
    * Finite Triangle Li2 sum.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput R3int(TOutput const& a, TOutput const& s1, TOutput const& s2, TOutput const& t1, TOutput const& t2, TOutput const& t3, TOutput const& t4) {
        
        const TOutput b = (s1 + s2) * (s1 - s2) - a;
        const TOutput c = s2 * s2;
        const TOutput d = Kokkos::sqrt((a - (s1 + s2) * (s1 + s2)) * (a - (s1 - s2) * (s1 - s2)));
        TOutput y[2]; //TODO:: change to hardcoded array of size 2
        TOutput s[2]; //TODO:: change to hardcoded array of size 2
        TOutput res;
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, d, y);
        ql::solveabcd<TOutput, TMass, TScale>(a, t2, t3, t4, s);

        const TOutput y0 = -(t1 + b * s[0]) / t4;
        const TOutput dq0 = (y0 - y[0]);
        const TOutput dq1 = (y0 - y[1]);

        const TOutput OneOdq0 = TOutput(1.0) / dq0;
        const TOutput OneMy0 = TOutput(1.0) - y[0];
        const TScale SignImagOneOdq0 = ql::Sign(ql::Imag(OneOdq0));

        const TOutput OneOdq1 = TOutput(1.0) / dq1;
        const TOutput OneMy1 = TOutput(1.0) - y[1];
        const TScale SignImagOneOdq1 = ql::Sign(ql::Imag(OneOdq1));

        res = ql::cspence<TOutput, TMass, TScale>(-y[0], ql::Sign(ql::Imag(-y[0])), OneOdq0, SignImagOneOdq0)
             -ql::cspence<TOutput, TMass, TScale>(OneMy0, ql::Sign(ql::Imag(OneMy0)), OneOdq0, SignImagOneOdq0)
             +ql::cspence<TOutput, TMass, TScale>(-y[1], ql::Sign(ql::Imag(-y[1])), OneOdq1, SignImagOneOdq1)
             -ql::cspence<TOutput, TMass, TScale>(OneMy1, ql::Sign(ql::Imag(OneMy1)), OneOdq1, SignImagOneOdq1);

        TOutput zz = y0 * (a * y0 + b);

        if (Kokkos::abs(ql::Real(zz)) * ql::Constants::_reps() * ql::Constants::_reps() <= Kokkos::abs(ql::Imag(zz)) * ql::Constants::_neglig() && Kokkos::abs(ql::Imag(zz)) <= Kokkos::abs(ql::Real(zz)) * ql::Constants::_neglig())
            zz = (TOutput(ql::Real(zz)) + c) / a;
        else
            zz = (zz + c) / a;

        // ajust complex logs
        TOutput extra = ql::eta3<TOutput, TMass, TScale>(-y[0], -y[1], c / a) - ql::eta3<TOutput, TMass, TScale>(dq0, dq1, zz);
        if (ql::Real(a) < 0.0 && ql::Imag(zz) < 0.0) 
            extra -= ql::Constants::_2ipi<TOutput, TMass, TScale>();
        if (extra != TOutput(0.0)) {
            const TOutput arg4 = (y0 - TOutput(1.0)) / y0;
            res += extra * ql::cLn<TOutput, TMass, TScale>(arg4, ql::Sign(ql::Imag(arg4)));
        }

        return res;
    }


    /*!
    * Finite Triangle Li2 sum.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput R3int(TOutput const& a, TOutput const& s1, TOutput const& s2, TOutput const& t1) {
        const TOutput b = (s1 + s2) * (s1 - s2) - a;
        const TOutput c = s2 * s2;
        const TOutput d = Kokkos::sqrt((a - (s1 + s2) * (s1 + s2)) * (a - (s1 - s2) * (s1 - s2)));
        
        TOutput y[2]; //TODO:: change to hardcoded array of size 2
        TOutput res;
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, d, y);

        const TOutput y0 = t1;
        const TOutput dq0 = (y0 - y[0]);
        const TOutput dq1 = (y0 - y[1]);

        const TOutput OneOdq0 = TOutput(1.0) / dq0;
        const TOutput OneMy0 = TOutput(1.0) - y[0];
        const TScale SignImagOneOdq0 = ql::Sign(ql::Imag(OneOdq0));

        const TOutput OneOdq1 = TOutput(1.0) / dq1;
        const TOutput OneMy1 = TOutput(1.0) - y[1];
        const TScale SignImagOneOdq1 = ql::Sign(ql::Imag(OneOdq1));

        res = ql::cspence<TOutput, TMass, TScale>(-y[0], ql::Sign(ql::Imag(-y[0])), OneOdq0, SignImagOneOdq0)
             -ql::cspence<TOutput, TMass, TScale>(OneMy0, ql::Sign(ql::Imag(OneMy0)), OneOdq0, SignImagOneOdq0)
             +ql::cspence<TOutput, TMass, TScale>(-y[1], ql::Sign(ql::Imag(-y[1])), OneOdq1, SignImagOneOdq1)
             -ql::cspence<TOutput, TMass, TScale>(OneMy1, ql::Sign(ql::Imag(OneMy1)), OneOdq1, SignImagOneOdq1);

        TOutput zz = y0 * (a * y0 + b);

        if (Kokkos::abs(ql::Real(zz)) * ql::Constants::_reps() * ql::Constants::_reps() <= Kokkos::abs(ql::Imag(zz)) * ql::Constants::_neglig() && Kokkos::abs(ql::Imag(zz)) <= Kokkos::abs(ql::Real(zz)) * ql::Constants::_neglig())
            zz = (TOutput(ql::Real(zz)) + c) / a;
        else
            zz = (zz + c) / a;

        // ajust complex logs
        TOutput extra = ql::eta3<TOutput, TMass, TScale>(-y[0], -y[1], c / a) - ql::eta3<TOutput, TMass, TScale>(dq0, dq1, zz);
        if (ql::Real(a) < 0.0 && ql::Imag(zz) < 0.0)
            extra -= ql::Constants::_2ipi<TOutput, TMass, TScale>();
        if (extra != TOutput(0.0)) {
            const TOutput arg4 = (y0 - TOutput(1.0))/y0;
            res += extra * ql::cLn<TOutput, TMass, TScale>(arg4, ql::Sign(ql::Imag(arg4)));
        }

        return res;
    }


    /*!
    * Finite Triangle Li2 sum.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput R2int(TOutput const& a, TOutput const& b, TOutput const& y0) {
        
        const TOutput y1 = -b / a;
        const TOutput dq0 = y0 - y1;

        const TOutput oneOdq0 = TOutput(1.0) / dq0;
        const TScale  SignImagOneOdq0 = ql::Sign(ql::Imag(oneOdq0));
        const TOutput oneMy1 = TOutput(1.0) - y1;

        TOutput res = ql::cspence<TOutput, TMass, TScale>(-y1, ql::Sign(ql::Imag(-y1)), oneOdq0, SignImagOneOdq0)
                     -ql::cspence<TOutput, TMass, TScale>(oneMy1, ql::Sign(ql::Imag(oneMy1)), oneOdq0, SignImagOneOdq0);

        TOutput extra = ql::eta5<TOutput, TMass, TScale>(a, -y1, b, dq0, a * dq0);
        if (extra != TOutput(0.0)) {
            const TOutput arg4 = (y0 - TOutput(1.0)) / y0;
            res += extra * ql::cLn<TOutput, TMass, TScale>(arg4, ql::Sign(ql::Imag(arg4)));
        }
        return res;
    }


    /*!
    * \param y
    * \param z
    * \param ieps
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Rint(TOutput const& y, TOutput const& z, TScale const& ieps) {
        
        const TOutput omz = TOutput(1.0) - z;
        const TOutput ymone = y - TOutput(1.0);
        const TOutput invymz = TOutput(1.0)/(y - z);
        const TOutput c[2] = { y * invymz, ymone * invymz }; 

        if (ql::Imag(z) != 0.0) {
            const TOutput c2ipi = TOutput{0.0, 2.0 * ql::Constants::_pi()};
            const TOutput a = -z;
            const TOutput b = invymz;
            const TOutput ab= -z * invymz;
            const TOutput eta1 = c2ipi * TOutput( ql::Htheta(-ql::Imag(a)) * ql::Htheta(-ql::Imag(b)) * ql::Htheta(Imag(ab))
                                                 -ql::Htheta(ql::Imag(a)) * ql::Htheta(Imag(b)) * ql::Htheta(-ql::Imag(ab)) );
            const TOutput a2 = omz;
            const TOutput ab2= omz * invymz;
            const TOutput eta2 = c2ipi * TOutput( ql::Htheta(-ql::Imag(a2)) * ql::Htheta(-ql::Imag(b)) * ql::Htheta(ql::Imag(ab2))
                                                 -ql::Htheta(ql::Imag(a2)) * ql::Htheta(ql::Imag(b)) * ql::Htheta(-ql::Imag(ab2)) );

            TOutput logc1 = TOutput(0.0), logc2 = TOutput(0.0);
            if (eta1 != TOutput(0.0)) logc1 = TOutput(Kokkos::log(c[0]));
            if (eta2 != TOutput(0.0)) logc2 = TOutput(Kokkos::log(c[1]));

            return ql::denspence<TOutput, TMass, TScale>(c[0], 0.0) - ql::denspence<TOutput, TMass, TScale>(c[1], 0.0) + eta1 * logc1 - eta2 * logc2;
        }
        else {
            const TScale ieps1 = -ieps * ql::Sign(ql::Real(y));
            const TScale ieps2 = -ieps * ql::Sign(ql::Real(ymone));
            return ql::denspence<TOutput, TMass, TScale>(c[0], ieps1) - ql::denspence<TOutput, TMass, TScale>(c[1], ieps2);
        }
    }


    /*!
    * \brief Tools<TOutput, TMass, TScale>::R
    * \param q
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void R(TOutput &r, TOutput &d, TOutput const& q) {
        d = Kokkos::sqrt(q * q - 4.0);
        r  = q + d;
        TOutput r2 = q - d;
        TScale a = Kokkos::abs(r), b = Kokkos::abs(r2);
        if (b > a) { 
            r = r2; 
            d = -d;
        }
        a = ql::Imag(q);
        b = ql::Imag(r);
        if (a == 0.0) {
            if (b <= 0.0)
                r /= TOutput(2.0);
            else {
                r = TOutput(2.0) / r;
                d = -d;
            }
        }
        else {
            const TScale ik = ql::Sign(ql::Real(a));
            const TScale ir = ql::Sign(ql::Real(b));
            if (ir == ik)
                r /= TOutput(2.0);
            else {
                r = TOutput(2.0) / r;
                d = -d;
            }
        }
        return;
    }


    /*!
    * Calculate the K-function give in Eq. 2.7 of \cite Beenakker:1988jr
    * \f[
    *  K(p^2,m,m_p) = \frac{1-\sqrt{1-4m m_p / (z-(m-m_p)^2)}}{1+\sqrt{1-4m m_p / (z-(m-m_p)^2)}}
    * \f]
    * and fill x[0] = -K, x[1] = 1+K, x[2] = 1-K, the roots are allowed to be imaginary.
    * ieps gives the sign of the imaginary part of -K: 1 -> +i*eps
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void kfn(Kokkos::View<TOutput[3]> &res, TScale& ieps, TMass const& xpi, TMass const& xm, TMass const& xmp) {
        
        if (xm == TMass(0.0) || xmp == TMass(0.0))
            Kokkos::printf("Error in kfn,xm,xmp");

        const TOutput xx1 = TOutput(xpi - (xm - xmp) * (xm - xmp));
        const TOutput rat = TOutput(xx1 / (4.0 * xm * xmp));
        if (ql::iszero<TOutput, TMass, TScale>(ql::Real(rat))) {
            res[1] = -TOutput(2.0) * Kokkos::sqrt(rat) * TOutput{0.0, 1.0} + TOutput(2.0) * rat;
            res[0] = TOutput(1.0) - res[1];
            res[2] = TOutput(2.0) - res[1];
        } 
        else {
            const TOutput root = Kokkos::sqrt((rat - TOutput(1.0)) / rat);
            const TOutput invopr = TOutput(1.0)/(TOutput(1.0) + root);
            res[0] = -invopr * invopr / rat;
            res[1] = TOutput(2.0) * invopr;
            res[2] = TOutput(2.0) * root * invopr;
        }
        ieps = 1.0;
    }

}
