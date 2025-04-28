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

    // template<typename TOutput, typename TMass, typename TScale>
    struct Constants {

        static Kokkos::View<double*> _C() {
            Kokkos::View<double*> d_vec("Bernoulli Const _C", 19);
    
            // Host mirror to initialize
            auto h_vec = Kokkos::create_mirror(d_vec);
    
            h_vec(0) = 0.4299669356081370;
            h_vec(1) = 0.4097598753307711;
            h_vec(2) = -0.0185884366501460;
            h_vec(3) = 0.0014575108406227;
            h_vec(4) = -0.0001430418444234;
            h_vec(5) = 0.0000158841554188;
            h_vec(6) = -0.0000019078495939;
            h_vec(7) = 0.0000002419518085;
            h_vec(8) = -0.0000000319334127;
            h_vec(9) = 0.0000000043454506;
            h_vec(10) = -0.0000000006057848;
            h_vec(11) = 0.0000000000861210;
            h_vec(12) = -0.0000000000124433;
            h_vec(13) = 0.0000000000018226;
            h_vec(14) = -0.0000000000002701;
            h_vec(15) = 0.0000000000000404;
            h_vec(16) = -0.0000000000000061;
            h_vec(17) = 0.0000000000000009;
            h_vec(18) = -0.0000000000000001;


            // Copy data to device
            Kokkos::deep_copy(d_vec, h_vec);
    
            return d_vec;
        }

        static Kokkos::View<double*> _B() {
            Kokkos::View<double*> d_vec("Bernoulli Const _B", 25);
    
            // Host mirror to initialize
            auto h_vec = Kokkos::create_mirror(d_vec);
    
            h_vec(0) = 0.02777777777777777777777777777777777777777778774E0;
            h_vec(1) = -0.000277777777777777777777777777777777777777777778E0;
            h_vec(2) = 4.72411186696900982615268329554043839758125472E-6;
            h_vec(3) = -9.18577307466196355085243974132863021751910641E-8;
            h_vec(4) = 1.89788699889709990720091730192740293750394761E-9;
            h_vec(5) = -4.06476164514422552680590938629196667454705711E-11;
            h_vec(6) = 8.92169102045645255521798731675274885151428361E-13;
            h_vec(7) = -1.993929586072107568723644347793789705630694749E-14;
            h_vec(8) = 4.51898002961991819165047655285559322839681901E-16;
            h_vec(9) = -1.035651761218124701448341154221865666596091238E-17;
            h_vec(10) = 2.39521862102618674574028374300098038167894899E-19;
            h_vec(11) = -5.58178587432500933628307450562541990556705462E-21;
            h_vec(12) = 1.309150755418321285812307399186592301749849833E-22;
            h_vec(13) = -3.087419802426740293242279764866462431595565203E-24;
            h_vec(14) = 7.31597565270220342035790560925214859103339899E-26;
            h_vec(15) = -1.740845657234000740989055147759702545340841422E-27;
            h_vec(16) = 4.15763564461389971961789962077522667348825413E-29;
            h_vec(17) = -9.96214848828462210319400670245583884985485196E-31;
            h_vec(18) = 2.394034424896165300521167987893749562934279156E-32;
            h_vec(19) = -5.76834735536739008429179316187765424407233225E-34;
            h_vec(20) = 1.393179479647007977827886603911548331732410612E-35;
            h_vec(21) = -3.372121965485089470468473635254930958979742891E-37;
            h_vec(22) = 8.17820877756210262176477721487283426787618937E-39;
            h_vec(23) = -1.987010831152385925564820669234786567541858996E-40;
            h_vec(24) = 4.83577851804055089628705937311537820769430091E-42;


            // Copy data to device
            Kokkos::deep_copy(d_vec, h_vec);
    
            return d_vec;
        }
        
        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static double _qlonshellcutoff() { return 1e-10; }
        KOKKOS_INLINE_FUNCTION static double _pi() { return M_PI; }
        KOKKOS_INLINE_FUNCTION static double _pi2() { return _pi() * _pi(); }
        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static double _pio3() { return _pi() / TScale(3); }
        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static double _pio6() { return _pi() / TScale(6); }
        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static double _pi2o3() { return _pi() * _pio3<TOutput, TMass, TScale>(); }
        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static double _pi2o6() { return _pi() * _pio6<TOutput, TMass, TScale>(); }
        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static double _pi2o12() { return _pi2() / TScale(12); }
        KOKKOS_INLINE_FUNCTION static complex _ipio2() { complex temp(0.0, 0.5 * M_PI); return temp; }
        KOKKOS_INLINE_FUNCTION static double _eps() { return 1e-6; }
        KOKKOS_INLINE_FUNCTION static double _eps4() { return 1e-4; }
        KOKKOS_INLINE_FUNCTION static double _eps7() { return 1e-7; }
        KOKKOS_INLINE_FUNCTION static double _eps15() { return 1e-15; }
        KOKKOS_INLINE_FUNCTION static double _eps14() { return 1e-14; }
        KOKKOS_INLINE_FUNCTION static double _neglig() { return 1e-14; }
        KOKKOS_INLINE_FUNCTION static double _reps() { return 1e-16; }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static double _ieps2() { return TOutput{0.0, _reps() * _reps()}; }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _2ipi() { return TOutput{0.0, 2.0 * M_PI}; }
    };

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput kPow(TOutput const& base, int const& exponent) {
        TOutput temp = TOutput(1.0);

        for (int i = 0; i < exponent; i++)
            temp *= base;

        return temp;
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION bool iszero(TScale const& x) {
        return (x < ql::Constants::_qlonshellcutoff<TOutput, TMass, TScale>()) ? true : false;
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

    
    KOKKOS_INLINE_FUNCTION double Max(double const& a, double const& b) {
        if (Kokkos::abs(a) > Kokkos::abs(b)) 
            return a;
        else 
            return b;
    }


    KOKKOS_INLINE_FUNCTION complex Max(complex const& a, complex const& b) {
        if (Kokkos::abs(a) > Kokkos::abs(b)) 
            return a;
        else 
            return b;
    }

    
    KOKKOS_INLINE_FUNCTION double Min(double const& a, double const& b) {
        if (Kokkos::abs(a) > Kokkos::abs(b)) 
            return b;
        else 
            return a;
    }


    KOKKOS_INLINE_FUNCTION complex Min(complex const& a, complex const& b) {
        if (Kokkos::abs(a) > Kokkos::abs(b)) 
            return b;
        else 
            return a;
    }

    KOKKOS_INLINE_FUNCTION double Htheta(double const& x) { 
        return 0.5 * (1 + ql::Sign(x)); 
    }

}