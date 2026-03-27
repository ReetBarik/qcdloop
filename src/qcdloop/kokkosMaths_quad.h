//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Quad precision version of kokkosMaths

#pragma once

#include <math.h>
#include <type_traits>

#ifdef KOKKOS_ENABLE_CUDA
#include "quad_math.hpp"
#include "quad_complex.hpp"
#endif

namespace ql
{
    using complex = ql::quad_complex;

    template<typename T>
    struct Constants {

        // Number of Chebyshev coefficients for ddilog (must match coeffs array in _C)
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_C() { return 43; }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _C(int i) {
            // Quad precision Chebyshev coefficients (43 terms)
            constexpr T coeffs[43] = {
                T(+0.4299669356081369720370336786993879911681q),  // 47
                T(+0.4097598753307710584682637109252552780893q),  // 28
                T(-0.0185884366501459196476416491402122676020q),  // 391
                T(+0.0014575108406226785536739284164594926527q),  // 2055
                T(-0.0001430418444234004877488301200908765223q),  // 11267
                T(+0.0000158841554187955323619055047167739548q),  // 391004
                T(-0.0000019078495938658272271963211420884105q),  // 8015588
                T(+0.0000002419518085416474949946146434289843q),  // 47407588
                T(-0.0000000319334127425178346049601414286296q),  // 365082445
                T(+0.0000000043454506267691229879571784781516q),  // 3020705163
                T(-0.0000000006057848011840744442970533090714q),  // 01310493274
                T(+0.0000000000861209779935949824428368451935q),  // 499015050068
                T(-0.0000000000124433165993886798964242137051q),  // 456159308256
                T(+0.0000000000018225569623573633006554773579q),  // 2903333133341
                T(-0.0000000000002700676604911465180857222938q),  // 07277348726616
                T(+0.0000000000000404220926315266464883285777q),  // 966819039605138
                T(-0.0000000000000061032514526918795037782652q),  // 6125857640983405
                T(+0.0000000000000009286297533019575861302980q),  // 12120905726364786
                T(-0.0000000000000001422602085511244683974902q),  // 77390005481676598
                T(+0.0000000000000000219263171815395735398033q),  // 409420375519846175
                T(-0.0000000000000000033979732421589786339948q),  // 7449166351657333035
                T(+0.0000000000000000005291954244833147145510q),  // 39155749919426092154
                T(-0.0000000000000000000827858081427899765284q),  // 374428975307431576970
                T(+0.0000000000000000000130037173454556037430q),  // 510286627563116086246
                T(-0.0000000000000000000020502222425528249238q),  // 9003899608272738595180
                T(+0.0000000000000000000003243578549148930334q),  // 49269968207481971234178
                T(-0.0000000000000000000000514779990334320717q),  // 855020023960694971647731
                T(+0.0000000000000000000000081938774771715779q),  // 3800068495056627245722365
                T(-0.0000000000000000000000013077835405712667q),  // 2290049437436409702417541
                T(+0.0000000000000000000000002092562930579890q),  // 91245949861835427981622465
                T(-0.0000000000000000000000000335616615054383q),  // 751790629092426343180650287
                T(+0.0000000000000000000000000053946577714317q),  // 5180550927103490540268384425
                T(-0.0000000000000000000000000008689193208690q),  // 9968479986614298596382685532
                T(+0.0000000000000000000000000001402281686966q),  // 58818398547877502045652862885
                T(-0.0000000000000000000000000000226715578131q),  // 8277249048765387314926406643
                T(+0.0000000000000000000000000000036717416991q),  // 368554683868436131912480666
                T(-0.0000000000000000000000000000005956151695q),  // 8775943870885460187487415
                T(+0.0000000000000000000000000000000967662432q),  // 50130423384987872105375089
                T(-0.0000000000000000000000000000000157438595q),  // 0010115002774284579478285
                T(+0.0000000000000000000000000000000025650460q),  // 002467956741653059479711
                T(-0.0000000000000000000000000000000004184519q),  // 8993748074077563959674
                T(+0.0000000000000000000000000000000000683494q),  // 09303501707906999012185
                T(-0.0000000000000000000000000000000000111772q)   // 939575215366434467459690877448
            };
            return coeffs[i];
        }

        // Number of Bernoulli coefficients for li2series (must match coeffs array in _B)
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_B() { return 25; }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _B(int i) {
            // Quad precision Bernoulli coefficients (25 terms)
            constexpr T coeffs[25] = {
                T(0.02777777777777777777777777777777777777777778774E0q),
                T(-0.000277777777777777777777777777777777777777777778E0q),
                T(4.72411186696900982615268329554043839758125472E-6q),
                T(-9.18577307466196355085243974132863021751910641E-8q),
                T(1.89788699889709990720091730192740293750394761E-9q),
                T(-4.06476164514422552680590938629196667454705711E-11q),
                T(8.92169102045645255521798731675274885151428361E-13q),
                T(-1.993929586072107568723644347793789705630694749E-14q),
                T(4.51898002961991819165047655285559322839681901E-16q),
                T(-1.035651761218124701448341154221865666596091238E-17q),
                T(2.39521862102618674574028374300098038167894899E-19q),
                T(-5.58178587432500933628307450562541990556705462E-21q),
                T(1.309150755418321285812307399186592301749849833E-22q),
                T(-3.087419802426740293242279764866462431595565203E-24q),
                T(7.31597565270220342035790560925214859103339899E-26q),
                T(-1.740845657234000740989055147759702545340841422E-27q),
                T(4.15763564461389971961789962077522667348825413E-29q),
                T(-9.96214848828462210319400670245583884985485196E-31q),
                T(2.394034424896165300521167987893749562934279156E-32q),
                T(-5.76834735536739008429179316187765424407233225E-34q),
                T(1.393179479647007977827886603911548331732410612E-35q),
                T(-3.372121965485089470468473635254930958979742891E-37q),
                T(8.17820877756210262176477721487283426787618937E-39q),
                T(-1.987010831152385925564820669234786567541858996E-40q),
                T(4.83577851804055089628705937311537820769430091E-42q)
            };
            return coeffs[i];
        }
        
        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _qlonshellcutoff() {
            return T(1e-20q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _pi() {
            return T(3.14159265358979323846264338327950288419716939937510q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _pi2() { 
            return _pi() * _pi(); 
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pio3() { 
            return _pi() / ql::Constants<TScale>::_three(); 
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pio6() { 
            return _pi() / ql::Constants<TScale>::_six(); 
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pi2o3() { 
            return _pi() * _pio3<TOutput, TMass, TScale>(); 
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pi2o6() { 
            return _pi() * _pio6<TOutput, TMass, TScale>(); 
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pi2o12() { 
            return _pi2() / ql::Constants<TScale>::_twelve(); 
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _zero() {
            return T(0.0q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _half() {
            return T(0.5q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _one() {
            return T(1.0q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _two() {
            return T(2.0q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _three() {
            return T(3.0q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _four() {
            return T(4.0q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _five() {
            return T(5.0q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _six() {
            return T(6.0q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _ten() {
            return T(10.0q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _twelve() {
            return T(12.0q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps() {
            return T(1e-12q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps4() {
            return T(1e-4q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps7() {
            return T(1e-7q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps10() {
            return T(1e-10q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps14() {
            return T(1e-14q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps15() {
            return T(1e-15q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _xloss() {
            return T(0.125q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _neglig() {
            return T(1e-28q);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _reps() {
            return T(1e-32q);
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _2ipi() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_two() * Constants<TScale>::_pi()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ipio2() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_pi() * Constants<TScale>::_half()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ipi() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_pi()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ieps() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_reps()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ieps2() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_reps() * Constants<TScale>::_reps()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ieps50() {
            return TOutput{Constants<TScale>::_zero(), 1e-50q};
        }
    };

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput kPow(TOutput const& base, int const& exponent) {
        TOutput temp = TOutput(1.0q);

        for (int i = 0; i < exponent; i++)
            temp *= base;

        return temp;
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TMass kPow(TMass const& base, int const& exponent) {
        TMass temp = TMass(1.0q);

        for (int i = 0; i < exponent; i++)
            temp *= base;

        return temp;
    }

    // Math dispatch functions - base templates
    template<typename T>
    KOKKOS_INLINE_FUNCTION
    T kAbs(T const& x) {
        return Kokkos::abs(x);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION
    T kLog(T const& x) {
        return Kokkos::log(x);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION
    T kSqrt(T const& x) {
        return Kokkos::sqrt(x);
    }

    // Quad precision specializations
    template<>
    KOKKOS_INLINE_FUNCTION
    fp128_t kAbs(fp128_t const& x) {
        return ql::quad::abs(x);
    }

    template<>
    KOKKOS_INLINE_FUNCTION
    fp128_t kLog(fp128_t const& x) {
        return ql::quad::log(x);
    }

    template<>
    KOKKOS_INLINE_FUNCTION
    fp128_t kSqrt(fp128_t const& x) {
        return ql::quad::sqrt(x);
    }

    // Overloads for quad_complex
    KOKKOS_INLINE_FUNCTION
    fp128_t kAbs(quad_complex const& z) {
        return ql::quad::abs(z);
    }

    KOKKOS_INLINE_FUNCTION
    quad_complex kLog(quad_complex const& z) {
        return ql::quad::log(z);
    }

    KOKKOS_INLINE_FUNCTION
    quad_complex kSqrt(quad_complex const& z) {
        return ql::quad::sqrt(z);
    }

    KOKKOS_INLINE_FUNCTION
    quad_complex kConj(quad_complex const& z) {
        return ql::quad::conj(z);
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION bool iszero(TScale const& x) {
        return (ql::kAbs(x) < ql::Constants<TScale>::template _qlonshellcutoff<TOutput, TMass, TScale>()) ? true : false;
    }

    KOKKOS_INLINE_FUNCTION fp128_t Imag(fp128_t const& x) {
        return fp128_t(0.0q);
    }

    KOKKOS_INLINE_FUNCTION fp128_t Imag(quad_complex const& x) {
        return x.imag();
    }

    KOKKOS_INLINE_FUNCTION fp128_t Real(fp128_t const& x) {
        return x;
    }

    KOKKOS_INLINE_FUNCTION fp128_t Real(quad_complex const& x) {
        return x.real();
    }

    KOKKOS_INLINE_FUNCTION fp128_t Sign(fp128_t const& x) {
        const fp128_t zero = ql::Constants<fp128_t>::_zero();
        const fp128_t one = ql::Constants<fp128_t>::_one();
        if (zero < x) {
            return one;
        } else if (x < zero) {
            return -one;
        } else {
            return zero;
        }
    }

    KOKKOS_INLINE_FUNCTION quad_complex Sign(quad_complex const& x) {
        return x / ql::kAbs(x);
    }

    KOKKOS_INLINE_FUNCTION fp128_t Max(fp128_t const& a, fp128_t const& b) {
        if (ql::kAbs(a) > ql::kAbs(b)) 
            return a;
        else 
            return b;
    }

    KOKKOS_INLINE_FUNCTION quad_complex Max(quad_complex const& a, quad_complex const& b) {
        if (ql::kAbs(a) > ql::kAbs(b)) 
            return a;
        else 
            return b;
    }

    KOKKOS_INLINE_FUNCTION fp128_t Min(fp128_t const& a, fp128_t const& b) {
        if (ql::kAbs(a) > ql::kAbs(b)) 
            return b;
        else 
            return a;
    }

    KOKKOS_INLINE_FUNCTION quad_complex Min(quad_complex const& a, quad_complex const& b) {
        if (ql::kAbs(a) > ql::kAbs(b)) 
            return b;
        else 
            return a;
    }

    KOKKOS_INLINE_FUNCTION fp128_t Htheta(fp128_t const& x) { 
        return fp128_t(0.5q) * (fp128_t(1.0q) + fp128_t(ql::Sign(x))); 
    }

}
