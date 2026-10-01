// Compiled only with host g++. Do not include Kokkos headers here.
#include "dd_quad_inputs.h"

#include <cstdlib>

extern "C" {
#include <quadmath.h>
}

static DdLimbs q_to_limbs(__float128 q) {
    double hi = (double)q;
    double lo = (double)(q - (__float128)hi);
    return DdLimbs{hi, lo};
}

// Same successive split QuadFloat(double) uses, starting from the quad
// value so the later words are not discarded.
static QuadWords q_to_words(__float128 q) {
    __float128 r = q;
    float w[4];
    for (int i = 0; i < 4; ++i) {
        float c = (float)r;
        w[i] = c;
        r -= (__float128)c;
    }
    return QuadWords{w[0], w[1], w[2], w[3]};
}

DdLimbs dd_mu2() {
    return q_to_limbs(91.2q * 91.2q);
}

DdLimbs dd_mass_10() {
    return q_to_limbs(10.0q);
}

DdLimbs dd_mass_4p9_sq() {
    return q_to_limbs(4.9q * 4.9q);
}

DdLimbs dd_mass_50_sq() {
    return q_to_limbs(50.0q * 50.0q);
}

DdLimbs dd_rand(double min, double max) {
    __float128 lo = min;
    __float128 hi = max;
    return q_to_limbs(lo + std::rand() * 1.0 / RAND_MAX * (hi - lo));
}

DdLimbs dd_rands(double min, double max) {
    __float128 lo = min;
    __float128 hi = max;
    __float128 r1 = lo + std::rand() * 1.0 / RAND_MAX * (hi - lo);
    __float128 draw = std::rand() * 1.0 / RAND_MAX;
    if (draw < 0.5)
        return q_to_limbs(-r1);
    else
        return q_to_limbs(r1);
}

QuadWords qw_mu2() {
    return q_to_words(91.2q * 91.2q);
}

QuadWords qw_mass_10() {
    return q_to_words(10.0q);
}

QuadWords qw_mass_4p9_sq() {
    return q_to_words(4.9q * 4.9q);
}

QuadWords qw_mass_50_sq() {
    return q_to_words(50.0q * 50.0q);
}

QuadWords qw_rand(double min, double max) {
    __float128 lo = min;
    __float128 hi = max;
    return q_to_words(lo + std::rand() * 1.0 / RAND_MAX * (hi - lo));
}

QuadWords qw_rands(double min, double max) {
    __float128 lo = min;
    __float128 hi = max;
    __float128 r1 = lo + std::rand() * 1.0 / RAND_MAX * (hi - lo);
    __float128 draw = std::rand() * 1.0 / RAND_MAX;
    if (draw < 0.5)
        return q_to_words(-r1);
    else
        return q_to_words(r1);
}
