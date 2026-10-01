// Host-only DoubleDouble inputs. This header must not include quadmath.h:
// the kernel translation unit is compiled by nvcc or hipcc, and those
// compilers reject __float128. The definitions live in dd_quad_inputs.cc,
// which is compiled with host g++.
#pragma once

struct DdLimbs {
    double hi;
    double lo;
};

// Four float words of the same quad value, most significant first.
// FloatFloat keeps w0,w1; TripleFloat keeps three; QuadFloat keeps all four.
struct QuadWords {
    float w0, w1, w2, w3;
};

DdLimbs dd_mu2();
DdLimbs dd_mass_10();
DdLimbs dd_mass_4p9_sq();
DdLimbs dd_mass_50_sq();
DdLimbs dd_rand(double min, double max);
DdLimbs dd_rands(double min, double max);

QuadWords qw_mu2();
QuadWords qw_mass_10();
QuadWords qw_mass_4p9_sq();
QuadWords qw_mass_50_sq();
QuadWords qw_rand(double min, double max);
QuadWords qw_rands(double min, double max);
