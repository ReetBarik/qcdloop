//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//          Stefan Hoeche   : shoeche@fnal.gov
//          Max Knobbe      : mknobbe@fnal.gov
//
// Wrapper header that conditionally includes the appropriate precision version
// of kokkosMaths based on USE_QUAD_COMPLEX define

#pragma once

#if defined(XPMATH_BACKEND_dd) || defined(XPMATH_BACKEND_ff) || defined(XPMATH_BACKEND_qf) || defined(XPMATH_BACKEND_tf)
#include "kokkosMaths_xp.h"
#elif defined(USE_QUAD_COMPLEX)
#ifdef KOKKOS_ENABLE_CUDA
#include "kokkosMaths_quad.h"
#else
#error "USE_QUAD_COMPLEX requires KOKKOS_ENABLE_CUDA to be defined"
#endif
#else
#include "kokkosMaths.h"
#endif
