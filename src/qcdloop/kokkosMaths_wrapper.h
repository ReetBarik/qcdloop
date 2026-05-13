//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Wrapper header that conditionally includes the appropriate precision version
// of kokkosMaths based on USE_DD_COMPLEX / USE_QUAD_COMPLEX defines.
//
//   USE_DD_COMPLEX   -> ql::ddfun double-double (~30-31 digits, all backends)
//   USE_QUAD_COMPLEX -> CUDA __nv_fp128 quad (~33 digits, CUDA sm_100 only)
//   neither          -> Kokkos::complex<double>

#pragma once

#if defined(USE_DD_COMPLEX)
#include "kokkosMaths_dd.h"
#elif defined(USE_QUAD_COMPLEX)
#ifdef KOKKOS_ENABLE_CUDA
#include "kokkosMaths_quad.h"
#else
#error "USE_QUAD_COMPLEX requires KOKKOS_ENABLE_CUDA to be defined"
#endif
#else
#include "kokkosMaths.h"
#endif
