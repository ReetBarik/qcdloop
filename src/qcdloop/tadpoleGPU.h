//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#pragma once

using complex = Kokkos::complex<double>;

namespace ql
{
  /**
   * @brief The TadPoleGPU class, inheriting from TadPole.
   *
   * Specializes the TadPole class for GPU-based computations.
   */
  template<typename TOutput = complex, typename TMass = double, typename TScale = double>
  class TadPoleGPU 
  {

      Kokkos::View<TOutput* [3]> res;
    public:
      TadPoleGPU(Kokkos::View<TOutput* [3]>& res);  //!< Constructor.
      ~TadPoleGPU(); //!< Destructor.

      //! Computes the tadpole integral on a GPU
      KOKKOS_INLINE_FUNCTION
      void integral(const TScale& mu2, const Kokkos::View<TMass*>& m, const Kokkos::View<TScale*>& p, const int i = {}) const;

  };
}
