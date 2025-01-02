//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#pragma once

#include "tadpole.h"

namespace ql
{
  /**
   * @brief The TadPoleGPU class, inheriting from TadPole.
   *
   * Specializes the TadPole class for GPU-based computations.
   */
  template<typename TOutput = complex, typename TMass = double, typename TScale = double>
  class TadPoleGPU : public TadPole<TOutput, TMass, TScale>
  {
  public:
    TadPoleGPU();  //!< Constructor.
    ~TadPoleGPU(); //!< Destructor.

    //! Computes the tadpole integral on a GPU
    void integral(vector<TOutput>& res, const TScale& mu2, vector<TMass> const& m, vector<TScale> const& p = {}) override;
  };
}
