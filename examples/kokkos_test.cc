//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <qcdloop/qcdloop.h>
#include <qcdloop/cache.h>

using std::vector;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;
using ql::complex;
using ql::qcomplex;
using ql::qdouble;

struct squaresum {
  // Specify the type of the reduction value with a "value_type"
  // alias.  In this case, the reduction value has type int.
  using value_type = int;

  // The reduction functor's operator() looks a little different than
  // the parallel_for functor's operator().  For the reduction, we
  // pass in both the loop index i, and the intermediate reduction
  // value lsum.  The latter MUST be passed in by nonconst reference.
  // (If the reduction type is an array like int[], indicating an
  // array reduction result, then the second argument is just int[].)
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, int& lsum) const {
    lsum += i * i;  // compute the sum of squares
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  const double mu2 = ql::Pow(1.7,2.0);
  vector<double> p   = {};
  vector<double>   m = {5.0};
  vector<complex> cm = {{5.0,0.0}};
  vector<complex> res(3);

  ql::Timer tt;
  ql::TadPole<complex,double> tp;
  cout << scientific << setprecision(32);

  tt.start();
  for (int i = 0; i < 1e7; i++) tp.integral(res, mu2, m, p);
  tt.printTime(tt.stop());

  for (size_t i = 0; i < res.size(); i++)
  cout << "eps" << i << "\t" << res[i] << endl;

  Kokkos::finalize();
  return 0;
}