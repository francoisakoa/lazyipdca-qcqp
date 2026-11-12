#include "lazyipdca_qcqp/zero_barrier_forcing.hpp"
#include <stdexcept>
#include <limits>
using lazyipdca::qcqp::detail::zero_barrier_epoch_forcing;
int main() {
  auto check=[](bool v){if(!v)throw std::runtime_error("zero barrier forcing");};
  double last=5;
  for(int k=0;k<100;++k) {
    const double v=zero_barrier_epoch_forcing(5,k,1e-6);
    check(v<=last&&v>=1e-6);last=v;
  }
  check(last==1e-6);
  check(zero_barrier_epoch_forcing(5,0,1e-6)==5);
  check(zero_barrier_epoch_forcing(0,0,1e-6)==1e-6);
  check(zero_barrier_epoch_forcing(std::numeric_limits<double>::infinity(),0,1e-6)==1e-6);
  check(zero_barrier_epoch_forcing(5,-1,1e-6)==1e-6);
  check(zero_barrier_epoch_forcing(5,1,0)==0);
}
