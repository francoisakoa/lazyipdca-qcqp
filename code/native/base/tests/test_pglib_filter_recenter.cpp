#include "lazyipdca_qcqp/majorant_filter_recenter.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
using namespace lazyipdca::qcqp;
static void check(bool p,const char* m){if(!p){std::cerr<<m<<"\n";std::exit(1);}}
int main(){
  using namespace detail;
  check(!Settings{}.majorant_filter_recenter,"must be opt-in");
  check(filter_recenter_needed(false,44.1,1.56e6,58.29,1e-6,1e-6),"observed growth");
  check(!filter_recenter_needed(true,44.1,1.56e6,58.29,1e-6,1e-6),"one-shot guard");
  // A physically feasible primal point is not an original KKT certificate:
  // huge complementarity after an exhausted original filter still needs the
  // same one-shot auxiliary reset. Healthy feasible points remain excluded.
  check(filter_recenter_needed(false,1e-8,1.56e6,58.29,1e-6,1e-6),"feasible primal with runaway complementarity");
  check(filter_recenter_needed(false,0.,1.56e6,58.29,1e-6,1e-6),"zero primal still has bad dual state");
  check(!filter_recenter_needed(false,1e-8,1e-7,1e-7,1e-6,1e-6),"healthy feasible point");
  check(!filter_recenter_needed(false,-1.,1.56e6,58.29,1e-6,1e-6),"invalid negative primal");
  check(!filter_recenter_needed(false,44.1,100,58.29,1e-6,1e-6),"small growth");
  const auto nan=std::numeric_limits<double>::quiet_NaN();
  check(!filter_recenter_needed(false,44.1,nan,58.29,1e-6,1e-6),"finite gate");
  bool used=true;
  check(!filter_recenter_new_barrier_epoch(used,.03,.03)&&used,"same barrier replenished permission");
  check(!filter_recenter_new_barrier_epoch(used,.03,.1)&&used,"larger barrier replenished permission");
  check(!filter_recenter_new_barrier_epoch(used,.03,0)&&used,"zero barrier replenished permission");
  check(!filter_recenter_new_barrier_epoch(used,.03,nan)&&used,"invalid barrier replenished permission");
  check(filter_recenter_new_barrier_epoch(used,.03,.006)&&!used,"strictly lower barrier did not reopen epoch");
  check(filter_recenter_needed(used,0.,1.,.006,1e-6,1e-6),"new epoch unavailable");
  used=true;
  check(!filter_recenter_needed(used,0.,1.,.006,1e-6,1e-6),"same epoch reset repeated");
  used=true;
  check(!filter_recenter_new_barrier_epoch(used,1e-8,1e-8)&&used,"barrier floor replenished permission");
  used=true;
  check(!filter_recenter_new_primal_epoch(used,4e-5,3e-5,1e-6)&&used,"small primal progress replenished permission");
  check(!filter_recenter_new_primal_epoch(used,4e-5,4e-5,1e-6)&&used,"same primal anchor replenished permission");
  check(!filter_recenter_new_primal_epoch(used,4e-5,std::nextafter(2e-5,INFINITY),1e-6)&&used,"unverified half decrease reopened epoch");
  check(filter_recenter_new_primal_epoch(used,4e-5,2e-5,1e-6)&&!used,"halved original primal did not reopen epoch");
  used=true;
  check(!filter_recenter_new_primal_epoch(used,4e-5,1e-6,1e-6)&&used,"feasible original primal reopened same barrier");
  check(!filter_recenter_new_primal_epoch(used,4e-5,nan,1e-6)&&used,"invalid primal reopened epoch");
  const double physical_mu=.0012/.25;
  check(filter_recenter_needed(false,0.,.6,std::min(3.,physical_mu),1e-6,1e-6),"physical scaled barrier reference ignored");
  check(!filter_recenter_needed(false,0.,.6,3.,1e-6,1e-6),"obsolete startup reference control");
  std::vector<double> g{.35,-2.,0.},s{1e-8,3.,4.},z{4e6,2.,0.},ns{7.},nz{8.};
  std::vector<int> active{1,1,0};
  check(make_filter_recenter_state(g,active,s,z,1.,ns,nz),"central state");
  check(ns[0]==1.&&ns[1]==2.&&nz[0]==1.&&nz[1]==.5,"central products");
  check(ns[2]==s[2]&&nz[2]==z[2],"inactive row");
  auto saved_s=ns,saved_z=nz;g[0]=nan;
  check(!make_filter_recenter_state(g,active,s,z,1.,ns,nz),"reject invalid value");
  check(ns==saved_s&&nz==saved_z,"transactional state");
  check(!make_filter_recenter_state(g,active,s,z,0.,ns,nz),"reject zero barrier");
  std::cout<<"FILTER_RECENTER_POLICY_TEST=PASS\n";
}
