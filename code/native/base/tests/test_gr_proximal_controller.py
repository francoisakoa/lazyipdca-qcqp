#!/usr/bin/env python3
"""Compile the actual engine retry lambda against an inert controller fixture.

No optimization, factorization or native solve is performed. This test checks
that a blocked proposal consumes no retry and changes no accepted/model state.
"""
from pathlib import Path
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    source = args.source.resolve()
    engine = (source / 'src/dca_inequality_majorant_engine.hpp').read_text()
    start = engine.index('auto retry_filter_proximal=[&]() {')
    end = engine.index('\n            };', start) + len('\n            };')
    actual_lambda = engine[start:end]
    # The test exercises the implementation in the shipped engine, not a
    # second transcription of the retry policy.
    cpp = r'''
#include "failure_recovery_policy.hpp"
#include "gr_proximal_policy.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace detail=lazyipdca::qcqp::detail;
static int checks=0;
void require(bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);}
struct ModelFixture {
  bool hybrid=false,study_trial_pending=false;int discarded=0,rolled_back=0;
  bool study_hybrid_controller_enabled()const{return hybrid;}
  void discard(){++discarded;}
  void study_rollback_trial(){++rolled_back;study_trial_pending=false;}
  const char* study_metric_digest()const{return "fixture";}
};
void scenario(bool gr,const char* policy,bool hybrid,bool atomic=false,bool initial_restoration=false){
  Settings st;st.gr_proximal_policy=policy;
  st.lazy_newton_atomic_regularization=atomic;
  st.majorant_initial_restoration=initial_restoration;
  st.lazy_newton_regularization_initial=1e-8;
  ModelFixture model;model.hybrid=hybrid;model.study_trial_pending=hybrid;auto* models=&model;
  struct WorkspaceFixture {Vec x{1.,2.},rd{.01,.02};double regularization=.001;}w;
  const Vec pristine_x=w.x;
  struct InnerFixture {double tau_prox=.001;}hybrid_inner;
  bool square_feasibility=false,gr_globalization=gr,hybrid_skip_trial_once=false;
  int filter_proximal_retries=0,outer=17,level=0;
  double rho_memory=.001,study_margin_proximal_floor=0;
  __ACTUAL_RETRY_LAMBDA__
  int admitted=0;
  while(retry_filter_proximal()){
    ++admitted;require(admitted<=4,"actual retry controller exceeded four attempts");
  }
  const bool frozen=gr&&std::string(policy)=="frozen";
  if(frozen||atomic||initial_restoration){
    require(admitted==0&&filter_proximal_retries==0,"disabled/frozen retry consumed budget");
    require(rho_memory==.001,"frozen remembered proximal changed");
    require(model.discarded==0&&model.rolled_back==0,"blocked retry mutated model state");
    require(model.study_trial_pending==hybrid,"blocked retry discarded pending metric");
    require(!hybrid_skip_trial_once,"blocked retry changed controller flags");
  }else{
    require(admitted==4&&filter_proximal_retries==4,"historical retry bound changed");
    require(rho_memory==10.,"historical tenfold memory changed");
    require(model.discarded==4&&model.rolled_back==4,"historical rollback changed");
    require(hybrid_skip_trial_once==hybrid,"historical hybrid flag changed");
  }
  require(w.x==pristine_x,"proximal retry changed original primal anchor");
  require(w.regularization==.001&&hybrid_inner.tau_prox==.001,
      "globalization policy altered numerical/model regularization in place");
}
int main(){try{
  for(bool gr:{false,true})for(const char* p:{"historical","frozen"})for(bool hybrid:{false,true})
    scenario(gr,p,hybrid);
  scenario(true,"frozen",false,true,false);
  scenario(true,"historical",false,true,false);
  scenario(true,"frozen",false,false,true);
  scenario(true,"historical",false,false,true);
  std::cout<<"gr_proximal_actual_retry_controller PASS checks="<<checks<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}
'''.replace('__ACTUAL_RETRY_LAMBDA__', actual_lambda)
    args.output.mkdir(parents=True, exist_ok=True)
    cpp_path = args.output / 'actual_retry.cpp'
    binary = args.output / 'actual_retry'
    cpp_path.write_text(cpp)
    subprocess.run([args.cxx, '-std=c++17', '-O2', '-Wall', '-Wextra',
                    '-I' + str(source / 'include'), '-I' + str(source / 'src'),
                    str(cpp_path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
