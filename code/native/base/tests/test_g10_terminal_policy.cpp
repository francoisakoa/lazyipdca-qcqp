#include "../src/dca_filter_contract.hpp"
#include "../src/dca_terminal_policy.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp::detail;
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
OriginalFilterPoint point(double kkt,double primal,double theta=0.01){
  OriginalFilterPoint p{1e6,theta,kkt,1e6};p.primal_inf=primal;p.normalized_kkt=kkt;return p;
}
int main(){try{
  OriginalDcaFilter f(1,1e-6);
  auto from=point(1.8e-3,4.3e-7),to=point(1.48e-2,3e-12,1e-8);
  auto decision=f.diagnose(from,to,1,1e-6);
  require(decision.stationary_phase&&!decision.accepted,
          "case30000 terminal theta gain paid for dual explosion");
  to=point(1.7e-3,8e-7,0.1);
  require(f.acceptable(from,to,0,1e-6),"terminal contraction depended on L1 constraint count");
  to=point(1.7e-3,1.01e-6,1e-10);
  require(!f.acceptable(from,to,1,1e-6),"terminal candidate lost certified primal feasibility");
  to=point(0.9995*from.kkt,4.3e-7);
  require(!f.acceptable(from,to,0,1e-6),"0.05 percent contraction passed terminal filter");
  to.kkt=to.normalized_kkt=0.999*from.kkt;
  require(f.acceptable(from,to,0,1e-6),"exact terminal contraction boundary rejected");
  // Primal feasibility is not a small-KKT phase: retain global objective moves.
  from=point(100,0);to=point(110,0);to.objective=from.objective-1;
  decision=f.diagnose(from,to,1,1e-6);
  require(!decision.stationary_phase&&decision.accepted,"global feasible objective progress disabled");
  require(dca_terminal_radius(1e-6)==0.01,"wrong normalized terminal radius");
  require(!dca_stationary_phase(0,INFINITY,1e-6,1e-6),"nonfinite normalized error is terminal");
  // The full incumbent audit is necessary; the candidate caps and a new state
  // remain necessary too. The 1.8e-3 plateau does not qualify as acceptable.
  from=point(8e-6,1e-10);to=point(9e-6,1e-10);
  to.within_acceptable_caps=true;to.distinct_from_anchor=true;
  require(!f.acceptable(from,to,0,1e-6),"uncertified incumbent enabled acceptable continuation");
  from.acceptable_certified=true;
  require(f.diagnose(from,to,0,1e-6).acceptable_continuation,
          "genuine new acceptable iterate rejected for lack of KKT monotonicity");
  to.distinct_from_anchor=false;
  require(!f.acceptable(from,to,0,1e-6),"same incumbent counted as a new acceptable point");
  to.distinct_from_anchor=true;to.within_acceptable_caps=false;to.kkt=1.8e-3;
  require(!f.acceptable(from,to,0,1e-6),"case30000 plateau entered acceptable continuation");
  DcaStationarityEvidence e;e.certificate(0,false);e.certificate(0.02,true);
  e.candidate(0.02,1.8e-3,0.9995*1.8e-3);
  require(e.needs_rescue(),"rejected sub-0.1-percent gain did not arm exact KKT rescue");
  e.certificate(0.005,true);e.certificate(10,true);e.candidate(10,1.8e-3,1e-3);
  require(e.needs_rescue()&&e.minimum_certified_rho==0.005,
          "later probe overwrote causal evidence or minimum certificate");
  e={};e.certificate(0,false);e.candidate(1,1.8e-3,0.999*1.8e-3);
  require(!e.needs_rescue(),"accepted contraction boundary armed stagnation");
  e={};e.certificate(0,true);e.candidate(1,1.8e-3,1.8e-3);
  require(!e.needs_rescue(),"missing zero-shift failure armed causal rescue");
  require(dca_equality_step(1e-7,1e-6,0.8,0.01)==0.8,"equality multiplier bound by inequality fraction");
  require(dca_equality_step(1e-4,1e-6,0.8,0.01)==0.01,"pre-feasible historical equality step changed");
  std::cout<<"PASS terminal norm, local/global boundary, acceptable distinct iterates, shared contraction, causal accumulation, rho memory, equality step\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
