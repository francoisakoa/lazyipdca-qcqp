#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/restoration.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int main(){try{
  QCQPProblem p;p.name="infeasible_quadratic_shared_phase1_budget";p.n=1;p.x0={0.5};
  p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={0};p.objective.hessian.n=1;
  QuadraticFunction h;h.constant=1;h.linear={0};h.hessian.n=1;h.hessian.entries={{0,0,2}};
  p.equalities.push_back(h);p.validate();
  Settings st;st.restoration_shared_iteration_budget=true;st.restoration_failed_linearization_context=true;
  st.restoration_max_restarts=0;st.restoration_minimax_enabled=false;st.feasibility_tolerance=1e-12;
  st.linear_backend="mumps";st.linear_metis=true;
  int calls=0,used=0;constexpr int total=12;
  for(int budget:{0,1,2,3,6,12,240}) {
    st.restoration_max_iterations=budget;
    st.restoration_deadline_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()+2;
    const auto r=FeasibilityRestoration(st).solve(p,p.x0);
    check(r.iterations<=budget,"legacy+exact exceeded shared iteration budget");
    check(!r.success,"infeasible original equality fabricated a phase-I success");
    check(p.x0[0]==0.5,"phase I changed frozen original start");
    std::cout<<"budget="<<budget<<" used="<<r.iterations<<" linear="<<r.linear_solves<<" exact="<<r.exact_hessian_solves<<"\n";
  }
  Vec private_start=p.x0;
  while(used<total&&calls<4) {
    st.restoration_max_iterations=total-used;
    const auto r=FeasibilityRestoration(st).solve(p,private_start);used+=r.iterations;++calls;
    check(used<=total,"private stages restarted global phase-I iteration budget");
    if(!r.success||r.iterations==0)break;private_start=r.x;
  }
  st.restoration_deadline_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()-1;
  const auto expired=FeasibilityRestoration(st).solve(p,p.x0);
  check(expired.iterations==0,"expired shared deadline spent restoration iterations");
  std::cout<<"PASS shared native Phase-I budget and deadline; no original success or input mutation\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
