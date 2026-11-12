#include "lazyipdca_qcqp/dca_filter.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(){try{
  QCQPProblem p;p.n=2;p.lb={-2,-2};p.ub={2,2};p.x0={0.999,0.01};
  p.objective.linear={-2,0};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,-1},{1,1,2}};
  QuadraticFunction q;q.linear={0,0};q.constant=-1;q.hessian.n=2;q.hessian.entries={{0,0,2},{1,1,2}};
  p.inequalities={q,q};
  Settings s;s.linear_backend="native";s.dca_engine="filter";s.live_display_level=0;s.max_outer_iterations=200;
  s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
  auto base=solve_filter_dca(p,s);s.dca_local_active=true;s.dca_local_trigger=1e6;
  auto trial=solve_filter_dca(p,s);
  std::cout<<"baseline="<<base.message<<" iterations="<<base.outer_iterations<<" local="<<trial.message<<" iterations="<<trial.outer_iterations
           <<" trials="<<trial.dca_filter.local_trials<<" acceptances="<<trial.dca_filter.local_acceptances<<" equal="<<(base.x==trial.x)<<"\n";
  if(base.status!=Status::Solved||trial.status!=Status::Solved||trial.dca_filter.local_trials==0||
     trial.dca_filter.local_acceptances!=0||base.outer_iterations!=trial.outer_iterations||base.x!=trial.x)
    throw std::runtime_error("rejected rank-deficient manifold changed baseline transaction");
  std::cout<<"PASS dependent active rows: exact baseline trajectory preserved\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
