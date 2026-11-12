#include "lazyipdca_qcqp/dca_filter.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(){try{
  QCQPProblem p;p.n=3;p.lb=Vec(3,-2);p.ub=Vec(3,2);p.x0={0.2,0.3,0.04};
  p.objective.linear=Vec(3,0);p.objective.hessian.n=3;
  p.objective.hessian.entries={{0,0,-2},{1,1,-4}};
  QuadraticFunction q;q.linear={0,0,1};q.hessian.n=3;q.hessian.entries={{0,0,-2}};
  p.equalities.push_back(q);
  Settings s;s.linear_backend="native";s.dca_engine="filter";s.live_display_level=0;
  s.max_outer_iterations=1;
  const auto reference=solve_filter_dca(p,s);
  s.dca_aggregate=true;s.dca_aggregate_backend="native";s.dca_aggregate_rho_max=0;
  const auto rejected=solve_filter_dca(p,s);
  if(rejected.dca_filter.aggregate_trials!=1 || rejected.dca_filter.aggregate_acceptances!=0 ||
     rejected.dca_filter.aggregate_rejections!=1 || rejected.dca_filter.aggregate_certificates!=1)
    throw std::runtime_error("test did not exercise the unshifted inertia rejection");
  if(reference.x!=rejected.x || reference.status!=rejected.status ||
     reference.certificate_equality_multipliers!=rejected.certificate_equality_multipliers ||
     reference.certificate_inequality_multipliers!=rejected.certificate_inequality_multipliers ||
     reference.certificate_lower_multipliers!=rejected.certificate_lower_multipliers ||
     reference.certificate_upper_multipliers!=rejected.certificate_upper_multipliers ||
     reference.lazy_newton.factor_attempts!=rejected.lazy_newton.factor_attempts ||
     reference.dca_filter.models!=rejected.dca_filter.models ||
     reference.dca_filter.curvature_factors!=rejected.dca_filter.curvature_factors)
    throw std::runtime_error("rejected aggregate model changed the DCA endpoint or factor counts");
  std::cout<<"Rejected aggregate model preserves the original DCA point and multipliers\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
