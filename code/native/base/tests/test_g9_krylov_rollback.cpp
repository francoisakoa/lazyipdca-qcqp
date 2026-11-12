#include "lazyipdca_qcqp/dca_filter.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(){try{
  QCQPProblem p;p.n=6;p.lb=Vec(6,-2);p.ub=Vec(6,2);p.x0={0.5,0.6,0.7,0.8,0.9,0.4};
  p.objective.linear=Vec(6,0);p.objective.hessian.n=6;
  for(int j=0;j<6;++j){p.objective.hessian.entries.push_back({j,j,double(2+3*j)});
    QuadraticFunction q;q.linear=Vec(6,0);q.hessian.n=6;q.hessian.entries={{j,j,-2}};q.constant=1;p.inequalities.push_back(q);}
  Settings s;s.linear_backend="native";s.dca_engine="filter";s.live_display_level=0;
  s.max_outer_iterations=1;
  const auto reference=solve_filter_dca(p,s);
  s.dca_krylov=true;s.dca_krylov_max_iterations=1;s.dca_krylov_failure_cooldown=0;
  const auto rejected=solve_filter_dca(p,s);
  if(rejected.dca_filter.krylov_trials!=1 || rejected.dca_filter.krylov_acceptances!=0 ||
     rejected.dca_filter.krylov_linear_rejections!=1)
    throw std::runtime_error("test did not exercise rejected original-system acceleration");
  if(reference.x!=rejected.x || reference.status!=rejected.status ||
     reference.certificate_equality_multipliers!=rejected.certificate_equality_multipliers ||
     reference.certificate_inequality_multipliers!=rejected.certificate_inequality_multipliers ||
     reference.certificate_lower_multipliers!=rejected.certificate_lower_multipliers ||
     reference.certificate_upper_multipliers!=rejected.certificate_upper_multipliers ||
     reference.lazy_newton.factor_attempts!=rejected.lazy_newton.factor_attempts ||
     reference.dca_filter.models!=rejected.dca_filter.models ||
     reference.dca_filter.curvature_factors!=rejected.dca_filter.curvature_factors)
    throw std::runtime_error("rejected Krylov trial changed reference endpoint or factorizations");
  std::cout<<"Rejected acceleration preserves exact DCA endpoint, multipliers and factor counts\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
