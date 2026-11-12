#include "lazyipdca_qcqp/dca_filter.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
QuadraticFunction quadratic(Vec linear,std::vector<SymTriplet> hessian,double constant=0) {
  QuadraticFunction q;q.linear=std::move(linear);q.constant=constant;
  q.hessian.n=q.linear.size();q.hessian.entries=std::move(hessian);return q;
}
void original_certificate(const QCQPProblem& p,const Result& r) {
  if(r.certificate_equality_multipliers.size()!=p.equalities.size() ||
     r.certificate_inequality_multipliers.size()!=p.inequalities.size())
    throw std::runtime_error("missing original multipliers");
  Vec gradient=p.objective.gradient(r.x);double primal=0,comp=0;
  for(std::size_t e=0;e<p.equalities.size();++e){
    primal=std::max(primal,std::abs(p.equalities[e].value(r.x)));
    axpy(r.certificate_equality_multipliers[e],p.equalities[e].gradient(r.x),gradient);
  }
  for(std::size_t i=0;i<p.inequalities.size();++i){
    const double value=p.inequalities[i].value(r.x),z=r.certificate_inequality_multipliers[i];
    if(z < -1e-12)throw std::runtime_error("negative original multiplier");
    primal=std::max(primal,value);comp=std::max(comp,std::abs(z*value));
    axpy(z,p.inequalities[i].gradient(r.x),gradient);
  }
  for(int j=0;j<p.n;++j){
    gradient[j]+=r.certificate_upper_multipliers[j]-r.certificate_lower_multipliers[j];
    primal=std::max({primal,p.lb[j]-r.x[j],r.x[j]-p.ub[j]});
    comp=std::max({comp,std::abs(r.certificate_lower_multipliers[j]*(r.x[j]-p.lb[j])),
                        std::abs(r.certificate_upper_multipliers[j]*(p.ub[j]-r.x[j]))});
  }
  if(primal>1e-6 || comp>1e-6 || inf_norm(gradient)>1e-6)
    throw std::runtime_error("independent original KKT certificate failed");
}
}
int main(){try{
  for(const auto* backend:{"native","auto"})for(bool reduce:{false,true}){
    QCQPProblem p;p.n=4;p.lb={-2,-2,-2,0.25};p.ub={2,2,2,0.25};p.x0={0.6,0.2,0.4,0.25};
    // Analytical local optimum in the positive component (1,0,1,0.25). Both equality curvatures and the
    // nonconvex inequality enter the ORIGINAL Lagrangian operator.
    p.objective=quadratic({2,0,0,0},{{0,0,2},{1,1,2}});
    p.equalities.push_back(quadratic({0,0,1,0},{{0,0,-2}}));
    p.inequalities.push_back(quadratic({0,0,0,0},{{0,0,-2}},1));
    p.inequalities.push_back(quadratic({0,0,0,0},{{1,1,2}},-1));
    Settings s;s.linear_backend="native";s.dca_engine="filter";s.dca_aggregate=true;s.dca_aggregate_backend=backend;
    s.lazy_newton_lazy_constraints=false;s.lazy_newton_reduction=reduce;
    s.max_outer_iterations=250;s.live_display_level=0;
    s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
    auto r=solve_filter_dca(p,s);
    std::cout<<"backend="<<backend<<" reduction="<<reduce<<" "<<r.message<<" accelerated="<<r.dca_filter.aggregate_acceptances<<" x="<<r.x[0]<<"\n";
    if(r.status!=Status::Solved||!r.dca_filter.aggregate_acceptances||std::abs(r.x[0]-1)>1e-5||std::abs(r.x[2]-1)>1e-5)
      throw std::runtime_error("original aggregate analytical constrained solution failed");
    if(r.dca_filter.aggregate_max_accepted_residual>s.kkt_backward_error_reject)
      throw std::runtime_error("original expanded Newton residual gate violated");
    original_certificate(p,r);
  }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
