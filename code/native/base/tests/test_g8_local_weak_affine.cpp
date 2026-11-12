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
  for(bool pairs:{false,true})for(bool weakly_active:{false,true}){
    QCQPProblem p;p.n=3;p.lb=Vec(3,-2);p.ub=Vec(3,2);
    // The unique minimizer is (1, target, 0). The x constraint is strongly
    // active. The curved y constraints exercise both signs of curvature;
    // the optional first y row is active with zero limiting multiplier.
    const double target=weakly_active?0:0.3;
    p.x0={0.999,weakly_active?-0.02:0.2,0};
    p.objective=quadratic({-2,-2*target,0},{{0,0,-1},{1,1,2},{2,2,1}});
    p.equalities.push_back(quadratic({0,0,1},{}));
    p.inequalities.push_back(quadratic({0,0,0},{{0,0,2}},-1));
    if(weakly_active)p.inequalities.push_back(quadratic({0,1,0},{{1,1,1}}));
    else p.inequalities.push_back(quadratic({0,0,0},{{1,1,2}},-0.5));
    p.inequalities.push_back(quadratic({0,0,0},{{1,1,-2}},-0.25));
    Settings s;s.linear_backend="native";s.dca_engine="filter";
    s.dca_local_active=true;s.dca_local_weak_affine=true;s.dca_pair_elimination=pairs;
    s.dca_local_trigger=1e6;s.max_outer_iterations=200;s.live_display_level=0;
    s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=
      s.complementarity_tolerance=1e-6;
    const auto r=solve_filter_dca(p,s);
    std::cout<<"weakly_active="<<weakly_active<<" pairs="<<pairs<<" "<<r.message
             <<" local="<<r.dca_filter.local_acceptances<<" x="<<r.x[0]<<" y="<<r.x[1]<<"\n";
    if(r.status!=Status::Solved || !r.dca_filter.local_acceptances ||
       std::abs(r.x[0]-1)>1e-5 || std::abs(r.x[1]-target)>(weakly_active?2e-3:1e-5))
      throw std::runtime_error("affine weak model missed analytical solution");
    original_certificate(p,r);
    if(weakly_active && r.certificate_inequality_multipliers[1]>5e-3)
      throw std::runtime_error("weakly active multiplier did not approach zero");
  }
  for(bool pairs:{false,true})for(bool with_equality:{false,true}){
    QCQPProblem p;p.n=3;p.lb=Vec(3,-2);p.ub=Vec(3,2);p.x0={-0.4,0.2,0.1};
    p.objective=quadratic({-0.6,0,0},{{0,0,2},{1,1,2},{2,2,2}});
    if(with_equality)p.equalities.push_back(quadratic({0,0,1},{}));
    // Both nonlinear rows remain strictly interior throughout the local
    // trajectory. Thus no selected inequality can supply the certificate
    // manifold: it is the equality tangent, or the complete primal space.
    p.inequalities.push_back(quadratic({0,0,0},{{0,0,-2}},-0.25));
    p.inequalities.push_back(quadratic({0,0,0},{{1,1,2}},-1));
    Settings s;s.linear_backend="native";s.dca_engine="filter";
    s.dca_local_active=true;s.dca_local_weak_affine=true;s.dca_pair_elimination=pairs;
    s.dca_local_trigger=1e6;s.dca_local_primal_trigger=100;
    s.max_outer_iterations=200;s.live_display_level=0;
    s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=
      s.complementarity_tolerance=1e-6;
    const auto r=solve_filter_dca(p,s);
    std::cout<<"no_strong with_equality="<<with_equality<<" pairs="<<pairs<<" "<<r.message
             <<" local="<<r.dca_filter.local_acceptances<<" max_strong="<<r.dca_filter.local_strong<<"\n";
    if(r.status!=Status::Solved || !r.dca_filter.local_acceptances || r.dca_filter.local_strong!=0 ||
       std::abs(r.x[0]-0.3)>1e-5 || std::abs(r.x[1])>1e-5 || std::abs(r.x[2])>1e-5)
      throw std::runtime_error("affine weak model without strong rows failed");
    original_certificate(p,r);
  }
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
