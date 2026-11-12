#include "lazyipdca_qcqp/dca_filter.hpp"
#include "../src/dca_curvature.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void check_success(const Result& result) {
  if(result.status==Status::Solved) {
    check(result.ipopt_desired_convergence,
          "strict success lacks its original KKT certificate");
    return;
  }
  if(result.status==Status::SolvedAcceptable) {
    check(result.acceptable_convergence && result.acceptable_point &&
              result.termination_audit_valid &&
              result.acceptable_streak>=result.acceptable_required,
          "acceptable success lacks its original KKT streak certificate");
    return;
  }
  throw std::runtime_error(result.message);
}
QuadraticFunction quadratic(int n, Vec g, std::vector<SymTriplet> h, double c=0) {
  QuadraticFunction q; q.linear=std::move(g); q.hessian.n=n;
  q.hessian.entries=std::move(h); q.constant=c; return q;
}
int main() {
 try {
  Settings s; s.dca_engine="filter"; s.linear_backend="native"; s.max_outer_iterations=500;
  s.dca_filter_inner_max=8; s.dca_filter_max_atoms=4;
  s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
  detail::DcaCurvature curvature(s);
  int budget=4;
  std::vector<std::vector<std::pair<int,double>>> normal={{{7,1}}};
  for(int step=0;step<16;++step) {
    const int n=8;
    Vec rotation(n*n,0); for(int i=0;i<n;++i)rotation[i+n*i]=1;
    // Vary both negative directions and their coupling to the uncovered space.
    for(int plane=0;plane<4;++plane) {
      int a=plane,b=plane+3; double angle=0.17*step*(plane+1),c=std::cos(angle),t=std::sin(angle);
      for(int i=0;i<n;++i){double x=rotation[i+n*a],y=rotation[i+n*b];rotation[i+n*a]=c*x+t*y;rotation[i+n*b]=-t*x+c*y;}
    }
    const Vec spectrum={-2.0-0.03*step,-0.6,0.2,0.5,1,2,3,4};
    std::vector<SymTriplet> h;
    for(int j=0;j<n;++j)for(int i=0;i<=j;++i){double v=0;for(int k=0;k<n;++k)v+=rotation[i+n*k]*spectrum[k]*rotation[j+n*k];h.push_back({i,j,v});}
    curvature.ensure(n,h,1e-6,true,budget,2,1e-8,&normal);
    check(curvature.stored_spectral_atoms()<=4,"stored atom cap exceeded by basis rotation");
    check(curvature.spectral_atoms.size()<=curvature.stored_spectral_atoms(),"applied rank exceeds stored space");
    Vec dense(n*n,0),values(n),work(3*n);
    for(auto e:curvature.corrected(n,h,curvature.rho))dense[e.i+n*e.j]=dense[e.j+n*e.i]=e.v;
    for(const auto &a:curvature.spectral_atoms)for(int j=0;j<n;++j)for(int i=0;i<n;++i)dense[i+n*j]+=a.beta*a.direction[i]*a.direction[j];
    dense[7+n*7]+=curvature.normal_weight;
    const char vectors='N',triangle='U'; int info=0,lwork=work.size();
    dsyev_(&vectors,&triangle,&n,dense.data(),&n,values.data(),work.data(),&lwork,&info);
    check(info==0&&values.front()>0.5e-6,"rotated atoms left an uncertified scalar complement");
    const int factors=curvature.certificates;
    curvature.ensure(n,h,1e-6,true,budget,2,1e-8,&normal);
    check(curvature.certificates==factors,"unchanged matrix certificate not reused");
  }
  std::vector<SymTriplet> positive;for(int i=0;i<8;++i)positive.push_back({i,i,2});
  curvature.ensure(8,positive,1e-6,false,budget,2,1e-8,&normal);
  check(curvature.rho==0&&curvature.spectral_atoms.empty(),"unneeded collective correction retained");
  check(curvature.subspace_spectra>0,"collective reweighting untested");

  // Independent changing eigendirections force cache compaction. The current
  // correction must remain PSD-certified, including the update which retires
  // search directions and the following update which reuses that capacity.
  detail::DcaCurvature recycling(s);
  std::vector<std::vector<std::pair<int,double>>> no_normals;
  for(int step=0;step<36;++step) {
    const int n=40;
    std::vector<SymTriplet> h;
    for(int j=0;j<n;++j)h.push_back({j,j,j==step?-2.0:2.0});
    int available=40-int(recycling.stored_spectral_atoms());
    recycling.ensure(n,h,1e-6,true,available,1,1e-8,&no_normals);
    check(recycling.last_peak_stored_atoms<=40,"temporary atom allocation exceeded cap");
    check(recycling.spectral_atoms.size()<=recycling.stored_spectral_atoms(),"compaction discarded an active direction");
    Vec dense(n*n,0),values(n),work(3*n);
    for(auto e:recycling.corrected(n,h,recycling.rho))dense[e.i+n*e.j]=dense[e.j+n*e.i]=e.v;
    for(const auto &a:recycling.spectral_atoms)for(int j=0;j<n;++j)for(int i=0;i<n;++i)dense[i+n*j]+=a.beta*a.direction[i]*a.direction[j];
    const char vectors='N',triangle='U';int info=0,lwork=work.size();
    dsyev_(&vectors,&triangle,&n,dense.data(),&n,values.data(),work.data(),&lwork,&info);
    check(info==0&&values.front()>0.5e-6,"compaction invalidated the full PSD certificate");
  }
  check(recycling.basis_compressions>0&&recycling.retired_directions>0,"inactive cache compaction untested");
  check(recycling.additions>int(recycling.stored_spectral_atoms()),"cumulative admissions confused with current storage");

  QCQPProblem p; p.n=2;p.name="g4_persistent_convex_model";p.x0={0.2,0.3};p.lb={0,0};p.ub={3,3};
  p.objective=quadratic(2,{-2,-4},{{0,0,2},{1,1,2}},5);
  auto reused=solve_filter_dca(p,s);
  check_success(reused);
  check(reused.dca_filter.model_reuses>0,"accepted endpoints never reused their valid majorant");
  check(reused.dca_filter.maximum_model_steps<=s.dca_filter_inner_max,"per-model Newton budget exceeded");
  check(reused.dca_filter.corrector_steps>0,"corrector never used");
  s.dca_filter_model_reuse=false;
  auto fresh=solve_filter_dca(p,s);
  check_success(fresh);
  for(int i=0;i<2;++i)check(std::abs(fresh.x[i]-reused.x[i])<1e-5,"model reuse changed the convex minimizer");
  check(fresh.dca_filter.model_reuses==0,"reuse control ignored");

  // Activation while a model is retained must rebuild at the committed point,
  // with each slack transported exactly once.
  s.dca_filter_model_reuse=true;s.lazy_newton_activation_margin=0;
  p.name="g4_activation_transport";
  p.inequalities={quadratic(2,{0,0},{{0,0,2}},-0.25)};
  auto activated=solve_filter_dca(p,s);
  check_success(activated);
  check(activated.lazy_newton.constraint_activations>0,"inactive nonlinear row was not activated");
  check(std::abs(activated.x[0]-0.5)<1e-5&&std::abs(activated.x[1]-2)<1e-5,"activation/transport endpoint wrong");
  check(activated.feasibility.total_inf<=1e-6&&activated.dual_infeasibility_inf<=1e-6,"original activation KKT failed");
  // A slowly contracting nonconvex inequality model exercises the filtered
  // extrapolation while retaining the original KKT acceptance conditions.
  p.name="g4_slow_dc_inequality";p.x0={0.7,0.8};p.lb={-2,-2};p.ub={2,2};
  p.objective=quadratic(2,{0,0},{{0,0,1},{1,1,1.1}});
  p.inequalities={quadratic(2,{0,0},{{0,0,-2},{1,1,-2}},1)};
  s.lazy_newton_activation_margin=1;s.dca_filter_extrapolation=true;
  auto boosted=solve_filter_dca(p,s);
  check_success(boosted);
  s.dca_filter_extrapolation=false;
  auto unboosted=solve_filter_dca(p,s);
  check_success(unboosted);
  check(std::abs(boosted.objective-0.5)<1e-5&&std::abs(unboosted.objective-0.5)<1e-5,"extrapolation original objective wrong");
  check(boosted.dca_filter.extrapolation_acceptances>0,"guarded extrapolation not exercised");
  check(boosted.outer_iterations<unboosted.outer_iterations,"extrapolation did not accelerate the slow DC mode");
  std::cout<<"PASS collective PSD recertification, persistent models, budgets and activation transport\n";
 } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
