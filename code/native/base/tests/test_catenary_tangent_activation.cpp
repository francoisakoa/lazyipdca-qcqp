#include "lazyipdca_qcqp/restoration.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace lazyipdca::qcqp;

namespace {
int idx(int joint,int coordinate) { return 3*joint+coordinate; }
void add_square_difference(QuadraticFunction& h,int a,int b) {
  h.hessian.entries.push_back({std::min(a,b),std::min(a,b),2.0});
  h.hessian.entries.push_back({std::max(a,b),std::max(a,b),2.0});
  h.hessian.entries.push_back({std::min(a,b),std::max(a,b),-2.0});
}
QCQPProblem make_problem(int beams) {
  constexpr double fraction=0.6;
  constexpr double total_mass=500.0;
  constexpr double gravity=9.81;
  const double mg=(total_mass/beams)*gravity;
  QCQPProblem p;
  p.name="CATENARY_tangent_activation";
  p.n=3*(beams+1);
  p.x0.assign(p.n,0.0);
  p.lb.assign(p.n,-std::numeric_limits<double>::infinity());
  p.ub.assign(p.n, std::numeric_limits<double>::infinity());
  p.objective.name="gravity";
  p.objective.linear.assign(p.n,0.0);
  p.objective.hessian.n=p.n;
  p.objective.linear[idx(0,1)]=0.5*mg;
  for(int i=1;i<beams;++i) p.objective.linear[idx(i,1)]=mg;
  p.objective.linear[idx(beams,1)]=0.5*mg;
  for(int c=0;c<3;++c) p.lb[idx(0,c)]=p.ub[idx(0,c)]=0.0;
  p.lb[idx(beams,0)]=p.ub[idx(beams,0)]=fraction*beams;
  for(int i=1;i<=beams;++i) {
    QuadraticFunction h;
    h.name="beam_"+std::to_string(i);
    h.constant=-1.0;
    h.linear.assign(p.n,0.0);
    h.hessian.n=p.n;
    add_square_difference(h,idx(i,0),idx(i-1,0));
    add_square_difference(h,idx(i,1),idx(i-1,1));
    add_square_difference(h,idx(i,0),idx(i-1,2));
    p.equalities.push_back(std::move(h));
  }
  for(int i=1;i<=beams;++i) p.x0[idx(i,0)]=fraction;
  p.x0[idx(beams,0)]=fraction*beams;
  return p;
}
}

int main() {
  constexpr int beams=200;
  const QCQPProblem p=make_problem(beams);
  Settings st;
  st.verbose=false;
  st.linear_backend="native";
  st.kkt_adaptive_regularization=true;
  st.restoration_max_iterations=80;
  st.restoration_max_restarts=0;
  st.restoration_minimax_enabled=false;
  st.restoration_tangent_activation=true;
  st.feasibility_tolerance=1e-6;

  const RestorationResult r=FeasibilityRestoration(st).solve(p,p.x0);
  if(!r.success || r.final_feasibility>st.feasibility_tolerance) {
    std::cerr << "[FAIL] tangent-activated restoration failed theta="
              <<r.final_feasibility<<" message="<<r.message<<"\n";
    return 1;
  }
  if(r.tangent_activation_attempts!=1 ||
     r.tangent_activation_acceptances!=1 ||
     r.tangent_activation_components<1 ||
     !(r.tangent_activation_feasibility<r.harmonic_feasibility)) {
    std::cerr << "[FAIL] tangent activation diagnostics att/acc/components="
              <<r.tangent_activation_attempts<<"/"
              <<r.tangent_activation_acceptances<<"/"
              <<r.tangent_activation_components<<" theta="
              <<r.harmonic_feasibility<<"->"
              <<r.tangent_activation_feasibility<<"\n";
    return 2;
  }
  double min_edge=std::numeric_limits<double>::infinity();
  double max_edge=0.0;
  for(int i=1;i<=beams;++i) {
    const double dy=std::abs(r.x[idx(i,1)]-r.x[idx(i-1,1)]);
    min_edge=std::min(min_edge,dy);
    max_edge=std::max(max_edge,dy);
  }
  if(!(min_edge>1e-4) || !(max_edge>1e-2)) {
    std::cerr << "[FAIL] Y chain remains first-order degenerate min/max edge="
              <<min_edge<<"/"<<max_edge<<"\n";
    return 3;
  }

  Settings off=st;
  off.restoration_tangent_activation=false;
  const RestorationResult baseline=FeasibilityRestoration(off).solve(p,p.x0);
  double baseline_min=std::numeric_limits<double>::infinity();
  for(int i=1;i<=beams;++i)
    baseline_min=std::min(baseline_min,
        std::abs(baseline.x[idx(i,1)]-baseline.x[idx(i-1,1)]));
  if(!(baseline_min<1e-12)) {
    std::cerr << "[FAIL] ablation no longer reproduces the flat-Y pathology: "
              <<baseline_min<<"\n";
    return 4;
  }
  std::cout << "[PASS] objective tangent activation breaks flat Y chain"
            << " theta="<<r.final_feasibility
            << " edge="<<min_edge<<"/"<<max_edge
            << " baseline="<<baseline_min<<"\n";
  return 0;
}
