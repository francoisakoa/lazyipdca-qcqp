#include "lazyipdca_qcqp/restoration.hpp"
#include <cmath>
#include <iomanip>
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

QCQPProblem make_problem(bool erroneous) {
  constexpr int beams=200;
  constexpr double fraction=0.6;
  QCQPProblem p;
  p.name=erroneous?"CATENARY_like":"CATENA_like";
  p.n=3*(beams+1);
  p.x0.assign(p.n,0.0);
  p.lb.assign(p.n,-std::numeric_limits<double>::infinity());
  p.ub.assign(p.n, std::numeric_limits<double>::infinity());
  p.objective.name="gravity";
  p.objective.linear.assign(p.n,0.0);
  p.objective.hessian.n=p.n;
  for(int i=0;i<=beams;++i) p.objective.linear[idx(i,1)]=1.0;
  for(int c=0;c<3;++c) p.lb[idx(0,c)]=p.ub[idx(0,c)]=0.0;
  const double length=fraction*beams;
  p.lb[idx(beams,0)]=p.ub[idx(beams,0)]=length;

  for(int i=1;i<=beams;++i) {
    QuadraticFunction h;
    h.name="beam_"+std::to_string(i);
    h.constant=-1.0;
    h.linear.assign(p.n,0.0);
    h.hessian.n=p.n;
    add_square_difference(h,idx(i,0),idx(i-1,0));
    add_square_difference(h,idx(i,1),idx(i-1,1));
    if(erroneous) add_square_difference(h,idx(i,0),idx(i-1,2));
    else add_square_difference(h,idx(i,2),idx(i-1,2));
    p.equalities.push_back(std::move(h));
  }

  for(int i=1;i<=beams;++i) {
    if(erroneous) {
      p.x0[idx(i,0)]=fraction;
    } else {
      p.x0[idx(i,0)]=fraction*i;
      p.x0[idx(i,1)]=-fraction*i;
    }
  }
  p.x0[idx(beams,0)]=length;
  return p;
}
}

int main() {
  Settings st;
  st.verbose=false;
  st.linear_backend="native";
  st.kkt_adaptive_regularization=true;
  st.restoration_max_iterations=80;
  st.restoration_max_restarts=0;
  st.restoration_minimax_enabled=false;
  st.feasibility_tolerance=1e-6;

  for(bool erroneous:{false,true}) {
    const QCQPProblem p=make_problem(erroneous);
    const RestorationResult r=FeasibilityRestoration(st).solve(p,p.x0);
    if(r.harmonic_attempts!=1 || (erroneous && r.harmonic_acceptances!=1)) {
      std::cerr << "[FAIL] unexpected harmonic decision for " << p.name
                << " att/acc=" << r.harmonic_attempts << "/"
                << r.harmonic_acceptances
                << " theta=" << r.initial_feasibility << "->"
                << r.harmonic_feasibility << "\n";
      return 1;
    }
    if(erroneous && !(r.harmonic_feasibility<0.01*r.initial_feasibility)) {
      std::cerr << "[FAIL] insufficient harmonic improvement for " << p.name
                << " theta=" << r.initial_feasibility << "->"
                << r.harmonic_feasibility << "\n";
      return 2;
    }
    const int expected_activation=erroneous?1:0;
    if(r.tangent_activation_acceptances!=expected_activation) {
      std::cerr << "[FAIL] unexpected tangent activation for "<<p.name
                << " accepted="<<r.tangent_activation_acceptances
                << " expected="<<expected_activation<<"\n";
      return 3;
    }
    if(!r.success || r.final_feasibility>st.feasibility_tolerance) {
      std::cerr << "[FAIL] restoration failed for " << p.name
                << " message=" << r.message << " theta="
                << r.final_feasibility << " it=" << r.iterations << "\n";
      return 4;
    }
    if(r.iterations>=st.restoration_max_iterations) {
      std::cerr << "[FAIL] harmonic start did not avoid full restoration budget for "
                << p.name << " it=" << r.iterations << "\n";
      return 5;
    }
    std::cout << std::setprecision(17)
              << "[PASS] " << p.name << " theta=" << r.initial_feasibility
              << " -> " << r.harmonic_feasibility << " -> "
              << r.final_feasibility << " iterations=" << r.iterations << "\n";
  }
  return 0;
}
