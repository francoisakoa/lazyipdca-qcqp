#include "lazyipdca_qcqp/restoration.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static QCQPProblem lifted_boundary_problem(int links) {
  QCQPProblem p;
  p.name = "dca_soc_defect_target_abs_reduced";
  p.n = 2*links + 1;
  const int y = 2*links;
  p.x0.assign(p.n, 0.0);
  p.lb.assign(p.n, -2.0);
  p.ub.assign(p.n,  2.0);
  for (int i = 0; i < links; ++i) {
    p.x0[i] = 1.0;
    p.x0[links+i] = 1.0;
    p.lb[links+i] = 0.0;
  }
  p.objective.linear.assign(p.n,0.0);
  p.objective.hessian.n=p.n;

  for (int i = 0; i < links; ++i) {
    QuadraticFunction h;
    h.name = "abs_link_" + std::to_string(i);
    h.linear.assign(p.n,0.0);
    h.hessian.n=p.n;
    h.hessian.entries={{i,i,-2.0},{links+i,links+i,2.0}};
    p.equalities.push_back(std::move(h));
  }

  constexpr double a = 2.0e-3;
  constexpr double current_residual = 0.99e-6;
  QuadraticFunction coupling;
  coupling.name="weak_lifted_coupling";
  coupling.constant=-2.0*links*a + current_residual;
  coupling.linear.assign(p.n,0.0);
  for(int i=0;i<links;++i) {
    coupling.linear[i]=a;
    coupling.linear[links+i]=a;
  }
  coupling.hessian.n=p.n;
  coupling.hessian.entries={{y,y,2.0}}; // + y^2
  p.equalities.push_back(std::move(coupling));
  return p;
}

int main() {
  constexpr int links=6;
  constexpr double alpha=1.0e-4;
  QCQPProblem p=lifted_boundary_problem(links);
  const int y=2*links;
  const auto& coupling=p.equalities.back();
  const double h0=coupling.value(p.x0);

  // Full SQP direction: every link has dt=dq and the aggregate equality has
  // h(x)+Jd=0.  The free tangent component dy=1 creates an O(alpha^2)
  // nonlinear defect in a scaled line-search trial.
  Vec d(p.n,0.0);
  const double dq=-h0/(4.0e-3*links);
  for(int i=0;i<links;++i) {
    d[i]=dq;
    d[links+i]=dq;
  }
  d[y]=1.0;

  Vec z=p.x0;
  axpy(alpha,d,z);
  Vec step=z;
  axpy(-1.0,p.x0,step);

  Vec targets(p.equalities.size(),0.0);
  for(int i=0;i<(int)p.equalities.size();++i)
    targets[i]=p.equalities[i].value(p.x0)+
               dot(p.equalities[i].gradient(p.x0),step);

  Vec full=retract_equalities(p,z,20,1e-12,1e-12,0.995);
  Vec defect=retract_equalities_to_targets(
      p,z,targets,20,1e-12,1e-12,0.995);
  Vec full_corr=full;
  Vec defect_corr=defect;
  axpy(-1.0,z,full_corr);
  axpy(-1.0,z,defect_corr);

  const double full_inf=inf_norm(full_corr);
  const double defect_inf=inf_norm(defect_corr);
  const double target_error=std::abs(coupling.value(defect)-targets.back());
  double link_error=0.0;
  for(int i=0;i<links;++i)
    link_error=std::max(link_error,std::abs(p.equalities[i].value(defect)));

  if(!(full_inf>50.0*defect_inf)) {
    std::cerr << "[FAIL] defect SOC retained the current residual floor: full="
              << full_inf << " defect=" << defect_inf << "\n";
    return 1;
  }
  if(target_error>1e-10 || link_error>1e-12) {
    std::cerr << "[FAIL] target/reduced retraction inaccurate: target="
              << target_error << " links=" << link_error << "\n";
    return 2;
  }
  if(!(defect_inf<1e-5 && full_inf>1e-5)) {
    std::cerr << "[FAIL] regression scales not discriminating: full="
              << full_inf << " defect=" << defect_inf << "\n";
    return 3;
  }

  std::cout << "[PASS] scaled-linearization defect SOC removes only O(alpha^2)"
            << " full_corr=" << full_inf
            << " defect_corr=" << defect_inf
            << " target_error=" << target_error
            << " links=" << links << "\n";
  return 0;
}
