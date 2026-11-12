#include "lazyipdca_qcqp/dca_filter.hpp"
#include "../src/dca_curvature.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
double minimum_eigenvalue(int n, const std::vector<SymTriplet> &h,
                          const detail::DcaCurvature &c,
                          const std::vector<std::vector<std::pair<int,double>>> &normals) {
  Vec a(n*n,0), eigenvalues(n), work(3*n);
  for (auto e:c.corrected(n,h,c.rho)) a[e.i+n*e.j]=a[e.j+n*e.i]=e.v;
  for (const auto &atom:c.spectral_atoms)
    for(int i=0;i<n;++i) for(int j=0;j<n;++j)
      a[i+n*j]+=atom.beta*atom.direction[i]*atom.direction[j];
  for(const auto &row:normals) for(auto i:row) for(auto j:row)
    a[i.first+n*j.first]+=c.normal_weight*i.second*j.second;
  int info=0,lwork=work.size(); const char vectors='N',triangle='U';
  dsyev_(&vectors,&triangle,&n,a.data(),&n,eigenvalues.data(),work.data(),&lwork,&info);
  require(info==0,"independent eigensystem failed"); return eigenvalues.front();
}
int main() {
 try {
  Settings s; s.linear_backend="native";
  // A Gershgorin completion of two is valid, but unnecessarily large. The
  // normal path must instead start from the exact 20-dimensional Ritz value
  // and certify it by inertia; Gershgorin remains an unused emergency escape.
  std::vector<SymTriplet> h;
  for(int j=0;j<20;++j) h.push_back({j,j,j==0?-1.0:3.0});
  h.push_back({0,1,1});
  std::vector<std::vector<std::pair<int,double>>> no_normals;
  int budget=0;
  detail::DcaCurvature refined(s);
  refined.ensure(20,h,1e-6,false,budget,2,1e-8,&no_normals);
  s.dca_filter_rho_refinements=0; detail::DcaCurvature coarse(s);
  coarse.ensure(20,h,1e-6,false,budget,2,1e-8,&no_normals);
  const double exact_shift=std::sqrt(5.0)-1.0+1e-6;
  require(std::abs(refined.rho-exact_shift)<1e-8&&
          std::abs(coarse.rho-exact_shift)<1e-8,
          "Ritz/inertia complement missed the smallest eigenvalue");
  require(minimum_eigenvalue(20,h,refined,no_normals)>0.5e-6,"refined scalar lost positive definiteness");
  require(refined.rho_refinement_trials==3,"refinement accounting missing");
  require(refined.ritz_shift_proposals>0&&refined.gershgorin_fallback_uses==0,
          "Gershgorin replaced the normal Ritz/inertia path");
  require(refined.additions==0&&refined.stored_spectral_atoms()==0,"zero cap bypassed");
  int factors=refined.certificates;
  refined.ensure(20,h,1e-6,true,budget,2,1e-8,&no_normals);
  require(factors==refined.certificates,"unchanged certificate was recomputed");

  // A previously captured tangent mode acquires a small normal component.
  // The artificial normal weight must not erase its physical negative curvature.
  s.dca_filter_rho_refinements=3;
  detail::DcaCurvature physical(s);
  s.dca_filter_normal_atom_weighting=true; detail::DcaCurvature legacy(s);
  std::vector<SymTriplet> initial={{0,0,-1},{1,1,2},{2,2,2},{3,3,2}};
  std::vector<std::vector<std::pair<int,double>>> normals={{{3,1}}};
  int bp=1,bl=1;
  physical.ensure(4,initial,1e-6,true,bp,1,1e-8,&normals);
  legacy.ensure(4,initial,1e-6,true,bl,1,1e-8,&normals);
  require(physical.additions==1&&legacy.additions==1,"initial negative witness missing");
  initial[0].v=-1.01; normals={{{0,0.002},{3,1}}};
  physical.ensure(4,initial,1e-6,false,bp,1,1e-8,&normals);
  legacy.ensure(4,initial,1e-6,false,bl,1,1e-8,&normals);
  require(physical.rho<legacy.rho,"normal curvature still erases captured modes");
  require(!physical.spectral_atoms.empty(),"physical atom was discarded by the normal completion");
  require(physical.additions==1,"atom added outside Hessian event");
  require(minimum_eigenvalue(4,initial,physical,normals)>0.25e-6,"normal drift invalidated certificate");
  require(minimum_eigenvalue(4,initial,legacy,normals)>0.25e-6,"legacy control certificate invalid");

  QCQPProblem p;p.name="g5_infeasible_affine_start";p.n=2;p.x0={5,0.3};p.lb={-10,-10};p.ub={10,10};
  p.objective.linear={0,0};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction eq;eq.linear={1,0};eq.constant=-1;eq.hessian.n=2;p.equalities={eq};
  s=Settings{};s.linear_backend="native";s.dca_engine="filter";s.max_outer_iterations=100;
  s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
  auto direct=solve_filter_dca(p,s);
  require(direct.status==Status::Solved,direct.message.c_str());
  require(direct.dca_filter.restorations==0,"default performed compulsory restoration");
  s.dca_filter_initial_restoration=true;
  auto restored=solve_filter_dca(p,s);
  require(restored.status==Status::Solved,restored.message.c_str());
  require(restored.dca_filter.restorations>0,"explicit restoration control ignored");
  require(std::abs(direct.objective-0.5)<1e-6&&std::abs(restored.objective-0.5)<1e-6,"original objective changed");
  s.dca_filter_rho_refinements=13;
  bool rejected=false;try{solve_filter_dca(p,s);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"invalid refinement budget accepted");
  std::cout<<"PASS certified scalar refinement, normal drift and infeasible start\n";
 } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
