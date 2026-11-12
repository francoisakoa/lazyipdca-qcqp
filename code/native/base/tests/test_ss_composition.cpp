#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
int checks=0;
void require(bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);}
bool same(const std::vector<SymTriplet>& a,const std::vector<SymTriplet>& b){
  if(a.size()!=b.size())return false;
  for(std::size_t k=0;k<a.size();++k)if(a[k].i!=b[k].i||a[k].j!=b[k].j||a[k].v!=b[k].v)return false;
  return true;
}
struct Snapshot {study::Correction objective,inequality;std::vector<SymTriplet> installed_objective,installed_inequality;};
Snapshot construct(const std::string& backend,const std::string& mode) {
  Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
  st.linear_backend=backend;st.linear_metis=true;st.dca_pair_elimination=false;
  st.convexification_study=mode;st.study_controller="r3";st.verbose=false;st.live_display_level=0;
  QCQPProblem p;p.name="ss_role_composition";p.n=4;p.x0={1,1,1,.1};p.lb=Vec(4,-5);p.ub=Vec(4,5);
  p.objective.linear={1,0,0,0};p.objective.hessian.n=4;
  p.objective.hessian.entries={{0,0,-2},{0,1,1},{1,1,1}};
  QuadraticFunction g;g.constant=-10;g.linear=Vec(4,0);g.hessian.n=4;
  g.hessian.entries={{1,1,1},{1,2,2},{2,2,-2}};p.inequalities={g};p.validate();
  LazyNewtonStatistics ns;DcaFilterStatistics ds;detail::NewtonCostLedger ledger(ns,Clock::now());
  Workspace w(p,st,ns,ledger);const auto anchor=w.x;DcaInequalityModels models(w,ds);
  models.study_prepare();models.study_reanchor();
  require(w.x==anchor,"study changed frozen primal point");
  require(models.study_hybrid_enabled()!=(mode=="SS_legacy"),"study support/controller architecture is incorrect");
  require(models.study_r3_controller()!=(mode=="SS_legacy"),"study changed common R3 outer controller");
  return {models.study_objective,models.study_inequalities[0],w.objective.model_hessian,w.inequalities[0].model_hessian};
}
void compare(const study::Correction& a,const study::Correction& b,const char* message){
  require(a.family==b.family&&a.rho==b.rho&&a.support_scalar==b.support_scalar&&a.support==b.support&&
      a.hybrid_atoms.size()==b.hybrid_atoms.size()&&a.atoms==b.atoms&&same(a.original,b.original)&&
      same(a.correction,b.correction)&&same(a.hessian,b.hessian),message);
}
}
int main(){try{
  for(const std::string backend:{"mumps"}) {
    require(linear_backend_compiled(backend),"requested MUMPS backend must be compiled");
    const auto hh=construct(backend,"HH"),hs=construct(backend,"HS"),sh=construct(backend,"SH"),ss=construct(backend,"SS"),legacy=construct(backend,"SS_legacy");
    compare(legacy.objective,study::construct(4,legacy.objective.original,"scalar",false),"SS_legacy objective differs from original SS constructor");
    compare(legacy.inequality,study::construct(4,legacy.inequality.original,"scalar",false),"SS_legacy inequality differs from original SS constructor");
    require(!legacy.objective.support_scalar&&!legacy.inequality.support_scalar,"SS_legacy lost original ambient scalar regularization");
    require(legacy.objective.rank_upper==4&&legacy.inequality.rank_upper==4,"SS_legacy lost its ambient correction rank");
    require(hh.objective.family=="hybrid"&&hh.inequality.family=="hybrid","HH role definition changed");
    require(hs.objective.family=="hybrid"&&hs.inequality.family=="scalar","HS role definition changed");
    require(sh.objective.family=="scalar"&&sh.inequality.family=="hybrid","SH role definition changed");
    require(ss.objective.family=="scalar"&&ss.inequality.family=="scalar","SS is not scalar/scalar");
    compare(ss.objective,sh.objective,"SS objective differs from scalar objective of SH");
    compare(ss.inequality,hs.inequality,"SS inequality differs from scalar inequality of HS");
    require(same(ss.installed_objective,sh.installed_objective),"SS installed objective differs from SH");
    require(same(ss.installed_inequality,hs.installed_inequality),"SS installed inequality differs from HS");
    require(ss.objective.support_scalar&&ss.inequality.support_scalar,"SS reverted to ambient scalar regularization");
    require(ss.objective.hybrid_atoms.empty()&&ss.inequality.hybrid_atoms.empty(),"SS constructed hybrid atoms");
    require(ss.objective.support.size()==2&&ss.inequality.support.size()==2,"scalar support geometry changed");
    require(study::hybrid_certificate(ss.objective,ss.objective.rho)&&study::hybrid_certificate(ss.inequality,ss.inequality.rho),"SS lost separate PSD certificates");
  }
  std::cout<<"SS_COMPOSITION_PASS checks="<<checks<<" mumps_backend=1 objective_matches_SH=1 inequalities_match_HS=1\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
