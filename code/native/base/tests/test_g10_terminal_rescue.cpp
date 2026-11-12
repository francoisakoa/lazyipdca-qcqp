#include <stdexcept>
#include <cmath>
#include <algorithm>
static bool compare_rho_fixture=false;
template<class A> void install_precision_certificate(A& a);
#define G10_AGGREGATE_TEST_CONFIGURE install_precision_certificate
#include "../src/atomic_newton.cpp"
using namespace lazyipdca::qcqp;
// Simulate loss of the exact inertia certificate on an ill-scaled, but truly
// convex QP. All accepted factors and solves remain real native factorizations.
// The exact root must use the majorant factor, NEVER this rejected factor.
class CertificateThreshold final : public SymmetricFactorization {
  std::unique_ptr<SymmetricFactorization> real_;
  bool valid_=false;
 public:
  explicit CertificateThreshold(Settings s){
    s.linear_backend="native";std::string m;real_=make_symmetric_factorization(s,m);
  }
  bool factor(int n,int np,const std::vector<SymmetricTriplet>& a,std::string& m) override {
    Vec diagonal(np,0);for(auto t:a)if(t.i==t.j && t.i<np)diagonal[t.i]+=t.value;
    if(*std::min_element(diagonal.begin(),diagonal.end())<1){
      valid_=false;m="native inertia mismatch: simulated precision threshold";return false;
    }
    valid_=real_->factor(n,np,a,m);return valid_;
  }
  bool solve(Vec& b,std::string& m)const override {
    if(!valid_)throw std::runtime_error("rejected aggregate factor used for a solve");
    return real_->solve(b,m);
  }
  const char* name()const override{return "certificate-threshold-native";}
};
template<class A> void install_precision_certificate(A& a){
  if(a.settings.dca_aggregate){
    a.factor=std::make_unique<CertificateThreshold>(a.settings);
    if(compare_rho_fixture)a.accepted_rho=a.certified_rho=1;
  }
}
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
QCQPProblem ill_scaled(int n,double h){
  QCQPProblem p;p.n=n;p.name="g10_terminal_rescue_convex";
  p.lb=Vec(n,-INFINITY);p.ub=Vec(n,INFINITY);p.x0.resize(n);
  p.objective.linear=Vec(n,0);p.objective.hessian.n=n;
  for(int j=0;j<n;++j){double hj=h*(j+1);p.objective.hessian.entries.push_back({j,j,hj});p.x0[j]=0.002/hj;}
  return p;
}
Settings setup(){
  Settings s;s.linear_backend="native";s.dca_engine="filter";s.dca_aggregate=true;
  s.dca_aggregate_backend="native";s.dca_aggregate_rho_max=100;
  s.dca_krylov=false;s.dca_aggregate_progress_window=0;
  s.live_display_level=0;s.max_outer_iterations=5;
  s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
  return s;
}
int main(){try{
  auto p=ill_scaled(1,1e-10);auto s=setup();
  for(bool compare:{false,true}){
  compare_rho_fixture=compare;
  auto solved=solve_filter_dca(p,s);
  std::cout<<"success: status="<<status_string(solved.status)<<" trigger="<<solved.dca_filter.terminal_rescue_triggers
           <<" trials="<<solved.dca_filter.terminal_rescue_trials<<" accepted="<<solved.dca_filter.terminal_rescue_acceptances
           <<" raw="<<solved.raw_kkt_error<<" outer="<<solved.outer_iterations<<"\n";
  require(solved.status==Status::Solved,"forced original KKT rescue did not solve convex fixture");
  require(solved.dca_filter.terminal_rescue_triggers==1 && solved.dca_filter.terminal_rescue_acceptances==1,
          "causal rescue not exercised with accelerator and progress monitor disabled");
  require(solved.dca_filter.original_line_trials==0,"rescue was placed after majorant line search");
  require(inf_norm(p.objective.gradient(solved.x))<1e-6,"independent original gradient certificate failed");
  require(solved.dca_filter.krylov_max_accepted_residual<=s.kkt_backward_error_reject,
          "original expanded Newton residual gate was bypassed");
  }
  // Hard iteration limit on a multidimensional GMRES: reject its inexact
  // direction, retry with EVERY new majorant factor despite cooldown, and
  // leave the actual accepted iterate identical to a no-Krylov control.
  p=ill_scaled(3,1e-14);s=setup();s.max_outer_iterations=1;s.dca_krylov_max_iterations=0;
  const auto control=solve_filter_dca(p,s);
  s.dca_krylov_max_iterations=1;s.dca_krylov_failure_cooldown=100;
  const auto failed=solve_filter_dca(p,s);
  std::cout<<"rollback: status="<<status_string(failed.status)<<" trials="<<failed.dca_filter.terminal_rescue_trials
           <<" rejects="<<failed.dca_filter.krylov_linear_rejections<<" skips="<<failed.dca_filter.krylov_skips<<"\n";
  require(failed.dca_filter.aggregate_root_trials==1 &&
          failed.dca_filter.terminal_rescue_trials==s.dca_filter_model_retries+2,
          "forced rescue not retried with each rebuilt preconditioner");
  require(failed.dca_filter.krylov_linear_rejections==failed.dca_filter.terminal_rescue_trials &&
          failed.dca_filter.krylov_skips==0 && failed.dca_filter.terminal_rescue_acceptances==0,
          "inexact direction accepted or retry hidden by accelerator cooldown");
  require(control.x==failed.x && control.status==failed.status &&
          control.certificate_equality_multipliers==failed.certificate_equality_multipliers &&
          control.certificate_inequality_multipliers==failed.certificate_inequality_multipliers &&
          control.certificate_lower_multipliers==failed.certificate_lower_multipliers &&
          control.certificate_upper_multipliers==failed.certificate_upper_multipliers,
          "failed original KKT transaction altered incumbent or multipliers");
  require(control.lazy_newton.factor_attempts==failed.lazy_newton.factor_attempts,
          "Krylov rescue added a factorization instead of reusing the majorant factor");
  std::cout<<"PASS real convex root solve before line search, original residual audit, bounded per-rebuild retries, exact rollback, factor reuse\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
