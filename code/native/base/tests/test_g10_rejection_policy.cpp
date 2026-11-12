#include "../src/atomic_newton.cpp"
#include <stdexcept>
using namespace lazyipdca::qcqp;

// Observe real controller transactions while isolating a specific cause. No
// false direction can pass acceptance: injected precision faults never solve.
class ControllerFailure final:public SymmetricFactorization {
 public:
  enum class Mode { backend,precision,inertia_then_precision,refinement_precision };
  Mode mode;std::vector<double> shifts;
  mutable int solves_since_factor=0,total_solves=0;
  int quality_requests=0;
  explicit ControllerFailure(Mode value):mode(value){}
  bool factor(int,int,const std::vector<SymmetricTriplet>& a,std::string& message)override{
    solves_since_factor=0;
    // The test has one primal coordinate. Its objective Hessian is one; the
    // final diagonal triplet is the explicit rho slot emitted by aggregate.
    double rho=0;
    for(const auto& t:a)if(t.i==0 && t.j==0)rho=t.value;
    shifts.push_back(rho);
    if(mode==Mode::backend){message="MUMPS factorization failed INFOG(1)=-10";return false;}
    if(mode==Mode::inertia_then_precision && rho==0){message="native inertia mismatch: injected";return false;}
    message.clear();return true;
  }
  bool solve(Vec& rhs,std::string& message)const override{
    ++solves_since_factor;++total_solves;
    if(mode==Mode::refinement_precision && solves_since_factor==1){
      // A finite but inaccurate first solve forces the aggregate wrapper to
      // request a residual correction; only that second solve fails.
      std::fill(rhs.begin(),rhs.end(),0.0);message.clear();return true;
    }
    message="backward error injected by rejection-policy regression test";return false;
  }
  bool increase_quality(std::string&)override{++quality_requests;return false;}
  const char* name()const override{return "controller-failure";}
};
static void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
int main(){try{
  for(bool separated:{false,true})for(auto mode:{ControllerFailure::Mode::backend,ControllerFailure::Mode::precision,ControllerFailure::Mode::inertia_then_precision,ControllerFailure::Mode::refinement_precision}){
    QCQPProblem p;p.n=1;p.lb={-2};p.ub={2};p.x0={0.5};
    p.objective.linear={0};p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,1}};
    Settings st;st.linear_backend="native";st.dca_aggregate=true;st.dca_aggregate_backend="native";
    st.dca_aggregate_only=true;st.dca_aggregate_retry_max=4;st.dca_aggregate_rho_initial=1e-6;
    st.dca_aggregate_rho_growth=10;st.dca_aggregate_rho_decay=0.2;
    // Keep the rho cap out of this retry-budget test: 0,40,400,4000.
    // The independent controller test checks clipping the seed to its cap.
    st.dca_aggregate_rho_max=1e9;
    st.dca_aggregate_rejection_policy=separated?"separated":"legacy";
    st.dca_aggregate_rho_memory=true;st.dca_aggregate_adaptive_dual=false;
    st.dca_aggregate_adaptive_pivot=mode==ControllerFailure::Mode::refinement_precision;st.live_display_level=0;
    LazyNewtonStatistics ws;detail::NewtonCostLedger costs(ws,Clock::now());Workspace w(p,st,ws,costs);
    DcaFilterStatistics stats;DcaModels models(w,stats);auto outer=dca_original(w,w.x,w.lambda,w.z);
    detail::OriginalDcaFilter filter(std::max(1.0,10*outer.theta));DcaAggregate aggregate(st);
    aggregate.positive_rho_memory.certificate(200,true);
    aggregate.positive_rho_memory.certificate(0,true);
    auto owned=std::make_unique<ControllerFailure>(mode);auto* probe=owned.get();aggregate.factor=std::move(owned);
    const auto x=w.x,s=w.s,z=w.z,lambda=w.lambda;
    require(!aggregate.attempt_single_loop(w,models,outer,filter,1,Clock::now()),"injected error unexpectedly accepted");
    require(w.x==x && w.s==s && w.z==z && w.lambda==lambda,"rejected attempt mutated primal-dual anchor");
    require(probe->shifts.front()==0,"historical seed displaced mandatory zero probe");
    if(separated){
      const bool inertia_first=mode==ControllerFailure::Mode::inertia_then_precision;
      require(probe->shifts.size()==(inertia_first?2u:1u),"separated controller inflated rho after backend/precision failure");
      if(inertia_first){
        require(probe->shifts[1]==40,"zero inertia failure did not use retained positive history");
        require(stats.aggregate_reject_inertia==1 && stats.aggregate_reject_numerical_direction==1,"cause counters incorrect");
      }else require(aggregate.positive_rho_memory.value==200,"zero certificate erased historical rho");
    }else require(probe->shifts.size()==4,"legacy retry budget changed");
    if(mode==ControllerFailure::Mode::refinement_precision){
      require(probe->total_solves==2*static_cast<int>(probe->shifts.size()),"refinement failure was not exercised");
      require(probe->quality_requests==static_cast<int>(probe->shifts.size()),"late precision failure skipped numerical quality recovery");
    }
    if(mode==ControllerFailure::Mode::backend)
      require(aggregate.last_rejection_kind==detail::DcaAggregateRejection::backend || !separated,"backend error mislabeled as curvature");
    else require(aggregate.last_rejection_kind==detail::DcaAggregateRejection::numerical_direction,"precision error mislabeled as curvature");
  }
  std::cout<<"PASS aggregate controller: distinct failures, bounded legacy retries, zero-first retained memory, rollback\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
