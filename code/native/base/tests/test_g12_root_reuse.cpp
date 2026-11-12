#include "../src/atomic_newton.cpp"
#include <stdexcept>
#include <filesystem>
using namespace lazyipdca::qcqp;
namespace {
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void phase_units(const std::string& profile){
  QCQPProblem p;p.name="terminal_internal_units";p.n=1;p.x0={10000};p.lb={-INFINITY};p.ub={INFINITY};
  p.objective.linear={0};p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,1}};
  Settings s;s.termination_test=profile;s.linear_backend="native";s.lazy_newton_atomic_regularization=false;
  s.feasibility_tolerance=s.stationarity_tolerance=1e-6;
  LazyNewtonStatistics ls;detail::NewtonCostLedger costs(ls,Clock::now());Workspace w(p,s,ls,costs);
  w.x[0]=0.066;auto point=dca_original(w,w.x,w.lambda,w.z);
  require(std::abs(point.dual-0.066)<1e-12,"raw dual altered by the phase metric");
  require(std::abs(point.normalized_kkt-0.00066)<1e-12,"phase metric not in original Workspace internal units");
  require(detail::dca_stationary_phase(point.primal,point.normalized_kkt,1e-6,1e-6),"valid terminal phase hidden by raw units");
}
void aggregate_reuse(bool dense,bool eliminated,double slack){
  QCQPProblem p;p.name="aggregate_root_layout";p.n=eliminated?3:2;p.x0={0.002,-0.002};
  if(eliminated)p.x0.push_back(0.000004);
  p.lb=Vec(p.n,-INFINITY);p.ub=Vec(p.n,INFINITY);p.objective.linear=Vec(p.n,0);
  p.objective.linear[0]=p.objective.linear[1]=-1;p.objective.hessian.n=p.n;p.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction q;q.linear=Vec(p.n,0);q.linear[0]=q.linear[1]=1;q.constant=-slack;q.hessian.n=p.n;p.inequalities.push_back(q);
  q.constant=-10;p.inequalities.push_back(q);
  if(eliminated){QuadraticFunction e;e.linear={0,0,1};e.hessian.n=3;e.hessian.entries={{0,0,-2}};p.equalities.push_back(e);}
  Settings s;s.linear_backend="native";s.dca_aggregate=true;s.dca_aggregate_backend="native";
  s.lazy_newton_atomic_regularization=false;s.dca_krylov_max_iterations=24;s.kkt_backward_error_reject=1e-7;
  s.feasibility_tolerance=s.stationarity_tolerance=s.complementarity_tolerance=s.dual_infeasibility_tolerance=1e-6;
  if(dense)s.lazy_newton_dense_threshold=1;
  LazyNewtonStatistics ls;detail::NewtonCostLedger costs(ls,Clock::now());Workspace w(p,s,ls,costs);
  w.s[0]=slack;w.z[0]=1;w.active[0]=1;w.s[1]=10;w.z[1]=0;w.active[1]=0;
  w.evaluate();w.update_geometry();DcaFilterStatistics st;DcaModels models(w,st);DcaAggregate aggregate(s);
  auto outer=dca_original(w,w.x,w.lambda,w.z);detail::OriginalDcaFilter filter;
  const auto x=w.x,lambda=w.lambda,z=w.z,slacks=w.s;
  DcaAggregate::Candidate unused;aggregate.attempt_one(w,models,outer,filter,1,1,&unused,true,false);
  require(aggregate.last_attempt_certified,"fixture aggregate factor lacks certificate");
  require(aggregate.root_factor_compatible(w),"fresh certified aggregate factor unavailable");
  const int factors=st.aggregate_certificates;
  Direction seed;seed.x.assign(w.n,0);seed.lambda.assign(w.me,0);seed.s.assign(w.mi,0);seed.z.assign(w.mi,0);
  DcaKrylov krylov;double alpha=0;
  bool ok=krylov.attempt(w,st,outer,filter,seed,seed,1e-12,0,Clock::now(),alpha,outer,true,&aggregate);
  std::cout<<"dense="<<dense<<" eliminated="<<eliminated<<" slack="<<slack<<" accepted="<<ok<<" reason="<<st.krylov_last_rejection<<"\n";
  require(ok,"aggregate-preconditioned root did not contract original KKT");
  require(st.krylov_max_accepted_residual<=s.kkt_backward_error_reject,"residual threshold bypassed");
  require(alpha==krylov.accepted_dual_alpha,"terminal root used split fractions");
  if(eliminated)require(st.soc_acceptances>0,"nonlinear equality was not corrected by original-KKT SOC");
  require(st.aggregate_certificates==factors,"root performed an aggregate factorization");
  require(w.x==x&&w.lambda==lambda&&w.z==z&&w.s==slacks,"root mutated anchor before controller commit");
  require(krylov.proposal.z[1]==0,"inactive row acquired a multiplier");
  w.x[0]+=1e-9;require(!aggregate.root_factor_compatible(w),"stale aggregate factor accepted");w.x=x;
  w.s[0]*=2;require(!aggregate.root_factor_compatible(w),"factor accepted after slack change");w.s=slacks;
  aggregate.root_factor_valid=false;require(!aggregate.root_factor_compatible(w),"invalidated factor remained usable");
}

void negative_curvature_descent(bool nonlinear=false){
  QCQPProblem p;p.name="negative_curvature_minimization";p.n=1;p.x0={0.002};p.lb={-INFINITY};p.ub={INFINITY};
  p.objective.linear={0};p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,-1}};
  if(nonlinear){p.n=2;p.x0={0.002,0.000004};p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
    p.objective.linear={0,0};p.objective.hessian.n=2;
    QuadraticFunction eq;eq.linear={0,1};eq.hessian.n=2;eq.hessian.entries={{0,0,-2}};p.equalities.push_back(eq);}
  Settings s;s.linear_backend="native";s.dca_aggregate_backend="native";s.dca_aggregate=true;s.lazy_newton_atomic_regularization=false;
  s.feasibility_tolerance=s.stationarity_tolerance=1e-6;
  LazyNewtonStatistics ls;detail::NewtonCostLedger costs(ls,Clock::now());Workspace w(p,s,ls,costs);
  DcaFilterStatistics st;DcaModels models(w,st);DcaAggregate aggregate(s);detail::OriginalDcaFilter filter;
  auto from=dca_original(w,w.x,w.lambda,w.z);DcaAggregate::Candidate candidate;
  require(aggregate.attempt_one(w,models,from,filter,1,2,&candidate,true,false),"valid negative-curvature DCA descent blocked");
  require(aggregate.last_attempt_certified,"negative curvature step without convexified certificate");
  if(nonlinear)require(candidate.soc && candidate.point.primal<=s.feasibility_tolerance,"DCA negative-curvature step not retracted onto nonlinear equality");
  require(candidate.point.kkt>from.kkt && candidate.point.objective<from.objective,"fixture must distinguish stationarity from minimization");
  require(w.x==p.x0,"candidate DCA descent mutated incumbent");
  require(filter.diagnose(from,candidate.point,1e-6,1e-6,1,true).history_checked,"curvature descent bypassed filter history");
  auto dominating=from;dominating.theta=0;dominating.objective=candidate.point.objective-1;
  filter.insert(dominating);
  // At theta=0 the historical filter deliberately permits theta=0 through
  // its non-strict feasibility arm. Use a strictly positive trial violation
  // to exercise domination, rather than silently changing that convention.
  auto dominated=candidate.point;dominated.theta=1e-10;
  const auto blocked=filter.diagnose(from,dominated,1e-6,1e-6,1,true);
  require(blocked.history_checked && !blocked.accepted && std::string(blocked.reason)=="filter-history",
          "curvature descent bypassed a dominating history entry");
  detail::OriginalDcaFilter fresh;aggregate.accepted_rho=aggregate.certified_rho=2;
  const int before=st.aggregate_curvature_shortcuts;
  require(aggregate.attempt(w,models,from,fresh,3,false),"productive curvature descent not committed");
  require(st.aggregate_curvature_shortcuts==before+1,"productive DCA descent still ran exhaustive rho contest");
  detail::OriginalFilterPoint a,b;a.objective=16000;a.lagrangian=16000;a.theta=1e-12;a.kkt=0.064;
  a.primal_inf=1e-12;a.normalized_kkt=0.0006;b=a;b.objective+=2e-5;b.lagrangian+=2e-5;b.kkt=0.023;b.primal_inf=1e-10;
  require(!filter.acceptable(a,b,1e-5,1e-6,0.64),"uphill exact root falsely admitted");
}
void endpoint_validation(){
  QCQPProblem p;p.name="endpoint_import";p.n=3;p.x0={10000,0,4};
  p.lb={-INFINITY,-INFINITY,4};p.ub={INFINITY,INFINITY,4};
  p.objective.linear={0,0,0};p.objective.hessian.n=3;p.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction q;q.constant=-1;q.linear={0,1,0};q.hessian.n=3;p.inequalities.push_back(q);
  Settings settings;settings.linear_backend="native";settings.lazy_newton_atomic_regularization=false;
  LazyNewtonStatistics st;detail::NewtonCostLedger costs(st,Clock::now());Workspace w(p,settings,st,costs);
  const auto path=std::filesystem::temp_directory_path()/("g12_endpoint_"+std::to_string(Clock::now().time_since_epoch().count())+".bin");
  auto write=[&](std::vector<Vec> vectors,const std::string& magic="DCAEP001",bool trailing=false){
    std::ofstream out(path,std::ios::binary);out.write(magic.data(),8);
    for(const auto& v:vectors){const std::uint64_t n=v.size();out.write(reinterpret_cast<const char*>(&n),8);out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(double));}
    if(trailing)out.put('x');
  };
  const std::vector<Vec> valid={{0.066,0.5,4},{},{2},{0,0,0},{0,0,0}};
  write(valid);dca_endpoint_start(w,path.string());
  require(std::abs(w.objective.scale-0.01)<1e-15,"warm start changed frozen QCQ scaling");
  require(w.x[0]==0.066&&w.x[1]==0.5&&std::abs(w.z[0]-0.02)<1e-15,"primal-dual import lost its units");
  require(w.s[0]==0.5,"warm slacks were not reconstructed explicitly");
  auto reject=[&](std::vector<Vec> vectors,const std::string& magic="DCAEP001",bool trailing=false){
    write(vectors,magic,trailing);bool threw=false;
    try{dca_endpoint_start(w,path.string());}catch(const std::runtime_error&){threw=true;}
    require(threw,"malformed endpoint file accepted");
  };
  reject(valid,"BADMAGIC");reject(valid,"DCAEP001",true);
  auto bad=valid;bad[0].pop_back();reject(bad);
  bad=valid;bad[0][2]=3;reject(bad);
  bad=valid;bad[2][0]=-1;reject(bad);
  bad=valid;bad[0][0]=std::numeric_limits<double>::quiet_NaN();reject(bad);
  std::filesystem::remove(path);
}
}
int main(){try{
  phase_units("project");phase_units("ipopt");endpoint_validation();negative_curvature_descent();negative_curvature_descent(true);
  require(!detail::dca_terminal_contracts(0.06429,0.064277),"full-step contraction was silently loosened");
  require(detail::dca_terminal_contracts(0.06429,0.064277,0.0002),"damped exact root was blocked by a fixed gain quota");
  require(!detail::dca_terminal_contracts(0.06429,0.06429,0.0002),"duplicate KKT accepted");
  require(!detail::dca_terminal_contracts(0.06429,0.065,0.0002),"KKT growth accepted");
  require(!detail::dca_terminal_contracts(0.06429,0.001,0),"zero fraction accepted");
  aggregate_reuse(false,false,1e-16);aggregate_reuse(true,false,1e-16);
  aggregate_reuse(false,true,1e-16);aggregate_reuse(false,false,1e-3);
  std::cout<<"PASS internal phase units, aggregate root reuse, numerical/dense borders, original equality reconstruction, inactive rows, rollback and stale-factor rejection\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
