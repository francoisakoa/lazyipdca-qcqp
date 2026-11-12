#include "source_under_test.hpp"
#include <cstring>
namespace gr=lazyipdca::qcqp::detail::gr;
static int checks=0;
static void require(bool p,const char* message){++checks;if(!p)throw std::runtime_error(message);}
static bool same(double a,double b){return std::memcmp(&a,&b,sizeof a)==0;}
static bool same_vector(const Vec& a,const Vec& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0);}
static Quadratic affine(double constant,double coefficient,const Vec& x){Quadratic q;q.constant=constant;q.support={0};q.coefficients={coefficient};q.at_anchor(x);return q;}
static Workspace sample(double mu,double objective_coefficient){Workspace w;w.n=1;w.me=1;w.mi=3;w.x={100000001};w.lambda={0};w.rd={0};w.objective=affine(0,objective_coefficient,w.x);w.equalities={affine(0,0,w.x)};w.inequalities={affine(-3,0,w.x),affine(100000000,-1,w.x),affine(-100000010,1,w.x)};w.s={3,1,9};w.z={-.2,mu,mu/9};w.active={1,1,1};w.free_source={1,0,0};w.inequality_kind={0,-1,1};return w;}
static Direction first(const Workspace& w){Direction d;d.x={.1};d.lambda={1e-5};d.s={0,.1,-.1};d.z={0,-w.z[1]*.1,w.z[2]*.1/9};return d;}
static gr::State state(const Workspace& w,const DcaMajorantBarrierPath& path,double alpha,double mu){Vec x,s,z,l;path.write_primal(alpha,x);path.write_slacks(alpha,s,&x);path.write_duals(alpha,mu,z,&s);path.write_equality_duals(alpha,l);gr::State out;out.objective=w.objective.original_at(x);for(int e=0;e<w.me;++e){out.c.push_back(w.equalities[e].original_at(x));out.y.push_back(l[e]);}for(int i=0;i<w.mi;++i)if(w.active[i]){out.s.push_back(s[i]);double c=static_cast<double>(static_cast<long double>(w.inequalities[i].original_at(x))+s[i]);if(w.inequality_kind[i]==0){out.c.push_back(c);out.y.push_back(z[i]);}else out.b.push_back(c);}return out;}
struct Run {Workspace endpoint;Control result;std::vector<bool> paths;std::vector<long double> changes;std::vector<Vec> parameters;bool committed=false;};
static Run run(const Workspace& initial,const Direction& direction,bool master,bool on_failure,int timeout=0){
  Run out;out.endpoint=initial;const double mu=1e-12;
  gr::Parameters parameters;parameters.mu=mu;parameters.rho=1e-4;parameters.kappa=144076;parameters.center={0,initial.z[0]};
  auto attempt=[&](bool active,int attempt_index,bool& accepted,bool& stop,double& elapsed){
    (void)stop;out.paths.push_back(active);out.parameters.push_back({parameters.mu,parameters.rho,parameters.kappa,parameters.center[0],parameters.center[1]});
    const DcaMajorantBarrierPath path(initial,direction,false,true,nullptr,active);
    const double alpha=active?1:initial.fraction(direction,.995,1);
    const auto anchor=state(initial,path,0,mu),candidate=state(initial,path,alpha,mu);
    const auto slope=actual_engine_slope(initial,direction,path,anchor,parameters,active);
    Vec x;path.write_primal(alpha,x);Vec displacement={x[0]-initial.x[0]};const auto objective=DcaMajorantBarrierPath::prepare_original(initial.objective,displacement,initial.x);
    const auto change=gr::change(anchor,candidate,parameters,objective.change(1));out.changes.push_back(change.value);
    accepted=gr::armijo(change,slope,alpha,1e-4);
    if(accepted){
      if(timeout==3)elapsed=1;
      if(!actual_precommit_deadline(active,elapsed)){out.endpoint=actual_engine_commit(initial,direction,path,alpha,mu,active);out.committed=true;}
    }
    if(timeout==2&&attempt_index==0&&!accepted)elapsed=1;
  };
  out.result=actual_control(initial,direction,master,on_failure,timeout==1,attempt);
  return out;
}
int main(){try{
  require(!actual_parse_trigger(nullptr)&&!actual_parse_trigger("always")&&actual_parse_trigger("on_failure"),"actual trigger enum/default");
  for(const char* invalid:{"","1","ON_FAILURE","on-failure"}){bool rejected=false;try{actual_parse_trigger(invalid);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid trigger accepted");}
  const double mu=1e-12;auto initial=sample(mu,-1e-8);auto d=first(initial);
  auto precision_origin=initial;auto precision_direction=first(precision_origin);
  precision_direction.x[0]=1e-12;precision_direction.s[1]=1e-12;
  precision_direction.z[1]=-mu*1e-12;
  DcaMajorantBarrierPath product_path(precision_origin,precision_direction,false,true,nullptr,true,false);
  DcaMajorantBarrierPath representable_path(precision_origin,precision_direction,false,true,nullptr,true,true);
  Vec physical_slacks;representable_path.write_slacks(1,physical_slacks);
  require(representable_path.dual(1,1,mu,&physical_slacks)==precision_origin.z[1]+precision_direction.z[1],"rounding-limited bound lost audited affine dual");
  require(product_path.dual(1,1,mu,&physical_slacks)!=representable_path.dual(1,1,mu,&physical_slacks),"representability diagnostic premise missing");
  require(actual_line_budget(true,true,28,true)==28,"measured-stall path lacks its full unchanged Armijo search");

  auto repaired=run(initial,d,true,true);
  require(repaired.paths==std::vector<bool>({false,true})&&repaired.result.accepted&&repaired.result.consistent&&repaired.committed,"legacy failure did not activate accepted repair");
  require(repaired.changes[0]>1e-4L&&repaired.changes[1]<0,"actual GR ULP rejection/repair premise");
  require(repaired.parameters[0]==repaired.parameters[1],"repair changed frozen merit parameters");
  require(static_cast<long double>(initial.inequalities[1].original_at(repaired.endpoint.x))+repaired.endpoint.s[1]==0,"repaired endpoint lacks consistent bound link");
  auto strong=sample(mu,-1);d=first(strong);auto accepted=run(strong,d,true,true);
  require(accepted.paths==std::vector<bool>({false})&&accepted.result.accepted&&!accepted.result.consistent&&accepted.committed,"already accepted legacy trial entered repair");
  Vec expected_x=strong.x,expected_s=strong.s,expected_l=strong.lambda;axpy(1,d.x,expected_x);axpy(1,d.s,expected_s);axpy(1,d.lambda,expected_l);
  const DcaMajorantBarrierPath old_path(strong,d,false,true,nullptr,false);Vec expected_z;old_path.write_duals(1,mu,expected_z,&expected_s);
  require(same_vector(accepted.endpoint.x,expected_x)&&same_vector(accepted.endpoint.s,expected_s)&&same_vector(accepted.endpoint.z,expected_z)&&same_vector(accepted.endpoint.lambda,expected_l),"accepted legacy four-vector state changed by enabled fallback");
  require(same(accepted.endpoint.s[1],1.1),"accepted legacy slack silently made consistent");
  auto disabled=run(initial,first(initial),false,true);require(disabled.paths==std::vector<bool>({false})&&!disabled.committed,"disabled master enabled repair");
  auto always=run(initial,first(initial),true,false);require(always.paths==std::vector<bool>({true})&&always.committed,"always control changed");
  auto both_initial=initial;both_initial.z[0]=0;auto both_d=first(both_initial);both_d.s[0]=.001;
  auto both=run(both_initial,both_d,true,true);require(both.paths==std::vector<bool>({false,true})&&!both.result.accepted&&!both.committed,"both rejected paths falsely committed");
  require(same_vector(both.endpoint.x,both_initial.x)&&same_vector(both.endpoint.s,both_initial.s)&&same_vector(both.endpoint.z,both_initial.z)&&same_vector(both.endpoint.lambda,both_initial.lambda),"rejected attempts modified state");
  auto before=run(initial,first(initial),true,true,1);require(before.paths.empty()&&!before.committed,"initial stop entered path or committed");
  auto between=run(initial,first(initial),true,true,2);require(between.paths==std::vector<bool>({false})&&between.result.stop&&between.result.status==Status::TimeLimit&&!between.committed,"expired legacy failure attempted repair");
  auto late=run(initial,first(initial),true,true,3);require(late.paths==std::vector<bool>({false,true})&&late.result.accepted&&late.result.consistent&&!late.committed,"expired repair trial committed");
  require(actual_line_budget(true,true,28)==1,"repair still has a backtracking trial");
  require(actual_line_budget(false,true,28)==28&&actual_line_budget(true,false,28)==28,"legacy or always profile search budget changed");
  auto half_initial=sample(mu,-.075);half_initial.z[0]=0;auto half_d=first(half_initial);half_d.s[0]=.001;
  DcaMajorantBarrierPath half_path(half_initial,half_d,false,true,nullptr,true);
  gr::Parameters hp;hp.mu=mu;hp.rho=1e-4;hp.kappa=144076;hp.center={0,0};const auto ha=state(half_initial,half_path,0,mu);
  const auto hs=actual_engine_slope(half_initial,half_d,half_path,ha,hp,true);
  auto merit_at=[&](double a){Vec x;half_path.write_primal(a,x);Vec displacement={x[0]-half_initial.x[0]};const auto objective=DcaMajorantBarrierPath::prepare_original(half_initial.objective,displacement,half_initial.x);return gr::change(ha,state(half_initial,half_path,a,mu),hp,objective.change(1));};
  const auto full_change=merit_at(1),half_change=merit_at(.5);
  require(!gr::armijo(full_change,hs,1,1e-4)&&gr::armijo(half_change,hs,.5,1e-4),"full-reject/half-accept actual merit counterexample missing");
  auto declined=run(half_initial,half_d,true,true);
  require(declined.paths==std::vector<bool>({false,true})&&!declined.result.accepted&&!declined.committed,"full-only repair accepted an otherwise useful half step");
  auto fractional_d=first(half_initial);fractional_d.s[0]=-6;
  require(half_initial.fraction(fractional_d,.995,1)<1,"nominal fraction premise missing");
  auto skipped=run(half_initial,fractional_d,true,true);
  require(skipped.paths==std::vector<bool>({false})&&!skipped.committed,"fraction below one still entered repair");
  std::cout<<"R10_actual_full_step_controller_GR_commit PASS checks="<<checks<<" legacy_change="<<double(repaired.changes[0])<<" repair_change="<<double(repaired.changes[1])<<"\n";return 0;
}catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<" checks="<<checks<<"\n";return 1;}}
