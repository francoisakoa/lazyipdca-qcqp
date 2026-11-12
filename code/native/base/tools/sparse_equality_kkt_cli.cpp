#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/acceptable.hpp"
#include <dmumps_c.h>
#include "../src/mumps_ordering_guard.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <functional>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace lazyipdca::qcqp;
namespace {
using Clock=std::chrono::steady_clock;
using Row=std::vector<std::pair<int,double>>;
struct Triplet {int i,j;double value;};
double l2(const Vec& x){long double s=0;for(double v:x)s+=(long double)v*v;return std::sqrt(s);}
bool finite(const Vec& x){for(double v:x)if(!std::isfinite(v))return false;return true;}
void require(bool good,const std::string& why){if(!good)throw std::runtime_error(why);}
struct Point {Vec x,y;};
struct State {Vec g,h,rd;std::vector<Row> rows;double objective=0,primal=0,dual=0,merit=0;};
struct Context {
 Clock::time_point begin=Clock::now();double seconds=30;int factors=0,solves=0,newton=0,normal=0,fits=0,trials=0,accepted=0,audits=0;
 double max_linear_audit=0,max_forcing_ratio=0,max_normal_residual=0;std::function<bool()> external_open;
 bool open()const{return elapsed()<seconds&&(!external_open||external_open());} double elapsed()const{return std::chrono::duration<double>(Clock::now()-begin).count();}
 void check()const{require(open(),"private deadline");}
};
long double value(const QuadraticFunction& q,const Vec& x){
 long double v=q.constant;q.for_each_linear_nonzero([&](int j,double a){v+=(long double)a*x[j];});
 for(const auto&e:q.hessian.entries)v+=(e.i==e.j?.5L:1.L)*e.v*x[e.i]*x[e.j];
 require(std::isfinite(v),"nonfinite original quadratic value");return v;
}
struct Problem {
 const QCQPProblem& p;double sf=1;Vec hs;mutable std::vector<long double> work;mutable std::vector<char> seen;mutable std::vector<int> touched;
 Problem(const QCQPProblem& model,const Point& point):p(model){
  // These guards precede structural work arrays, dense-gradient scratch,
  // Jacobian rows and every numerical factor allocation.
  require(p.n>0&&p.n<=100000&&p.equalities.size()<=100000&&
      p.equalities.size()<std::size_t(p.n),"sparse equality finisher dimensions");
  require(p.inequalities.empty()&&!p.has_signed_square(),"only smooth equality-only QCQP supported");
  p.validate();
  require(point.x.size()==std::size_t(p.n)&&point.y.size()==p.equalities.size()&&
      finite(point.x)&&finite(point.y),"invalid sparse starting coordinates/multipliers");
  for(int j=0;j<p.n;++j)require(p.lb[j]==-std::numeric_limits<double>::infinity()&&
      p.ub[j]==std::numeric_limits<double>::infinity(),"canonical unbounded sides required");
  long double support=0;
  auto count=[&](const QuadraticFunction& q){
    q.for_each_linear_nonzero([&](int,double){support+=1;});
    support+=2.0L*q.hessian.entries.size();
  };
  count(p.objective);for(const auto&q:p.equalities)count(q);
  const long double auxiliary_bytes=2048.0L*p.n+256.0L*p.equalities.size()+512.0L*support;
  require(support<=4000000&&auxiliary_bytes<=384.0L*1024*1024,
      "sparse construction storage budget");
  work.assign(p.n,0);seen.assign(p.n,0);
  auto g=gradient(p.objective,point.x);double gn=0;
  for(auto e:g)gn=std::max(gn,std::abs(e.second));
  sf=1/std::max(1.,gn);
  for(const auto&q:p.equalities){double n=0;
   for(auto e:gradient(q,point.x))n=std::max(n,std::abs(e.second));
   hs.push_back(1/std::max(1.,n));
  }
  require(std::isfinite(sf)&&sf>0&&finite(hs),"invalid frozen original scaling");
 }
 Row gradient(const QuadraticFunction& q,const Vec& x)const {
  const auto add=[&](int j,long double v){if(!seen[j]){seen[j]=1;touched.push_back(j);}work[j]+=v;};
  q.for_each_linear_nonzero([&](int j,double a){add(j,a);});
  for(const auto&e:q.hessian.entries){add(e.i,(long double)e.v*x[e.j]);if(e.i!=e.j)add(e.j,(long double)e.v*x[e.i]);}
  std::sort(touched.begin(),touched.end());Row row;row.reserve(touched.size());
  for(int j:touched){require(std::isfinite(work[j])&&std::isfinite(double(work[j])),"nonfinite original gradient");if(work[j]!=0)row.push_back({j,double(work[j])});work[j]=0;seen[j]=0;}touched.clear();return row;
 }
 State state(const Point& t)const {
  require(t.x.size()==std::size_t(p.n)&&t.y.size()==p.equalities.size()&&finite(t.x)&&finite(t.y),"invalid sparse point");
  State s;s.g.assign(p.n,0);s.h.resize(p.equalities.size());s.rows.reserve(p.equalities.size());
  for(auto e:gradient(p.objective,t.x))s.g[e.first]=e.second;
  std::vector<long double> rd(s.g.begin(),s.g.end());
  for(std::size_t e=0;e<p.equalities.size();++e){s.h[e]=double(value(p.equalities[e],t.x));s.primal=std::max(s.primal,std::abs(s.h[e]));auto row=gradient(p.equalities[e],t.x);for(auto a:row)rd[a.first]+=(long double)t.y[e]*a.second;s.rows.push_back(std::move(row));}
  s.rd.resize(p.n);long double merit=0;
  for(int j=0;j<p.n;++j){s.rd[j]=double(rd[j]);s.dual=std::max(s.dual,std::abs(s.rd[j]));merit+=(long double)sf*sf*rd[j]*rd[j];}
  for(std::size_t e=0;e<s.h.size();++e)merit+=(long double)hs[e]*hs[e]*s.h[e]*s.h[e];
  s.merit=std::sqrt(merit);s.objective=double(value(p.objective,t.x));
  require(finite(s.g)&&finite(s.h)&&finite(s.rd)&&std::isfinite(s.merit)&&
      std::isfinite(s.objective)&&std::isfinite(s.primal)&&std::isfinite(s.dual),
      "nonfinite original KKT state");return s;
 }
 std::vector<Triplet> normal_matrix(const State& s,double delta)const {
  std::vector<Triplet> a;a.reserve(p.n+s.h.size()*8);for(int j=0;j<p.n;++j)a.push_back({j,j,1});
  for(std::size_t e=0;e<s.h.size();++e){for(auto v:s.rows[e])a.push_back({v.first,p.n+int(e),hs[e]*v.second});a.push_back({p.n+int(e),p.n+int(e),-delta});}return a;
 }
 std::vector<Triplet> newton_matrix(const Point& t,const State& s,double regularization)const {
  std::map<std::pair<int,int>,long double> h;
  for(auto e:p.objective.hessian.entries)h[{e.i,e.j}]+=(long double)sf*e.v;
  for(std::size_t k=0;k<p.equalities.size();++k)for(auto e:p.equalities[k].hessian.entries)h[{e.i,e.j}]+=(long double)sf*t.y[k]*e.v;
  std::vector<Triplet>a;a.reserve(h.size()+p.n+s.h.size()*8);
  for(auto e:h)if(e.second!=0)a.push_back({e.first.first,e.first.second,double(e.second)});
  for(int j=0;j<p.n;++j)a.push_back({j,j,regularization});
  for(std::size_t e=0;e<s.h.size();++e){for(auto v:s.rows[e])a.push_back({v.first,p.n+int(e),hs[e]*v.second});a.push_back({p.n+int(e),p.n+int(e),-regularization});}return a;
 }
 Vec rhs(const State& s)const {Vec b(p.n+s.h.size());for(int j=0;j<p.n;++j)b[j]=-sf*s.rd[j];for(std::size_t e=0;e<s.h.size();++e)b[p.n+e]=-hs[e]*s.h[e];return b;}
};
Vec apply(int n,const std::vector<Triplet>& a,const Vec& x){std::vector<long double> v(n,0);for(auto e:a){v[e.i]+=(long double)e.value*x[e.j];if(e.i!=e.j)v[e.j]+=(long double)e.value*x[e.i];}return Vec(v.begin(),v.end());}
bool solve(Context& c,int n,const std::vector<Triplet>& a,const Vec& b,Vec& x,const char* role){
 c.check();require(n>0&&n<=200000&&a.size()<=4000000&&b.size()==std::size_t(n)&&finite(b),"linear storage/input budget");
 for(auto e:a)require(e.i>=0&&e.j>=e.i&&e.j<n&&std::isfinite(e.value),"invalid sparse KKT triplet");
 DMUMPS_STRUC_C id{};id.job=-1;id.par=1;id.sym=2;id.comm_fortran=-987654;dmumps_c(&id);
 struct Finish{DMUMPS_STRUC_C& id;~Finish(){id.job=-2;dmumps_c(&id);}} finish{id};
 id.icntl[0]=id.icntl[1]=id.icntl[2]=-1;id.icntl[3]=0;id.icntl[6]=5;id.icntl[27]=1;id.icntl[12]=1;id.icntl[13]=20;id.icntl[22]=128;id.cntl[0]=.01;
 std::vector<MUMPS_INT>irn,jcn;Vec values;irn.reserve(a.size());jcn.reserve(a.size());values.reserve(a.size());for(auto e:a){irn.push_back(e.i+1);jcn.push_back(e.j+1);values.push_back(e.value);}
 id.n=n;id.nnz=a.size();id.irn=irn.data();id.jcn=jcn.data();id.a=values.data();++c.factors;id.job=1;dmumps_c(&id);c.check();
 std::cout<<"[MUMPS-ORDERING] role="<<role<<" requested_ordering=5 requested_analysis=1 actual_ordering="<<id.infog[6]<<" actual_analysis="<<id.infog[31]<<'\n';
 if(id.infog[0]<0)return false;
 std::string ordering_message;
 if(!lazyipdca::qcqp::detail::mumps_requested_ordering_verified(true,id.infog[6],id.infog[31],ordering_message)){std::cout<<"[MUMPS-ORDERING-REJECT] "<<ordering_message<<'\n';return false;}
 id.job=2;dmumps_c(&id);c.check();
 std::cout<<"[SPARSE-ORIGINAL-FACTOR] role="<<role<<" n="<<n<<" nnz="<<a.size()<<" code="<<id.infog[0]<<" detail="<<id.infog[1]<<" negative="<<id.infog[11]<<" margin=20 workspace_mb=128 signed_inertia_not_convexity=1\n";
 if(id.infog[0]<0)return false;
 x=b;id.rhs=x.data();id.nrhs=1;id.lrhs=n;id.job=3;dmumps_c(&id);++c.solves;c.check();if(id.infog[0]<0||!finite(x))return false;
 for(int refine=0;refine<=4;++refine){Vec residual=apply(n,a,x);double err=0,den=inf_norm(b);for(int j=0;j<n;++j){residual[j]=b[j]-residual[j];err=std::max(err,std::abs(residual[j]));}den=std::max(den,inf_norm(apply(n,a,x)));double audit=err/std::max(1.,den);std::cout<<"[SPARSE-ORIGINAL-SOLVE] role="<<role<<" refinement="<<refine<<" residual="<<err<<" normwise="<<audit<<'\n';if(audit<=1e-10){c.max_linear_audit=std::max(c.max_linear_audit,audit);return true;}if(refine==4)return false;id.rhs=residual.data();id.job=3;dmumps_c(&id);++c.solves;c.check();if(id.infog[0]<0||!finite(residual))return false;axpy(1,residual,x);}
 return false;
}
bool fit(Context& c,const Problem& p,Point& t){
 c.check();const auto before=p.state(t);Vec b(p.p.n+before.h.size(),0),z;
 for(int j=0;j<p.p.n;++j)b[j]=-p.sf*before.g[j];
 ++c.fits;
 if(!solve(c,b.size(),p.normal_matrix(before,0),b,z,"free-dual-fit"))return false;
 Point trial=t;
 for(std::size_t e=0;e<before.h.size();++e)trial.y[e]=p.hs[e]/p.sf*z[p.p.n+e];
 const auto after=p.state(trial);c.check();
 // A numerically inferior fit does not replace already better multipliers.
 if(after.dual<before.dual)t.y=std::move(trial.y);
 return true;
}

bool retract(Context& c,const Problem& p,Point& t,double correction_cap){
 require(std::isfinite(correction_cap)&&correction_cap>=0,"invalid normal correction cap");
 const Vec initial=t.x;
 for(int k=0;k<5;++k){
  c.check();const auto s=p.state(t);if(s.primal<=1e-10)return true;
  Vec b(p.p.n+s.h.size(),0),z;
  for(std::size_t e=0;e<s.h.size();++e)b[p.p.n+e]=-p.hs[e]*s.h[e];
  ++c.normal;
  if(!solve(c,b.size(),p.normal_matrix(s,0),b,z,"normal-retraction"))return false;
  Vec delta(z.begin(),z.begin()+p.p.n);long double error2=0;
  for(std::size_t e=0;e<s.h.size();++e){
   long double r=p.hs[e]*s.h[e];
   for(auto a:s.rows[e])r+=(long double)p.hs[e]*a.second*delta[a.first];
   error2+=r*r;
  }
  const double error=std::sqrt(error2);Vec scaled_h=s.h;
  for(std::size_t e=0;e<s.h.size();++e)scaled_h[e]*=p.hs[e];
  if(!std::isfinite(error)||error>std::max(1e-12,1e-8*l2(scaled_h)))return false;
  c.max_normal_residual=std::max(c.max_normal_residual,error);
  axpy(1,delta,t.x);Vec total=t.x;axpy(-1,initial,total);
  if(!finite(t.x)||l2(total)>correction_cap)return false;
 }
 c.check();return p.state(t).primal<=1e-10;
}

Settings audit_settings(){
 Settings s;s.termination_test="project";
 s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=
     s.complementarity_tolerance=1e-6;
 s.acceptable_iterations=0;s.acceptable_on_failure=false;return s;
}
bool direct_original_audit(const QCQPProblem& p,Result& r){
 // This tool always holds the ORIGINAL input. Avoid the mapped/presolve
 // dispatcher and its unused ordinary-optimizer link dependency.
 const auto settings=audit_settings();detail::DirectOriginalAuditView view{p,r};
 return detail::audit_original_termination_impl(p,settings,r,view,nullptr,false,nullptr);
}
Result payload(const Problem& p,const Point& t){
 Result r;r.x=t.x;r.certificate_equality_multipliers=t.y;
 r.certificate_lower_multipliers.assign(p.p.n,0);
 r.certificate_upper_multipliers.assign(p.p.n,0);
 direct_original_audit(p.p,r);
 r.message="Explicit optional sparse ORIGINAL signed-Lagrangian KKT finisher; ordinary DCA is unchanged";
 return r;
}
bool strict(const Result& r){
 return r.termination_audit_valid&&std::isfinite(r.raw_kkt_error)&&
     r.feasibility.total_inf<=1e-6&&r.termination_dual_inf<=1e-6&&
     r.termination_compl_inf<=1e-6&&r.raw_kkt_error<=1e-6;
}
struct Outcome {Point trial,point;bool committed=false;std::string reason="iteration-budget";};
Outcome finish(Context& c,const Problem& p,const Point& primary,int maximum_newton){
 Outcome out;out.point=out.trial=primary;Point current=primary;
 try{
  c.check();require(maximum_newton>=0&&maximum_newton<=256,"signed Newton attempt budget outside0..256");
  if(maximum_newton==0)return out;
  if(!fit(c,p,current)){out.reason="free-dual-fit-factor-failure";return out;}
  out.trial=current;
  while(c.newton<maximum_newton){
   c.check();const auto s=p.state(current);
   std::cout<<std::setprecision(17)<<"[SPARSE-ORIGINAL-POINT] iteration="<<c.newton
       <<" f="<<s.objective<<" primal="<<s.primal<<" dual="<<s.dual
       <<" merit="<<s.merit<<" seconds="<<c.elapsed()<<'\n';
   ++c.audits;const auto audited=payload(p,current);c.check();
   if(strict(audited)){out.committed=true;out.reason="strict-original-QCQP";break;}
   const auto b=p.rhs(s);const auto original=p.newton_matrix(current,s,0);Vec d;
   // One fresh, signed, unshifted ORIGINAL matrix per attempted Newton step.
   // Singular/inaccurate factors do not acquire a hidden convexification.
   ++c.newton;
   if(!solve(c,b.size(),original,b,d,"signed-Lagrangian-Newton")){
    out.reason="original-Newton-factor-failure";break;
   }
   Vec residual=apply(b.size(),original,d);axpy(-1,b,residual);
   const double forcing=l2(residual)/std::max(1e-300,l2(b));
   std::cout<<"[SPARSE-ORIGINAL-FORCING] regularization=0 ratio="<<forcing<<'\n';
   if(!std::isfinite(forcing)||forcing>.25){out.reason="original-Newton-forcing-rejected";break;}
   c.max_forcing_ratio=std::max(c.max_forcing_ratio,forcing);
   const int n=p.p.n;Vec dx(d.begin(),d.begin()+n);
   const double radius=.1*(1+l2(current.x))/std::sqrt(double(n));
   require(std::isfinite(radius)&&radius>0,"invalid predictor radius");
   double alpha=std::min(1.,radius/std::max(1e-300,l2(dx)));bool accepted=false;
   for(int ls=0;ls<24;++ls){
    c.check();++c.trials;Point trial=current;
    for(int j=0;j<n;++j)trial.x[j]+=alpha*d[j];
    for(std::size_t e=0;e<s.h.size();++e)trial.y[e]+=alpha*p.hs[e]/p.sf*d[n+e];
    if(!finite(trial.x)||!finite(trial.y)||
        !retract(c,p,trial,.5*alpha*l2(dx))||!fit(c,p,trial)){
     alpha*=.5;continue;
    }
    const auto after=p.state(trial);c.check();
    std::cout<<"[SPARSE-ORIGINAL-TRIAL] alpha="<<alpha<<" primal="<<after.primal
        <<" dual="<<after.dual<<" merit="<<after.merit<<" previous="<<s.merit
        <<" regularization=0\n";
    if(after.primal<=std::max(1e-8,s.primal)&&after.merit<(1-1e-4*alpha)*s.merit){
     current=std::move(trial);out.trial=current;accepted=true;++c.accepted;break;
    }
    alpha*=.5;
   }
   if(!accepted){out.reason="original-KKT-merit-rejected";break;}
  }
  // A strict last permitted step may commit without spending another step.
  c.check();++c.audits;const auto audited=payload(p,current);c.check();
  if(strict(audited)){out.committed=true;out.reason="strict-original-QCQP";}
 }catch(const std::exception& e){out.committed=false;out.reason=e.what();}
 if(!c.open()){out.committed=false;out.reason="private-deadline";}
 if(out.committed)out.point=current;
 return out;
}

Vec read_vector(std::istream& in){
 std::uint64_t n=0;in.read(reinterpret_cast<char*>(&n),sizeof(n));
 require(bool(in)&&n<=100000,"invalid warm vector length");Vec v(n);
 in.read(reinterpret_cast<char*>(v.data()),n*sizeof(double));
 require(bool(in)&&finite(v),"invalid nonfinite/truncated warm vector");return v;
}
Point read_warm(const std::string& path,const QCQPProblem& p,Result& raw){
 std::ifstream in(path,std::ios::binary);char header[8];in.read(header,8);
 require(bool(in)&&std::string(header,8)=="DCAEP001","invalid warm header");
 raw.x=read_vector(in);raw.certificate_equality_multipliers=read_vector(in);
 raw.certificate_inequality_multipliers=read_vector(in);
 raw.certificate_lower_multipliers=read_vector(in);raw.certificate_upper_multipliers=read_vector(in);
 require(raw.x.size()==std::size_t(p.n)&&raw.certificate_equality_multipliers.size()==p.equalities.size()&&
     raw.certificate_inequality_multipliers.size()==p.inequalities.size()&&
     raw.certificate_lower_multipliers.size()==std::size_t(p.n)&&
     raw.certificate_upper_multipliers.size()==std::size_t(p.n),"warm dimensions");
 require(in.peek()==std::char_traits<char>::eof(),"trailing warm content");
 return {raw.x,raw.certificate_equality_multipliers};
}
double requested_seconds(const std::string& text){
 std::size_t used=0;const double v=std::stod(text,&used);
 require(used==text.size()&&std::isfinite(v)&&v>=0,"invalid finite time budget");
 return std::min(60.,v);
}
int requested_iterations(const std::string& text){
 std::size_t used=0;const int n=std::stoi(text,&used);
 require(used==text.size()&&n>=0&&n<=256,"signed Newton budget outside0..256");return n;
}

void unit(){
 int checks=0;const auto check=[&](bool b,const char* why){++checks;require(b,why);};
 QCQPProblem raw;raw.name="anonymous_sparse_terminal_fixture";raw.n=600;
 raw.x0.assign(600,1.0001);raw.lb.assign(600,-INFINITY);raw.ub.assign(600,INFINITY);
 raw.objective.compact_linear_dimension=600;raw.objective.hessian.n=600;raw.objective.constant=600;
 for(int j=0;j<600;++j){raw.objective.compact_linear.push_back({j,-2});raw.objective.hessian.entries.push_back({j,j,2});}
 for(int j=0;j<300;++j){QuadraticFunction h;h.constant=-1;h.compact_linear_dimension=600;
  h.hessian.n=600;h.hessian.entries={{j,j,2}};raw.equalities.push_back(h);}
 Point primary{raw.x0,Vec(300,0)};Problem p(raw,primary);Context c;c.seconds=10;
 const auto success=finish(c,p,primary,16);
 check(success.committed&&strict(payload(p,success.point)),"anonymous sparse original-KKT fixture not strict");
 check(c.newton<=16&&c.normal>0&&c.factors>=c.newton,"native sparse counters are inconsistent");
 for(int kind=0;kind<3;++kind){Context stopped;stopped.seconds=10;
  if(kind==0)stopped.seconds=0;
  if(kind==1)stopped.external_open=[&](){return stopped.solves==0;};
  const auto result=finish(stopped,p,primary,kind==2?0:16);
  check(!result.committed&&result.point.x==primary.x&&result.point.y==primary.y,
      "failed/expired/exhausted sparse finish did not roll back primary exactly");
  if(kind!=1)check(stopped.newton==0&&stopped.factors==0,"closed budget started a factor");
 }
 auto dependent=raw;for(auto& h:dependent.equalities)h=raw.equalities.front();
 Problem rank_problem(dependent,primary);Context rank;rank.seconds=10;
 const auto deficient=finish(rank,rank_problem,primary,16);
 check(!deficient.committed&&deficient.point.x==primary.x&&deficient.point.y==primary.y,
     "rank-deficient sparse normal solve silently shifted or committed");
 for(int kind=0;kind<4;++kind){bool refused=false;try{
  auto invalid=raw;Point start=primary;
  if(kind==0)invalid.n=std::numeric_limits<int>::max();
  if(kind==1)invalid.lb[0]=INFINITY;
  if(kind==2)start.x[0]=std::numeric_limits<double>::quiet_NaN();
  if(kind==3)invalid.equalities[0].hessian.entries[0].v=INFINITY;
  Problem guarded(invalid,start);
 }catch(const std::exception&){refused=true;}
 check(refused,"invalid sparse input reached gradient/structural allocation");}
 for(const auto& text:{"nan","inf","-1","1x"}){bool refused=false;
  try{requested_seconds(text);}catch(const std::exception&){refused=true;}
  check(refused,"invalid sparse time budget admitted");}
 for(const auto& text:{"-1","257","1x"}){bool refused=false;
  try{requested_iterations(text);}catch(const std::exception&){refused=true;}
  check(refused,"invalid sparse Newton budget admitted");}
 Context bad;bad.seconds=10;bool refused=false;try{Vec x;
  solve(bad,1,{{0,0,std::numeric_limits<double>::quiet_NaN()}},{1},x,"invalid-fixture");
 }catch(const std::exception&){refused=true;}
 check(refused&&bad.factors==0,"nonfinite sparse matrix reached MUMPS");
 std::cout<<"sparse_original_kkt_unit PASS checks="<<checks<<" anonymous_n=600 signed_newton="
     <<c.newton<<" normal="<<c.normal<<" no_dense_arrays=1 exact_rollback=1"<<std::endl;
}
}

int main(int argc,char**argv){try{
 for(const char*k:{"OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"})setenv(k,"1",1);
#ifdef _OPENMP
 omp_set_num_threads(1);
#endif
 if(argc==2&&std::string(argv[1])=="--self-test"){unit();return 0;}
 require(argc==7||argc==8,"usage input.qcq warm.bin private_trial.json committed_or_primary.json report.json seconds [iterations<=256]");
 Context c;c.seconds=requested_seconds(argv[6]);const int maximum=argc==8?requested_iterations(argv[7]):256;
 const auto raw=read_qcq_compact(argv[1]);Result primary_result;
 const Point primary=read_warm(argv[2],raw,primary_result);Problem p(raw,primary);
 // Preserve all supplied multipliers on failure, including absent-bound sides.
 direct_original_audit(raw,primary_result);++c.audits;
 const auto out=finish(c,p,primary,maximum);Result trial=payload(p,out.trial);++c.audits;
 bool commit=out.committed&&strict(trial)&&c.open();
 auto write_outputs=[&](){
  trial.status=commit?Status::Solved:Status::MaxOuterIterations;
  auto final=commit?trial:primary_result;
  if(!commit)final.status=Status::MaxOuterIterations;
  write_result_json(trial,raw.name,argv[3]);
  if(!c.open()){commit=false;final=primary_result;final.status=Status::MaxOuterIterations;}
  write_result_json(final,raw.name,argv[4]);
 };
 write_outputs();
 if(!c.open()){commit=false;write_outputs();}
 auto write_report=[&](){std::ofstream f(argv[5]);require(bool(f),"cannot write sparse finisher report");
  f<<std::setprecision(17)<<"{\"stage\":\"feasible_original_kkt_finisher\",\"implementation\":\"sparse-equality-signed-Lagrangian\",\"committed_by_native\":"<<(commit?"true":"false")
   <<",\"reason\":\""<<(c.open()?out.reason:"private-deadline")<<"\",\"seconds_all_inclusive\":"<<c.elapsed()
   <<",\"Newton_directions\":"<<c.newton<<",\"normal_directions\":"<<c.normal<<",\"dual_fits\":"<<c.fits
   <<",\"dual_basis_solves\":0,\"factorizations\":"<<c.factors<<",\"linear_solves\":"<<c.solves
   <<",\"line_search_trials\":"<<c.trials<<",\"accepted_private_steps\":"<<c.accepted<<",\"full_original_audits\":"<<c.audits
   <<",\"max_expanded_linear_audit\":"<<c.max_linear_audit<<",\"max_normal_linear_residual\":"<<c.max_normal_residual
   <<",\"max_original_Newton_forcing_ratio\":"<<c.max_forcing_ratio
   <<",\"original_Newton_forcing_ratio_threshold\":0.25"
   <<",\"backend_shifts\":0,\"Cq\":0,\"ordinary_DCA_acceptance\":false,\"global_convexity_claimed\":false,\"production_enabled\":false}"<<std::endl;
 };
 write_report();
 if(commit&&!c.open()){commit=false;write_outputs();write_report();}
 return commit?0:1;
 }catch(const std::exception&e){std::cerr<<"[sparse-original-kkt fatal] "<<e.what()<<std::endl;return 2;}}
