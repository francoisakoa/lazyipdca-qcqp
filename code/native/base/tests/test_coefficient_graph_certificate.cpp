#include "coefficient_graph_certificate.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
using namespace lazyipdca::qcqp::detail;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
QCQPProblem model(int N=5,int d=2){
 QCQPProblem p;p.n=N*d;p.name="anonymous";p.x0.assign(p.n,1);p.lb.assign(p.n,-INFINITY);p.ub.assign(p.n,INFINITY);
 p.objective.hessian.n=p.n;p.objective.linear.assign(p.n,0);
 for(int a=0;a<d;++a)for(int i=0;i<N;++i)for(int j=i;j<N;++j)p.objective.hessian.entries.push_back({i*d+a,j*d+a,-2.*(a+1)});
 QuadraticFunction sphere;sphere.hessian.n=p.n;sphere.linear.assign(p.n,0);sphere.constant=-2;
 for(int j=0;j<p.n;++j)sphere.hessian.entries.push_back({j,j,4});p.equalities.push_back(sphere);
 for(int i=0;i<N-1;++i){QuadraticFunction e;e.hessian.n=p.n;e.linear.assign(p.n,0);for(int a=0;a<d;++a)e.hessian.entries.push_back({i*d+a,(i+1)*d+a,-3});p.equalities.push_back(e);}
 p.validate();return p;
}
void audit(const QCQPProblem& p,const CoefficientGraphCertificate& r){
 check(r.constructed,"coefficient-domain construction missing");
 check(feasibility(p,r.x).total_inf<=1e-12,"original feasibility");
 auto g=p.objective.gradient(r.x);
 for(int e=0;e<int(p.equalities.size());++e)axpy(r.multipliers[e],p.equalities[e].gradient(r.x),g);
 check(inf_norm(g)<=1e-12,"original stationarity");
}
int main(){
 Settings st;st.linear_backend="mumps";st.dca_engine="majorants";st.native_al_fallback="off";
 auto now=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
 auto p=model();auto frozen=p.x0;auto r=coefficient_graph_certificate(p,st,now+30);audit(p,r);
 check(p.x0==frozen&&r.independent==3,"start mutation or graph coverage");
 auto renamed=p;renamed.name="another_catalogue_entry";for(auto& q:renamed.equalities)q.name="different";
 check(coefficient_graph_certificate(renamed,st,now+30).x==r.x,"name-dependent dispatch");
 for(int d:{1,2,3})audit(model(7,d),coefficient_graph_certificate(model(7,d),st,now+30));
 auto invalid=p;invalid.equalities[1].constant=1e-300;
 check(!coefficient_graph_certificate(invalid,st,now+30).constructed,"tiny equality constant ignored");
 invalid=p;invalid.objective.linear[0]=1e-300;
 check(!coefficient_graph_certificate(invalid,st,now+30).constructed,"tiny affine objective ignored");
 invalid=p;invalid.objective.hessian.entries.push_back(invalid.objective.hessian.entries[0]);
 check(!coefficient_graph_certificate(invalid,st,now+30).constructed,"duplicate Hessian entry admitted");
 invalid=p;invalid.equalities[1].hessian.entries[0].v=-3.0000000000000004;
 check(!coefficient_graph_certificate(invalid,st,now+30).constructed,"inexact block alignment admitted");
 invalid=p;invalid.lb[0]=0;
 check(!coefficient_graph_certificate(invalid,st,now+30).constructed,"finite bound admitted");
 invalid=p;invalid.x0=r.x;
 check(!coefficient_graph_certificate(invalid,st,now+30).constructed,"already feasible start replaced");
 auto explicit_off=st;explicit_off.standard_r10_policy="off";
 check(!coefficient_graph_certificate(p,explicit_off,now+30).constructed,"historical control changed");
 for(const char* config:{"HH","HS","SH","SS","SS_legacy"}){auto study=st;study.convexification_study=config;check(!coefficient_graph_certificate(p,study,now+30).constructed,"curvature study silently converted to reference certificate path");}
 check(!coefficient_graph_certificate(p,st,0).constructed,"expired construction");
 auto native=Solver(st).solve(p);
 check(native.status==Status::Solved&&native.coefficient_graph_certificate&&native.termination_audit_valid,"native original certificate wiring");
 check(native.outer_iterations==0&&native.restoration_iterations==1&&!native.native_al_fallback.attempted,"provenance or work counts");
 std::cout<<"coefficient_graph_certificate PASS original KKT, exact domain, names, deadlines, six policies and native audit\n";
}
