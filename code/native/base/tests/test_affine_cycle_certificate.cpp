#include "affine_cycle_certificate.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;using namespace lazyipdca::qcqp::detail;
void check(bool v,const char* s){if(!v)throw std::runtime_error(s);}
QCQPProblem model(){
 QCQPProblem p;p.name="anonymous";p.n=3;p.x0={0,0,0};p.lb.assign(3,-INFINITY);p.ub.assign(3,INFINITY);
 p.objective.hessian.n=3;p.objective.linear.assign(3,0);
 QuadraticFunction q;q.hessian.n=3;q.linear={-30,0,30};p.equalities.push_back(q);
 q.constant=1;q.linear={0,-30,30};p.equalities.push_back(q);
 q.constant=0;q.linear={-30,30,0};p.equalities.push_back(q);p.validate();return p;
}
int main(){
 const double end=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()+20;
 auto p=model();const auto c=affine_cycle_certificate(p,end);
 check(c.verified&&c.rows.size()==3&&c.primal_lower_bound=="1/3","exact cycle contradiction missing");
 using Q=boost::multiprecision::cpp_rational;std::vector<Q> a(3);Q b=0,norm=0;
 for(std::size_t k=0;k<c.rows.size();++k){Q w(c.weights[k]);const auto& row=p.equalities[c.rows[k]];
  b+=w*Q(row.constant);norm+=w<0?-w:w;row.for_each_linear_nonzero([&](int j,double v){a[j]+=w*Q(v);});}
 check(b!=0&&a==std::vector<Q>(3)&&Q(c.primal_lower_bound)==abs(b)/norm,"exported proof does not sum to contradiction");
 auto renamed=p;renamed.name="unrelated";std::reverse(renamed.equalities.begin(),renamed.equalities.end());check(affine_cycle_certificate(renamed,end).verified,"name or row-order dispatch");
 auto consistent=p;consistent.equalities[1].constant=0;check(!affine_cycle_certificate(consistent,end).verified,"consistent cycle declared infeasible");
 auto curved=p;curved.equalities[1].hessian.entries={{0,0,1e-300}};check(!affine_cycle_certificate(curved,end).verified,"tiny nonlinear term discarded");
 auto tiny=p;tiny.equalities[1].constant=1e-300;check(affine_cycle_certificate(tiny,end).verified,"tiny contradictory constant discarded");
 check(!affine_cycle_certificate(p,0).verified,"expired search");
 Settings st;st.native_al_fallback="off";st.dca_engine="majorants";auto r=Solver(st).solve(p);
 check(r.status==Status::Infeasible&&r.affine_infeasibility_certificate&&!r.termination_audit_valid&&!r.native_al_fallback.attempted,"infeasibility misreported as KKT success");
 auto rt=Solver(st).solve(tiny);
 check(rt.status==Status::Solved&&!rt.affine_infeasibility_certificate,"sub-tolerance contradiction blocked a valid residual certificate");
 std::cout<<"affine_cycle_certificate PASS exact identity, norm lower bound, tiny terms, row order, names, native failure status\n";
}
