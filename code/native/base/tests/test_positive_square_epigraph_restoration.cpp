#include "positive_square_epigraph_restoration.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
using namespace lazyipdca::qcqp::detail;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
QCQPProblem model(){
 QCQPProblem p;p.name="anonymous";p.n=3;p.x0={1,0,0};p.lb={0,-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY,INFINITY};
 p.objective.linear={1,0,0};p.objective.hessian.n=3;
 QuadraticFunction h;h.hessian.n=3;h.constant=-2;h.linear={0,1,0};p.equalities.push_back(h);
 h.constant=-3;h.linear={0,0,1};p.equalities.push_back(h);
 QuadraticFunction g;g.hessian.n=3;g.linear={0,0,0};g.constant=-0x1p-53;g.hessian.entries={{0,0,-2},{1,1,2},{2,2,2}};p.inequalities.push_back(g);p.validate();return p;
}
int main(){
 Settings st;st.linear_backend="mumps";st.linear_metis=true;
 const double deadline=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()+10;
 auto p=model();const auto frozen=p.x0;
 auto r=restore_positive_square_epigraph(p,st,deadline);
 std::cerr<<"diagnostic reason="<<r.reason<<" factors="<<r.factors<<" solves="<<r.solves<<" after="<<r.after<<"\n";
 check(r.accepted&&r.factors==1&&r.solves==1,"valid original-domain restoration rejected");
 check(p.x0==frozen&&r.x[1]==2&&r.x[2]==3&&feasibility(p,r.x).total_inf<=1e-6,"start modified or original feasibility lost");
 using Exact=boost::multiprecision::cpp_rational;
 check(Exact(r.x[1])*Exact(r.x[1])+Exact(r.x[2])*Exact(r.x[2])+Exact(p.inequalities[0].constant)-Exact(r.x[0])*Exact(r.x[0])<=0,"tiny constant not retained exactly");
 auto renamed=p;renamed.name="totally_unrelated";renamed.inequalities[0].name="arbitrary";
 check(restore_positive_square_epigraph(renamed,st,deadline).x==r.x,"label-dependent restoration");
 auto changed=p;changed.equalities[0].linear[0]=1;
 check(!restore_positive_square_epigraph(changed,st,deadline).eligible,"epigraph-dependent equality admitted");
 changed=p;changed.inequalities[0].hessian.entries.push_back({1,2,1e-30});
 check(!restore_positive_square_epigraph(changed,st,deadline).eligible,"tiny off-diagonal term ignored");
 changed=p;changed.lb[1]=0;
 check(!restore_positive_square_epigraph(changed,st,deadline).eligible,"other bounds ignored");
 changed=p;changed.inequalities[0].hessian.entries[1].v=-1e-30;
 check(!restore_positive_square_epigraph(changed,st,deadline).eligible,"tiny negative quadratic term ignored");
 changed=p;changed.inequalities[0].constant=0x1p-53;
 r=restore_positive_square_epigraph(changed,st,deadline);
 check(r.accepted,"positive constant incorrectly refused");
 check(Exact(r.x[1])*Exact(r.x[1])+Exact(r.x[2])*Exact(r.x[2])+Exact(changed.inequalities[0].constant)-Exact(r.x[0])*Exact(r.x[0])<=0,"positive tiny constant lost");
 check(!restore_positive_square_epigraph(p,st,0).attempted,"expired work admitted");
 std::cout<<"positive_square_epigraph_restoration PASS exact constants, original audit, affine projection, domain rejections, labels and deadlines\n";
}
