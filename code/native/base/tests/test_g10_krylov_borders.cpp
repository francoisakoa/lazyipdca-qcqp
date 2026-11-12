#include "../src/atomic_newton.cpp"
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void run(double slack,bool dense,bool eliminated,bool atom_borders,int budget,bool success){
  QCQPProblem p;p.name="short_row_gram_cancellation";p.n=eliminated?3:2;
  p.x0={0.002,-0.002};if(eliminated)p.x0.push_back(0.002*0.002);
  p.lb=Vec(p.n,-INFINITY);p.ub=Vec(p.n,INFINITY);
  p.objective.linear=Vec(p.n,0);p.objective.linear[0]=p.objective.linear[1]=-1;
  p.objective.hessian.n=p.n;p.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction q;q.linear=Vec(p.n,0);q.linear[0]=q.linear[1]=1;
  q.constant=-slack;q.hessian.n=p.n;p.inequalities.push_back(q);
  // Inactive rows must never acquire a numerical border.
  QuadraticFunction inactive=q;inactive.constant=-10;p.inequalities.push_back(inactive);
  if(eliminated){QuadraticFunction e;e.linear={0,0,1};e.hessian.n=3;
    e.hessian.entries={{0,0,-2}};p.equalities.push_back(e);}
  Settings s;s.linear_backend="native";s.dca_engine="filter";
  s.lazy_newton_atomic_regularization=false;s.live_display_level=1;
  s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
  s.kkt_backward_error_reject=1e-7;s.dca_krylov_max_iterations=budget;
  if(dense)s.lazy_newton_dense_threshold=1;
  LazyNewtonStatistics ls;detail::NewtonCostLedger costs(ls,Clock::now());
  Workspace w(p,s,ls,costs);w.s[0]=slack;w.z[0]=1;w.active[0]=true;
  w.s[1]=10;w.z[1]=0;w.active[1]=false;
  w.evaluate();w.update_matrix();
  for(std::size_t k=0;k<w.kkt.size();++k)w.kkt[k].value=w.base_values[k];
  // A real, strongly regularized SPD majorant, factored by the native backend.
  // This emulates the valid factor available to the terminal controller; it
  // does not inject a failed solve or change the ORIGINAL quadratic data.
  for(int k:w.diagonal)w.kkt[k].value+=1e6;
  if(atom_borders){
    w.atom_border_begin=w.kkt.size();w.has_model_borders=true;
    for(int a=0;a<2;++a){int b=w.dim++;w.kkt.push_back({b,b,-1});w.kkt.push_back({a,b,0.5});}
  }
  std::string message;
  require(w.factor->factor(w.dim,w.nr,w.kkt,message),message.c_str());w.current_factor=w.factor.get();
  const auto x=w.x,lambda=w.lambda,z=w.z,slacks=w.s;const auto matrix=w.kkt;
  const int dim=w.dim,factors=ls.factor_attempts;
  Vec rhs(w.dim,1),before=rhs;require(w.current_factor->solve(before,message),"factor solve before rescue failed");
  auto outer=dca_original(w,w.x,w.lambda,w.z);detail::OriginalDcaFilter filter;
  DcaFilterStatistics st;DcaKrylov krylov;Direction seed;
  seed.x.assign(w.n,0);seed.lambda.assign(w.me,0);seed.s.assign(w.mi,0);seed.z.assign(w.mi,0);
  double alpha=0;
  bool ok=krylov.attempt(w,st,outer,filter,seed,seed,1e-12,0,Clock::now(),alpha,outer,true);
  std::cout<<"slack="<<slack<<" dense="<<dense<<" eliminated="<<eliminated<<" atoms="<<atom_borders
           <<" budget="<<budget<<" accepted="<<ok<<" lifted="<<st.krylov_last_lifted_rows
           <<" reason="<<st.krylov_last_rejection<<"\n";
  require(ok==success,"unexpected original KKT rescue outcome");
  require(st.krylov_last_lifted_rows==(!dense && slack<1e-4?1:0),"wrong numerical border selection");
  require(w.x==x && w.lambda==lambda && w.z==z && w.s==slacks && w.dim==dim,
          "Krylov transaction altered the public workspace");
  require(w.kkt.size()==matrix.size(),"Krylov changed preconditioner pattern");
  for(std::size_t k=0;k<matrix.size();++k)
    require(w.kkt[k].i==matrix[k].i && w.kkt[k].j==matrix[k].j && w.kkt[k].value==matrix[k].value,
            "Krylov changed preconditioner coefficients");
  Vec after=rhs;require(w.current_factor->solve(after,message) && before==after && factors==ls.factor_attempts,
                      "Krylov refactored or damaged the certified preconditioner");
  if(ok){
    require(st.krylov_max_accepted_residual<=s.kkt_backward_error_reject,
            "original expanded residual gate bypassed");
    const auto& c=krylov.proposal;
    require(c.point.primal<=s.feasibility_tolerance && c.point.kkt<=0.999*outer.kkt,
            "accepted proposal lacks original nonlinear progress");
    Vec gradient=p.objective.gradient(c.x);axpy(c.z[0],p.inequalities[0].gradient(c.x),gradient);
    for(std::size_t e=0;e<p.equalities.size();++e)axpy(c.lambda[e],p.equalities[e].gradient(c.x),gradient);
    require(inf_norm(gradient)<=0.999*outer.kkt && p.inequalities[0].value(c.x)<=s.feasibility_tolerance,
            "independent original certificate failed");
    require(c.s[0]>0 && c.z[0]>0 && c.z[1]==0,"positivity or inactive multiplier corrupted");
  }else require(st.krylov_linear_rejections==1 && st.krylov_last_rejection=="affine-original-residual",
                 "budget test did not reject the inexact affine direction");
}
}
int main(){try{
  // In double precision the condensed matrix [[1+1e16,1e16],[1e16,1+1e16]]
  // loses both unit diagonal contributions and its tangent eigenvalue.
  require(1e16+1==1e16,"fixture does not exhibit the intended double rounding");
  run(1e-16,false,false,false,24,true);
  run(1e-16,false,false,true,24,true);
  run(1e-16,false,true,false,24,true);
  run(1e-16,true,false,false,24,true);
  run(1e-3,false,false,false,24,true);
  run(1e-16,false,false,true,1,false);
  std::cout<<"PASS short-row Gram cancellation, numerical/dense borders, eliminated equality, atom layout, inactive row, residual audit, rollback and factor reuse\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
