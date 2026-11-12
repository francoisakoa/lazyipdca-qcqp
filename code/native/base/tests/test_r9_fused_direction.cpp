#include <cstddef>
#include "../src/atomic_newton.cpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {
void check(bool ok,const char* message) {
  if(!ok)throw std::runtime_error(message);
}
void close(double actual,long double expected,const char* message) {
  check(std::isfinite(actual)&&std::abs(static_cast<long double>(actual)-expected)<1e-10L,message);
}
Settings settings(const std::string& engine) {
  Settings st;
  st.dca_engine=engine;st.linear_backend="native";
  st.lazy_newton_reduction=true;st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_atomic_regularization=false;
  st.lazy_newton_objective_gradient_target=100;
  st.lazy_newton_constraint_gradient_target=100;
  st.kkt_backward_error_reject=1e-7;
  st.verbose=false;st.live_display_level=0;
  return st;
}
QCQPProblem problem(double pivot_value) {
  QCQPProblem p;p.name="direction_fused_complete_and_refinement";p.n=3;
  p.x0={1,0.5,pivot_value};p.lb=Vec(3,-INFINITY);p.ub=Vec(3,INFINITY);
  p.objective.linear={0,0,0};p.objective.hessian.n=3;
  p.objective.hessian.entries={{0,0,2},{1,1,4},{2,2,6}};
  QuadraticFunction pivot;pivot.name="pivot_with_nonlinear_remainder";
  pivot.linear={0,0,1};pivot.hessian.n=3;
  pivot.hessian.entries={{0,0,-2},{1,1,-2}};
  p.equalities.push_back(pivot); // x2-x0^2-x1^2 = 0
  QuadraticFunction retained;retained.name="retained_equality";
  retained.constant=-2;retained.linear={1,1,0};retained.hessian.n=3;
  p.equalities.push_back(retained); // x0+x1-2 = 0
  QuadraticFunction g;g.name="concave_native_inequality";
  g.constant=1;g.linear={0,0,0.1};g.hessian.n=3;
  g.hessian.entries={{0,0,-2}};p.inequalities.push_back(g);
  p.validate();return p;
}

// Independent dense seven-equation oracle. It does not use Workspace's
// condensation, assembled matrix, residual helper or Jacobian products.
std::vector<long double> analytic_direction(double pivot_value,double rc) {
  constexpr int n=7;
  std::vector<std::vector<long double>> a(n,std::vector<long double>(n+1,0));
  const long double x0=1,x1=0.5L,x2=pivot_value;
  const long double s=0.8L,z=0.7L,lambda0=0.3L,lambda1=-0.2L;
  const long double j0[3]={-2*x0,-2*x1,1},j1[3]={1,1,0},jg[3]={-2*x0,0,0.1L};
  const long double b[3]={2-2*lambda0,4-2*lambda0,6};
  const long double objective_gradient[3]={2*x0,4*x1,6*x2};
  for(int j=0;j<3;++j) {
    a[j][j]=b[j];a[j][3]=j0[j];a[j][4]=j1[j];a[j][6]=jg[j];
    a[j][n]=-(objective_gradient[j]+lambda0*j0[j]+lambda1*j1[j]+z*jg[j]);
    a[3][j]=j0[j];a[4][j]=j1[j];a[5][j]=jg[j];
  }
  a[3][n]=-(x2-x0*x0-x1*x1);a[4][n]=-(x0+x1-2);
  a[5][5]=1;a[5][n]=-(1-x0*x0+0.1L*x2+s);
  a[6][5]=z;a[6][6]=s;a[6][n]=-static_cast<long double>(rc);
  for(int k=0;k<n;++k) {
    int pivot=k;
    for(int i=k+1;i<n;++i)if(std::abs(a[i][k])>std::abs(a[pivot][k]))pivot=i;
    check(std::abs(a[pivot][k])>1e-12L,"independent analytic KKT is singular");
    std::swap(a[k],a[pivot]);
    const long double diagonal=a[k][k];
    for(int j=k;j<=n;++j)a[k][j]/=diagonal;
    for(int i=0;i<n;++i)if(i!=k) {
      const long double factor=a[i][k];
      for(int j=k;j<=n;++j)a[i][j]-=factor*a[k][j];
    }
  }
  std::vector<long double> result(n);
  for(int i=0;i<n;++i)result[i]=a[i][n];
  return result;
}
void compare_complete(const Direction& d,const std::vector<long double>& expected) {
  for(int j=0;j<3;++j)close(d.x.at(j),expected[j],"incorrect analytic primal direction");
  for(int e=0;e<2;++e)close(d.lambda.at(e),expected[3+e],"incorrect analytic equality multiplier");
  close(d.s.at(0),expected[5],"incorrect analytic slack direction");
  close(d.z.at(0),expected[6],"incorrect analytic inequality multiplier");
}

void complete_direction_and_fused_audit(double pivot_value) {
  auto p=problem(pivot_value);auto st=settings("majorants");
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics diagnostics;
  DcaInequalityModels models(w,diagnostics);
  w.s={0.8};w.z={0.7};w.lambda={0.3,-0.2};w.evaluate();
  models.prepare_curvature_once();models.reanchor_from_cache();w.update_matrix();
  check(w.pivot_variable[0]==2&&w.pivot_variable[1]<0&&w.mr==1,
        "fixture lacks both pivot and retained equalities");
  check(w.mi==1&&w.active[0],"fixture lacks an active native inequality");
  check(w.objective.scale==1&&w.equalities[0].scale==1&&w.inequalities[0].scale==1,
        "analytic fixture unexpectedly scaled");
  const bool nonzero_offset=pivot_value!=1.25;
  check((w.h[0]!=0)==nonzero_offset,"fixture has the wrong offset condition");
  check(w.factorize(0)&&w.regularization==0,"analytic auxiliary KKT failed to factor");
  const Vec rc={w.s[0]*w.z[0]};
  const auto expected=analytic_direction(pivot_value,rc[0]);
  const auto x=w.x,s=w.s,z=w.z,lambda=w.lambda,h=w.h,rp=w.rp,rd=w.rd;
  const auto initial_products=ns.lagrangian_hessian_products;
  const auto full=w.direction(rc,w.rd,w.h,w.rp,0,false);
  compare_complete(full,expected);
  const auto full_products=ns.lagrangian_hessian_products-initial_products;
  check(full_products==(nonzero_offset?2:1),
        "complete raw direction missed the exact-zero offset shortcut");
  check(w.x==x&&w.s==s&&w.z==z&&w.lambda==lambda&&w.h==h&&w.rp==rp&&w.rd==rd,
        "complete direction changed the original state or residual cache");

  const auto audits=ns.direction_residual_evaluations;
  const auto refinements=ns.direction_corrections;
  const auto validated_products=ns.lagrangian_hessian_products;
  const auto complete=w.direction(rc,w.rd,w.h,w.rp);
  compare_complete(complete,expected);
  check(complete.residual<=st.kkt_backward_error_reject&&ns.direction_corrections==refinements,
        "analytic direction failed the unchanged expanded tolerance");
  check(ns.direction_residual_evaluations==audits+1,
        "majorant direction still performed duplicate double/extended audits");
  check(ns.lagrangian_hessian_products-validated_products==(nonzero_offset?2:1),
        "validated direction duplicated the Hessian product used to reconstruct multipliers");

  auto wrong=complete;wrong.z[0]+=1e-3;
  Vec ed(w.n),eh(w.me),ep(w.mi),ec(w.mi);double de=0,ee=0,pe=0,ce=0;
  const double error=w.majorant_direction_residual_extended(wrong,rc,w.rd,w.h,w.rp,
      ed,eh,ep,ec,de,ee,pe,ce);
  close(ce,0.0008L,"extended audit hid a real complementarity residual");
  check(error/std::max(1.0,inf_norm(w.rd))>st.kkt_backward_error_reject,
        "fused audit relaxed the configured Newton residual threshold");

  // Exercise exactly one internal numerical correction of all four blocks.
  // This is a solve of the residual with the SAME factorized matrix, followed
  // by one full audit. The independent dense oracle supplies the final answer.
  auto refined=complete;
  refined.x[0]+=0.02;refined.lambda[1]+=0.03;
  refined.s[0]+=0.04;refined.z[0]-=0.05;
  const auto correction_audits=ns.direction_residual_evaluations;
  const double before=w.majorant_direction_residual_extended(refined,rc,w.rd,w.h,w.rp,
      ed,eh,ep,ec,de,ee,pe,ce);
  check(before>1e-3,"refinement fixture did not have a material expanded residual");
  const auto correction_products=ns.lagrangian_hessian_products;
  const auto correction_count=ns.direction_corrections;
  const auto correction=w.direction(ec,ed,eh,ep,1,true,DirectionPurpose::ResidualCorrection);
  check(correction.lambda.at(0)==0,
        "internal correction duplicated the parent pivot-multiplier reconstruction");
  check(ns.direction_residual_evaluations==correction_audits+1 &&
        ns.direction_corrections==correction_count+1,
        "internal correction performed an unnecessary expanded audit");
  check(ns.lagrangian_hessian_products-correction_products==(eh[0]!=0?1:0),
        "internal correction retained a redundant Hessian product");
  axpy(1.0,correction.x,refined.x);axpy(1.0,correction.lambda,refined.lambda);
  axpy(1.0,correction.s,refined.s);axpy(1.0,correction.z,refined.z);
  const double after=w.majorant_direction_residual_extended(refined,rc,w.rd,w.h,w.rp,
      ed,eh,ep,ec,de,ee,pe,ce);
  check(after/std::max(1.0,inf_norm(w.rd))<=st.kkt_backward_error_reject,
        "corrected full direction failed the unchanged expanded tolerance");
  compare_complete(refined,expected);
  check(ns.direction_residual_evaluations==correction_audits+2,
        "refinement required more than its before/after full audits");
  check(w.x==x&&w.s==s&&w.z==z&&w.lambda==lambda&&w.h==h&&w.rp==rp&&w.rd==rd,
        "fused reconstruction or refinement changed the original state or cache");

  // A raw Complete call at depth>0 is used by existing reconstruction tests.
  // It must retain full multipliers despite the new internal partial mode.
  compare_complete(w.direction(rc,w.rd,w.h,w.rp,1),expected);
}

void majorant_cholesk_weak_primal_rescue() {
  auto st=settings("majorants");st.linear_backend="cholesk";
  st.cholesk_equilibration=false;st.lazy_newton_reduction=false;
  QCQPProblem p;p.name="weak_primal_rescue";p.n=2;p.x0={1,1};
  p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.linear={0,0};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,5e-16},{1,1,2}};p.validate();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);w.update_matrix();
  check(w.factorize(0),"majorant weak primal rescue failed");
  check(w.factor->last_regularization_attempts()==2,
        "weak primal must recover in one retry");
  check(w.factor->last_primal_regularization()>0 &&
        w.factor->last_primal_regularization()<=1e-8 &&
        w.factor->last_dual_regularization()==0 && w.regularization==0,
        "numerical rescue changed wrong block or outer rho");
  auto d=w.direction({},w.rd,w.h,w.rp);
  check(d.residual<=st.kkt_backward_error_reject,"rescued direction failed audit");
  // A shifted solution can have a tiny shifted residual while violating the
  // actual auxiliary equations. Verify the audit does not subtract it away.
  d.x={1e6,0};Vec dual={-(5e-16+w.factor->last_primal_regularization())*1e6,0};
  Vec ed(2),eh,ep,ec;double de,ee,pe,ce;
  const double error=w.majorant_direction_residual_extended(
      d,{},dual,{}, {},ed,eh,ep,ec,de,ee,pe,ce);
  check(error>5e-7,"audit hid the numerical primal shift");
  // A subsequent healthy matrix must return to an unshifted factorization.
  p.objective.hessian.entries[0].v=1;
  LazyNewtonStatistics clean_ns;detail::NewtonCostLedger clean_cost(clean_ns,Clock::now());
  Workspace clean(p,st,clean_ns,clean_cost);clean.update_matrix();
  check(clean.factorize(0)&&clean.factor->last_regularization_attempts()==1&&
        clean.factor->last_primal_regularization()==0,
        "healthy majorant KKT was unnecessarily regularized");
}

void objective_lazy_contract() {
  auto run=[](int n,const std::vector<SymTriplet>& h,bool corrected) {
    QCQPProblem p;p.n=n;p.x0=Vec(n,0.2);p.lb=Vec(n,-2);p.ub=Vec(n,2);
    p.objective.linear=Vec(n,0);p.objective.hessian.n=n;p.objective.hessian.entries=h;p.validate();
    auto st=settings("majorants");st.linear_backend="cholesk";
    st.objective_lazy_convexification=true;st.lazy_newton_reduction=false;
    LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
    Workspace w(p,st,ns,costs);DcaFilterStatistics ds;DcaInequalityModels m(w,ds);
    const double original=w.objective.value;
    m.prepare_curvature_once();m.reanchor_from_cache();
    if((m.objective_rho>0)!=corrected) {
      std::cerr<<"objective fixture n="<<n<<" nnz="<<h.size()<<" expected="<<corrected<<" rho="<<m.objective_rho<<" floor="<<m.objective_floor<<"\n";
      throw std::runtime_error("objective activation misclassified convexity");
    }
    check(w.objective.model_active==corrected,"wrong objective model activation");
    close(w.objective.at(w.x),original,"objective lost tangency");
    Vec trial=w.x;trial[0]+=0.3;
    auto correction00=[&]() {
      double c=0;
      const auto it=m.objective_diagonal_atoms.find(0);
      if(it!=m.objective_diagonal_atoms.end())c+=it->second;
      for(const auto& a:w.objective.model_atoms)c+=a.beta*a.direction[0]*a.direction[0];
      return c;
    };
    const double expected=(m.objective_rho+correction00())*0.3*0.3/2;
    close(w.objective.at(trial)-w.objective.original_at(trial),expected,"objective is not the claimed majorant");
    Direction d;d.x=Vec(n,0);d.x[0]=0.3;d.s=Vec(w.mi,0);d.z=Vec(w.mi,0);d.lambda=Vec(w.me,0);
    const DcaMajorantBarrierPath path(w,d);
    close(double(path.objective.at(1)),w.objective.original_at(trial),"filter used convexified objective");
    const double initial=m.objective_excess;
    for(int i=0;i<60;++i)m.objective_accepted_step();
    check(m.objective_floor==0&&m.objective_rho<=1e-15*std::max(1.0,initial),"rho did not tend to zero after atoms covered curvature");
    if(corrected)check(initial>0&&m.objective_admitted>0,"nonconvex objective lacks persistent atoms");
    m.discard();check(!w.objective.model_active,"original objective not restored");
    m.reanchor_from_cache();w.update_matrix();
    if(corrected) {
      Vec e(n,0);e[0]=1;
      auto hv=w.hessian_product(e);
      double original_h00=0;for(auto a:h)if(a.i==0&&a.j==0)original_h00+=a.v*w.objective.scale;
      close(hv[0],original_h00+m.objective_rho+correction00(),"auxiliary Hessian omitted objective correction");
    }
  };
  run(2,{},false); // affine
  run(2,{{0,0,1},{0,1,2},{1,1,4}},false); // PSD, not diagonally dominant
  run(2,{{0,0,-2},{1,1,1}},true);
  run(2,{{0,1,1}},true); // absent diagonal slots
  std::vector<SymTriplet> large={{0,0,1},{0,1,2},{1,1,4}};
  for(int i=2;i<257;++i)large.push_back({i,i,1});
  run(257,large,false); // sparse PSD certificate, not a dense n-by-n spectrum
  large[0].v=-1;run(257,large,true);
  QCQPProblem p;p.n=1;p.x0={0.2};p.lb={-1};p.ub={1};
  p.objective.linear={0};p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,-2}};
  auto st=settings("majorants");st.objective_lazy_convexification=true;
  st.max_outer_iterations=300;st.feasibility_tolerance=1e-6;
  st.stationarity_tolerance=st.dual_infeasibility_tolerance=st.complementarity_tolerance=1e-6;
  const auto result=solve_inequality_majorants(p,st);
  check(result.status==Status::Solved,"nonconvex objective did not converge to original KKT");
  check(std::abs(result.x[0])>0.999,"nonconvex fixture converged to wrong stationary point");

}

void conditional_restoration_permission_and_rollback() {
  auto st=settings("majorants");st.majorant_initial_restoration=true;
  st.majorant_restoration_iterations=2;st.max_outer_iterations=1;
  QCQPProblem p;p.n=2;p.x0={2,0};p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.linear={0,0};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,2},{1,1,2}};
  QuadraticFunction h;h.constant=-1;h.linear={1,0};h.hessian.n=2;p.equalities={h};
  QuadraticFunction g;g.constant=-3;g.linear={0,1};g.hessian.n=2;p.inequalities={g};
  auto result=solve_inequality_majorants(p,st);
  check(result.restoration_iterations==0,"restoration permission caused a systematic startup");
  // With one trial permitted, the Newton point for x^2=1 from x=.1
  // violates the filter cap. The feasibility retry also fails at full step:
  // all primal-dual state must roll back to the original accepted point.
  p.n=1;p.x0={0.1};p.lb={-INFINITY};p.ub={INFINITY};
  p.objective.linear={0};p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,2}};
  h.linear={0};h.hessian.n=1;h.hessian.entries={{0,0,2}};p.equalities={h};
  g.constant=-10;g.linear={0};g.hessian.n=1;p.inequalities={g};
  st.line_search_max_steps=1;
  result=solve_inequality_majorants(p,st);
  check(result.restoration_iterations==1,"blocked filter did not trigger bounded restoration");
  check(result.status==Status::LineSearchFailure,"unsuccessful restoration hid filter failure");
  close(result.x[0],0.1L,"unsuccessful restoration failed to roll back x");
  // A large tangential objective step is blocked by the equality defect;
  // the objective-free recovery improves the normal component instead.
  p.n=2;p.x0={0.8,0};p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.linear={0,100};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,1},{1,1,1}};
  h.linear={0,0};h.hessian.n=2;h.hessian.entries={{0,0,2},{1,1,2}};p.equalities={h};
  g.linear={0,0};g.hessian.n=2;p.inequalities={g};
  result=solve_inequality_majorants(p,st);
  check(result.restoration_iterations==2,"successful recovery did not respect its budget");
  check(result.restoration_final_feasibility<0.01,"conditional recovery failed to reduce feasibility");
  close(result.x[1],0,"feasibility recovery followed the rejected tangential step");
}

void equality_soc_reuses_factor_and_transports_duals() {
  auto p=problem(1.25);auto st=settings("majorants");
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics diagnostics;
  DcaInequalityModels models(w,diagnostics);
  w.s={0.8};w.z={0.7};w.lambda={0.3,-0.2};w.evaluate();
  models.prepare_curvature_once();models.reanchor_from_cache();w.update_matrix();
  check(w.factorize(0),"SOC fixture factorization failed");
  auto d=w.direction(Vec{0.1},w.rd,w.h,w.rp);
  DcaMajorantBarrierPath path(w,d);
  constexpr double alpha=1e-3;
  Vec defect(w.me,0);
  for(int e=0;e<w.me;++e)defect[e]=double(alpha*alpha*path.equalities[e].quadratic);
  const auto factors=ns.factor_attempts;
  auto c=w.direction(Vec(w.mi,0),Vec(w.n,0),defect,Vec(w.mi,0));
  check(ns.factor_attempts==factors,"SOC rebuilt factors");
  for(int e=0;e<w.me;++e)close(w.equalities[e].directional(c.x),-defect[e],"SOC equality defect not cancelled");
  for(int i=0;i<w.mi;++i) {
    close(w.inequalities[i].directional(c.x)+c.s[i],0,"SOC slack linearization inconsistent");
    close(w.s[i]*c.z[i]+w.z[i]*c.s[i],0,"SOC complementarity transport inconsistent");
  }
}

void equality_only_flat_chain_start() {
  constexpr int beams=200;
  QCQPProblem p;p.name="flat_equality_chain";p.n=3*(beams+1);
  p.x0=Vec(p.n,0);p.lb=Vec(p.n,-INFINITY);p.ub=Vec(p.n,INFINITY);
  p.objective.linear=Vec(p.n,0);p.objective.hessian.n=p.n;
  const double gravity=500.0*9.81/beams;
  for(int i=0;i<=beams;++i)p.objective.linear[3*i+1]=(i==0||i==beams)?gravity/2:gravity;
  for(int c=0;c<3;++c)p.lb[c]=p.ub[c]=0;
  p.lb[3*beams]=p.ub[3*beams]=0.6*beams;
  for(int i=1;i<=beams;++i) {
    p.x0[3*i]=0.6;
    QuadraticFunction h;h.constant=-1;h.linear=Vec(p.n,0);h.hessian.n=p.n;
    for(auto pair: {std::pair<int,int>{3*i,3*(i-1)},
                   std::pair<int,int>{3*i+1,3*(i-1)+1},
                   std::pair<int,int>{3*i,3*(i-1)+2}}) {
      h.hessian.entries.push_back({pair.first,pair.first,2});
      h.hessian.entries.push_back({pair.second,pair.second,2});
      h.hessian.entries.push_back({std::min(pair.first,pair.second),std::max(pair.first,pair.second),-2});
    }
    p.equalities.push_back(std::move(h));
  }
  const Vec frozen=p.x0;
  auto st=settings("majorants");st.max_outer_iterations=300;
  st.stationarity_tolerance=st.feasibility_tolerance=1e-6;
  st.dual_infeasibility_tolerance=st.complementarity_tolerance=1e-6;
  st.objective_lazy_convexification=true;
  const auto result=solve_inequality_majorants(p,st);
  check(result.status==Status::Solved,"flat equality chain did not recover original KKT");
  check(result.restoration_tangent_activation_acceptances==1,"flat chain omitted tangent activation");
  check(p.x0==frozen,"restoration modified the supplied initial point");
}

void dual_minimax_contract() {
  close(dca_dual_minimax_step({1,0},{0,2}),1.0L/3,"incorrect minimax kink");
  close(dca_dual_minimax_step({2,-2},{1,-1}),1,"incorrect full minimax step");
  close(dca_dual_minimax_step({1,-1},{2,-2}),0,"incorrect zero minimax step");
  for(int seed=1;seed<30;++seed) {
    Vec a(7),b(7);
    for(int j=0;j<7;++j){a[j]=std::sin(seed*(j+1))*10;b[j]=std::cos((seed+2)*(j+1))*7;}
    const double alpha=dca_dual_minimax_step(a,b);
    auto merit=[&](double t){double f=0;for(int j=0;j<7;++j)f=std::max(f,std::abs(a[j]+t*(b[j]-a[j])));return f;};
    check(alpha>=0&&alpha<=1,"dual alpha outside segment");
    for(int k=0;k<=1000;++k)check(merit(alpha)<=merit(k/1000.0)+1e-10,"minimax failed independent mesh oracle");
  }
}

void lambda_interpolation_contract() {
  QCQPProblem p;p.name="partial_dual_fit";p.n=2;p.x0={0,0};
  p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.linear={0.1,1};p.objective.hessian.n=2;
  QuadraticFunction e;e.linear={1,2};e.hessian.n=2;p.equalities.push_back(e);
  auto st=settings("majorants");st.lazy_newton_reduction=false;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);w.lambda={0};w.evaluate();
  auto metric=dca_outer_metrics(w);DcaFilterStatistics stats;
  check(dca_majorant_recalc_lambda(w,metric,stats,40,Clock::now(),10,true),"partial dual correction rejected");
  close(w.lambda[0],-11.0L/30,"partial multiplier is not minimax");
  close(metric.dual,4.0L/15,"partial stationarity is not minimax");
  check(stats.majorant_lambda_last_alpha>0&&stats.majorant_lambda_last_alpha<1,"partial alpha not used");
  check(w.x==p.x0&&ns.factor_attempts==0,"interpolation moved primal or factored");
}

void lambda_projection_contract() {
  QCQPProblem p;p.name="lambda_projection";p.n=2;p.x0={0,0};
  p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.linear={3,4};p.objective.hessian.n=2;
  QuadraticFunction e;e.linear={1,0};e.hessian.n=2;p.equalities.push_back(e);
  e.linear={0,2};p.equalities.push_back(e);
  auto st=settings("majorants");st.lazy_newton_reduction=false;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  w.lambda={10,10};w.evaluate();auto current=dca_outer_metrics(w);
  const auto x=w.x,s=w.s,z=w.z;
  check(dca_majorant_recalc_lambda(w,current,stats,40,Clock::now(),10),"lambda projection did not accept analytic fit");
  check(current.dual<1e-7&&std::abs(w.lambda[0]+3)<1e-7&&std::abs(w.lambda[1]+2)<1e-7,"incorrect least-squares equality fit");
  check(w.x==x&&w.s==s&&w.z==z&&ns.factor_attempts==0,"lambda fit changed primal/slack/z or factorized");
  // An uncorrectable stationarity component must reject transactionally.
  // Use a separate problem with only the first row, leaving residual 4 in x1.
  p.equalities.resize(1);Workspace v(p,st,ns,costs);v.lambda={-3};v.evaluate();
  auto metric=dca_outer_metrics(v);const auto old_lambda=v.lambda,old_rd=v.rd;
  check(!dca_majorant_recalc_lambda(v,metric,stats,40,Clock::now(),10),"uncorrectable dual accepted");
  check(v.lambda==old_lambda&&v.rd==old_rd,"rejected lambda fit changed state");
  // Same projection with duplicate rows: regularization handles rank deficiency.
  p.equalities.push_back(p.equalities[0]);Workspace rank(p,st,ns,costs);rank.lambda={10,10};rank.evaluate();
  auto rank_metric=dca_outer_metrics(rank);
  check(dca_majorant_recalc_lambda(rank,rank_metric,stats,40,Clock::now(),10),"rank-deficient fit failed");
  close(rank_metric.dual,4,"rank-deficient fit damaged orthogonal stationarity");
}

void legacy_direction_keeps_complete_multiplier_contract() {
  auto p=problem(2);auto st=settings("filter");
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics diagnostics;
  DcaInequalityModels models(w,diagnostics);
  w.s={0.8};w.z={0.7};w.lambda={0.3,-0.2};w.evaluate();
  models.prepare_curvature_once();models.reanchor_from_cache();w.update_matrix();
  check(w.factorize(0)&&w.regularization==0,"legacy analytic KKT failed to factor");
  const Vec rc={w.s[0]*w.z[0]};const auto expected=analytic_direction(2,rc[0]);
  compare_complete(w.direction(rc,w.rd,w.h,w.rp,0,false),expected);
  compare_complete(w.direction(rc,w.rd,w.h,w.rp),expected);
}
}

int main() {
  try {
    lambda_interpolation_contract();
    dual_minimax_contract();
    lambda_projection_contract();
    conditional_restoration_permission_and_rollback();
    equality_soc_reuses_factor_and_transports_duals();
    equality_only_flat_chain_start();
    objective_lazy_contract();
    majorant_cholesk_weak_primal_rescue();
    complete_direction_and_fused_audit(2); // nonzero nonlinear-pivot residual
    complete_direction_and_fused_audit(1.25); // exactly zero pivot offset
    legacy_direction_keeps_complete_multiplier_contract();
    std::cout<<"PASS fused direction: independent full/refined auxiliary KKT, one extended audit, strict tolerance and legacy full multipliers\n";
  } catch(const std::exception& e) {
    std::cerr<<"FAIL fused direction: "<<e.what()<<'\n';return 1;
  }
}
