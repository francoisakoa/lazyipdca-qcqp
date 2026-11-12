#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main() {
  int checks=0;auto check=[&](bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);};
  for(double alpha:{1.0,.5})for(bool full:{false,true}) {
    QCQPProblem p;p.name="adversarial_full_primal_soc";p.n=2;p.x0={1,0};
    p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
    p.objective.linear={0,-.3};p.objective.hessian.n=2;
    p.objective.hessian.entries={{0,0,1},{1,1,1}};
    QuadraticFunction h;h.constant=-1;h.linear={0,0};h.hessian.n=2;
    h.hessian.entries={{0,0,2},{1,1,2}};p.equalities.push_back(h);
    QuadraticFunction g;g.constant=-4;g.linear={0,0};g.hessian.n=2;
    g.hessian.entries={{1,1,2}};p.inequalities.push_back(g);
    Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
    st.linear_backend="mumps";st.linear_metis=true;st.verbose=false;st.live_display_level=0;
    st.lazy_newton_reduction=false;st.lazy_newton_atomic_regularization=false;st.lazy_newton_lazy_constraints=false;
    st.lazy_newton_objective_gradient_target=1000;st.lazy_newton_constraint_gradient_target=1000;
    LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());Workspace w(p,st,stats,costs);
    w.s={4};w.z={.25};w.lambda={0};w.evaluate();w.update_matrix();check(w.factorize(0),"factorization failed");
    check(w.active[0],"adversarial inequality was not active in barrier");
    const auto anchor_x=w.x,anchor_s=w.s,anchor_z=w.z,anchor_lambda=w.lambda;
    auto* factor=w.current_factor;const auto attempts=stats.factor_attempts;const double tau=w.regularization;
    const auto retained=w.retained;const double mu=1;
    const Direction source=w.direction({0},w.rd,w.h,w.rp,0,true,DirectionPurpose::Complete,true);
    std::cout<<"fixture alpha="<<alpha<<" full="<<full<<" dx="<<source.x[0]<<","<<source.x[1]<<" scales="<<w.objective.scale<<","<<w.inequalities[0].scale<<" tau="<<tau<<std::endl;
    check(std::abs(source.x[0])<1e-12&&std::abs(source.x[1]-.2)<1e-12,"true Newton tangent fixture changed");
    Direction working=source;Vec total(w.n,0);double last_h=INFINITY,last_rp=0;
    const DcaMajorantBarrierPath source_path(w,source);
    const auto from=DcaMajorantBarrierPath::current(w,mu),source_point=source_path.point(alpha,mu);
    DcaMajorantBarrierFilter filter(from.theta);
    check(filter.diagnose(from,source_point,source_path.slope(mu),alpha,1e-4).accepted,"source filter acceptance missing");
    for(int k=0;k<3;++k) {
      Vec x=w.x,s=w.s;axpy(alpha,working.x,x);axpy(alpha,working.s,s);
      const double hv=w.equalities[0].original_at(x),rp=w.inequalities[0].original_at(x)+s[0];
      const double eq_rhs=hv-(1-alpha)*w.equalities[0].value;
      const double primal_rhs=full?rp-(1-alpha)*(w.inequalities[0].value+w.s[0]):0;
      const auto c=w.direction({0},Vec(w.n,0),{eq_rhs},{primal_rhs},0,true,DirectionPurpose::Complete,true);
      axpy(1,c.x,total);check(inf_norm(total)<=alpha*inf_norm(source.x),"accumulated size bound failed");
      axpy(1/alpha,c.x,working.x);axpy(1/alpha,c.s,working.s);
      axpy(1/alpha,c.z,working.z);axpy(1/alpha,c.lambda,working.lambda);
      check(w.fraction(working,st.lazy_newton_fraction_to_boundary)>=alpha,"combined domain rejected");
      x=w.x;s=w.s;axpy(alpha,working.x,x);axpy(alpha,working.s,s);
      const double after=std::abs(w.equalities[0].original_at(x));last_rp=std::abs(w.inequalities[0].original_at(x)+s[0]);
      check(after<std::abs(hv)&&after<last_h,"actual residual did not contract incrementally");last_h=after;
      check(!full||last_rp<1e-12,"full primal residual was not corrected");
      const DcaMajorantBarrierPath corrected(w,working);const auto point=corrected.point(alpha,mu);
      check(point.positive&&point.finite&&point.theta<source_point.theta,"corrected original theta/domain failed");
      check(filter.diagnose(from,point,source_path.slope(mu),alpha,1e-4).accepted,"corrected original filter failed");
      check(w.current_factor==factor&&stats.factor_attempts==attempts&&w.regularization==tau&&w.retained==retained,"factor or rho changed");
      check(w.x==anchor_x&&w.s==anchor_s&&w.z==anchor_z&&w.lambda==anchor_lambda,"private probe mutated anchor");
    }
    check(last_h<1e-6,"three exact provisional corrections failed target");
    check(full?last_rp<1e-12:last_rp>1e-3,"adversarial slack defect not distinguished");
  }
  // An infeasible anchor with alpha<1 has a NONZERO affine target. This
  // rejects the tempting but incorrect iteration towards h=0 at each probe.
  {
    QCQPProblem p;p.name="nonzero_anchor_soc_target";p.n=2;p.x0={.9,0};
    p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
    p.objective.linear={0,-1.8};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,1},{1,1,1}};
    QuadraticFunction h;h.constant=-1;h.linear={0,0};h.hessian.n=2;h.hessian.entries={{0,0,2},{1,1,2}};p.equalities.push_back(h);
    QuadraticFunction g;g.constant=-4;g.linear={0,0};g.hessian.n=2;g.hessian.entries={{1,1,2}};p.inequalities.push_back(g);
    Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";st.linear_backend="mumps";st.linear_metis=true;
    st.verbose=false;st.live_display_level=0;st.lazy_newton_reduction=false;st.lazy_newton_atomic_regularization=false;st.lazy_newton_lazy_constraints=false;
    st.lazy_newton_objective_gradient_target=1000;st.lazy_newton_constraint_gradient_target=1000;
    LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());Workspace w(p,st,stats,costs);w.s={4};w.z={.25};w.lambda={0};w.evaluate();w.update_matrix();
    check(w.factorize(0),"nonzero target factorization failed");
    const auto attempts=stats.factor_attempts;auto* factor=w.current_factor;const double alpha=.5;
    Direction working=w.direction({0},w.rd,w.h,w.rp,0,true,DirectionPurpose::Complete,true);
    const double target=(1-alpha)*w.equalities[0].value;check(std::abs(target+.095)<1e-14,"nonzero affine target missing");
    double previous=INFINITY,last_actual=0;
    for(int k=0;k<3;++k) {
      Vec x=w.x,slack=w.s;axpy(alpha,working.x,x);axpy(alpha,working.s,slack);
      const double actual=w.equalities[0].original_at(x),rhs=actual-target;
      const double rp=w.inequalities[0].original_at(x)+slack[0]-(1-alpha)*(w.inequalities[0].value+w.s[0]);
      const auto c=w.direction({0},Vec(w.n,0),{rhs},{rp},0,true,DirectionPurpose::Complete,true);
      check(std::abs(1.8*c.x[0]+rhs)<1e-12,"increment sign or nonzero target incorrect");
      axpy(1/alpha,c.x,working.x);axpy(1/alpha,c.s,working.s);axpy(1/alpha,c.z,working.z);axpy(1/alpha,c.lambda,working.lambda);
      x=w.x;axpy(alpha,working.x,x);last_actual=w.equalities[0].original_at(x);
      const double error=std::abs(last_actual-target);check(error<std::abs(rhs)&&error<previous,"nonzero affine-target residual did not contract");previous=error;
      check(stats.factor_attempts==attempts&&w.current_factor==factor,"nonzero target refactored");
    }
    check(std::abs(last_actual)>.05&&std::abs(last_actual-target)<.002,"iteration incorrectly targeted zero instead of source affine residual");
  }
  std::cout<<"trial_soc_full_primal PASS checks="<<checks<<" alpha_half_and_one=1 fixed_factor=1 actual_residual=1 full_primal=1\n";
}
