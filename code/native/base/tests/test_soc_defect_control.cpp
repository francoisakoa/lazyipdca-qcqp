#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;

int main() {
  int checks=0;
  auto check=[&](bool condition,const char* why){++checks;if(!condition)throw std::runtime_error(why);};
  auto close=[&](long double actual,long double expected,long double tolerance,const char* why) {
    check(std::isfinite(actual)&&std::abs(actual-expected)<=tolerance,why);
  };
  check(!Settings{}.majorant_soc_defect_control,"disabled R10 default changed");
  Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
  st.linear_backend="mumps";st.linear_metis=true;st.verbose=false;st.live_display_level=0;
  st.lazy_newton_reduction=false;st.lazy_newton_atomic_regularization=false;st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_objective_gradient_target=1000;st.lazy_newton_constraint_gradient_target=1000;
  auto sphere=[](int n) {
    QCQPProblem p;p.name="coefficient_levelset_fixture";p.n=n;p.x0.assign(n,0);p.x0[0]=1;
    p.lb.assign(n,-INFINITY);p.ub.assign(n,INFINITY);
    p.objective.linear.assign(n,0);p.objective.hessian.n=n;
    p.objective.hessian.entries={{0,0,1},{1,1,1}};
    QuadraticFunction h;h.constant=-1;h.linear.assign(n,0);h.hessian.n=n;
    h.hessian.entries={{0,0,2},{1,1,2}};p.equalities.push_back(h);
    return p;
  };
  auto direction=[](const Workspace& w) {
    Direction d;d.x.assign(w.n,0);d.s.assign(w.mi,0);d.z.assign(w.mi,0);d.lambda.assign(w.me,0);return d;
  };
  auto inspect=[&](QCQPProblem p) {
    LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());
    Workspace w(p,st,stats,costs);detail::SocDefectGeometry geometry;
    return geometry.inspect(w);
  };

  // A genuine MUMPS KKT generates a unit tangent. Three accepted SOCs
  // improve the source while leaving h=.09283447. The coefficient-levelset
  // proposal instead repairs the original SOURCE at alpha=1 directly.
  auto p=sphere(2);p.objective.linear[1]=-1.000000005;
  QuadraticFunction g;g.constant=-4;g.linear={0,0};g.hessian.n=2;
  g.hessian.entries={{1,1,2}};p.inequalities.push_back(g);
  LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());
  Workspace w(p,st,stats,costs);w.s={4};w.z={2.5e-9};w.lambda={0};w.evaluate();w.update_matrix();
  check(w.factorize(0),"genuine KKT factorization failed");
  const auto anchor_x=w.x,anchor_s=w.s,anchor_z=w.z,anchor_lambda=w.lambda;
  const auto attempts=stats.factor_attempts;auto* factor=w.current_factor;
  const double tau=w.regularization;const auto retained=w.retained;
  const Direction source=w.direction({0},w.rd,w.h,w.rp,0,true,DirectionPurpose::Complete,true);
  close(source.x[0],0,1e-12L,"source not tangent to original circle");
  close(source.x[1],1,1e-12L,"fixture did not generate unit tangent");
  const DcaMajorantBarrierPath source_path(w,source);
  const auto from=DcaMajorantBarrierPath::current(w,1e-8);
  DcaMajorantBarrierFilter filter(from.theta);
  const auto source_point=source_path.rounded_point(1,1e-8);
  check(filter.diagnose(from,source_point,source_path.slope(1e-8),1,st.line_search_armijo).accepted,"source not accepted by original filter");
  Direction working=source;
  double previous_theta=source_point.theta;
  for(int correction=0;correction<3;++correction) {
    Vec x=w.x,s=w.s;axpy(1,working.x,x);axpy(1,working.s,s);
    const double eq=w.equalities[0].original_at(x),gp=w.inequalities[0].original_at(x)+s[0];
    const Direction increment=w.direction({0},Vec(w.n,0),{eq},{gp},0,true,DirectionPurpose::Complete,true);
    axpy(1,increment.x,working.x);axpy(1,increment.s,working.s);
    axpy(1,increment.z,working.z);axpy(1,increment.lambda,working.lambda);
    const DcaMajorantBarrierPath corrected(w,working);
    const auto point=corrected.rounded_point(1,1e-8);
    check(w.fraction(working,st.lazy_newton_fraction_to_boundary)>=1,"fixture SOC failed domain guard");
    check(point.finite&&point.positive&&point.theta<previous_theta,"fixture SOC failed to improve source");
    check(filter.diagnose(from,point,source_path.slope(1e-8),1,st.line_search_armijo).accepted,"fixture SOC not filter accepted");
    check(detail::soc_defect_trial_residuals(w,working,1).equality>st.feasibility_tolerance,"fixture SOC unexpectedly recovered original equality");
    previous_theta=point.theta;
  }
  close(detail::soc_defect_trial_residuals(w,working,1).equality,.09283447265625L,1e-12L,"three-SOC residual identity changed");
  const auto solves=stats.linear_solves;
  detail::SocDefectGeometry geometry;
  const bool circle_supported=geometry.inspect(w);
  if(!circle_supported) {
    std::cerr<<"[SOC-DEFECT-UNIT-GEOMETRY] reason="<<geometry.reason
        <<" n="<<w.n<<" me="<<w.me<<" mi="<<w.mi<<'\n';
    for(int e=0;e<w.me;++e) {
      const auto& q=w.equalities[e];
      std::cerr<<" equality="<<e<<" constant="<<q.constant<<" scale="<<q.scale
          <<" signed_square="<<q.signed_square.size()<<" H=";
      for(const auto& t:q.hessian)std::cerr<<'('<<t.i<<','<<t.j<<','<<t.v<<')';
      std::cerr<<" linear=";
      for(const auto& t:q.linear)std::cerr<<'('<<t.first<<','<<t.second<<')';
      std::cerr<<'\n';
    }
  }
  check(circle_supported&&geometry.supported,"circle coefficient geometry not supported");
  check(geometry.blocks.size()==1&&geometry.leaves.empty(),"circle structural metadata incorrect");
  Direction combined;
  const auto projection=geometry.project(w,source,source_path,1,combined);
  check(projection.valid,"valid source circle retraction rejected");
  check(projection.block_count==1&&projection.leaf_count==0,"proposal geometry counts incorrect");
  const auto repaired=detail::soc_defect_trial_residuals(w,combined,1);
  check(repaired.valid&&repaired.equality<1e-12&&repaired.slack<1e-12,"source-levelset geometry failed original primal equations");
  const DcaMajorantBarrierPath repaired_path(w,combined);
  const auto repaired_point=repaired_path.rounded_point(1,1e-8);
  check(repaired_point.finite&&repaired_point.positive&&repaired_point.theta<source_point.theta,"retraction failed actual original theta/domain");
  check(filter.diagnose(from,repaired_point,source_path.slope(1e-8),1,st.line_search_armijo).accepted,"retraction failed unchanged original filter");
  check(w.fraction(combined,st.lazy_newton_fraction_to_boundary)>=1,"retraction failed actual fraction-to-boundary");
  check(detail::trial_soc_arc_finite_update(w,combined,1),"retraction nonfinite in one primal-dual block");
  check(projection.correction_inf<=inf_norm(source.x),"circle retraction exceeds original SOC correction bound");
  check(combined.z==source.z&&combined.lambda==source.lambda,"retraction altered pure source dual direction");
  check(stats.linear_solves==solves,"geometric proposal used an extra linear solve");
  check(stats.factor_attempts==attempts&&w.current_factor==factor&&w.regularization==tau&&w.retained==retained,"geometric proposal changed frozen factor or model");
  check(w.x==anchor_x&&w.s==anchor_s&&w.z==anchor_z&&w.lambda==anchor_lambda,"geometric proposal mutated original accepted anchor");
  check(filter.size()==0,"private retraction changed filter history");
  // A later geometric line-search trial may already be below the requested
  // feasibility tolerance. It still improves the originally admitted source
  // and must not need a fresh per-alpha nonlinear-loss admission.
  {
    const double alpha=1e-4;
    detail::NonlinearTrialSocEvidence tiny_source(st.feasibility_tolerance);
    for(int e=0;e<w.me;++e) {
      const auto& q=source_path.equalities[e];
      tiny_source.add_row(q.constant,q.constant+alpha*q.linear,q.at(alpha),
          q.nonlinear_change(alpha),w.equalities[e].scale);
    }
    check(!tiny_source.eligible(),"small-alpha source unexpectedly exceeds admission tolerance");
    Direction tiny;check(geometry.project(w,source,source_path,alpha,tiny).valid,"small-alpha geometric improvement incorrectly requires renewed source admission");
    const auto before=source_path.rounded_point(alpha,1e-8);
    const DcaMajorantBarrierPath tiny_path(w,tiny);
    const auto after=tiny_path.rounded_point(alpha,1e-8);
    check(after.finite&&after.positive&&after.theta<before.theta,"small-alpha realized projection did not improve original theta");
    check(detail::soc_defect_trial_residuals(w,tiny,alpha).equality<tiny_source.trial_inf,"small-alpha geometry failed actual equality improvement");
    check(filter.diagnose(from,after,source_path.slope(1e-8),alpha,st.line_search_armijo).accepted,"small-alpha geometric point failed original filter");
    check(w.fraction(tiny,st.lazy_newton_fraction_to_boundary)>=alpha,"small-alpha geometry failed fraction boundary");
    check(stats.linear_solves==solves&&stats.factor_attempts==attempts,"small-alpha geometric trial used an extra solve or factorization");
  }
  for(double alpha:{0.0,-1.0,1.01,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    Direction ignored;check(!geometry.project(w,source,source_path,alpha,ignored).valid,"invalid alpha admitted by private geometry");
  }
  auto broken=source;broken.lambda[0]=INFINITY;
  check(!geometry.project(w,broken,source_path,.5,combined).valid,"nonfinite equality multiplier update admitted");
  broken=source;broken.x[0]=std::numeric_limits<double>::quiet_NaN();
  check(!geometry.project(w,broken,source_path,.5,combined).valid,"nonfinite source coordinate admitted");
  auto inward=source;inward.x={-1,0};
  const DcaMajorantBarrierPath inward_path(w,inward);
  check(!geometry.project(w,inward,inward_path,.75,combined).valid,"negative affine-target radius admitted");
  check(!geometry.project(w,inward,inward_path,1,combined).valid,"source at radial center admitted");

  // A second genuine KKT produces the large tangent from the recorded
  // refusal case. Its FIRST frozen SOC exceeds the correction bound and
  // leaves the slack domain. Geometry uses the source directly, so no
  // selected SOC or successful increment is a prerequisite for recovery.
  {
    auto refused=p;refused.objective.linear[1]=-3.15;
    LazyNewtonStatistics local_stats;detail::NewtonCostLedger local_costs(local_stats,Clock::now());
    Workspace rw(refused,st,local_stats,local_costs);
    rw.s={4};rw.z={.25};rw.lambda={0};rw.evaluate();rw.update_matrix();
    check(rw.factorize(0),"refused-SOC fixture KKT factorization failed");
    const auto saved_x=rw.x,saved_s=rw.s,saved_z=rw.z,saved_lambda=rw.lambda;
    const auto factor_attempts=local_stats.factor_attempts;auto* frozen_factor=rw.current_factor;
    const auto frozen_retained=rw.retained;const double frozen_tau=rw.regularization;
    const Direction raw=rw.direction({0},rw.rd,rw.h,rw.rp,0,true,DirectionPurpose::Complete,true);
    close(raw.x[0],0,1e-12L,"refusal source lost circle tangency");
    close(raw.x[1],2.1L,1e-12L,"refusal fixture did not produce large tangent");
    const DcaMajorantBarrierPath raw_path(rw,raw);
    const auto origin=DcaMajorantBarrierPath::current(rw,1);
    DcaMajorantBarrierFilter raw_filter(origin.theta);
    const auto raw_point=raw_path.rounded_point(1,1);
    check(raw_filter.diagnose(origin,raw_point,raw_path.slope(1),1,st.line_search_armijo).accepted,"large source fixture not filter accepted");
    Vec x=rw.x,s=rw.s;axpy(1,raw.x,x);axpy(1,raw.s,s);
    const double eq=rw.equalities[0].original_at(x),gp=rw.inequalities[0].original_at(x)+s[0];
    const Direction increment=rw.direction({0},Vec(rw.n,0),{eq},{gp},0,true,DirectionPurpose::Complete,true);
    check(inf_norm(increment.x)>inf_norm(raw.x),"first SOC did not exceed source correction-size bound");
    Direction full=raw;axpy(1,increment.x,full.x);axpy(1,increment.s,full.s);
    axpy(1,increment.z,full.z);axpy(1,increment.lambda,full.lambda);
    check(rw.fraction(full,st.lazy_newton_fraction_to_boundary)<1,"first SOC did not fail fraction-to-boundary guard");
    check(rw.s[0]+full.s[0]<0,"first SOC did not leave the positive slack domain");
    const auto linear_solves=local_stats.linear_solves;
    detail::SocDefectGeometry geom;check(geom.inspect(rw),"refused-SOC source geometry not supported");
    Direction trial;const auto out=geom.project(rw,raw,raw_path,1,trial);
    check(out.valid,"source geometry incorrectly requires a selected SOC");
    const auto residual=detail::soc_defect_trial_residuals(rw,trial,1);
    check(residual.valid&&residual.equality<1e-12&&residual.slack<1e-12,"refused-SOC source geometry failed original primal equations");
    check(out.correction_inf<inf_norm(raw.x),"refused-SOC geometry exceeds original correction bound");
    check(rw.fraction(trial,st.lazy_newton_fraction_to_boundary)>=1,"refused-SOC source geometry failed fraction boundary");
    const DcaMajorantBarrierPath repaired_path(rw,trial);
    const auto point=repaired_path.rounded_point(1,1);
    check(point.finite&&point.positive&&point.theta<raw_point.theta,"refused-SOC source geometry failed actual theta/domain improvement");
    check(raw_filter.diagnose(origin,point,raw_path.slope(1),1,st.line_search_armijo).accepted,"refused-SOC geometric proposal failed original source-slope filter");
    check(trial.z==raw.z&&trial.lambda==raw.lambda,"refused-SOC geometry did not preserve source dual update");
    check(local_stats.linear_solves==linear_solves,"refused-SOC geometric recovery used another solve");
    check(local_stats.factor_attempts==factor_attempts&&rw.current_factor==frozen_factor&&
        rw.retained==frozen_retained&&rw.regularization==frozen_tau,"refused-SOC geometric recovery changed factor or model");
    check(rw.x==saved_x&&rw.s==saved_s&&rw.z==saved_z&&rw.lambda==saved_lambda,"refused-SOC geometric recovery changed accepted anchor");
    check(raw_filter.size()==0,"refused-SOC geometric recovery changed original filter history");
  }

  // The affine leaf is transported after the nonlinear block. All lifted
  // coordinates and the nonlinear inequality/slack equation must be tested.
  auto lift=sphere(3);lift.x0[2]=2;lift.objective.linear[1]=-1.5;
  QuadraticFunction affine;affine.constant=0;affine.linear={-2,-3,1};affine.hessian.n=3;
  lift.equalities.push_back(affine);
  QuadraticFunction lifted_g;lifted_g.constant=-16;lifted_g.linear={0,0,0};lifted_g.hessian.n=3;
  lifted_g.hessian.entries={{2,2,2}};lift.inequalities.push_back(lifted_g);
  {
    LazyNewtonStatistics local_stats;detail::NewtonCostLedger local_costs(local_stats,Clock::now());
    Workspace lw(lift,st,local_stats,local_costs);lw.s={12};lw.z={1e-8};lw.lambda={0,0};lw.evaluate();
    auto d=direction(lw);d.x={0,1,3};d.s={-12};d.z={.0001};d.lambda={.2,-.3};
    const DcaMajorantBarrierPath path(lw,d);detail::SocDefectGeometry geom;
    check(geom.inspect(lw)&&geom.blocks.size()==1&&geom.leaves.size()==1,"independent affine unit leaf not detected");
    Direction trial;const auto out=geom.project(lw,d,path,.5,trial);
    check(out.valid&&out.leaf_count==1,"source geometry plus affine transport rejected");
    Vec x=lw.x,s=lw.s;axpy(.5,trial.x,x);axpy(.5,trial.s,s);
    close(x[0],1/std::sqrt(1.25L),1e-13L,"radial weight coordinate incorrect");
    close(x[1],.5L/std::sqrt(1.25L),1e-13L,"radial second coordinate incorrect");
    close(x[2],2*x[0]+3*x[1],1e-13L,"affine leaf not relifted from repaired coordinates");
    close(lw.equalities[0].original_at(x),0,1e-12L,"lifted retraction missed original norm equation");
    close(lw.equalities[1].original_at(x),0,1e-12L,"lifted retraction missed original affine equation");
    close(lw.inequalities[0].original_at(x)+s[0],0,1e-12L,"lifted retraction missed actual nonlinear slack equation");
    check(trial.z==d.z&&trial.lambda==d.lambda,"affine transport changed source dual direction");
    check(out.correction_inf<=.5*inf_norm(d.x),"lifted retraction exceeds original source correction bound");
    check(lw.fraction(trial,st.lazy_newton_fraction_to_boundary)>=.5,"lifted retraction failed fraction boundary");
    const DcaMajorantBarrierPath trial_path(lw,trial);
    const auto origin=DcaMajorantBarrierPath::current(lw,4e-8);
    DcaMajorantBarrierFilter actual_filter(origin.theta);
    const auto point=trial_path.rounded_point(.5,4e-8),same=path.rounded_point(.5,4e-8);
    check(point.positive&&point.finite&&point.theta<same.theta,"lifted retraction failed same-alpha actual point improvement");
    check(actual_filter.diagnose(origin,point,path.slope(4e-8),.5,st.line_search_armijo).accepted,"lifted retraction failed original source-slope filter");
    // Deliberately inexact linear direction. The correct affine h target is
    // .01 at alpha=.5, not the idealized zero residual.
    auto inexact=d;inexact.x={.01,1,3.02};inexact.s={-12.08};
    const DcaMajorantBarrierPath inexact_path(lw,inexact);
    check(geom.project(lw,inexact,inexact_path,.5,trial).valid,"inexact-linear source affine target rejected");
    x=lw.x;s=lw.s;axpy(.5,trial.x,x);axpy(.5,trial.s,s);
    const long double htarget=inexact_path.equalities[0].constant+.5L*inexact_path.equalities[0].linear;
    close(htarget,.01L,1e-14L,"inexact source fixture target changed");
    close(lw.equalities[0].original_at(x),htarget,1e-12L,"retraction used ideal Newton target instead of actual source affine target");
    check(std::abs(lw.equalities[0].original_at(x))>1e-3,"inexact source target was silently set to zero");
    close(lw.equalities[1].original_at(x),0,1e-12L,"inexact source affine relift lost its target");
    close(lw.inequalities[0].original_at(x)+s[0],0,1e-12L,"inexact source slack affine target incorrect");
    // Structural padding is mathematically zero, even outside a block or
    // on an off-diagonal. It must not alter the coefficient certificate.
    lw.equalities[0].hessian.push_back({0,2,0});
    lw.equalities[0].linear.push_back({2,0});
    lw.equalities[1].hessian={{0,0,0},{1,2,0},{2,2,0}};
    detail::SocDefectGeometry padded;
    check(padded.inspect(lw)&&padded.blocks.size()==1&&padded.leaves.size()==1,"exact structural zero padding changed nonlinear/affine classification");
    check(padded.project(lw,d,path,.5,trial).valid,"zero-padded equivalent geometry rejected its source proposal");
    x=lw.x;axpy(.5,trial.x,x);
    close(lw.equalities[0].original_at(x),0,1e-12L,"structural zero padding changed actual nonlinear target");
    close(lw.equalities[1].original_at(x),0,1e-12L,"zero-Hessian affine row was not relifted");
  }

  // Shifted, unequal diagonal coefficients and heterogeneous internal row
  // scales must not turn an ellipse into a unit sphere by assumption.
  for(int sign:{1,-1}) {
    auto ellipse=sphere(2);ellipse.x0={1.5,-2};
    ellipse.equalities[0].constant=sign*39.;ellipse.equalities[0].linear={sign*-8.,sign*36.};
    ellipse.equalities[0].hessian.entries={{0,0,sign*8.},{1,1,sign*18.}};
    LazyNewtonStatistics local_stats;detail::NewtonCostLedger local_costs(local_stats,Clock::now());
    Workspace ew(ellipse,st,local_stats,local_costs);ew.equalities[0].scale=sign>0?100:.001;ew.evaluate();
    auto d=direction(ew);d.x={0,1./3};d.lambda={.3};
    const DcaMajorantBarrierPath path(ew,d);detail::SocDefectGeometry geom;
    check(geom.inspect(ew),"definite shifted ellipse rejected");
    Direction trial;check(geom.project(ew,d,path,1,trial).valid,"shifted scaled ellipse projection rejected");
    Vec x=ew.x;axpy(1,trial.x,x);
    close(x[0],1+.5L/std::sqrt(2.0L),1e-12L,"ellipse center or first weight was lost");
    close(x[1],-2+(1.L/3)/std::sqrt(2.0L),1e-12L,"ellipse center or second weight was lost");
    check(detail::soc_defect_trial_residuals(ew,trial,1).equality<1e-11,"heterogeneous scale changed physical target residual");
    check(trial.lambda==d.lambda,"ellipse projection altered source equality multiplier direction");
    // A non-tangent source makes the sign and INTERNAL/ORIGINAL conversion
    // observable: Jp is sign*.04 in physical units, not an idealized zero.
    auto non_tangent=d;non_tangent.x[0]=.01;
    const DcaMajorantBarrierPath non_tangent_path(ew,non_tangent);
    check(geom.project(ew,non_tangent,non_tangent_path,1,trial).valid,"non-tangent shifted ellipse source rejected");
    x=ew.x;axpy(1,trial.x,x);
    const long double internal_target=non_tangent_path.equalities[0].constant+non_tangent_path.equalities[0].linear;
    close(internal_target/ew.equalities[0].scale,sign*.04L,1e-12L,"scaled non-tangent ellipse source target incorrect");
    close(ew.equalities[0].original_at(x)/ew.equalities[0].scale,sign*.04L,1e-11L,"ellipse retraction mixed sign/scale or discarded actual source affine target");
    const long double radius_ratio=std::sqrt(1.04L/2.0404L);
    close(x[0],1+.51L*radius_ratio,1e-12L,"non-tangent ellipse first coordinate incorrect");
    close(x[1],-2+(1.L/3)*radius_ratio,1e-12L,"non-tangent ellipse second coordinate incorrect");
    check(trial.lambda==non_tangent.lambda,"non-tangent ellipse proposal changed source dual direction");
  }

  // A mathematical proposal is not an acceptance certificate. Relifted
  // coordinates may violate size, and exact slack transport may leave the
  // positive domain; the caller must retain both guards.
  {
    auto large=sphere(3);large.x0[2]=10;
    QuadraticFunction a;a.linear={-10,0,1};a.hessian.n=3;large.equalities.push_back(a);
    LazyNewtonStatistics local_stats;detail::NewtonCostLedger local_costs(local_stats,Clock::now());
    Workspace lw(large,st,local_stats,local_costs);lw.evaluate();auto d=direction(lw);d.x={0,1,0};
    const DcaMajorantBarrierPath path(lw,d);detail::SocDefectGeometry geom;check(geom.inspect(lw),"large independent affine leaf not recognized");
    Direction trial;const auto out=geom.project(lw,d,path,1,trial);
    check(out.valid&&out.correction_inf>inf_norm(d.x),"full lifted-coordinate size guard counterexample was not reproduced");
    check(std::abs((lw.x[0]+trial.x[0])-1)<inf_norm(d.x),"size counterexample does not distinguish block-only from full-coordinate check");
  }
  {
    auto bad_slack=sphere(2);auto q=g;q.constant=-.01;bad_slack.inequalities.push_back(q);
    LazyNewtonStatistics local_stats;detail::NewtonCostLedger local_costs(local_stats,Clock::now());
    Workspace sw(bad_slack,st,local_stats,local_costs);sw.s={.02};sw.z={.5};sw.lambda={0};sw.evaluate();
    auto d=direction(sw);d.x={0,1};const DcaMajorantBarrierPath path(sw,d);detail::SocDefectGeometry geom;
    check(geom.inspect(sw),"slack-domain counterexample geometry not recognized");
    Direction trial;check(geom.project(sw,d,path,1,trial).valid,"finite geometric domain counterexample unexpectedly rejected before caller guards");
    const DcaMajorantBarrierPath repaired_path(sw,trial);
    check(!repaired_path.rounded_point(1,.01).positive&&sw.fraction(trial,st.lazy_newton_fraction_to_boundary)<1,"positive-domain guard counterexample not reproduced");
  }
  {
    auto centered=sphere(2);centered.x0={0,0};
    LazyNewtonStatistics local_stats;detail::NewtonCostLedger local_costs(local_stats,Clock::now());
    Workspace cw(centered,st,local_stats,local_costs);cw.evaluate();auto d=direction(cw);d.x={0,1};
    const DcaMajorantBarrierPath path(cw,d);detail::SocDefectGeometry geom;check(geom.inspect(cw),"centered fixture coefficient geometry rejected before radial check");
    Direction trial;check(!geom.project(cw,d,path,.5,trial).valid,"singular centered anchor admitted as tangent-preserving radial retraction");
  }

  auto overlap=sphere(2);overlap.equalities.push_back(overlap.equalities[0]);
  check(!inspect(overlap),"overlapping nonlinear equality blocks admitted");
  auto indefinite=sphere(2);indefinite.equalities[0].hessian.entries[1].v=-2;
  check(!inspect(indefinite),"indefinite diagonal equality admitted");
  auto offdiag=sphere(2);offdiag.equalities[0].hessian.entries.push_back({0,1,.5});
  check(!inspect(offdiag),"off-diagonal equality admitted as diagonal geometry");
  auto native=sphere(2);native.equalities[0].signed_square={{0,.1}};
  check(!inspect(native),"native signed-square equality admitted as an exact quadratic ellipse");
  auto linear_outside=sphere(3);linear_outside.equalities[0].linear[2]=.1;
  check(!inspect(linear_outside),"linear coordinate outside definite block admitted");
  auto bounded=lift;bounded.ub[2]=10;
  check(!inspect(bounded),"bounded affine leaf admitted for automatic transport");
  auto nonunique=lift;nonunique.equalities.push_back(nonunique.equalities.back());
  check(!inspect(nonunique),"nonunique affine leaf admitted");
  auto nonunit=lift;nonunit.equalities[1].linear[2]=2;
  check(!inspect(nonunit),"nonunit affine pivot admitted without its coefficient certificate");
  auto empty=sphere(2);empty.equalities.clear();check(!inspect(empty),"model without nonlinear equality geometry admitted");
  {
    LazyNewtonStatistics local_stats;detail::NewtonCostLedger local_costs(local_stats,Clock::now());
    Workspace malformed(p,st,local_stats,local_costs);malformed.equalities[0].hessian[0].v=INFINITY;
    detail::SocDefectGeometry geom;check(!geom.inspect(malformed),"nonfinite original diagonal coefficient admitted");
  }
  check(w.x==anchor_x&&w.s==anchor_s&&w.z==anchor_z&&w.lambda==anchor_lambda,"unit probes changed original KKT fixture anchor");
  std::cout<<"soc_defect_control PASS checks="<<checks
      <<" genuine_frozen_KKT=1 accepted_SOC3_insufficient=1 source_affine_levelset=1"
      <<" exact_affine_leaf_transport=1 nonlinear_slack_equation=1 shifted_ellipses=1 internal_row_scaling=1"
      <<" refused_SOC_source_recovery=1 small_alpha_recovery=1"
      <<" no_extra_solve=1 unchanged_original_filter=1 unsupported_geometry_rejected=1 caller_size_and_domain_guards=1\n";
}
