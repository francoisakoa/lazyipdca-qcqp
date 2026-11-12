// Exact original-QCQP line polynomials for the majorant filter. Included after
// DcaPoint and Workspace, inside their anonymous namespace.
// PRECONDITION: q.value/q.gradient are current at the tangent majorant centre,
// and Workspace/direction remain unchanged for the lifetime of this object.
// Preparation visits original Hessians once; point(alpha) visits only these
// scalar polynomials and the three stationarity coefficient vectors.
struct DcaMajorantFilterPath {
  using Acc = long double;
  struct Polynomial {
    Acc constant=0, linear=0, quadratic=0;
    Acc at(Acc alpha) const {
      return constant+alpha*(linear+alpha*quadratic);
    }
  };
  struct RowPoint { Acc value, multiplier, fixed_multiplier; };

  const Workspace& w;
  const Direction& direction;
  Polynomial objective;
  std::vector<Polynomial> equalities, inequalities;
  std::vector<Acc> rd0, rd1, rd2;

  DcaMajorantFilterPath(const Workspace& work,const Direction& step)
      : w(work),direction(step),equalities(w.me),inequalities(w.mi),
        rd0(w.n,0),rd1(w.n,0),rd2(w.n,0) {
    auto prepare=[&](const Quadratic& q,Acc weight,Acc weight_step,
                     bool contributes) {
      Polynomial p;
      p.constant=q.value;
      for(std::size_t k=0;k<q.support.size();++k) {
        const int j=q.support[k];
        const Acc gradient=q.gradient[k];
        p.linear+=gradient*direction.x[j];
        if(contributes) {
          rd0[j]+=weight*gradient;
          rd1[j]+=weight_step*gradient;
        }
      }
      // q.hessian is the ORIGINAL unscaled QCQP Hessian. In particular,
      // neither q.model_hessian nor rhoKKT belongs in this original filter.
      for(const auto& e:q.hessian) {
        const Acc hij=Acc(q.scale)*e.v;
        const Acc hi_dxj=hij*direction.x[e.j];
        p.quadratic+=(e.i==e.j?Acc(0.5):Acc(1))*hi_dxj*direction.x[e.i];
        if(contributes) {
          rd1[e.i]+=weight*hi_dxj;
          rd2[e.i]+=weight_step*hi_dxj;
          if(e.i!=e.j) {
            const Acc hj_dxi=hij*direction.x[e.i];
            rd1[e.j]+=weight*hj_dxi;
            rd2[e.j]+=weight_step*hj_dxi;
          }
        }
      }
      return p;
    };
    objective=prepare(w.objective,1,0,true);
    for(int e=0;e<w.me;++e)
      equalities[e]=prepare(w.equalities[e],w.lambda[e],direction.lambda[e],true);
    for(int i=0;i<w.mi;++i)
      inequalities[i]=prepare(w.inequalities[i],w.z[i],direction.z[i],w.active[i]);
  }

  double objective_slope() const { return double(objective.linear); }

  // Common normalization with dca_original, including INTERNAL units for
  // phase detection and ORIGINAL units for strict residual/cap reporting.
  template<class EqualityPoint,class InequalityPoint>
  static DcaPoint reduce(const Workspace& w,Acc objective,Acc dual_internal,
                         bool distinct,EqualityPoint equality_point,
                         InequalityPoint inequality_point) {
    DcaPoint p;
    p.objective=double(objective);
    Acc theta=0,primal=0,comp=0,lagrangian=objective,current_lagrangian=objective;
    Acc y_l1=0,z_l1=0,yi=0,zi=0,pi=0;
    int native_inequalities=0;
    bool values_finite=std::isfinite(objective)&&std::isfinite(dual_internal);
    for(int e=0;e<w.me;++e) {
      const auto r=equality_point(e);
      const Acc scale=w.equalities[e].scale;
      values_finite=values_finite&&std::isfinite(r.value)&&std::isfinite(r.multiplier);
      theta+=std::abs(r.value);
      primal=std::max(primal,std::abs(r.value)/scale);
      lagrangian+=r.fixed_multiplier*r.value;
      current_lagrangian+=r.multiplier*r.value;
      y_l1+=std::abs(r.multiplier*scale/w.objective.scale);
      yi+=std::abs(r.multiplier);
      pi=std::max(pi,std::abs(r.value));
    }
    for(int i=0;i<w.mi;++i) {
      const auto r=inequality_point(i);
      const Acc scale=w.inequalities[i].scale;
      const bool native=w.inequality_kind[i]==0;
      if(native)++native_inequalities;
      values_finite=values_finite&&std::isfinite(r.value)&&std::isfinite(r.multiplier);
      const Acc violation=std::max(Acc(0),r.value);
      theta+=violation;
      primal=std::max(primal,violation/scale);
      pi=std::max(pi,native?violation:violation/scale);
      if(w.active[i]) {
        lagrangian+=r.fixed_multiplier*r.value;
        current_lagrangian+=r.multiplier*r.value;
        comp=std::max(comp,std::abs(r.multiplier*r.value)/w.objective.scale);
        const Acc original_multiplier=std::abs(r.multiplier*scale/w.objective.scale);
        z_l1+=original_multiplier;
        if(native)y_l1+=original_multiplier;
        const Acc internal_multiplier=std::abs(native?r.multiplier:r.multiplier*scale);
        zi+=internal_multiplier;
        if(native)yi+=internal_multiplier;
      }
    }
    p.theta=double(theta);
    p.lagrangian=double(lagrangian);
    p.current_lagrangian=double(current_lagrangian);
    p.primal=double(primal);
    p.dual=double(dual_internal/w.objective.scale);
    p.comp=double(comp);
    p.kkt=std::max({p.primal,p.dual,p.comp});
    if(!values_finite||!std::isfinite(p.objective)||!std::isfinite(p.theta)||
       !std::isfinite(p.kkt))p.kkt=INFINITY;
    p.primal_inf=p.primal;
    const auto original_metrics=ipopt_desired_convergence_metrics(
        p.dual,p.primal,p.comp,double(y_l1),double(z_l1),
        w.me+native_inequalities+w.mi,w.mi,w.st);
    p.within_acceptable_caps=values_finite&&original_metrics.finite&&
        original_metrics.overall_nlp_error<=w.st.acceptable_tolerance&&
        p.primal<=w.st.acceptable_constraint_tolerance&&
        p.dual<=w.st.acceptable_dual_tolerance&&
        p.comp<=w.st.acceptable_complementarity_tolerance;
    const auto internal_metrics=ipopt_desired_convergence_metrics(
        double(dual_internal),double(pi),double(comp*w.objective.scale),
        double(yi),double(zi),w.me+native_inequalities+w.mi,w.mi,w.st);
    p.normalized_kkt=values_finite&&internal_metrics.finite?
        internal_metrics.overall_nlp_error:INFINITY;
    if(w.st.termination_test=="ipopt")
      p.within_acceptable_caps=values_finite&&internal_metrics.finite&&
          p.normalized_kkt<=w.st.acceptable_tolerance&&
          p.primal<=w.st.acceptable_constraint_tolerance&&
          p.dual<=w.st.acceptable_dual_tolerance&&
          p.comp<=w.st.acceptable_complementarity_tolerance;
    p.distinct_from_anchor=p.within_acceptable_caps&&distinct;
    // A polynomial trial is never a certificate, even when it passes caps.
    p.acceptable_certified=false;
    p.guard_reference_kkt=p.kkt;
    p.feasible_reference_initialized=p.primal<=w.st.feasibility_tolerance;
    return p;
  }

  DcaPoint point(double alpha) const {
    const Acc a=alpha;
    Acc dual=0;
    for(int j=0;j<w.n;++j) {
      const Acc value=rd0[j]+a*(rd1[j]+a*rd2[j]);
      if(!std::isfinite(value)){dual=INFINITY;break;}
      dual=std::max(dual,std::abs(value));
    }
    // A nonzero direction need not change a floating-point iterate when its
    // step is tiny. Preserve the filter's exact distinct-iterate semantics.
    bool distinct=false;
    if(alpha!=0) {
      for(int j=0;j<w.n&&!distinct;++j)
        distinct=w.x[j]+alpha*direction.x[j]!=w.x[j];
      for(int e=0;e<w.me&&!distinct;++e)
        distinct=w.lambda[e]+alpha*direction.lambda[e]!=w.lambda[e];
      for(int i=0;i<w.mi&&!distinct;++i)
        distinct=w.z[i]+alpha*direction.z[i]!=w.z[i];
    }
    return reduce(w,objective.at(a),dual,distinct,
        [&](int e) {return RowPoint{equalities[e].at(a),
            Acc(w.lambda[e])+a*direction.lambda[e],w.lambda[e]};},
        [&](int i) {return RowPoint{inequalities[i].at(a),
            Acc(w.z[i])+a*direction.z[i],w.z[i]};});
  }

  static DcaPoint current(const Workspace& w) {
    // Cached tangent gradients already define the original stationarity at
    // the anchor. Avoid all original coefficient/gradient reevaluations.
    const Acc dual=finite(w.rd)?inf_norm(w.rd):INFINITY;
    return reduce(w,w.objective.value,dual,false,
        [&](int e) {return RowPoint{w.equalities[e].value,w.lambda[e],w.lambda[e]};},
        [&](int i) {return RowPoint{w.inequalities[i].value,w.z[i],w.z[i]};});
  }
};
