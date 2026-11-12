#pragma once
// Included beside Workspace/Direction and DcaMajorantBarrierPath. This
// optional globalization probe never edits their anchor or factorization.
// Its quadratic correction may touch Hessian columns: every trial is
// evaluated from the ORIGINAL coefficients at the actual rounded vectors.
struct DcaGrKktSoc {
  struct Staged { Vec x,s,z,lambda; };
  struct EqualityNorms { long double inf=0,l1=0;bool finite=true; };
  bool attempted=false,prepared=false;
  Direction correction;
  double defect_inf=0,correction_inf=0;
  int backsolves=0;
  std::string reason="not-attempted";

  static bool time_open(Clock::time_point start,double limit) {
    return std::isfinite(limit)&&limit>0&&seconds(start)<limit;
  }
  static long double quadratic_defect(const Quadratic& q,const Vec& dx) {
    if(!q.signed_square.empty())throw std::runtime_error("GR KKT SOC requires polynomial QCQP rows");
    detail::AccurateProductSum sum;
    for(const auto& h:q.hessian)
      sum.add_triple_product(h.v,dx[h.i],dx[h.j],
          (h.i==h.j?.5L:1.L)*q.scale);
    return sum.value();
  }
  // One solve request, at most the existing eight audited refinement solves.
  // The same O(alpha^2) correction can be tested at smaller line-search steps.
  bool prepare(Workspace& w,const Direction& direction,Clock::time_point start,double limit) {
    if(attempted)return prepared;
    attempted=true;
    if(!time_open(start,limit)){reason="deadline-before-correction";return false;}
    if(w.current_factor!=w.factor.get()) {reason="configured-frozen-factor-unavailable";return false;}
    const auto* factor=w.current_factor;
    const int factors=w.stats.factor_attempts;
    const int solves=w.stats.linear_solves;
    const double tau=w.regularization;
    Vec equality(w.me,0),primal(w.mi,0);
    try {
      for(int e=0;e<w.me;++e) {
        const long double defect=quadratic_defect(w.equalities[e],direction.x);
        equality[e]=static_cast<double>(defect);
        if(!std::isfinite(defect)||!std::isfinite(equality[e]))
          throw std::runtime_error("nonfinite GR SOC equality defect");
        defect_inf=std::max(defect_inf,std::abs(equality[e]));
      }
      for(int i=0;i<w.mi;++i)if(w.active[i]) {
        const long double defect=quadratic_defect(w.inequalities[i],direction.x);
        primal[i]=static_cast<double>(defect);
        if(!std::isfinite(defect)||!std::isfinite(primal[i]))
          throw std::runtime_error("nonfinite GR SOC inequality defect");
      }
      if(!(defect_inf>0)){reason="zero-equality-curvature-defect";return false;}
      if(!time_open(start,limit)){reason="deadline-before-backsolve";return false;}
      correction=w.direction(Vec(w.mi,0),Vec(w.n,0),equality,primal,
          0,true,DirectionPurpose::Complete,true);
      backsolves=w.stats.linear_solves-solves;
      if(w.current_factor!=factor||w.stats.factor_attempts!=factors||w.regularization!=tau)
        throw std::logic_error("GR KKT SOC changed its frozen factor or proximal metric");
      if(backsolves<1||backsolves>8)
        throw std::logic_error("GR KKT SOC exceeded its audited correction-solve bound");
      correction_inf=inf_norm(correction.x);
      prepared=std::isfinite(correction.residual)&&
          correction.residual<=w.st.kkt_backward_error_reject&&
          finite(correction.x)&&finite(correction.s)&&finite(correction.z)&&finite(correction.lambda)&&
          time_open(start,limit);
      reason=prepared?"original-KKT-audited-frozen-correction":"correction-or-deadline-rejected";
    } catch(const std::runtime_error& error) {
      backsolves=w.stats.linear_solves-solves;
      // Frozen direction failures cannot refactor, replace the iterate or
      // change rho. A numerical rejection just preserves the ordinary trial.
      if(w.current_factor!=factor||w.stats.factor_attempts!=factors||w.regularization!=tau)
        throw std::logic_error("failed GR KKT SOC changed its frozen factor or proximal metric");
      reason=error.what();prepared=false;
    }
    return prepared;
  }
  static EqualityNorms original_equalities(const Workspace& w,const Vec& x) {
    EqualityNorms out;
    for(const auto& q:w.equalities) {
      const long double value=std::abs(static_cast<long double>(q.original_at(x))/q.scale);
      out.finite=out.finite&&std::isfinite(value);
      out.inf=std::max(out.inf,value);out.l1+=value;
    }
    out.finite=out.finite&&std::isfinite(out.l1);return out;
  }
  // Stage all four vectors. No private-pivot geometry or truncated path
  // polynomial is assumed, and no vector will be reconstructed after audit.
  bool stage(const DcaMajorantBarrierPath& path,double alpha,double mu,
      Staged& base,Staged& corrected,double& relative_correction) const {
    if(!prepared||!(alpha>0)||!std::isfinite(alpha))return false;
    path.write_primal(alpha,base.x);
    path.write_slacks(alpha,base.s,&base.x);
    path.write_duals(alpha,mu,base.z,&base.s);
    path.write_equality_duals(alpha,base.lambda);
    const auto& w=path.w;
    double displacement=0;
    for(int j=0;j<w.n;++j)displacement=std::max(displacement,std::abs(base.x[j]-w.x[j]));
    for(int i=0;i<w.mi;++i)if(w.active[i])
      displacement=std::max(displacement,std::abs(base.s[i]-w.s[i]));
    const double size=alpha*alpha*std::max(inf_norm(correction.x),inf_norm(correction.s));
    relative_correction=displacement>0?size/displacement:INFINITY;
    if(!std::isfinite(size)||!std::isfinite(relative_correction)||relative_correction>.5)return false;
    corrected=base;
    auto add=[&](Vec& values,const Vec& delta) {
      if(values.size()!=delta.size())throw std::logic_error("GR KKT SOC staged dimensions");
      for(std::size_t j=0;j<values.size();++j)values[j]+=(alpha*alpha)*delta[j];
    };
    add(corrected.x,correction.x);add(corrected.s,correction.s);
    add(corrected.z,correction.z);add(corrected.lambda,correction.lambda);
    return finite(corrected.x)&&finite(corrected.s)&&finite(corrected.z)&&finite(corrected.lambda);
  }
};
