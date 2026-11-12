// Included after dca_original. Signed residuals come from the SAME original
// coefficient stream, fixed-variable reconstruction and extended accumulator
// as the terminal certificate. They are not norms reinterpreted as a RHS.
struct DcaPreciseResidual {
  Result certificate;
  Vec dual,equality,primal,complementarity;
  DcaPoint point;
  bool strict=false,valid=false;
};
DcaPreciseResidual dca_precise_residual(
    Workspace& w,const Settings& stop,const Vec& x,const Vec& s,
    const Vec& z,const Vec& lambda,int iteration,
    const Vec* anchor_lambda=nullptr,const Vec* anchor_z=nullptr) {
  DcaPreciseResidual out;
  detail::OriginalSignedResidual original;
  auto& r=out.certificate;r.outer_iterations=iteration;
  w.export_state_at(r,x,s,z,lambda);
  out.strict=detail::audit_original_termination(w.original,stop,r,
      &w.original_linear_support,true,&original);
  if(!original.valid||!r.termination_audit_valid)return out;
  out.dual.resize(w.n);out.equality.resize(w.me);
  out.primal.assign(w.mi,0);out.complementarity.assign(w.mi,0);
  using L=long double;
  const L sf=w.objective.scale;
  for(int j=0;j<w.n;++j)out.dual[j]=double(sf*original.dual[w.free[j]]);
  auto& p=out.point;
  p.objective=double(sf*r.objective);
  L lag=sf*r.objective,current=lag,theta=0;
  const Vec& al=anchor_lambda?*anchor_lambda:lambda;
  const Vec& az=anchor_z?*anchor_z:z;
  for(int e=0;e<w.me;++e){
    const L h=L(w.equalities[e].scale)*original.equality[e];
    out.equality[e]=double(h);theta+=std::abs(h);
    lag+=L(al[e])*h;current+=L(lambda[e])*h;
  }
  for(int i=0;i<w.mi;++i){
    const int origin=w.inequality_origin[i];
    const L value=w.inequality_kind[i]==0?L(original.inequality[origin]):
        w.inequality_kind[i]<0?L(w.original.lb[origin])-r.x[origin]:
                              L(r.x[origin])-w.original.ub[origin];
    const L h=L(w.inequalities[i].scale)*value;
    theta+=std::max(L(0),h);
    if(w.active[i]){
      out.primal[i]=double(h+L(s[i]));
      out.complementarity[i]=double(L(s[i])*L(z[i]));
      lag+=L(az[i])*h;current+=L(z[i])*h;
    }
  }
  p.theta=double(theta);p.lagrangian=double(lag);p.current_lagrangian=double(current);
  p.primal=p.primal_inf=r.feasibility.total_inf;p.dual=r.dual_infeasibility_inf;
  p.comp=r.complementarity_inf;p.kkt=r.raw_kkt_error;
  p.normalized_kkt=r.scaled_nlp_error;
  p.within_acceptable_caps=r.acceptable_point;
  p.distinct_from_anchor=x!=w.x||lambda!=w.lambda||z!=w.z;
  p.guard_reference_kkt=p.kkt;
  p.feasible_reference_initialized=p.primal<=w.st.feasibility_tolerance;
  out.valid=finite(out.dual)&&finite(out.equality)&&finite(out.primal)&&finite(out.complementarity);
  return out;
}
