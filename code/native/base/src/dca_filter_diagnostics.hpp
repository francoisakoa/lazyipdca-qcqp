// Included inside the filter engine's anonymous namespace after DcaPoint.
// Read-only diagnostics: no state, model, filter or acceptance is modified.
struct DcaTraceContext { int outer=-1, rebuild=-1, inner=-1; };
void dca_trace_context(std::ostream& o,const DcaTraceContext& c) {
  o<<" outer="<<c.outer<<" rebuild="<<c.rebuild<<" inner="<<c.inner;
}

bool dca_filter_trace_enabled(const Workspace& w) {
  return w.st.dca_filter_diagnostics &&
      (w.st.live_display_level>0 || (w.st.live_display_level<0 && w.st.live_iterations));
}

double dca_original_hessian_gap(Workspace& w,const Vec& dx) {
  // Compare the matrix actually used by the majorant direction with the
  // original Lagrangian Hessian at the same state.  This includes atoms and
  // the explicit primal regularization in the model side, but no line-search
  // or multiplier effect.
  Vec model=w.hessian_product(dx),original(w.n,0.0);
  auto add=[&](const Quadratic& q,double weight) {
    const double a=weight*q.scale;
    for(const auto& e:q.hessian) {
      original[e.i]+=a*e.v*dx[e.j];
      if(e.i!=e.j)original[e.j]+=a*e.v*dx[e.i];
    }
  };
  add(w.objective,1.0);
  for(int e=0;e<w.me;++e)add(w.equalities[e],w.lambda[e]);
  for(int i=0;i<w.mi;++i)if(w.active[i])add(w.inequalities[i],w.z[i]);
  for(int j=0;j<w.n;++j)model[j]-=original[j];
  return inf_norm(model)/std::max(1.0,w.objective.scale);
}

void dca_trace_dual_decomposition(
    Workspace& w,const DcaPoint& from,const DcaPoint& combined,
    const Vec& x0,const Vec& lambda0,const Vec& z0,const Direction& d,
    double alpha_primal,double alpha_lambda,double alpha_z,
    const char* path,const char* stage,double direction_residual,
    double model_gap,DcaTraceContext context={}) {
  if(!w.st.dca_dual_decomposition)return;
  const double base=std::max(from.dual,1e-300);
  const double growth=combined.dual/base;
  if(w.st.dca_dual_decomposition_growth>0.0 &&
     growth<w.st.dca_dual_decomposition_growth)return;

  Vec xp=x0,ld=lambda0,zd=z0,ls=lambda0,zs=z0;
  axpy(alpha_primal,d.x,xp);
  axpy(alpha_lambda,d.lambda,ld);axpy(alpha_z,d.z,zd);
  // Counterfactual with the primal fraction also imposed on both multiplier
  // families.  This isolates alpha_du != alpha_pr without changing dx.
  axpy(alpha_primal,d.lambda,ls);axpy(alpha_primal,d.z,zs);
  const DcaPoint primal_only=dca_original(w,xp,lambda0,z0);
  const DcaPoint dual_only=dca_original(w,x0,ld,zd);
  const DcaPoint same_alpha=dca_original(w,xp,ls,zs);

  std::ostringstream o;
  o<<std::scientific<<std::setprecision(8)
   <<"[DCA-DUAL-DECOMP] path="<<path<<" stage="<<stage
   <<" inf_du0="<<from.dual<<" inf_du="<<combined.dual
   <<" growth="<<growth
   <<" primal_only="<<primal_only.dual
   <<" dual_only="<<dual_only.dual
   <<" same_alpha="<<same_alpha.dual
   <<" combined="<<combined.dual
   <<" alpha_pr="<<alpha_primal
   <<" alpha_lambda="<<alpha_lambda<<" alpha_z="<<alpha_z
   <<" dx_inf="<<inf_norm(d.x)<<" dlambda_inf="<<inf_norm(d.lambda)
   <<" dz_inf="<<inf_norm(d.z)
   <<" direction_residual="<<direction_residual
   <<" model_hessian_gap_dx="<<model_gap;
  dca_trace_context(o,context);
  std::cout<<o.str()<<std::endl;
}

void dca_trace_domain(const Workspace& w,const char* path,const char* stage,int ls,
                      double ap,double ad,const char* reason,int row=-1,DcaTraceContext context={}) {
  if(!dca_filter_trace_enabled(w))return;
  std::ostringstream o;
  o<<std::scientific<<std::setprecision(8)
   <<"[DCA-FILTER] path="<<path<<" stage="<<stage<<" ls="<<ls
   <<" alpha_pr="<<ap<<" alpha_lambda="<<ad<<" alpha_z="<<ad
   <<" decision=reject reason="<<reason;
  if(row>=0)o<<" row="<<row;
  dca_trace_context(o,context);
  std::cout<<o.str()<<std::endl;
}

void dca_trace_filter(const Workspace& w,const detail::OriginalDcaFilter& filter,
                     const DcaPoint& from,const DcaPoint& to,double predicted,
                     bool restoration,const char* path,const char* stage,int ls,
                     double ap,double ad,const char* extra_rejection=nullptr,int row=-1,DcaTraceContext context={}) {
  if(!dca_filter_trace_enabled(w))return;
  const auto f=filter.diagnose(from,to,predicted,w.st.feasibility_tolerance);
  const double ceiling=dca_kkt_ceiling(w,from);
  const double lambda_step=w.st.dca_dual_step=="ipopt" ? ap :
      detail::dca_equality_step(from.primal,w.st.feasibility_tolerance,ap,ad);
  const bool kkt_ok=dca_kkt_allowed(w,from,to);
  const bool accept=f.accepted && kkt_ok && !extra_rejection;
  const char* reason=extra_rejection ? extra_rejection : (!kkt_ok ? "kkt-growth" : f.reason);
  std::ostringstream o;
  o<<std::scientific<<std::setprecision(8)
   <<"[DCA-FILTER] path="<<path<<" stage="<<stage<<" ls="<<ls
   <<" alpha_pr="<<ap<<" alpha_lambda="<<lambda_step<<" alpha_z="<<ad
   <<" restoration="<<restoration
   <<" decision="<<(accept?"accept":"reject")<<" reason="<<reason
   <<" f0="<<from.objective/w.objective.scale<<" f="<<to.objective/w.objective.scale
   <<" df="<<(to.objective-from.objective)/w.objective.scale
   <<" objective_scale="<<w.objective.scale
   <<" theta0_s="<<from.theta<<" theta_s="<<to.theta
   <<" inf_pr0="<<from.primal<<" inf_pr="<<to.primal<<" inf_du="<<to.dual<<" compl="<<to.comp
   <<" kkt0="<<from.kkt<<" kkt="<<to.kkt<<" kkt_ceiling="<<ceiling
   <<" pred_s="<<predicted<<" noise_s="<<f.noise
   <<" stationary_phase="<<f.stationary_phase
   <<" normalized_kkt0="<<from.normalized_kkt
   <<" acceptable_continuation="<<f.acceptable_continuation
   <<" feas_ok="<<f.feasibility<<" obj_ok="<<f.objective<<" terminal_ok="<<f.terminal
   <<" filter_ok="<<f.accepted<<" kkt_ok="<<kkt_ok
   <<" history_checked="<<f.history_checked<<" history_bypassed="<<f.history_bypassed
   <<" feas_theta_limit_s="<<f.feasibility_theta_limit
   <<" obj_limit_s="<<f.objective_limit<<" theta_cap_s="<<f.theta_cap
   <<" feas_requires_kkt="<<(from.primal<=w.st.feasibility_tolerance)
   <<" feas_kkt_limit="<<0.99*from.kkt
   <<" terminal_kkt_limit="<<f.terminal_kkt_limit
   <<" terminal_f_limit_s="<<from.objective+f.noise
   <<" lagrangian0_s="<<from.lagrangian<<" lagrangian_s="<<to.lagrangian
   <<" feasibility_tol="<<w.st.feasibility_tolerance;
  if(f.history_index>=0)
    o<<" history_index="<<f.history_index<<" history_theta_limit_s="<<f.history_theta_limit
     <<" history_obj_limit_s="<<f.history_objective_limit;
  if(row>=0)o<<" row="<<row;
  dca_trace_context(o,context);
  std::cout<<o.str()<<std::endl;
}

void dca_trace_private(const Workspace& w,int ls,double ap,double ad,bool positive,
                       bool finite_state,bool accepted,double merit,double trial,
                       double slope,double theta,double target,double noise,
                       double before,double after,bool residual_evaluated,DcaTraceContext context={}) {
  if(!dca_filter_trace_enabled(w))return;
  const char* reason=accepted?"accepted-private":(!positive?"nonpositive-state":
      (!finite_state || !std::isfinite(trial)?"nonfinite":
       (slope>=0?"non-descent/residual":(ap<1e-6?"armijo-step-floor/residual":"private-merit/residual"))));
  std::ostringstream o;
  o<<std::scientific<<std::setprecision(8)
   <<"[DCA-PRIVATE] ls="<<ls<<" alpha_pr="<<ap<<" alpha_lambda="<<ad<<" alpha_z="<<ad
   <<" decision="<<(accepted?"eligible-private":"reject")<<" reason="<<reason
   <<" merit0_s="<<merit<<" merit_s="<<trial
   <<" armijo_limit_s="<<(merit+w.st.line_search_armijo*ap*slope+noise)
   <<" slope_s="<<slope<<" theta0_s="<<theta<<" penalty="<<w.penalty<<" mu_target="<<target
   <<" armijo_alpha_floor=1e-6 armijo_eligible="<<(ap>=1e-6 && slope<0)
   <<" residual_space=model-scaled residual0_s="<<before<<" residual_trial_s="<<after
   <<" residual_evaluated="<<residual_evaluated
   <<" residual_limit_s="<<((1-1e-4*std::min(ap,ad))*before)
   <<" noise_s="<<noise;
  dca_trace_context(o,context);
  std::cout<<o.str()<<std::endl;
}

void dca_trace_unavailable(const Workspace& w,const char* path,const char* stage,
                           const char* reason,const std::string& detail,
                           DcaTraceContext context={}) {
  if(!dca_filter_trace_enabled(w))return;
  std::ostringstream o;
  o<<"[DCA-FILTER] path="<<path<<" stage="<<stage
   <<" decision=not-tested reason="<<reason<<" detail="<<std::quoted(detail);
  dca_trace_context(o,context);
  std::cout<<o.str()<<std::endl;
}
