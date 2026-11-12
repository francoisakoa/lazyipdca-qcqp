// A bounded finish AFTER an incumbent's full audit and BEFORE the next DC
// step/acceptable exit. Failed proposals restore the entire workspace state.
// The original Hessian is used only with its unshifted inertia certificate;
// the main/global algorithm remains Lazy-DCA.
bool dca_strict_finish(Workspace& w,DcaModels& models,DcaAggregate& aggregate,
    DcaPoint& outer,detail::OriginalDcaFilter& filter,const Settings& stop,
    const Vec& original_slacks,int iteration) {
  auto& stats=models.stats;
  ++stats.strict_finish_attempts;
  detail::DcaTimedScope timer(stats.strict_finish_seconds);
  const Vec old_x=w.x,old_s=w.s,old_z=w.z,old_lambda=w.lambda;
  const bool objective_model=w.objective.model_active;
  std::vector<bool> eq_models,ineq_models;
  for(const auto& q:w.equalities)eq_models.push_back(q.model_active);
  for(const auto& q:w.inequalities)ineq_models.push_back(q.model_active);
  auto rollback=[&](){
    w.x=old_x;w.s=old_s;w.z=old_z;w.lambda=old_lambda;
    w.objective.model_active=objective_model;
    for(int e=0;e<w.me;++e)w.equalities[e].model_active=eq_models[e];
    for(int i=0;i<w.mi;++i)w.inequalities[i].model_active=ineq_models[i];
    w.evaluate();w.update_geometry();aggregate.root_factor_valid=false;
  };
  auto reject=[&](const std::string& reason){
    rollback();++stats.strict_finish_rejections;stats.strict_finish_last_rejection=reason;
    if(w.st.live_display_level>0)std::cout<<"[DCA-STRICT-FINISH] outer="<<iteration
        <<" decision=reject reason=\""<<reason<<"\" incumbent_preserved=1\n";
    return false;
  };
  try {
    w.s=original_slacks;dca_disable_models(w);
    auto from=dca_precise_residual(w,stop,w.x,w.s,w.z,w.lambda,iteration-1);
    ++stats.strict_finish_certificates;
    if(!from.valid)return reject("invalid-original-anchor");
    if(from.strict)return reject("anchor-already-strict");
    // The caller supplies a fully audited acceptable point. Repeat the raw
    // caps/error checks locally as a defensive guard, independent of streak.
    if(from.certificate.scaled_nlp_error>stop.acceptable_tolerance ||
       from.certificate.feasibility.total_inf>stop.acceptable_constraint_tolerance ||
       from.certificate.termination_dual_inf>stop.acceptable_dual_tolerance ||
       from.certificate.termination_compl_inf>stop.acceptable_complementarity_tolerance)
      return reject("anchor-outside-acceptable-caps");
    from.point.acceptable_certified=true;
    stats.strict_finish_error_before=from.certificate.scaled_nlp_error;
    if(w.st.live_display_level>0)std::cout<<"[DCA-STRICT-FINISH] outer="<<iteration
        <<" phase=attempt rhs=original-full-extended rho=0 E="<<from.certificate.scaled_nlp_error
        <<" budget="<<stats.strict_finish_attempts<<'/'<<w.st.dca_strict_finish_max<<"\n";
    DcaAggregate::Candidate candidate;
    if(!aggregate.attempt_one(w,models,from.point,filter,iteration,0.0,&candidate,true,false,nullptr,true))
      return reject(aggregate.last_rejection);
    auto after=dca_precise_residual(w,stop,candidate.x,candidate.s,candidate.z,candidate.lambda,
        iteration,&old_lambda,&old_z);
    ++stats.strict_finish_certificates;
    if(!after.valid)return reject("invalid-original-candidate");
    if(!after.strict && !(after.certificate.scaled_nlp_error<from.certificate.scaled_nlp_error*(1-1e-3*candidate.ap)))
      return reject("no-audited-error-contraction");
    // A failed strict finish may not exchange a certified acceptable point
    // for a candidate outside its configured acceptable caps.
    if(!after.strict && (after.certificate.scaled_nlp_error>stop.acceptable_tolerance ||
       after.certificate.feasibility.total_inf>stop.acceptable_constraint_tolerance ||
       after.certificate.termination_dual_inf>stop.acceptable_dual_tolerance ||
       after.certificate.termination_compl_inf>stop.acceptable_complementarity_tolerance))
      return reject("candidate-outside-acceptable-caps");
    if(candidate.point.theta>w.st.feasibility_tolerance)filter.insert(from.point);
    w.x=std::move(candidate.x);w.s=std::move(candidate.s);
    w.z=std::move(candidate.z);w.lambda=std::move(candidate.lambda);
    dca_inherit_guard(w,from.point,after.point);outer=after.point;
    aggregate.accepted_rho=0;stats.aggregate_last_rho=0;
    ++stats.strict_finish_acceptances;if(after.strict)++stats.strict_finish_solved;
    stats.strict_finish_error_after=after.certificate.scaled_nlp_error;
    stats.strict_finish_last_rejection.clear();
    if(w.st.live_display_level>0)std::cout<<"[DCA-STRICT-FINISH] outer="<<iteration
        <<" decision=commit strict="<<after.strict<<" E_before="<<from.certificate.scaled_nlp_error
        <<" E_after="<<after.certificate.scaled_nlp_error<<" dual="<<after.certificate.termination_dual_inf
        <<" primal="<<after.certificate.feasibility.total_inf<<" compl="<<after.certificate.termination_compl_inf
        <<" alpha="<<candidate.ap<<" linear_residual="<<candidate.backward_error<<"\n";
    return true;
  }catch(...){rollback();throw;}
}
