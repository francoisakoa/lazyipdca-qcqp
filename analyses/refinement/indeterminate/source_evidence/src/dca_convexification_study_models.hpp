// Included inside DcaInequalityModels. No state is shared with historical R3.
  study::Correction study_objective;
  std::vector<study::Correction> study_inequalities;
  int study_accepted=0,study_builds=0,study_reuses=0;
  double study_seconds=0,study_uniform_rho=0;
  std::vector<std::pair<int,study::Correction>> study_trial_backup;
  std::vector<double> study_hybrid_priority;
  Vec study_last_displacement;
  bool study_trial_pending=false,study_hybrid_due=false,study_model_failure_due=false;
  std::string factorial_initial_metric_sha256,factorial_current_metric_sha256;
  std::uint64_t factorial_phase_metric_index=0;
  double study_recovery_original_theta=0;
  int study_rejections=0,study_trial_id=0,study_commits=0,study_rollbacks=0;
  bool study_hybrid_enabled() const {return study::support_study_mode(w.st.convexification_study);}

  bool study_r3_controller() const {return study_hybrid_enabled()&&w.st.study_controller=="r3";}
  bool study_hybrid_controller_enabled() const {return study_hybrid_enabled()&&!study_r3_controller();}

  bool study_enabled() const {return w.st.convexification_study!="off";}
  std::string study_family(bool objective) const {
    return study::correction_family(w.st.convexification_study,objective);
  }
  void study_trace(const char* event,const char* role,int index,const study::Correction& c) const {
    std::ostringstream out;out<<std::setprecision(17);
    out<<"[CONVEXIFICATION] {\"event\":\""<<event<<"\",\"configuration\":\""<<w.st.convexification_study
       <<"\",\"schedule\":\""<<w.st.convexification_schedule<<"\",\"role\":\""<<role
       <<"\",\"index\":"<<index<<",\"family\":\""<<c.family<<"\",\"rho\":"<<c.rho
       <<",\"atom_batches\":"<<c.atom_batches<<",\"atom_certificate_checks\":"<<c.atom_certificate_checks<<",\"atoms\":"<<c.atoms<<",\"rank_upper\":"<<c.rank_upper<<",\"residual_scalar\":"<<(c.support_scalar?c.rho:0)
       <<",\"scale\":"<<(index<0?w.objective.scale:w.inequalities[index].scale)<<",\"ambient_dimension\":"<<w.n<<",\"support_dimension\":"<<c.support.size()
       <<",\"certificate\":\""<<c.certificate<<"\",\"correction_psd\":true,\"model_psd\":true"
       <<",\"original_convex\":"<<(c.original_convex?"true":"false")
       <<",\"classification_known\":"<<(c.classification_known?"true":"false")
       <<",\"certificate_padding\":"<<c.padding<<",\"refined\":"<<(c.refined?"true":"false")
       <<",\"curvature_seconds_total\":"<<study_seconds;
    if(c.support.size()>16&&c.support.size()<=128&&study::hybrid_cached_interval_certificate(c,c.rho,w.st.study_hybrid_margin)) {
      const auto& proof=*c.implicit_interval_certificate;
      out<<",\"implicit_interval_psd\":{\"rho_threshold\":"<<proof.rho<<",\"precision_bits\":"<<proof.precision_bits
        <<",\"exact_Q_plus_beta_vv\":true,\"correction_psd_separate\":true,\"rounded_diagnostic_H_used\":false}";
    }
    if(c.sparse_psd_proof.precision_bits>0) {
      const auto& proof=c.sparse_psd_proof;
      out<<",\"sparse_interval_psd\":{\"certified\":"<<(proof.certified?"true":"false")
        <<",\"reason\":\""<<proof.reason<<"\",\"precision_bits\":"<<proof.precision_bits
        <<",\"positive_pivots\":"<<proof.positive_pivots<<",\"exact_zero_rows\":"<<proof.exact_zero_rows
        <<",\"updates\":"<<proof.updates<<",\"peak_entries\":"<<proof.peak_entries<<",\"seconds\":"<<proof.seconds;
      if(std::isfinite(proof.minimum_pivot_lower))out<<",\"minimum_pivot_lower\":"<<proof.minimum_pivot_lower;
      out<<",\"ordering\":\"symmetric-minimum-degree\",\"numeric_backend_shift\":0}";
    }
    if(c.support_scalar) {
      out<<",\"scalar_scope\":\"function_support\",\"scalar_support\":[";
      for(std::size_t k=0;k<c.support.size();++k){if(k)out<<',';out<<c.support[k];}out<<"],\"hybrid_atoms\":[";
      for(std::size_t k=0;k<c.hybrid_atoms.size();++k){if(k)out<<',';const auto& a=c.hybrid_atoms[k];out<<"{\"weight\":"<<a.weight<<",\"strikes\":"<<a.strikes<<",\"indices\":[";for(std::size_t j=0;j<a.indices.size();++j){if(j)out<<',';out<<a.indices[j];}out<<"],\"values\":[";for(std::size_t j=0;j<a.values.size();++j){if(j)out<<',';out<<a.values[j];}out<<"]}";}out<<']';
    }
    if(c.support.size()<=16) {
      auto matrix=[&](const char* key,const std::vector<SymTriplet>& h){out<<",\""<<key<<"\":[";bool first=true;for(auto e:h){if(!first)out<<',';first=false;out<<'['<<e.i<<','<<e.j<<','<<e.v<<']';}out<<']';};
      matrix("Q",c.original);matrix("C_atoms",c.correction);matrix("H",c.hessian);
    }
    out<<'}';std::cout<<out.str()<<std::endl;
  }
  void study_classify_large(study::Correction& c) {
    if(c.classification_known||c.support.size()<=16)return;
    Settings cert=w.factor_settings;
    cert.linear_backend=w.factor_settings.linear_backend=="ma27"?"ma27":"mumps";cert.linear_metis=true;
    cert.kkt_adaptive_regularization=false;cert.cholesk_adaptive_regularization=false;
    cert.kkt_primal_regularization=cert.kkt_dual_regularization=0;
    std::vector<SymmetricTriplet> matrix;
    for(auto e:c.original) {
      const int i=std::lower_bound(c.support.begin(),c.support.end(),e.i)-c.support.begin();
      const int j=std::lower_bound(c.support.begin(),c.support.end(),e.j)-c.support.begin();
      matrix.push_back({i,j,e.v});
    }
    std::string message;auto factor=make_symmetric_factorization(cert,message);
    if(!factor)throw std::runtime_error("study original curvature backend unavailable: "+message);
    const int n=c.support.size();const bool ok=factor->factor(n,n,matrix,message);
    if(ok&&factor->last_positive_pivots()==n&&factor->last_negative_pivots()==0) {
      c.rho=0;c.atoms=c.rank_upper=0;c.correction.clear();c.hessian=c.original;
      c.original_convex=c.classification_known=true;c.certificate=cert.linear_backend=="ma27"?"unregularized-MA27-positive-inertia":"unregularized-MUMPS-positive-inertia";
    } else if(factor->last_negative_pivots()>0) {
      c.original_convex=false;c.classification_known=true;
    } else throw std::runtime_error("study original curvature undecidable without regularization: "+message);
  }
  study::Correction study_construct(const Quadratic& q,bool objective,bool precise=false) {
    if(!q.signed_square.empty())throw std::runtime_error("study supports true QCQP only; native signed-square function unsupported");
    const auto start=Clock::now();
    auto c=study_hybrid_enabled()?(study_family(objective)=="hybrid"?
      study::construct_hybrid(w.n,q.hessian,w.st.study_hybrid_margin):
      study::construct_support_scalar(w.n,q.hessian,w.st.study_hybrid_margin)):
      study::construct(w.n,q.hessian,study_family(objective),precise);
    if(study_hybrid_enabled()&&c.rho>0){
      // construct_hybrid/construct_support_scalar already produced a
      // certified initial metric. Multiplying that installed metric by a
      // selection safety factor can make a previously feasible tangent
      // model infeasible. Safety remains a candidate-selection setting;
      // it must not add curvature after an exact certificate was obtained.
      if(!study::hybrid_certificate(c,c.rho,w.st.study_hybrid_margin))throw std::runtime_error("initial certified scalar model invalid");
    }
    if(!study_hybrid_enabled())study_classify_large(c);study_seconds+=seconds(start);++study_builds;
    return c;
  }
  void study_prepare() {
    if(curvature_ready){++study_reuses;return;}
    const bool atom_objective=study_family(true)=="atoms";
    if(!w.st.factorial_exact_objective())study_objective=study_construct(w.objective,true,atom_objective);
    study_inequalities.resize(w.mi);
    for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0)
      study_inequalities[i]=study_construct(w.inequalities[i],false,study_family(false)=="atoms");
    if(w.st.convexification_study=="U") {
      study_uniform_rho=study_objective.rho;
      for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0)study_uniform_rho=std::max(study_uniform_rho,study_inequalities[i].rho);
      auto uniform=[&](study::Correction& c){c.rho=study_uniform_rho;c.rank_upper=c.rho>0?w.n:0;c.certificate="common-max-of-certified-scalar-bounds";};
      uniform(study_objective);for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0)uniform(study_inequalities[i]);
    }
    if(!w.st.factorial_exact_objective())study_trace("construct","objective",-1,study_objective);
    for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0) {
      auto& cache=inequality[i];cache.ready=true;cache.scale=w.inequalities[i].scale;
      cache.rho=study_inequalities[i].rho;cache.support=study_inequalities[i].support;
      if(cache.rho>0||!study_inequalities[i].correction.empty())corrected_inequalities.push_back(i);
      study_trace("construct","inequality",i,study_inequalities[i]);
    }
    curvature_ready=true;objective_ready=true;
    if(w.st.factorial_refinement_active()) {
      factorial_initial_metric_sha256=factorial_current_metric_sha256=study_factorial_metric_sha256();
      CostScope observed(w.costs,NewtonCost::Diagnostics);
      factorial_phase_metric_index=detail::factorial_trace_metric_initial(w.st,
          factorial_initial_metric_sha256,w.n,w.mi,w.me);
    }
    // Eager runs request the same certified correction family before their
    // first model is installed. This remains a rollback-capable metric trial.
    if(study_hybrid_enabled()&&w.st.convexification_schedule=="eager"&&!w.st.factorial_refinement_frozen())study_hybrid_due=true;
    curvature_builds=study_builds;curvature_seconds=study_seconds;
    stats.majorant_curvature_certificates=study_builds;stats.majorant_curvature_seconds=study_seconds;
    stats.model_build_seconds+=study_seconds;
    stats.majorant_curvature_threads=1;
  }
  void study_install(Quadratic& q,const study::Correction& c) {
    if(c.support_scalar) {
      if(c.family=="scalar"&&!c.hybrid_atoms.empty())throw std::runtime_error("scalar role cannot install atoms");
      std::map<Pair,double> base;for(auto e:c.original)base[upper(e.i,e.j)]+=q.scale*e.v;
      for(int j:c.support)base[upper(j,j)]+=q.scale*c.rho;
      std::vector<SymTriplet> h;for(auto e:base)h.push_back({e.first.first,e.first.second,e.second});
      q.model_atoms.clear();q.model_atom_support.clear();q.set_model(center,std::move(h),slots);q.study_model_scalar=0;
      for(const auto& a:c.hybrid_atoms){LowRankTerm atom;atom.beta=q.scale*a.weight;atom.direction.assign(w.n,0);for(std::size_t j=0;j<a.indices.size();++j)atom.direction[a.indices[j]]=a.values[j];q.model_atoms.push_back(std::move(atom));}
      return;
    }
    std::vector<SymTriplet> h=c.hessian;for(auto& e:h)e.v*=q.scale;
    q.model_atoms.clear();q.model_atom_support.clear();q.set_model(center,std::move(h),slots);
    q.study_model_scalar=q.scale*c.rho;
  }
  void study_anchor_trace(const Quadratic& q,const char* role,int index) const {
    if(w.n>16)return;
    std::ostringstream out;out<<std::setprecision(17);
    out<<"[CONVEXIFICATION-ANCHOR] {\"role\":\""<<role<<"\",\"index\":"<<index
       <<",\"scale\":"<<q.scale<<",\"model_constant\":"<<q.model_constant<<",\"center\":[";
    for(std::size_t j=0;j<center.size();++j){if(j)out<<',';out<<center[j];}out<<"],\"support\":[";
    for(std::size_t j=0;j<q.support.size();++j){if(j)out<<',';out<<q.support[j];}out<<"],\"model_gradient\":[";
    for(std::size_t j=0;j<q.model_gradient.size();++j){if(j)out<<',';out<<q.model_gradient[j];}out<<"],\"H\":[";
    for(std::size_t j=0;j<q.model_hessian.size();++j){if(j)out<<',';const auto e=q.model_hessian[j];out<<'['<<e.i<<','<<e.j<<','<<e.v<<']';}
    out<<"],\"scaled_rho\":"<<q.study_model_scalar<<'}';std::cout<<out.str()<<std::endl;
  }
  void study_accepted_trace(int iteration,double alpha,double equality_proximal) {
    try {
      long double norm=0;for(int j=0;j<w.n;++j){const long double d=(long double)w.x[j]-center[j];norm+=d*d;}
      auto gap=[&](const study::Correction& c) {
        std::map<Pair,long double> actual;for(auto e:c.hessian)actual[upper(e.i,e.j)]+=e.v;for(auto e:c.original)actual[upper(e.i,e.j)]-=e.v;
        long double scalar_norm=norm;if(c.support_scalar){scalar_norm=0;for(int j:c.support){const long double d=(long double)w.x[j]-center[j];scalar_norm+=d*d;}}
        long double value=0.5L*c.rho*scalar_norm;
        if(c.support_scalar){
          for(const auto& a:c.hybrid_atoms){long double dot=0;for(std::size_t j=0;j<a.indices.size();++j)dot+=a.values[j]*((long double)w.x[a.indices[j]]-center[a.indices[j]]);value+=0.5L*a.weight*dot*dot;}
          return value;
        }
        for(auto e:actual){const long double di=(long double)w.x[e.first.first]-center[e.first.first],dj=(long double)w.x[e.first.second]-center[e.first.second];value+=(e.first.first==e.first.second?0.5L:1.0L)*e.second*di*dj;}
        return value;
      };
      if(study_hybrid_enabled())study_hybrid_measure_accepted();
      const long double objective=gap(study_objective);long double inequalities=0;
      for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0)inequalities=std::max(inequalities,gap(study_inequalities[i]));
      const double numeric_primal=w.current_factor?w.current_factor->last_primal_regularization():0;
      const double numeric_dual=w.current_factor?w.current_factor->last_dual_regularization():0;
      std::ostringstream out;out<<std::setprecision(17);
      out<<"[CONVEXIFICATION-ACCEPTED] {\"iteration\":"<<iteration<<",\"alpha\":"<<alpha;
      auto number=[&](const char* key,long double value){out<<",\""<<key<<"\":";if(std::isfinite(value))out<<value;else out<<"null";};
      number("objective_correction",objective);number("inequality_correction_max",inequalities);
      number("equality_linearization_proximal",equality_proximal);
      number("equality_proximal_step_value",0.5L*equality_proximal*norm/w.objective.scale);
      number("outer_primal_shift_total",w.regularization);
      number("outer_other_proximal",std::max(0.0,w.regularization-equality_proximal));
      number("backend_numeric_primal",numeric_primal);number("backend_numeric_dual",numeric_dual);
      out<<'}';std::cout<<out.str()<<std::endl;
    }catch(const std::exception& error){std::cout<<"[CONVEXIFICATION-ACCEPTED] diagnostic_unavailable=1 reason="<<error.what()<<std::endl;}
  }
  void study_reanchor() {
    if(!curvature_ready)study_prepare();
    study_factorial_snapshot();
    if(installed)throw std::runtime_error("study previous model still installed");
    center=w.x;
    if(!w.st.factorial_exact_objective())study_install(w.objective,study_objective);
    else if(w.objective.model_active||!w.objective.model_atoms.empty()||w.objective.study_model_scalar!=0)
      throw std::runtime_error("factorial exact objective model was modified");
    for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0)study_install(w.inequalities[i],study_inequalities[i]);
    if(!study_r3_controller())for(auto& q:w.equalities){q.model_atoms.clear();q.study_model_scalar=0;q.set_model(center,{},slots);}
    else for(auto& q:w.equalities)if(q.model_active)throw std::runtime_error("R3 controller equality must remain original");
    if(builds==0) {
      study_anchor_trace(w.objective,"objective",-1);
      for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0)study_anchor_trace(w.inequalities[i],"inequality",i);
      for(int e=0;e<w.me;++e)study_anchor_trace(w.equalities[e],"equality",e);
    }
    installed=true;++builds;++stats.models;study_reuses+=study_builds;
    objective_rho=study_objective.rho;objective_admitted=study_objective.atoms;
    rho_max=0;int atoms=0;for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0){rho_max=std::max(rho_max,study_inequalities[i].rho);atoms+=study_inequalities[i].atoms;}
    stats.rho_lagrangian=study_objective.rho;stats.rho_inequalities_max=rho_max;
    stats.atoms_lagrangian=study_objective.atoms;stats.atoms_inequalities=atoms;
    stats.majorant_curvature_reuses=study_reuses;stats.curvature_reuses=study_reuses;
    if(builds==1)std::cout<<"[CONVEXIFICATION-MODEL] configuration="<<w.st.convexification_study
       <<" schedule="<<w.st.convexification_schedule<<" equality_model="<<(study_r3_controller()?"original-nonlinear-equalities":"linearized")<<" all_original_inequalities_retained=1"
       <<" activation_lazy=0 correction_lazy="<<(w.st.convexification_schedule=="lazy")
       <<" ambient_scalar="<<(study_hybrid_enabled()?0:1)
       <<" scalar_scope="<<(study_hybrid_enabled()?"function_support":"ambient")
       <<" native_bounds_uncorrected=1 original_curvature_probe="<<w.st.factorial_exact_objective()<<std::endl;
  }
  void study_accepted_step() {
    ++study_accepted;
    if(w.st.factorial_refinement_frozen())return;
    if(study_hybrid_enabled()){
      study_rejections=0;
      if(study_accepted==1||study_accepted%std::max(1,w.st.study_hybrid_event_interval)==0)study_hybrid_due=true;
      return;
    }
    // Constant Hessians: refine the certificate once, not because x moved or
    // because a KKT factor failed. No forced rho->0 and no repeated oracle.
    if(study_accepted!=1||w.st.convexification_study=="U"||w.st.convexification_study=="F")return;
    auto refine=[&](Quadratic& q,study::Correction& c,bool objective,int i) {
      if(c.family!="scalar"||c.rho==0||c.support.size()>16)return;
      const double old=c.rho;auto proposed=study_construct(q,objective,true);
      if(proposed.rho<=old){c=std::move(proposed);study_trace("refine",objective?"objective":"inequality",i,c);}
    };
    refine(w.objective,study_objective,true,-1);
    for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0)refine(w.inequalities[i],study_inequalities[i],false,i);
    stats.majorant_curvature_seconds=study_seconds;stats.majorant_curvature_certificates=study_builds;
  }
  long double study_gap(int i,const Vec& x,std::vector<long double>* dual,double multiplier) const {
    const auto& c=study_inequalities[i];const auto& q=w.inequalities[i];long double gap=0;
    if(c.support_scalar) {
      for(int j:c.support){const long double d=(long double)x[j]-center[j];gap+=0.5L*q.scale*c.rho*d*d;if(dual)(*dual)[j]+=(long double)multiplier*q.scale*c.rho*d;}
      for(const auto& a:c.hybrid_atoms){long double dot=0;for(std::size_t j=0;j<a.indices.size();++j)dot+=a.values[j]*((long double)x[a.indices[j]]-center[a.indices[j]]);gap+=0.5L*q.scale*a.weight*dot*dot;if(dual)for(std::size_t j=0;j<a.indices.size();++j)(*dual)[a.indices[j]]+=(long double)multiplier*q.scale*a.weight*dot*a.values[j];}
      return gap;
    }
    for(std::size_t j=0;j<x.size();++j) {
      const long double d=(long double)x[j]-center[j];gap+=0.5L*q.scale*c.rho*d*d;
      if(dual)(*dual)[j]+=(long double)multiplier*q.scale*c.rho*d;
    }
    std::map<Pair,long double> actual;for(auto e:c.hessian)actual[upper(e.i,e.j)]+=e.v;for(auto e:c.original)actual[upper(e.i,e.j)]-=e.v;
    for(auto entry:actual) {
      const SymTriplet e{entry.first.first,entry.first.second,0};
      const long double a=(long double)x[e.i]-center[e.i],b=(long double)x[e.j]-center[e.j],v=q.scale*entry.second;
      gap+=(e.i==e.j?0.5L:1.0L)*v*a*b;
      if(dual){(*dual)[e.i]+=multiplier*v*b;if(e.i!=e.j)(*dual)[e.j]+=multiplier*v*a;}
    }
    return gap;
  }

  study::Correction& study_hybrid_metric(int i){return i<0?study_objective:study_inequalities[i];}
  // Hash the mathematical inequality corrections, not mutable oracle caches
  // or atom activity counters. This has no effect on the numerical models.
  // Hash once per construction/change and at finalization; reanchoring only
  // reuses the digest, so diagnostic work cannot become O(Hessian nnz) at
  // every Newton step. Every metric writer below is guarded in frozen mode.
  std::string study_factorial_metric_sha256() const {
    CostScope observed(w.costs,NewtonCost::Diagnostics);
    detail::FactorialSHA256 hash;hash.text("hybrid-inequality-metric-binary64-v1");
    hash.word(w.n);hash.word(w.mi);hash.word(w.me);
    auto matrix=[&](const std::vector<SymTriplet>& a) {
      hash.word(a.size());for(const auto& e:a){hash.word(e.i);hash.word(e.j);hash.word(1);hash.number(e.v);}
    };
    for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0) {
      if(std::size_t(i)>=study_inequalities.size())throw std::runtime_error("unconstructed ablation inequality metric");
      const auto& c=study_inequalities[i];hash.word(i);hash.text(c.family);hash.word(c.support_scalar);
      hash.word(2);hash.number(w.inequalities[i].scale);hash.number(c.rho);hash.word(c.support.size());for(int j:c.support)hash.word(j);
      matrix(c.original);matrix(c.correction);hash.word(c.hybrid_atoms.size());
      for(const auto& atom:c.hybrid_atoms){hash.word(1);hash.number(atom.weight);hash.word(atom.indices.size());for(int j:atom.indices)hash.word(j);hash.vector(atom.values);}
    }
    if(w.st.factorial_trace_state)++w.st.factorial_trace_state->metric_hash_evaluations;
    return hash.finish();
  }
  void study_factorial_snapshot(bool metric_changed=false) {
    if(!w.st.factorial_refinement_active()||!curvature_ready)return;
    CostScope observed(w.costs,NewtonCost::Diagnostics);
    if(metric_changed)factorial_current_metric_sha256=study_factorial_metric_sha256();
    detail::factorial_trace_metric_current(w.st,factorial_phase_metric_index,
        factorial_initial_metric_sha256,factorial_current_metric_sha256);
  }
  void study_factorial_final_check() const {
    if(!w.st.factorial_refinement_active()||!curvature_ready)return;
    try {
      const auto digest=study_factorial_metric_sha256();
      detail::factorial_trace_metric_current(w.st,factorial_phase_metric_index,
          factorial_initial_metric_sha256,digest);
    }catch(const std::exception& error) {
      if(w.st.factorial_trace_state){auto& s=*w.st.factorial_trace_state;s.success=false;
        s.frozen_metric_invariant_verified=false;s.error=error.what();}
    }
  }
  std::string study_metric_digest() const {
    std::ostringstream repr;repr<<std::hexfloat;
    auto add=[&](const study::Correction& c){repr<<c.family<<':'<<c.support_scalar<<':'<<c.rho<<';';for(const auto& a:c.hybrid_atoms){repr<<a.weight<<':'<<a.strikes<<':';for(std::size_t j=0;j<a.indices.size();++j)repr<<a.indices[j]<<','<<a.values[j]<<';';}};
    add(study_objective);
    for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0) {
      // A first lazy construction may throw before this storage exists. A
      // diagnostic destructor must preserve that exception, not index past
      // the empty vector and terminate the process.
      if(std::size_t(i)<study_inequalities.size())add(study_inequalities[i]);
      else repr<<"unconstructed-inequality:"<<i<<';';
    }
    std::uint64_t h=1469598103934665603ULL;for(unsigned char c:repr.str()){h^=c;h*=1099511628211ULL;}
    std::ostringstream out;out<<std::hex<<h;return out.str();
  }
  std::string study_trial_accepted_digest;
  void study_controller_trace(const char* event,const char* reason,int changed=0) const {
    const auto current=study_metric_digest();
    std::cout<<"[HYBRID-CONTROLLER] {\"event\":\""<<event<<"\",\"reason\":\""<<reason
      <<"\",\"configuration\":\""<<w.st.convexification_study
      <<"\",\"objective_family\":\""<<study_family(true)<<"\",\"inequality_family\":\""<<study_family(false)
      <<"\",\"accepted_steps\":"<<study_accepted<<",\"rejection_count\":"<<study_rejections
      <<",\"trial_id\":"<<study_trial_id<<",\"changed_functions\":"<<changed
      <<",\"metrics_commit_count\":"<<study_commits<<",\"rollback_count\":"<<study_rollbacks
      <<",\"pending_trial\":"<<(study_trial_pending?"true":"false")
      <<",\"accepted_metric_digest\":\""<<(study_trial_pending?study_trial_accepted_digest:current)
      <<"\",\"current_metric_digest\":\""<<current<<"\"}"<<std::endl;
  }
  void study_hybrid_measure_accepted() {
    const double before=w.objective.original_at(center)/w.objective.scale,after=w.objective.original_at(w.x)/w.objective.scale;
    study_last_displacement.resize(w.n);for(int j=0;j<w.n;++j)study_last_displacement[j]=w.x[j]-center[j];
    study_hybrid_priority.assign(w.mi+1,0);
    auto measure=[&](study::Correction& c,double scale,double multiplier,int index){
      double maximum=0;
      int atom_index=0;
      for(auto& atom:c.hybrid_atoms){
        long double dot=0;for(std::size_t j=0;j<atom.indices.size();++j)dot+=atom.values[j]*study_last_displacement[atom.indices[j]];
        const auto activity=study::hybrid_downweight_activity(atom.weight,double(dot),scale,multiplier,before,after,
          w.st.study_hybrid_objective_floor,w.st.study_hybrid_dominance);
        maximum=std::max(maximum,activity.penalty);atom.strikes=activity.active?atom.strikes+1:0;
        std::cout<<std::setprecision(17)<<"[HYBRID-DOWNWEIGHT] {\"role\":\""<<(index<0?"objective":"inequality")
          <<"\",\"index\":"<<index<<",\"atom\":"<<atom_index++<<",\"objective_before\":"<<before<<",\"objective_after\":"<<after
          <<",\"objective_floor\":"<<w.st.study_hybrid_objective_floor<<",\"denominator\":"<<activity.denominator
          <<",\"beta\":"<<atom.weight<<",\"projection\":"<<double(dot)<<",\"scale\":"<<scale<<",\"multiplier\":"<<multiplier
          <<",\"penalty\":"<<activity.penalty<<",\"dominance\":"<<w.st.study_hybrid_dominance<<",\"active\":"<<(activity.active?"true":"false")
          <<",\"strikes\":"<<atom.strikes<<",\"patience\":"<<w.st.study_hybrid_downweight_patience
          <<",\"objective_units\":\"original_unscaled\",\"criterion\":\""<<(index<0?"paper-objective-decrease":"QCQP-positive-dual-weighted-penalty-objective-decrease")<<"\"}"<<std::endl;
      }
      study_hybrid_priority[index+1]=maximum+std::max(0.0,multiplier)*scale*c.rho;
    };
    measure(study_objective,1,1,-1);
    for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0)measure(study_inequalities[i],w.inequalities[i].scale/w.objective.scale,w.z[i],i);
  }
  study::HybridSelectionSettings study_selector_settings() const {
    study::HybridSelectionSettings s;
    s.candidates=w.st.study_hybrid_candidates;s.lanczos_steps=w.st.study_hybrid_lanczos_steps;s.lanczos_starts=w.st.study_hybrid_lanczos_starts;
    s.additions=w.st.study_hybrid_additions_per_event;s.support_energy=w.st.study_hybrid_support_energy;
    s.strength=w.st.study_hybrid_strength;s.beta_offset=w.st.study_hybrid_beta_offset;s.negative_tolerance=w.st.study_hybrid_negative_tolerance;
    s.scalar_safety=w.st.study_hybrid_scalar_safety;s.scalar_floor=w.st.study_hybrid_scalar_floor;
    s.score_activity=w.st.study_hybrid_score_activity;s.score_scalar=w.st.study_hybrid_score_scalar;
    s.score_support=w.st.study_hybrid_score_support;s.score_density=w.st.study_hybrid_score_density;s.score_weight=w.st.study_hybrid_score_weight;
    s.min_lift=w.st.study_hybrid_min_lift;s.min_activity=w.st.study_hybrid_min_activity;s.min_score=w.st.study_hybrid_min_score;return s;
  }
  void study_selection_trace(const study::HybridSelectionRecord& r,const study::HybridSelectionSettings& s,int index) const {
    std::cout<<std::setprecision(17)<<"[HYBRID-SELECTION] {\"role\":\""<<(index<0?"objective":"inequality")<<"\",\"index\":"<<index
      <<",\"candidate_basis\":\"two-pass-MGS-recomputed-implicit-Rayleigh\",\"pool_raw\":"<<r.pool_raw
      <<",\"pool_orthogonalized\":"<<r.pool_orthogonalized<<",\"pool_dependent_dropped\":"<<r.pool_dependent_dropped
      <<",\"pool_nonnegative_dropped\":"<<r.pool_nonnegative_dropped
      <<",\"trial_id\":"<<study_trial_id<<",\"candidate\":"<<r.candidate<<",\"greedy_iteration\":"<<r.greedy_iteration
      <<",\"lambda0\":"<<r.lambda0<<",\"lambda_plus\":"<<r.lambda_plus<<",\"rayleigh\":"<<r.rayleigh<<",\"beta\":"<<r.beta
      <<",\"alpha\":"<<r.alpha<<",\"chi\":"<<r.chi<<",\"nnz\":"<<r.nnz<<",\"n\":"<<r.n<<",\"nfree\":"<<r.nfree
      <<",\"support_cap\":"<<r.support_cap<<",\"rho_before\":"<<r.rho_before<<",\"rho_limit\":"<<r.rho_limit
      <<",\"l1_squared\":"<<r.l1_squared<<",\"retained_energy\":"<<r.retained_energy<<",\"lift\":"<<r.lift
      <<",\"theta\":"<<r.theta<<",\"sparsity\":"<<r.sparsity<<",\"density\":"<<r.density<<",\"rho_scalar\":"<<r.rho_scalar
      <<",\"residual_target\":"<<r.residual_target<<",\"certified_residual\":"<<r.certified_residual<<",\"installed_residual\":"<<r.installed_residual
      <<",\"score\":"<<r.score<<",\"evaluated\":"<<(r.evaluated?"true":"false")<<",\"admissible\":"<<(r.admissible?"true":"false")<<",\"selected\":"<<(r.selected?"true":"false")
      <<",\"certificate\":"<<(r.certificate?"true":"false")<<",\"redundant\":"<<(r.redundant?"true":"false")
      <<",\"energy_cap_rejected\":"<<(r.energy_cap_rejected?"true":"false")
      <<",\"screened_before_certificate\":"<<(r.screened_before_certificate?"true":"false")
      <<",\"negative_tolerance\":"<<s.negative_tolerance<<",\"scalar_safety\":"<<s.scalar_safety<<",\"scalar_floor\":"<<s.scalar_floor
      <<",\"strength\":"<<s.strength<<",\"beta_offset\":"<<s.beta_offset<<",\"support_energy\":"<<s.support_energy
      <<",\"residual_installation_policy\":\"certified-no-artificial-floor\""
      <<",\"score_weights\":["<<s.score_activity<<','<<s.score_scalar<<','<<s.score_support<<','<<s.score_density<<','<<s.score_weight
      <<"],\"thresholds\":["<<s.min_lift<<','<<s.min_activity<<','<<s.min_score<<"],\"spectral_mode\":\"fullspace-estimate-selection-only\"}"<<std::endl;
  }
  bool study_begin_trial(const char* reason="periodic") {
    if(w.st.factorial_refinement_frozen()){study_factorial_snapshot();return false;}
    if(!study_hybrid_enabled()||!curvature_ready||study_trial_pending)return false;
    if(installed)throw std::runtime_error("hybrid trial requires discarded model");
    if(!study_hybrid_due&&study_rejections<2)return false;
    if(w.st.factorial_refinement_active()&&w.st.factorial_trace_state)
      ++w.st.factorial_trace_state->refinement_attempt_events;
    const bool model_recovery=study_model_failure_due;
    study_model_failure_due=false;
    study_hybrid_due=false;study_trial_accepted_digest=study_metric_digest();++study_trial_id;
    int total=study_objective.hybrid_atoms.size();for(const auto& c:study_inequalities)total+=c.hybrid_atoms.size();
    std::vector<int> order;for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0&&!study_inequalities[i].original_convex)order.push_back(i);
    auto priority=[&](int i){if(model_recovery)return std::max(0.0,w.inequalities[i].original_at(w.x)/w.inequalities[i].scale);return study_hybrid_priority.size()==std::size_t(w.mi+1)?study_hybrid_priority[i+1]:std::max(0.0,w.z[i])*w.inequalities[i].scale*study_inequalities[i].rho;};
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return priority(a)>priority(b);});
    const bool initial_eager=builds==0&&w.st.convexification_schedule=="eager";
    // Recovery must be able to modify a coupled collection of violated
    // functions; the normal periodic event still examines at most four rows.
    if(!initial_eager&&!model_recovery&&order.size()>4)order.resize(4);
    if(!w.st.factorial_exact_objective())order.insert(order.begin(),-1);
    for(int index:order) {
      auto& current=study_hybrid_metric(index);
      if(current.original_convex)continue;
      if(study::correction_cancels_original(current)) {
        study_trace("minimal_correction_preserved",index<0?"objective":"inequality",index,current);
        continue;
      }
      auto candidate=current;bool changed=false;const char* event="scalar_refine";
      const auto start=Clock::now();
      if(model_recovery&&candidate.family=="hybrid"&&total<w.st.study_hybrid_total_atoms) {
        study::HybridRecoveryRecord record;
        const int limit=std::min(w.st.study_hybrid_max_atoms,int(candidate.hybrid_atoms.size())+w.st.study_hybrid_total_atoms-total);
        changed=study::hybrid_recover_negative_subspace(candidate,w.n,limit,w.st.study_hybrid_atom_support,
          study_selector_settings(),w.st.study_hybrid_margin,record)>0;
        if(changed)event="model_recovery_enrich";
        std::cout<<std::setprecision(17)<<"[HYBRID-MODEL-RECOVERY] {\"role\":\""<<(index<0?"objective":"inequality")
          <<"\",\"index\":"<<index<<",\"strategy\":\"negative-subspace-batch\",\"original_theta\":"<<study_recovery_original_theta
          <<",\"candidate_basis\":\"two-pass-MGS-recomputed-implicit-Rayleigh\",\"pool_raw\":"<<record.pool_raw
          <<",\"pool_orthogonalized\":"<<record.pool_orthogonalized<<",\"pool_dependent_dropped\":"<<record.pool_dependent_dropped
          <<",\"pool_nonnegative_dropped\":"<<record.pool_nonnegative_dropped
          <<",\"feasibility_tolerance\":"<<w.st.feasibility_tolerance
          <<",\"negative_candidates\":"<<record.negative_candidates<<",\"added\":"<<record.added
          <<",\"rho_before\":"<<record.rho_before<<",\"scalar_baseline\":"<<record.scalar_baseline<<",\"rho_after\":"<<record.rho_after
          <<",\"certificate\":"<<(record.certificate?"true":"false")<<",\"proposed\":"<<(changed?"true":"false")
          <<",\"accepted_original_step\":false}"<<std::endl;
      }
      for(std::size_t k=0;!changed&&k<candidate.hybrid_atoms.size();++k)if(candidate.hybrid_atoms[k].strikes>=w.st.study_hybrid_downweight_patience){
        // Test complete withdrawal before attenuation. With unchanged rho,
        // C_old-C_new is exactly a nonnegative rank-one term; installation
        // still requires the independent function PSD checks below.
        if(study::hybrid_downweight(candidate,w.n,k,0,w.st.study_hybrid_margin)){changed=true;event="remove_atom";break;}
        if(study::hybrid_downweight(candidate,w.n,k,w.st.study_hybrid_downweight_factor,w.st.study_hybrid_margin)){changed=true;event="downweight";break;}
        // The candidate object is unchanged on failure. Report the refused
        // attempt separately, never as a certified proposed metric.
        study_trace("downweight_rejected",index<0?"objective":"inequality",index,candidate);
        study_controller_trace("downweight_rejected","PSD-certificate-blocked");
      }
      if(!changed&&candidate.family=="hybrid"&&total<w.st.study_hybrid_total_atoms) {
        study::HybridSelectionContext context;context.ambient=w.n;context.gradient.assign(w.n,0);context.last_displacement=study_last_displacement;
        (index<0?w.objective:w.inequalities[index]).original_gradient_at(w.x,1,context.gradient);
        const auto selector=study_selector_settings();
        const int limit=std::min(w.st.study_hybrid_max_atoms,int(candidate.hybrid_atoms.size())+w.st.study_hybrid_total_atoms-total);
        if(study::hybrid_select_enrich(candidate,w.n,limit,w.st.study_hybrid_atom_support,selector,context,w.st.study_hybrid_margin,
          [&](const study::HybridSelectionRecord& record){study_selection_trace(record,selector,index);})>0){changed=true;event="enrich";}
        if(!changed&&!model_recovery) {
          study::HybridRecoveryRecord record;
          changed=study::hybrid_recover_negative_subspace(candidate,w.n,limit,w.st.study_hybrid_atom_support,
            selector,w.st.study_hybrid_margin,record,&context)>0;
          if(changed)event="spectral_cluster_enrich";
          std::cout<<std::setprecision(17)<<"[HYBRID-BATCH-SELECTION] {\"role\":\""<<(index<0?"objective":"inequality")
            <<"\",\"index\":"<<index<<",\"strategy\":\"negative-subspace-composite-batch\",\"lambda0\":"<<record.lambda0
            <<",\"candidate_basis\":\"two-pass-MGS-recomputed-implicit-Rayleigh\",\"pool_raw\":"<<record.pool_raw
            <<",\"pool_orthogonalized\":"<<record.pool_orthogonalized<<",\"pool_dependent_dropped\":"<<record.pool_dependent_dropped
            <<",\"pool_nonnegative_dropped\":"<<record.pool_nonnegative_dropped
            <<",\"lambda_plus\":"<<record.lambda_plus<<",\"lift\":"<<record.lift<<",\"alpha\":"<<record.alpha
            <<",\"chi\":"<<record.chi<<",\"theta\":"<<record.theta<<",\"score\":"<<record.score
            <<",\"support_cost\":"<<record.support_cost<<",\"density_cost\":"<<record.density_cost<<",\"weight_cost\":"<<record.weight_cost
            <<",\"n\":"<<w.n<<",\"negative_tolerance\":"<<selector.negative_tolerance<<",\"scalar_safety\":"<<selector.scalar_safety
            <<",\"score_weights\":["<<selector.score_activity<<','<<selector.score_scalar<<','<<selector.score_support<<','<<selector.score_density<<','<<selector.score_weight
            <<"],\"thresholds\":["<<selector.min_lift<<','<<selector.min_activity<<','<<selector.min_score<<']'
            <<",\"spectral_mode\":\"fullspace-estimate-selection-only\""
            <<",\"added\":"<<record.added<<",\"rho_before\":"<<record.rho_before<<",\"scalar_baseline\":"<<record.scalar_baseline
            <<",\"rho_after\":"<<record.rho_after<<",\"screened_before_certificate\":"<<(record.selection_screened?"true":"false")
            <<",\"certificate\":"<<(record.certificate?"true":"false")<<",\"proposed\":"<<(changed?"true":"false")
            <<",\"accepted_original_step\":false}"<<std::endl;
        }
      }
      if(!changed&&candidate.rho>0) {
        const double refined=study::hybrid_certified_residual(candidate,candidate.rho,w.st.study_hybrid_margin);
        const double proposed=std::max(refined,candidate.rho*w.st.study_hybrid_scalar_decay);
        if(proposed<candidate.rho*(1-1e-10)){candidate.rho=proposed;candidate.refined=true;candidate.certificate=candidate.family=="hybrid"?"hybrid-certified-residual-refinement":"support-scalar-certified-refinement";changed=true;}
      }
      // Lowering rho alone is not an improvement if new atoms restore
      // exactly the removed scalar matrix. Preserve the atom budget for
      // functions whose effective curvature can actually change.
      if(changed&&study::same_effective_correction(current,candidate)) {
        changed=false;
        study_trace("unchanged_effective_correction_rejected",index<0?"objective":"inequality",index,candidate);
        study_controller_trace("no_metric_change","exact-same-correction-atom-budget-preserved");
      }
      study_seconds+=seconds(start);
      if(changed){
        study_trial_backup.push_back({index,current});study_trial_pending=true;total+=int(candidate.hybrid_atoms.size())-int(current.hybrid_atoms.size());current=std::move(candidate);++study_builds;
        study_trace(event,index<0?"objective":"inequality",index,current);
      }
    }
    study_trial_pending=!study_trial_backup.empty();
    stats.majorant_curvature_seconds=study_seconds;stats.majorant_curvature_certificates=study_builds;
    curvature_builds=study_builds;curvature_seconds=study_seconds;
    study_controller_trace(study_trial_pending?"trial":"no_change",model_recovery?"infeasible-anchor-model-nonconvergence":reason,study_trial_backup.size());
    return study_trial_pending;
  }
  void study_commit_trial() {
    if(!study_hybrid_enabled()||!study_trial_pending)return;
    if(w.st.factorial_refinement_frozen())throw std::runtime_error("frozen refinement attempted a metric commit");
    const int changed=study_trial_backup.size();study_trial_backup.clear();study_trial_pending=false;++study_commits;
    if(w.st.factorial_refinement_active()&&w.st.factorial_trace_state)++w.st.factorial_trace_state->accepted_refinements;
    study_factorial_snapshot(true);
    study_controller_trace("commit","original-filter-accepted",changed);
  }
  void study_rollback_trial() {
    if(!study_hybrid_enabled()||!study_trial_pending)return;
    if(w.st.factorial_refinement_frozen())throw std::runtime_error("frozen refinement attempted a metric rollback");
    if(installed)throw std::runtime_error("hybrid rollback requires discarded model");
    const int changed=study_trial_backup.size();for(auto& old:study_trial_backup)study_hybrid_metric(old.first)=std::move(old.second);
    study_trial_backup.clear();study_trial_pending=false;++study_rollbacks;
    if(w.st.factorial_refinement_active()&&w.st.factorial_trace_state)++w.st.factorial_trace_state->refinement_rollback_events;
    study_factorial_snapshot(true);
    study_controller_trace("rollback","original-step-rejected",changed);
  }
  void study_note_rejection(const char* reason) {
    if(w.st.factorial_refinement_frozen())return;
    if(!study_hybrid_enabled())return;
    ++study_rejections;if(study_rejections>=2)study_hybrid_due=true;
    if(std::string(reason)=="inner-criterion") {
      // A failed subproblem is not proof of infeasibility. At an infeasible
      // ORIGINAL anchor it is, however, a valid demand for a different
      // certified function metric. Do not wait for a first accepted step.
      double theta=0;
      for(const auto& q:w.equalities)theta=std::max(theta,std::abs(q.original_at(w.x)/q.scale));
      for(const auto& q:w.inequalities)theta=std::max(theta,std::max(0.0,q.original_at(w.x)/q.scale));
      if(std::isfinite(theta)&&theta>w.st.feasibility_tolerance) {
        study_recovery_original_theta=theta;study_model_failure_due=true;study_hybrid_due=true;
      }
    }
    study_controller_trace("rejection",reason);
  }
  void study_summary() const {
    study_factorial_final_check();
    if(!study_enabled())return;
    if(study_hybrid_enabled())study_controller_trace("summary","finished");
    if(study_hybrid_enabled()) {
      std::uint64_t hits=0,misses=0,evictions=0,inconclusive=0,downweight=0,gram_hits=0,gram_misses=0,refusals_invalidated=0,spectrum_hits=0,spectrum_misses=0;
      std::size_t bytes=0,entries=0,spectrum_entries=0,spectrum_bytes=0;double lookup_seconds=0,certificate_seconds=0,spectrum_seconds=0;
      auto count=[&](const study::Correction& c) {
        if(!c.certificate_memo)return;const auto& m=*c.certificate_memo;
        hits+=m.hits;misses+=m.misses;evictions+=m.evictions;inconclusive+=m.inconclusive;
        downweight+=m.downweight_hits;gram_hits+=m.gram_hits;gram_misses+=m.gram_misses;
        bytes+=m.bytes;entries+=m.entries.size();lookup_seconds+=m.lookup_seconds;certificate_seconds+=m.certificate_seconds;
        refusals_invalidated+=m.refusals_invalidated;spectrum_hits+=m.spectrum_hits;spectrum_misses+=m.spectrum_misses;
        spectrum_entries+=m.spectrum_entries;spectrum_bytes+=m.spectrum_bytes;spectrum_seconds+=m.spectrum_seconds;
      };
      count(study_objective);for(const auto& c:study_inequalities)count(c);
      std::cout<<"[CURVATURE-CACHE] {\"configuration\":\""<<w.st.convexification_study
        <<"\",\"hits\":"<<hits<<",\"misses\":"<<misses<<",\"evictions\":"<<evictions
        <<",\"inconclusive_not_cached\":"<<inconclusive<<",\"rejected_downweight_hits\":"<<downweight
        <<",\"gram_hits\":"<<gram_hits<<",\"gram_misses\":"<<gram_misses
        <<",\"refusals_invalidated\":"<<refusals_invalidated<<",\"spectrum_hits\":"<<spectrum_hits<<",\"spectrum_misses\":"<<spectrum_misses
        <<",\"spectrum_entries\":"<<spectrum_entries<<",\"spectrum_bytes\":"<<spectrum_bytes<<",\"cold_spectrum_seconds\":"<<spectrum_seconds
        <<",\"entries\":"<<entries<<",\"accounted_bytes\":"<<bytes
        <<",\"lookup_seconds\":"<<std::setprecision(17)<<lookup_seconds<<",\"uncached_certificate_seconds\":"<<certificate_seconds
        <<",\"key\":\"exact-function-and-metric-bytes\",\"budget_failures_reused\":false}"<<std::endl;
    }
    std::cout<<"[CONVEXIFICATION-SUMMARY] models="<<builds<<" curvature_constructed="<<study_builds
      <<" curvature_reused="<<study_reuses<<" original_functions_never_requested="
      <<(curvature_ready?0:1+std::count(w.inequality_kind.begin(),w.inequality_kind.end(),0))
      <<" curvature_seconds="<<std::setprecision(17)<<study_seconds<<std::endl;
  }
