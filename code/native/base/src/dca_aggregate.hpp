// Included inside the filter engine's anonymous namespace, after DcaModels.
// Aggregate convex primal-dual quadratic model with one inertia certificate.
struct DcaAggregate {
  struct Candidate {
    Vec x,s,z,lambda;
    DcaPoint point;
    double ap=0,al=0,az=0,rho=0,backward_error=INFINITY,sigma=0;
    std::string stage;
    bool soc=false,curvature_descent=false;
  };
  // Lifetime: one attempt() / one frozen primal-dual anchor only. Values
  // and geometry in Workspace remain at that anchor during every rho probe;
  // candidate evaluations use original_at/original_gradient_at (const).
  // Store an exact key as a defensive invariant, not a hash of x alone.
  struct Preparation {
    bool ready=false;
    const Workspace* owner=nullptr;
    int nr=0,mr=0,dimension=0;
    Vec x,s,z,lambda,hessian;
    std::vector<unsigned char> active;
    std::vector<int> weak_border;
    std::vector<SymmetricTriplet> unshifted;
    bool matches(const Workspace& w) const {
      return ready && owner==&w && nr==w.nr && mr==w.mr &&
          x==w.x && s==w.s && z==w.z && lambda==w.lambda && active==w.active &&
          !w.objective.model_active &&
          std::none_of(w.equalities.begin(),w.equalities.end(),[](const Quadratic& q){return q.model_active;}) &&
          std::none_of(w.inequalities.begin(),w.inequalities.end(),[](const Quadratic& q){return q.model_active;});
    }
    std::uint64_t bytes() const {
      return (x.size()+s.size()+z.size()+lambda.size()+hessian.size())*sizeof(double)+
          active.size()*sizeof(unsigned char)+weak_border.size()*sizeof(int)+
          unshifted.size()*sizeof(SymmetricTriplet);
    }
  };
  Settings stop_settings;
  Settings settings;
  std::unique_ptr<SymmetricFactorization> factor;
  const char* last_rejection="not-attempted";
  detail::DcaAggregateRejection last_rejection_kind=detail::DcaAggregateRejection::none;
  detail::DcaPositiveRhoMemory positive_rho_memory;
  detail::DcaStableBorderSlots stable_borders;
  // Keep numerical certificate memory separate from accepted-iterate memory.
  // The former historically warmed the next inertia search even when a line
  // step was rejected; conflating both made the R7D7 transaction alter the
  // entire pre-feasible trajectory.
  double certified_rho=0,accepted_rho=0;
  double last_candidate_kkt=INFINITY,last_candidate_primal=INFINITY;
  double last_backward_error=INFINITY,last_attempt_rho=INFINITY;
  double last_curvature_rho=INFINITY;
  bool last_attempt_certified=false;
  detail::DcaStationarityEvidence stationarity_evidence;
  // A certified aggregate factor can precondition the ORIGINAL root without
  // rebuilding a partial-Lagrangian majorant or its atom bundle. The layouts
  // differ (dense and numerically lifted inequalities), so copy by row IDs,
  // never by an assumed common prefix beyond the primal/equality block.
  bool root_factor_valid=false;
  int root_dimension=0;
  std::vector<int> root_border;
  Vec root_x,root_s,root_z,root_lambda;
  std::vector<unsigned char> root_active;
  void capture_root_factor(const Workspace& w,int dimension,const std::vector<int>& borders,
                           const DcaPoint& outer) {
    root_factor_valid=detail::dca_stationary_phase(outer.primal,outer.normalized_kkt,
        w.st.feasibility_tolerance,w.st.stationarity_tolerance);
    if(!root_factor_valid)return;
    root_dimension=dimension;root_border=borders;
    root_x=w.x;root_s=w.s;root_z=w.z;root_lambda=w.lambda;root_active=w.active;
  }
  bool root_factor_compatible(const Workspace& w)const {
    return root_factor_valid && factor && root_x==w.x && root_s==w.s &&
        root_z==w.z && root_lambda==w.lambda && root_active==w.active;
  }
  bool root_precondition(const Workspace& w,const std::vector<int>& active,
                         const std::vector<int>& border,const Vec& b,Vec& out,
                         std::string& message) const {
    // Compatibility is checked ONCE at entry to the transaction; all its
    // operator/preconditioner calls leave the frozen anchor unchanged.
    if(!root_factor_valid||!factor){message="aggregate-root-factor-unavailable";return false;}
    const int base=w.nr+w.mr;
    if(int(b.size())<base || active.size()!=border.size()){
      message="aggregate-root-layout-dimension";return false;
    }
    Vec rhs(root_dimension,0);std::copy_n(b.begin(),base,rhs.begin());
    for(std::size_t k=0;k<active.size();++k){int original=active[k];
      if((root_border[original]>=0)!=(border[k]>=0)){
        message="aggregate-root-border-classification-mismatch";return false;
      }
      if(border[k]>=0)rhs[root_border[original]]=b[border[k]];
    }
    const bool solved=factor->solve(rhs,message);
    if(w.dca_stats)w.dca_stats->aggregate_backend_solve_seconds=factor->linear_solve_seconds();
    if(!solved||!finite(rhs))return false;
    out.assign(b.size(),0);std::copy_n(rhs.begin(),base,out.begin());
    // Inactive structural dense rows in the target are independent -1 pivots;
    // they do not exist in the smaller aggregate layout.
    for(int i=0;i<w.mi;++i)if(!w.active[i]&&w.jr[i].dense>=0){
      int row=base+w.jr[i].dense;out[row]=-b[row];
    }
    for(std::size_t k=0;k<active.size();++k)if(border[k]>=0)
      out[border[k]]=rhs[root_border[active[k]]];
    return finite(out);
  }
  explicit DcaAggregate(const Settings& s,const Settings* stop=nullptr):stop_settings(stop?*stop:s),settings(s) {
    if(!s.dca_aggregate)return;
    if(s.dca_aggregate_kkt_layout!="persistent" && s.dca_aggregate_kkt_layout!="dynamic")
      throw std::invalid_argument("dca aggregate KKT layout must be persistent or dynamic");
    if(!(s.dca_aggregate_border_reserve>=1 && s.dca_aggregate_border_reserve<=4))
      throw std::invalid_argument("dca aggregate border reserve must be in [1,4]");
    if(s.dca_aggregate_rejection_policy!="separated" && s.dca_aggregate_rejection_policy!="legacy")
      throw std::invalid_argument("dca aggregate rejection policy must be separated or legacy");
    settings.kkt_adaptive_regularization=true;
    settings.kkt_primal_regularization=settings.kkt_dual_regularization=0;
    settings.kkt_interleaved_equality_ordering=true;
    settings.kkt_primal_first_ordering=false;
    settings.kkt_preserve_structural_zeros=true;
    std::string message;
    if(s.dca_aggregate_backend=="auto"){
      settings.linear_backend="mumps";
      factor=make_symmetric_factorization(settings,message);
      if(factor)return;
      settings.linear_backend=s.linear_backend;
    }else settings.linear_backend=s.dca_aggregate_backend;
    factor=make_symmetric_factorization(settings,message);
  }
  void prepare(Workspace& w,DcaFilterStatistics& stats,Preparation& prepared,
               bool reuse,int outer_iteration) {
    detail::DcaTimedScope timer(stats.aggregate_preparation_seconds);
    ++stats.aggregate_preparation_requests;
    if(reuse && prepared.ready){
      const auto key_start=Clock::now();
      const bool hit=prepared.matches(w);
      stats.aggregate_preparation_key_seconds+=seconds(key_start);
      if(hit){
        ++stats.aggregate_preparation_hits;
        if(w.st.live_display_level>=3)std::cout<<"[DCA-AGGREGATE-CACHE] outer="<<outer_iteration
            <<" action=hit dimension="<<prepared.dimension<<"\n";
        return;
      }
      ++stats.aggregate_preparation_invalidations;
    }
    prepared.ready=false; // Exceptions can never leave a partial cache valid.
    ++stats.aggregate_preparation_builds;
    // Model caches (including spectral directions) are untouched by this raw test.
    const auto geometry_start=Clock::now();
    dca_disable_models(w);w.evaluate();w.update_geometry();
    stats.aggregate_preparation_geometry_seconds+=seconds(geometry_start);
    const auto operator_start=Clock::now();
    Vec partial(w.hpattern.size(),0);
    auto add_original=[&](Vec& h,const Quadratic& q,double weight){
      for(int k=0;k<(int)q.hessian.size();++k)h[q.hglobal[k]]+=weight*q.scale*q.hessian[k].v;
    };
    add_original(partial,w.objective,1);
    for(int e=0;e<w.me;++e)add_original(partial,w.equalities[e],w.lambda[e]);
    for(int i=0;i<w.mi;++i)if(w.active[i])
      add_original(partial,w.inequalities[i],w.z[i]);
    auto project=[&](const Vec& h){
      std::vector<SymmetricTriplet> out;Vec values(w.kkt.size(),0);
      for(auto m:w.hmap)values[m.k]+=m.multiplier*h[m.h]*w.tvalue[m.a]*w.tvalue[m.b];
      for(int k=0;k<(int)w.kkt.size();++k)if(w.kkt[k].i<w.nr&&w.kkt[k].j<w.nr)
        out.push_back({w.kkt[k].i,w.kkt[k].j,values[k]});
      return out;
    };
    const int equalities=w.mr,base_dim=w.nr+equalities;
    auto append_normals=[&](std::vector<SymmetricTriplet>& matrix){
      for(int e=0;e<equalities;++e){
        const auto& g=w.ar[e];
        matrix.push_back({w.nr+e,w.nr+e,0});
        for(int k=0;k<(int)g.support.size();++k)matrix.push_back({g.support[k],w.nr+e,g.values[k]});
      }
    };
    auto matrix=project(partial);append_normals(matrix);
    std::vector<int> weak_border(w.mi,-1);int dimension=base_dim;
    const bool persistent=settings.dca_aggregate_kkt_layout=="persistent";
    std::vector<int> requested;
    if(persistent){
      stable_borders.configure(w.nr,w.mr,w.mi,settings.dca_aggregate_border_reserve);
      for(int i=0;i<w.mi;++i)if(w.active[i]){
        const auto& g=w.jr[i];
        if(g.dense>=0 || w.z[i]/w.s[i]*dot(g.values,g.values)>1e4)requested.push_back(i);
      }
    }
    const auto generation_before=stable_borders.generation();
    if(persistent)stable_borders.update(requested);
    for(int i=0;i<w.mi;++i)if(w.active[i] || (persistent && w.jr[i].dense<0)) {
      const auto& g=w.jr[i];
      const bool bordered=w.active[i] && (g.dense>=0 || w.z[i]/w.s[i]*dot(g.values,g.values)>1e4);
      if(persistent){
        weak_border[i]=stable_borders.active_row_for(i);
        // The sparse condensed envelope survives changes of the numerical
        // border classification. Dense rows are always handled as borders.
        if(g.dense<0){
          const double weight=w.active[i]&&!bordered?w.z[i]/w.s[i]:0.0;
          for(int a=0;a<(int)g.support.size();++a)for(int b=a;b<(int)g.support.size();++b)
            matrix.push_back({g.support[a],g.support[b],weight*g.values[a]*g.values[b]});
        }
      }else if(bordered){
        int row=dimension++;weak_border[i]=row;
        matrix.push_back({row,row,-w.s[i]/w.z[i]});
        for(int k=0;k<(int)g.support.size();++k)matrix.push_back({g.support[k],row,g.values[k]});
      }else for(int a=0;a<(int)g.support.size();++a)for(int b=a;b<(int)g.support.size();++b)
        matrix.push_back({g.support[a],g.support[b],w.z[i]/w.s[i]*g.values[a]*g.values[b]});
    }
    if(persistent){
      stable_borders.append_borders(w.jr,w.s,w.z,matrix);
      dimension=stable_borders.dimension();
      if(stable_borders.generation()!=generation_before)++stats.aggregate_kkt_layout_rebuilds;
      stats.aggregate_kkt_border_capacity=std::max(stats.aggregate_kkt_border_capacity,stable_borders.capacity());
      stats.aggregate_kkt_border_assigned=std::max(stats.aggregate_kkt_border_assigned,stable_borders.assigned_count());
    }
    stats.aggregate_kkt_dimension=std::max(stats.aggregate_kkt_dimension,dimension);
    stats.aggregate_preparation_operator_seconds+=seconds(operator_start);
    prepared.hessian=std::move(partial);
    prepared.unshifted=std::move(matrix);
    prepared.weak_border=std::move(weak_border);
    prepared.dimension=dimension;prepared.nr=w.nr;prepared.mr=w.mr;
    if(reuse){
      prepared.owner=&w;prepared.x=w.x;prepared.s=w.s;prepared.z=w.z;
      prepared.lambda=w.lambda;prepared.active=w.active;prepared.ready=true;
      stats.aggregate_preparation_peak_bytes=std::max(stats.aggregate_preparation_peak_bytes,prepared.bytes());
    }
    if(w.st.live_display_level>=3)std::cout<<"[DCA-AGGREGATE-CACHE] outer="<<outer_iteration
        <<" action=build enabled="<<reuse<<" dimension="<<dimension
        <<" nnz="<<prepared.unshifted.size()<<"\n";
  }

  // Only called for an already filtered, positive, certified candidate.
  // Never increments the acceptable counter or changes the incumbent/result.
  bool strict_candidate(Workspace& w,const Candidate& c,DcaFilterStatistics& stats,
                        int outer_iteration) const {
    if(!settings.dca_aggregate_strict_probe_stop)return false;
    // Conservative cheap gate. A false gate only misses an optimization;
    // it never weakens the full certificate or rejects a candidate.
    const bool native=stop_settings.termination_test=="ipopt";
    if(!std::isfinite(c.point.kkt) || c.point.primal>stop_settings.feasibility_tolerance ||
       c.point.dual>stop_settings.dual_infeasibility_tolerance ||
       c.point.comp>stop_settings.complementarity_tolerance ||
       (native && c.point.normalized_kkt>stop_settings.stationarity_tolerance))return false;
    ++stats.aggregate_strict_probe_checks;
    detail::DcaTimedScope timer(stats.aggregate_strict_probe_seconds);
    CostScope cost(w.costs,NewtonCost::Certificate);
    Result certificate;
    certificate.outer_iterations=outer_iteration;
    w.export_state_at(certificate,c.x,c.s,c.z,c.lambda);
    const bool ok=detail::audit_original_termination(w.original,stop_settings,certificate,&w.original_linear_support);
    if(ok)++stats.aggregate_strict_probe_successes;
    else ++stats.aggregate_strict_probe_failures;
    if(w.st.live_display_level>0)std::cout<<"[DCA-RHO-STRICT] outer="<<outer_iteration
        <<" rho="<<c.rho<<" certificate="<<(ok?"passed":"failed")
        <<" test="<<stop_settings.termination_test<<" E="<<certificate.scaled_nlp_error
        <<" primal="<<certificate.feasibility.total_inf<<" dual="<<certificate.termination_dual_inf
        <<" compl="<<certificate.termination_compl_inf<<"\n";
    return ok;
  }
  // A single frozen anchor. Unlike attempt()'s terminal rho contest, an
  // infeasible candidate is allowed here only through the UNMODIFIED filter.
  // Curvature and globalization use separate decisions. A backend/precision
  // failure has already exhausted numerical recovery inside attempt_one;
  // increasing rho cannot silently relabel it as missing curvature.
  bool attempt_single_loop(Workspace& w,DcaModels& models,DcaPoint& outer,
      detail::OriginalDcaFilter& filter,int outer_iteration,Clock::time_point solve_start) {
    stationarity_evidence={};
    Preparation prepared;
    auto& stats=models.stats;++stats.aggregate_trials;
    const double cap=settings.dca_aggregate_rho_max;
    const bool separated=settings.dca_aggregate_rejection_policy=="separated";
    const double remembered=std::max(accepted_rho,certified_rho);
    const double seed=settings.dca_aggregate_rho_memory ?
        positive_rho_memory.seed(settings.dca_aggregate_rho_initial,settings.dca_aggregate_rho_decay,cap):
        std::min(cap,separated?settings.dca_aggregate_rho_initial:
          std::max(settings.dca_aggregate_rho_initial,remembered*settings.dca_aggregate_rho_decay));
    double rho=0,rho_curvature=0,rho_step=0;
    bool step_search=false;
    for(int trial=0;trial<settings.dca_aggregate_retry_max;++trial) {
      if(seconds(solve_start)>=settings.time_limit_seconds) {
        last_rejection="aggregate-only-time-limit";
        last_rejection_kind=detail::DcaAggregateRejection::globalization;return false;
      }
      ++stats.aggregate_only_attempts;
      const bool ok=attempt_one(w,models,outer,filter,outer_iteration,
                               rho,nullptr,false,true,&prepared,false,!separated || !step_search);
      if(last_attempt_certified && !step_search)rho_curvature=last_curvature_rho;
      if(w.st.live_display_level>0)std::cout<<"[DCA-SINGLE-LOOP] outer="<<outer_iteration
          <<" trial="<<trial+1<<"/"<<settings.dca_aggregate_retry_max
          <<" rho="<<rho<<" effective_rho="<<last_attempt_rho
          <<" certified="<<last_attempt_certified<<" accepted="<<ok
          <<" cause="<<detail::dca_aggregate_rejection_name(last_rejection_kind)
          <<" rho_curvature="<<rho_curvature<<" rho_step="<<rho_step
          <<" reason=\""<<last_rejection<<"\" majorants="<<stats.models<<"\n";
      if(ok)return true;
      if(last_rejection_kind==detail::DcaAggregateRejection::inertia)
        ++stats.aggregate_only_inertia_rejections;
      else ++stats.aggregate_only_step_rejections;
      if(!detail::dca_aggregate_retry_permitted(last_rejection_kind,separated))break;
      if(separated && last_rejection_kind!=detail::DcaAggregateRejection::inertia)step_search=true;
      if(rho>=cap)break;
      const double next=rho==0?seed:std::min(cap,rho*settings.dca_aggregate_rho_growth);
      if(!(next>rho)||!std::isfinite(next))break;
      if(rho==0 && settings.dca_aggregate_rho_memory && positive_rho_memory.value>0)
        ++stats.aggregate_rho_memory_uses;
      if(step_search){rho_step=std::max(0.0,next-rho_curvature);++stats.aggregate_step_regularizations;}
      rho=next;
    }
    ++stats.aggregate_only_exhaustions;
    return false;
  }

  bool attempt(Workspace& w,DcaModels& models,DcaPoint& outer,
               detail::OriginalDcaFilter& filter,int outer_iteration=-1,
               bool externally_acceptable=false) {
    stationarity_evidence={};
    const double anchor=accepted_rho;
    const bool compare_rhos=std::isfinite(outer.primal)&&
        outer.primal<=w.st.feasibility_tolerance&&anchor>0.0;
    if(!compare_rhos)return attempt_one(
        w,models,outer,filter,outer_iteration,-1.0,nullptr,true,true);

    auto& stats=models.stats;
    ++stats.aggregate_trials;
    const auto rhos=detail::dca_rho_probe_schedule(
        anchor,w.st.dca_aggregate_rho_max);
    std::optional<Candidate> best;
    std::vector<double> attempted_rhos;
    int eligible=0,probes_attempted=0;
    Preparation preparation; // destroyed before leaving this anchor
    bool strict_stop=false;
    auto probe=[&](double requested,const char* mode){
      ++probes_attempted;
      Candidate candidate;
      const bool ok=attempt_one(w,models,outer,filter,outer_iteration,
                                requested,&candidate,false,false,&preparation);
      if(last_attempt_certified&&std::isfinite(last_attempt_rho))
        attempted_rhos.push_back(last_attempt_rho);
      if(w.st.live_display_level>0){
        std::cout<<"[DCA-RHO-PROBE] outer="<<outer_iteration
                 <<" mode="<<mode<<" requested_rho=";
        if(requested<0.0)std::cout<<"adaptive";else std::cout<<requested;
        std::cout<<" effective_rho="<<last_attempt_rho
                 <<" certified="<<last_attempt_certified
                 <<" candidate_kkt="<<(ok?candidate.point.kkt:last_candidate_kkt)
                 <<" candidate_primal="<<(ok?candidate.point.primal:last_candidate_primal)
                 <<" outcome="<<(ok?"eligible":"reject")
                 <<" reason=\""<<(ok?"none":last_rejection)<<"\"\n";
      }
      if(!ok)return;
      ++eligible;
      if(strict_candidate(w,candidate,stats,outer_iteration)){
        strict_stop=true;best=std::move(candidate);return;
      }
      const bool both_curvature=best && candidate.curvature_descent && best->curvature_descent;
      const bool better=!best || (both_curvature ?
            candidate.point.objective<best->point.objective : detail::dca_rho_candidate_better(
            candidate.point.kkt,candidate.point.primal,candidate.rho,
            best->point.kkt,best->point.primal,best->rho,anchor));
      if(better)best=std::move(candidate);
    };
    // Always retain the historical adaptive inertia ladder as a reference
    // candidate, then compare explicit neighboring shifts transactionally.
    probe(-1.0,"adaptive");
    // Lazy policy: a certified objective-descent DCA step along measured
    // negative original curvature is already productive. Do not perform an
    // exhaustive stationarity-rho contest whose ranking is inappropriate in
    // that regime. Failed or stationarity-only candidates retain all probes.
    const bool productive_curvature=best && best->curvature_descent;
    if(productive_curvature && !strict_stop){
      ++stats.aggregate_curvature_shortcuts;
      if(w.st.live_display_level>0)std::cout<<"[DCA-RHO-POLICY] outer="<<outer_iteration
          <<" probes=1 reason=certified-objective-descent remaining_probes=deferred\n";
    }
    if(!productive_curvature)for(double rho:rhos){
      const bool duplicate=std::any_of(attempted_rhos.begin(),attempted_rhos.end(),
        [&](double tested){
          const double scale=std::max({1.0,std::abs(tested),std::abs(rho)});
          return std::abs(tested-rho)<=32.0*std::numeric_limits<double>::epsilon()*scale;
        });
      if(duplicate){
        if(w.st.live_display_level>0)
          std::cout<<"[DCA-RHO-PROBE] outer="<<outer_iteration
                   <<" mode=fixed requested_rho="<<rho
                   <<" effective_rho="<<rho
                   <<" certified=1 candidate_kkt=nan candidate_primal=nan"
                   <<" outcome=skip reason=\"duplicate-effective-rho\"\n";
        continue;
      }
      if(strict_stop){++stats.aggregate_strict_probe_skipped;continue;}
      probe(rho,"fixed");
    }
    if(std::isfinite(stationarity_evidence.minimum_certified_rho))
      certified_rho=stationarity_evidence.minimum_certified_rho;
    if(!best){
      if(externally_acceptable){
        if(w.st.live_display_level>0)
          std::cout<<"[DCA-RHO-SELECT] outer="<<outer_iteration
                   <<" decision=historical-fallback"
                   <<" reason=\"no-monotone-probe-at-acceptable-point\"\n";
        // Consecutive acceptable certificates still require a genuinely
        // accepted next iterate.  Reuse the historical filtered transaction;
        // never count the incumbent twice and never enable this escape while
        // the raw original tolerances are not yet satisfied.
        return attempt_one(w,models,outer,filter,outer_iteration,
                           -1.0,nullptr,false,true,&preparation);
      }
      ++stats.aggregate_rejections;
      stats.aggregate_last_rejection=last_rejection;
      return false;
    }
    if(best->point.theta>w.st.feasibility_tolerance)filter.insert(outer);
    w.x=std::move(best->x);w.s=std::move(best->s);
    w.z=std::move(best->z);w.lambda=std::move(best->lambda);
    accepted_rho=best->rho;
    last_rejection_kind=detail::DcaAggregateRejection::none;
    // Selection optimizes the endpoint; it must not overwrite the smallest
    // curvature certificate observed at this SAME accepted anchor.
    stats.aggregate_last_rho=best->rho;
    stats.aggregate_max_accepted_residual=std::max(
        stats.aggregate_max_accepted_residual,best->backward_error);
    stats.aggregate_max_accepted_rho=std::max(
        stats.aggregate_max_accepted_rho,best->rho);
    stats.last_sigma=best->sigma;
    if(best->soc)++stats.aggregate_soc;
    if(best->curvature_descent)++stats.aggregate_curvature_descents;
    dca_inherit_guard(w,outer,best->point);
    outer=best->point;++stats.aggregate_acceptances;last_rejection="none";
    if(w.st.live_display_level>0)
      std::cout<<"[DCA-RHO-SELECT] outer="<<outer_iteration
               <<" decision=commit rho="<<best->rho
               <<" original_kkt="<<outer.kkt
               <<" original_primal="<<outer.primal
               <<" eligible="<<eligible
               <<" probes="<<probes_attempted
               <<" strict_shortcut="<<strict_stop<<"\n";
    return true;
  }

  bool attempt_one(Workspace& w,DcaModels& models,DcaPoint& outer,
               detail::OriginalDcaFilter& filter,int outer_iteration,
               double requested_rho,Candidate* candidate_output,
               bool count_trial,bool record_certificate,Preparation* shared_preparation=nullptr,
               bool precise_finish=false,bool curvature_search=true) {
    const DcaTraceContext trace_context{outer_iteration,-1,-1};
    last_rejection="none";
    last_rejection_kind=detail::DcaAggregateRejection::none;
    last_candidate_kkt=last_candidate_primal=last_backward_error=INFINITY;
    last_attempt_rho=last_curvature_rho=INFINITY;last_attempt_certified=false;root_factor_valid=false;
    double local_rho=0.0;
    std::string message;
    auto& stats=models.stats;
    struct BackendStatistics {
      SymmetricFactorization* factor;DcaFilterStatistics& stats;
      ~BackendStatistics(){if(!factor)return;
        stats.aggregate_backend=factor->name();
        stats.aggregate_symbolic_analyses=factor->symbolic_analyses();
        stats.aggregate_symbolic_reuses=factor->symbolic_reuses();
        stats.aggregate_backend_assembly_seconds=factor->assembly_seconds();
        stats.aggregate_backend_symbolic_seconds=factor->symbolic_seconds();
        stats.aggregate_backend_numeric_seconds=factor->numeric_seconds();
        stats.aggregate_backend_solve_seconds=factor->linear_solve_seconds();
        stats.aggregate_scatter_builds=factor->numeric_scatter_builds();
        stats.aggregate_scatter_reuses=factor->numeric_scatter_reuses();
        stats.aggregate_scatter_invalidations=factor->numeric_scatter_invalidations();
      }
    } backend_statistics{factor.get(),stats};
    auto reject=[&](const char* reason,detail::DcaAggregateRejection kind=detail::DcaAggregateRejection::none){
      last_rejection=reason;
      last_rejection_kind=kind==detail::DcaAggregateRejection::none?
          detail::dca_aggregate_point_rejection(reason):kind;
      if(last_rejection_kind==detail::DcaAggregateRejection::none)
        last_rejection_kind=detail::DcaAggregateRejection::globalization;
      stats.aggregate_last_rejection_kind=detail::dca_aggregate_rejection_name(last_rejection_kind);
      if(last_rejection_kind==detail::DcaAggregateRejection::backend ||
         last_rejection_kind==detail::DcaAggregateRejection::numerical_direction)
        stats.aggregate_last_backend_message=message;
      switch(last_rejection_kind){
        case detail::DcaAggregateRejection::inertia:++stats.aggregate_reject_inertia;break;
        case detail::DcaAggregateRejection::backend:++stats.aggregate_reject_backend;break;
        case detail::DcaAggregateRejection::numerical_direction:++stats.aggregate_reject_numerical_direction;break;
        case detail::DcaAggregateRejection::domain:++stats.aggregate_reject_domain;break;
        case detail::DcaAggregateRejection::filter:++stats.aggregate_reject_filter;break;
        case detail::DcaAggregateRejection::kkt:++stats.aggregate_reject_kkt;break;
        case detail::DcaAggregateRejection::globalization:++stats.aggregate_reject_globalization;break;
        case detail::DcaAggregateRejection::none:break;
      }
      if(!candidate_output){
        ++models.stats.aggregate_rejections;
        models.stats.aggregate_last_rejection=reason;
      }
      if(w.st.live_display_level>=4 || dca_filter_trace_enabled(w))
        std::cout<<"[DCA-AGGREGATE-REJECT] outer="<<outer_iteration<<" reason="<<reason
                 <<" cause="<<detail::dca_aggregate_rejection_name(last_rejection_kind)<<" original_kkt="<<outer.kkt
                 <<" candidate_kkt="<<last_candidate_kkt<<" backward_error="<<last_backward_error
                 <<" backend_detail=\""<<message<<"\"\n";
      return false;
    };
    // Eligibility is owned by the outer controller. The transaction always
    // certifies its actual manifold and audits the original problem below.
    if(!factor)return reject("backend-unavailable");
    if(count_trial)++stats.aggregate_trials;
    detail::DcaTimedScope timer(stats.aggregate_seconds);
    CostScope cost(w.costs,NewtonCost::CoreOther);
    Preparation local_preparation;
    Preparation& prepared=shared_preparation?*shared_preparation:local_preparation;
    prepare(w,stats,prepared,shared_preparation && settings.dca_aggregate_preparation_cache,outer_iteration);
    Vec partial=prepared.hessian;
    const Vec& full=partial; // This branch uses one full Lagrangian Hessian.
    const int equalities=w.mr,dimension=prepared.dimension;
    const auto& weak_border=prepared.weak_border;
    const auto& unshifted=prepared.unshifted;
    auto matrix=unshifted;
    // Keep the triplet layout identical even for rho=0; backend scatter maps
    // and symbolic structure then survive every curvature retry.
    const std::size_t rho_slots=matrix.size();
    const bool stable_rho_layout=settings.dca_aggregate_kkt_layout=="persistent" ||
        settings.dca_aggregate_fast_assembly;
    if(stable_rho_layout)for(int j=0;j<w.nr;++j)matrix.push_back({j,j,0.0});
    // Once the original residual is local, preserve the last certified
    // curvature shift.  Restarting at zero and then at 0.2*certified_rho caused
    // case30000 to leave its best point (rho 1.6e-2 -> 3.2e-3).
    const bool local_guard=detail::dca_local_guard_active(
        outer.kkt,outer.primal,1.0,w.st.dca_common_step_primal_trigger);
    const bool exact_stationarity_probe=
        std::isfinite(outer.primal)&&outer.primal<=w.st.feasibility_tolerance;
    const bool fixed_rho=requested_rho>=0.0;
    local_rho=requested_rho>=0.0?requested_rho:
        detail::dca_aggregate_initial_rho(
          certified_rho,outer.primal,w.st.feasibility_tolerance,local_guard);
    bool certified=false;
    std::vector<double> rho_schedule;
    if(fixed_rho){if(local_rho<=w.st.dca_aggregate_rho_max)rho_schedule.push_back(local_rho);}
    else if(settings.dca_rho_complete_search)
      rho_schedule=detail::dca_complete_inertia_schedule(local_rho,certified_rho,
          w.st.dca_aggregate_rho_max,exact_stationarity_probe);
    else {
      double next=local_rho;
      for(int k=0;k<10&&next<=w.st.dca_aggregate_rho_max;++k){rho_schedule.push_back(next);
        next=detail::dca_aggregate_next_rho(next,certified_rho,exact_stationarity_probe);}
    }
    double legacy_next=local_rho;
    for(std::size_t trial=0;trial<rho_schedule.size()&&!certified;++trial){
      local_rho=rho_schedule[trial];
      const bool completion=!fixed_rho && local_rho!=legacy_next;
      if(completion)++stats.rho_grid_completions;
      if(stable_rho_layout){
        for(int j=0;j<w.nr;++j)matrix[rho_slots+j].value=local_rho;
      }else{
        // The all-off compatibility profile retains the historical triplet
        // insertion order, including the absence of extra zeros at rho=0.
        matrix=unshifted;
        if(local_rho>0)for(int j=0;j<w.nr;++j)matrix.push_back({j,j,local_rho});
      }
      last_attempt_rho=local_rho;
      ++stats.aggregate_certificates;
      { CostScope stage(w.costs,NewtonCost::Factorization);
        const auto stamp=Clock::now();
        certified=factor->factor(dimension,w.nr,matrix,message);
        stats.aggregate_factor_seconds+=seconds(stamp);
      }
      stationarity_evidence.certificate(local_rho,certified);
      if(w.st.live_display_level>=3 || (completion&&w.st.live_display_level>0))
        std::cout<<"[DCA-RHO-CERTIFICATE] outer="<<outer_iteration<<" rho="<<local_rho
                 <<" certified="<<certified<<" grid_completion="<<completion
                 <<" cap="<<w.st.dca_aggregate_rho_max<<" detail=\""<<(certified?"ok":message)<<"\"\n";
      legacy_next=detail::dca_aggregate_next_rho(local_rho,certified_rho,exact_stationarity_probe);
      if(!certified && settings.dca_aggregate_rejection_policy=="separated" &&
         detail::dca_aggregate_factor_rejection(message)!=detail::DcaAggregateRejection::inertia)
        return reject("aggregate-backend-factorization",detail::DcaAggregateRejection::backend);
    }
    if(!certified)return reject("aggregate-curvature-certificate");
    last_attempt_certified=true;last_attempt_rho=last_curvature_rho=local_rho;
    capture_root_factor(w,dimension,weak_border,outer);
    if(record_certificate){
      certified_rho=local_rho;
      positive_rho_memory.certificate(local_rho,curvature_search);
      stats.aggregate_last_positive_rho=positive_rho_memory.value;
    }
    stats.aggregate_backend=factor->name();
    stats.aggregate_symbolic_analyses=factor->symbolic_analyses();
    stats.aggregate_symbolic_reuses=factor->symbolic_reuses();
    for(int j:w.retained){int k=models.slots.at({j,j});partial[k]+=local_rho;}
    auto product=[&](const Vec& x){Vec y(w.n,0);for(int k=0;k<(int)full.size();++k){auto [a,b]=w.hpattern[k];y[a]+=full[k]*x[b];if(a!=b)y[b]+=full[k]*x[a];}return y;};
    // Keep the unperturbed curvature certificate. A dual shift below is a
    // numerical recovery, never a substitute for that certificate.
    double dual_shift=0;
    auto inequality_shift=[&](int i){return w.inequality_kind[i]==0?dual_shift:0.0;};
    bool reconstructed=false;
    bool precision_failed=false;
    double reduced_error=INFINITY,expanded_error=INFINITY;
    auto solve_once=[&](const Vec& dual,const Vec& eq,const Vec& primal,const Vec& comp,Direction& d,bool permit_inexact=false){
      d.x.assign(w.n,0);d.lambda.assign(w.me,0);d.s.assign(w.mi,0);d.z.assign(w.mi,0);
      Vec offset(w.n,0);
      for(int e=0;e<w.me;++e)if(w.pivot_variable[e]>=0)offset[w.pivot_variable[e]]=-eq[e]/w.pivot_coefficient[e];
      Vec hd=product(offset),rhs(dimension,0);
      for(int j=0;j<w.n;++j)for(int t=w.tbegin[j];t<w.tbegin[j+1];++t)
        rhs[w.tcolumn[t]]-=w.tvalue[t]*(hd[j]+dual[j]);
      for(int e=0;e<w.mr;++e)rhs[w.nr+e]=-eq[w.eq_retained[e]]-w.equalities[w.eq_retained[e]].directional(offset);
      for(int i=0;i<w.mi;++i)if(w.active[i]){
        const double r=primal[i]+w.inequalities[i].directional(offset)-comp[i]/w.z[i];
        if(weak_border[i]>=0)rhs[weak_border[i]]=-r;
        else for(int k=0;k<(int)w.jr[i].support.size();++k)rhs[w.jr[i].support[k]]-=w.jr[i].values[k]*w.z[i]/(w.s[i]+inequality_shift(i)*w.z[i])*r;
      }
      precision_failed=false;reconstructed=false;reduced_error=expanded_error=INFINITY;
      const Vec original_rhs=rhs;
      auto backsolve=[&](Vec& b){
        CostScope stage(w.costs,NewtonCost::LinearBackend);
        ++stats.aggregate_linear_solves;
        const bool solved=factor->solve(b,message);
        if(!solved||!finite(b)){
          // The initial and every refinement solve share the same failure
          // classification. A late backward-error failure still qualifies
          // for the existing numerical recovery at this certified matrix.
          precision_failed=(solved&&!finite(b)) || message.find("backward error")!=std::string::npos;
          reduced_error=factor->last_backward_error();return false;
        }
        return true;
      };
      if(!backsolve(rhs))return false;
      // Evaluate cancellation-prone bordered residuals in extended precision.
      // Keep the best iterate: a late refinement may hit the arithmetic floor,
      // and must not replace a more accurate direction with a worse one.
      Vec best_rhs=rhs;double best_error=INFINITY;
      for(int refinement=0;refinement<=5;++refinement){
        std::vector<long double> accumulated(original_rhs.begin(),original_rhs.end());
        for(auto t:matrix){
          accumulated[t.i]-=static_cast<long double>(t.value)*rhs[t.j];
          if(t.i!=t.j)accumulated[t.j]-=static_cast<long double>(t.value)*rhs[t.i];
        }
        Vec residual(accumulated.begin(),accumulated.end());
        const double error=inf_norm(residual)/std::max(1.0,inf_norm(original_rhs));
        if(!std::isfinite(error)){precision_failed=true;return false;}
        if(error<best_error){best_error=error;best_rhs=rhs;}
        else break;
        if(error<=1e-12 || refinement==5)break;
        if(!backsolve(residual))return false;
        axpy(1,residual,rhs);
      }
      rhs=std::move(best_rhs);last_backward_error=best_error;reduced_error=best_error;
      if(best_error>w.st.kkt_backward_error_reject && !permit_inexact){precision_failed=true;return false;}
      for(int j=0;j<w.n;++j){d.x[j]=offset[j];for(int t=w.tbegin[j];t<w.tbegin[j+1];++t)d.x[j]+=w.tvalue[t]*rhs[w.tcolumn[t]];}
      for(int e=0;e<w.mr;++e)d.lambda[w.eq_retained[e]]=rhs[w.nr+e];
      for(int i=0;i<w.mi;++i)if(w.active[i]){
        d.s[i]=-primal[i]-w.inequalities[i].directional(d.x);
        d.z[i]=weak_border[i]>=0?rhs[weak_border[i]]:(-comp[i]-w.z[i]*d.s[i])/(w.s[i]+inequality_shift(i)*w.z[i]);
        d.s[i]+=inequality_shift(i)*d.z[i];
      }
      hd=product(d.x);for(int j=0;j<w.n;++j)hd[j]+=dual[j];
      for(int i=0;i<w.mi;++i)if(w.active[i])w.inequalities[i].add_gradient(d.z[i],hd);
      for(int e:w.eq_retained)w.equalities[e].add_gradient(d.lambda[e],hd);
      for(int e=0;e<w.me;++e)if(w.pivot_variable[e]>=0)d.lambda[e]=-hd[w.pivot_variable[e]]/w.pivot_coefficient[e];
      for(int e=0;e<w.me;++e)if(w.pivot_variable[e]>=0)w.equalities[e].add_gradient(d.lambda[e],hd);
      double expanded=inf_norm(hd),scale=std::max(1.0,inf_norm(dual));
      for(int e=0;e<w.me;++e){expanded=std::max(expanded,std::abs(eq[e]+w.equalities[e].directional(d.x)-(w.pivot_variable[e]<0?dual_shift*d.lambda[e]:0)));scale=std::max(scale,std::abs(eq[e]));}
      for(int i=0;i<w.mi;++i)if(w.active[i]){
        expanded=std::max(expanded,std::abs(primal[i]+w.inequalities[i].directional(d.x)+d.s[i]-inequality_shift(i)*d.z[i]));
        expanded=std::max(expanded,std::abs(comp[i]+w.z[i]*d.s[i]+w.s[i]*d.z[i]));
        scale=std::max({scale,std::abs(primal[i]),std::abs(comp[i])});
      }
      reconstructed=true;d.residual=expanded/scale;expanded_error=d.residual;
      precision_failed=!finite(d.x)||!finite(d.lambda)||!finite(d.z)||!finite(d.s)||
          !std::isfinite(d.residual)||d.residual>w.st.kkt_backward_error_reject;
      last_backward_error=std::max(last_backward_error,d.residual);
      stats.aggregate_residual=std::max(stats.aggregate_residual,d.residual);
      return finite(d.x)&&finite(d.lambda)&&finite(d.z)&&std::isfinite(d.residual)&&d.residual<=w.st.kkt_backward_error_reject;
    };
    // Full-space residuals use extended accumulation to distinguish loss in
    // reconstruction from rounding in the audit itself. Nothing changes on a
    // historical successful solve.
    auto full_residual=[&](const Vec& dual,const Vec& eq,const Vec& primal,const Vec& comp,
                           const Direction& d,Direction& r){
      std::vector<long double> rd(dual.begin(),dual.end());
      for(int k=0;k<(int)full.size();++k){auto [a,b]=w.hpattern[k];
        rd[a]+=static_cast<long double>(full[k])*d.x[b];
        if(a!=b)rd[b]+=static_cast<long double>(full[k])*d.x[a];}
      auto gradient=[&](const Quadratic& q,double weight){
        const double* g=q.pooled_gradient?q.pooled_gradient:q.gradient.data();
        for(std::size_t k=0;k<q.support.size();++k)rd[q.support[k]]+=static_cast<long double>(g[k])*weight;
      };
      auto directional=[&](const Quadratic& q){
        const double* g=q.pooled_gradient?q.pooled_gradient:q.gradient.data();long double v=0;
        for(std::size_t k=0;k<q.support.size();++k)v+=static_cast<long double>(g[k])*d.x[q.support[k]];
        return v;
      };
      r.lambda.resize(w.me);r.s.assign(w.mi,0);r.z.assign(w.mi,0);
      for(int e=0;e<w.me;++e){gradient(w.equalities[e],d.lambda[e]);
        r.lambda[e]=eq[e]+directional(w.equalities[e])-(w.pivot_variable[e]<0?static_cast<long double>(dual_shift)*d.lambda[e]:0);}
      for(int i=0;i<w.mi;++i)if(w.active[i]){gradient(w.inequalities[i],d.z[i]);
        r.s[i]=primal[i]+directional(w.inequalities[i])+d.s[i]-static_cast<long double>(inequality_shift(i))*d.z[i];
        r.z[i]=comp[i]+static_cast<long double>(w.z[i])*d.s[i]+static_cast<long double>(w.s[i])*d.z[i];}
      r.x.assign(rd.begin(),rd.end());
      if(!finite(r.x)||!finite(r.lambda)||!finite(r.s)||!finite(r.z))return std::numeric_limits<double>::infinity();
      const double scale=std::max({1.0,inf_norm(dual),inf_norm(eq),inf_norm(primal),inf_norm(comp)});
      return std::max({inf_norm(r.x),inf_norm(r.lambda),inf_norm(r.s),inf_norm(r.z)})/scale;
    };
    auto solve_refined=[&](const Vec& dual,const Vec& eq,const Vec& primal,const Vec& comp,Direction& d){
      bool ok=solve_once(dual,eq,primal,comp,d,precise_finish);
      if(precise_finish){
        if(!reconstructed||!finite(d.x)||!finite(d.lambda)||!finite(d.s)||!finite(d.z))return false;
        const double rhs_norm=std::max({inf_norm(dual),inf_norm(eq),inf_norm(primal),inf_norm(comp)});
        const double scale=std::max(1.0,rhs_norm);
        const double requested_error=stop_settings.stationarity_tolerance*
            (stop_settings.termination_test=="ipopt"?1.0:w.objective.scale);
        const double target=std::min(w.st.kkt_backward_error_reject,
            settings.dca_strict_finish_eta*std::min(rhs_norm,requested_error)/scale);
        Direction residual;double error=full_residual(dual,eq,primal,comp,d,residual);
        for(int k=0;k<settings.dca_strict_finish_refinements&&error>target;++k){
          Direction correction;solve_once(residual.x,residual.lambda,residual.s,residual.z,correction,true);
          if(!reconstructed||!finite(correction.x)||!finite(correction.lambda)||!finite(correction.s)||!finite(correction.z))break;
          Direction trial=d;axpy(1,correction.x,trial.x);axpy(1,correction.lambda,trial.lambda);
          axpy(1,correction.s,trial.s);axpy(1,correction.z,trial.z);
          Direction next;const double candidate=full_residual(dual,eq,primal,comp,trial,next);
          ++stats.strict_finish_refinements;
          if(w.st.live_display_level>=2)std::cout<<"[DCA-STRICT-REFINE] outer="<<outer_iteration
              <<" refinement="<<k+1<<" before="<<error<<" after="<<candidate<<" target="<<target<<" same_factor=1\n";
          if(!std::isfinite(candidate)||candidate>=error)break;
          d=std::move(trial);residual=std::move(next);error=candidate;
        }
        d.residual=error;last_backward_error=error;expanded_error=error;
        stats.strict_finish_linear_residual=error;
        precision_failed=!std::isfinite(error)||error>target;
        if(w.st.live_display_level>=2)std::cout<<"[DCA-STRICT-LINEAR] outer="<<outer_iteration
            <<" residual="<<error<<" rhs_norm="<<rhs_norm<<" target="<<target<<" passed="<<!precision_failed<<"\n";
        return !precision_failed;
      }
      if(ok || !precision_failed || !reconstructed || !w.st.dca_aggregate_full_refinement)return ok;
      Direction residual;double error=full_residual(dual,eq,primal,comp,d,residual);
      if(w.st.live_display_level>0)std::cout<<"[DCA-FULL-REFINE] outer="<<outer_iteration
        <<" delta_c="<<dual_shift<<" initial="<<error<<" stationarity="<<inf_norm(residual.x)
        <<" equality="<<inf_norm(residual.lambda)<<" inequality="<<inf_norm(residual.s)
        <<" complementarity="<<inf_norm(residual.z)<<"\n";
      for(int k=0;k<10 && error>w.st.kkt_backward_error_reject;++k){
        Direction correction;
        solve_once(residual.x,residual.lambda,residual.s,residual.z,correction,true);
        if(!reconstructed || !finite(correction.x)||!finite(correction.lambda)||!finite(correction.s)||!finite(correction.z))break;
        Direction trial=d;axpy(1,correction.x,trial.x);axpy(1,correction.lambda,trial.lambda);
        axpy(1,correction.s,trial.s);axpy(1,correction.z,trial.z);
        Direction next;const double candidate=full_residual(dual,eq,primal,comp,trial,next);
        if(w.st.live_display_level>0)std::cout<<"[DCA-FULL-REFINE] outer="<<outer_iteration
          <<" refinement="<<k+1<<" before="<<error<<" after="<<candidate<<" same_factor=1\n";
        if(!std::isfinite(candidate)||candidate>=error)break;
        d=std::move(trial);residual=std::move(next);error=candidate;
      }
      expanded_error=error;d.residual=error;last_backward_error=error;
      precision_failed=!std::isfinite(error)||error>w.st.kkt_backward_error_reject;
      return !precision_failed;
    };
    auto solve=[&](const Vec& dual,const Vec& eq,const Vec& primal,const Vec& comp,Direction& d){
      bool ok=solve_refined(dual,eq,primal,comp,d);
      if(ok || !precision_failed)return ok;
      if(w.st.dca_aggregate_adaptive_pivot){
        const double before=factor->pivot_tolerance();
        bool refactored=false;
        {CostScope stage(w.costs,NewtonCost::Factorization);const auto stamp=Clock::now();
          refactored=factor->increase_quality(message);stats.aggregate_factor_seconds+=seconds(stamp);}
        if(w.st.live_display_level>0)std::cout<<"[DCA-PIVOT] outer="<<outer_iteration<<" before="<<before
          <<" after="<<factor->pivot_tolerance()<<" refactored="<<refactored<<" same_matrix=1\n";
        if(refactored){ok=solve_refined(dual,eq,primal,comp,d);if(ok)return true;}
        // A failed factorization must not be used for another backsolve.
        else if(factor->pivot_tolerance()>before){root_factor_valid=false;return false;}
      }
      if(!precision_failed || precise_finish || !w.st.dca_aggregate_adaptive_dual || dual_shift>0)return false;
      double mu=0;int count=0;
      for(int i=0;i<w.mi;++i)if(w.active[i]){mu+=w.s[i]*w.z[i];++count;}
      mu=count?mu/count:0;
      dual_shift=1e-8*std::pow(std::max(mu,1e-300),0.25);
      if(!(dual_shift>0)||!std::isfinite(dual_shift))return false;
      // Copy the matrix only on this exceptional numerical recovery path.
      // Before this point the certified coefficients are still in matrix.
      for(int e=0;e<equalities;++e)matrix.push_back({w.nr+e,w.nr+e,-dual_shift});
      for(int i=0;i<w.mi;++i)if(w.active[i]&&w.inequality_kind[i]==0){
        if(weak_border[i]>=0)matrix.push_back({weak_border[i],weak_border[i],-dual_shift});
        else {const auto& g=w.jr[i];const double old=w.z[i]/w.s[i];
          const double change=-old*(dual_shift*w.z[i])/(w.s[i]+dual_shift*w.z[i]);
          for(int a=0;a<(int)g.support.size();++a)for(int b=a;b<(int)g.support.size();++b)
            matrix.push_back({g.support[a],g.support[b],change*g.values[a]*g.values[b]});}
      }
      bool factored=false;
      const double certified_rho=local_rho;
      const auto dual_matrix=matrix;
      // Dual perturbation weakens the constraint Schur contribution. A
      // formerly certified primal shift can then have the wrong inertia.
      // As in Ipopt, repair the primal block rather than ignoring inertia.
      for(int retry=0;retry<8;++retry){
        matrix=dual_matrix;
        if(local_rho>certified_rho)for(int j=0;j<w.nr;++j)
          matrix.push_back({j,j,local_rho-certified_rho});
        ++stats.aggregate_certificates;
        {CostScope stage(w.costs,NewtonCost::Factorization);const auto stamp=Clock::now();
          root_factor_valid=false;message.clear();factored=factor->factor(dimension,w.nr,matrix,message);stats.aggregate_factor_seconds+=seconds(stamp);}
        if(factored)break;
        if(message.find("inertia mismatch")==std::string::npos)break;
        const double next=local_rho>0?8*local_rho:1e-4;
        if(next>w.st.dca_aggregate_rho_max || retry==7)break;
        if(w.st.live_display_level>0)std::cout<<"[DCA-DUAL-INERTIA] outer="<<outer_iteration
          <<" delta_c="<<dual_shift<<" rho_before="<<local_rho<<" rho_after="<<next
          <<" reason=wrong-inertia\n";
        local_rho=next;
      }
      if(factored && local_rho>certified_rho){
        const double extra=local_rho-certified_rho;
        for(int j:w.retained){int k=models.slots.at({j,j});partial[k]+=extra;}
      }
      if(factored){
        capture_root_factor(w,dimension,weak_border,outer);
        stationarity_evidence.certificate(local_rho,true);
        last_attempt_certified=true;last_attempt_rho=local_rho;
        if(record_certificate){
          this->certified_rho=local_rho;
        }
      }
      if(w.st.live_display_level>0)std::cout<<"[DCA-DUAL-PIVOT] outer="<<outer_iteration
        <<" delta_c="<<dual_shift<<" mu="<<mu<<" rho="<<local_rho<<" factored="<<factored
        <<" retained_eq="<<equalities<<" original_curvature_certificate=retained detail=\""<<(factored?"ok":message)<<"\"\n";
      if(!factored)return false;
      ok=solve_refined(dual,eq,primal,comp,d);
#ifdef G10_DUAL_TEST_OBSERVER
      if(ok)G10_DUAL_TEST_OBSERVER(w,d,dual_shift,local_rho,dual,eq,primal,comp);
#endif
      if(ok && w.st.live_display_level>0){
        Direction check;const double regularized=full_residual(dual,eq,primal,comp,d,check);
        for(int e=0;e<w.me;++e)if(w.pivot_variable[e]<0)check.lambda[e]+=dual_shift*d.lambda[e];
        for(int i=0;i<w.mi;++i)if(w.active[i])check.s[i]+=inequality_shift(i)*d.z[i];
        const double scale=std::max({1.0,inf_norm(dual),inf_norm(eq),inf_norm(primal),inf_norm(comp)});
        std::cout<<"[DCA-DUAL-AUDIT] outer="<<outer_iteration<<" regularized="<<regularized
          <<" unregularized="<<std::max({inf_norm(check.x),inf_norm(check.lambda),inf_norm(check.s),inf_norm(check.z)})/scale<<"\n";
      }
      if(w.st.live_display_level>0)std::cout<<"[DCA-DUAL-PIVOT] outer="<<outer_iteration
        <<" retry_ok="<<ok<<" reduced="<<reduced_error<<" reconstructed="<<expanded_error
        <<" dlambda_inf="<<inf_norm(d.lambda)<<" dz_inf="<<inf_norm(d.z)<<"\n";
      return ok;
    };
    Vec rhs_dual=w.rd,rhs_eq=w.h,primal=w.rp,comp(w.mi,0);
    if(precise_finish){
      auto original=dca_precise_residual(w,stop_settings,w.x,w.s,w.z,w.lambda,outer_iteration-1);
      if(!original.valid)return reject("precise-original-residual-invalid");
      rhs_dual=std::move(original.dual);rhs_eq=std::move(original.equality);
      primal=std::move(original.primal);
    }
    for(int i=0;i<w.mi;++i)if(w.active[i]){
      comp[i]=w.s[i]*w.z[i];
    }
    Direction d,certified_affine;
    double current_mu=0.0;
    for(int direction_retry=0;direction_retry<2;++direction_retry){
    for(int i=0;i<w.mi;++i)if(w.active[i])comp[i]=w.s[i]*w.z[i];
    if(!solve(rhs_dual,rhs_eq,primal,comp,d)){
      if(w.st.live_display_level>=4)std::cout<<"[DCA-AGGREGATE-DIRECTION] "<<message<<" residual="<<d.residual<<"\n";
      return reject("newton-direction",precision_failed?detail::DcaAggregateRejection::numerical_direction:
                    detail::DcaAggregateRejection::backend);
    }
    const double predictor_shift=dual_shift;const double predictor_rho=local_rho;
    {
      // Predictor and corrector share the same certified aggregate factor.
      const Direction affine=d;certified_affine=affine;
      double affine_primal=1,affine_dual=1,mu=0;int weak_count=0;
      for(int i=0;i<w.mi;++i)if(w.active[i]){
        if(affine.z[i]<0)affine_dual=std::min(affine_dual,-w.z[i]/affine.z[i]);
        {
          if(affine.s[i]<0)affine_primal=std::min(affine_primal,-w.s[i]/affine.s[i]);
          mu+=w.s[i]*w.z[i];++weak_count;
        }
      }
      if(weak_count){
        mu/=weak_count;current_mu=mu;double affine_mu=0;
        for(int i=0;i<w.mi;++i)if(w.active[i])
          affine_mu+=(w.s[i]+affine_primal*affine.s[i])*(w.z[i]+affine_dual*affine.z[i]);
        affine_mu/=weak_count;
        const double sigma=std::clamp(std::pow(std::max(0.0,affine_mu)/std::max(mu,1e-300),3),1e-8,0.95);
        // Do not drive every slack product to zero in a single affine finish.
        // A positive central target below BOTH actual strict error caps avoids
        // avoidable degeneracy; the UNPERTURBED certificate still owns success.
        const double finish_target=settings.dca_strict_finish_mu_fraction*std::min(
            w.objective.scale*stop_settings.complementarity_tolerance,
            stop_settings.stationarity_tolerance*(stop_settings.termination_test=="ipopt"?1.0:w.objective.scale));
        const double target=precise_finish ? finish_target : std::max(0.001*w.objective.scale*
            std::min(w.st.complementarity_tolerance,w.st.stationarity_tolerance),sigma*mu);
        if(precise_finish){stats.strict_finish_mu_target=target;
          if(w.st.live_display_level>=2)std::cout<<"[DCA-STRICT-CENTER] outer="<<outer_iteration
              <<" mu="<<mu<<" target="<<target<<" fraction="<<settings.dca_strict_finish_mu_fraction<<" same_factor=1\n";}
        for(int i=0;i<w.mi;++i)if(w.active[i])
          comp[i]=w.s[i]*w.z[i]-target+affine.s[i]*affine.z[i];
        ++stats.corrector_steps;stats.last_sigma=sigma;
        if(w.st.live_display_level>=4)
          std::cout<<"[DCA-AGGREGATE-CORRECTOR] weak="<<weak_count<<" mu="<<mu
                   <<" affine_mu="<<affine_mu<<" sigma="<<sigma<<" target="<<target<<"\n";
        // Reuse the unchanged factor and audit the corrected original bordered
        // equations, exactly as for the predictor. No second factorization.
        if(!solve(rhs_dual,rhs_eq,primal,comp,d))return reject("newton-corrector",
            precision_failed?detail::DcaAggregateRejection::numerical_direction:detail::DcaAggregateRejection::backend);
      }
    }
    if(dual_shift==predictor_shift && local_rho==predictor_rho)break;
    }
    const bool terminal_alternative=detail::dca_stationary_phase(
        outer.primal,outer.normalized_kkt,w.st.feasibility_tolerance,w.st.stationarity_tolerance);
    int terminal_soc_solves=0;
    for(int variant=0;variant<((terminal_alternative || (settings.dca_aggregate_only && settings.dca_aggregate_affine_retry))?2:1);++variant){
    if(variant==1){d=certified_affine;
      if(w.st.live_display_level>=2)std::cout<<"[DCA-AGGREGATE-ALTERNATIVE] outer="<<outer_iteration
          <<" direction=affine factor_reused=1 rho="<<local_rho<<"\n";
    }
    double primal_alpha=1,dual_alpha=1;
    for(int i=0;i<w.mi;++i)if(w.active[i]){
      if(d.z[i]<0)dual_alpha=std::min(dual_alpha,-0.995*w.z[i]/d.z[i]);
      if(d.s[i]<0)primal_alpha=std::min(primal_alpha,-0.995*w.s[i]/d.s[i]);
    }
    if(std::min(primal_alpha,dual_alpha)<1e-6){
      dca_trace_domain(w,"aggregate","fraction",-1,primal_alpha,dual_alpha,"fraction-too-small",-1,trace_context);
      last_rejection="multiplier-or-slack-step";continue;
    }
    if(precise_finish)primal_alpha=dual_alpha=std::min(primal_alpha,dual_alpha);
    const double initial_primal_alpha=primal_alpha;
    for(int ls=0;ls<12;++ls,primal_alpha*=0.5){
      const double alpha=primal_alpha;
      const auto step=detail::primal_dual_step(initial_primal_alpha,dual_alpha,std::pow(0.5,ls),settings.dca_dual_step);
      const double z_alpha=step.bound_dual;
      const double lambda_alpha=detail::dca_equality_step(
          outer.primal,w.st.feasibility_tolerance,primal_alpha,step.equality_dual);
      Vec x=w.x,z=w.z,lambda=w.lambda,s=w.s;
      axpy(primal_alpha,d.x,x);axpy(z_alpha,d.z,z);axpy(lambda_alpha,d.lambda,lambda);
      for(int i=0;i<w.mi;++i)if(w.active[i])s[i]=w.s[i]+primal_alpha*d.s[i];
      auto accept=[&](bool soc_trial=false){
        struct Audited {Vec x,s,z,lambda;DcaPoint point;double ap=0,al=0,az=0,predicted=0;const char* stage=nullptr;bool curvature_descent=false,normal_corrected=false;};
        auto audit=[&](Vec cx,Vec cs,Vec cz,Vec cl,double ap,double al,double az,
                       const char* candidate_stage)->std::optional<Audited>{
        for(int i=0;i<w.mi;++i)if(w.active[i]&&(!(cz[i]>0)||!(cs[i]>0))){
          last_rejection="nonpositive-local-state";
          dca_trace_domain(w,"aggregate",candidate_stage,ls,ap,az,last_rejection,i,trace_context);
          return std::nullopt;
        }
        auto candidate=dca_original(w,cx,cl,cz,&w.lambda,&w.z);
        double precise_primal_error=INFINITY;
        const double precise_primal_target=stop_settings.termination_test=="ipopt"?
            stop_settings.stationarity_tolerance:std::min(stop_settings.stationarity_tolerance,stop_settings.feasibility_tolerance);
        if(precise_finish){
          auto snapshot=dca_precise_residual(w,stop_settings,cx,cs,cz,cl,outer_iteration,&w.lambda,&w.z);
          ++stats.strict_finish_certificates;
          if(!snapshot.valid){last_rejection="precise-original-residual-invalid";return std::nullopt;}
          candidate=snapshot.point;
          precise_primal_error=stop_settings.termination_test=="ipopt"?
              snapshot.certificate.termination_internal_primal_inf:snapshot.certificate.feasibility.total_inf;
        }
        bool normal_corrected=false;
        const bool synchronized_trial=std::abs(ap-al)<=1e-15 && std::abs(ap-az)<=1e-15;
        // Retract a synchronized ORIGINAL DCA trial rather than correcting
        // only the rejected split trial. Reuse this same certified matrix;
        // every normal RHS is checked by the expanded residual audit.
        while((terminal_alternative||precise_finish) && !soc_trial && synchronized_trial &&
              (precise_finish ? precise_primal_error>precise_primal_target :
               (candidate.primal>w.st.feasibility_tolerance || normal_corrected)) && terminal_soc_solves<2){
          Vec eq_rhs(w.me),primal_rhs(w.mi,0),comp_rhs(w.mi,0),dual_rhs(w.n,0);
          for(int e=0;e<w.me;++e)eq_rhs[e]=w.equalities[e].original_at(cx);
          for(int i=0;i<w.mi;++i)if(w.active[i])primal_rhs[i]=w.inequalities[i].original_at(cx)+cs[i];
          if(precise_finish){
            auto rhs=dca_precise_residual(w,stop_settings,cx,cs,cz,cl,outer_iteration);
            if(!rhs.valid)break;
            eq_rhs=std::move(rhs.equality);primal_rhs=std::move(rhs.primal);
          }
          Direction normal;++terminal_soc_solves;++stats.soc_attempts;
          if(!solve(dual_rhs,eq_rhs,primal_rhs,comp_rhs,normal))break;
          Vec base_dx=cx;axpy(-1,w.x,base_dx);
          if(inf_norm(normal.x)>std::max(1e-8,0.5*inf_norm(base_dx)))break;
          Vec nx=cx,nl=cl,nz=cz,ns=cs;axpy(1,normal.x,nx);axpy(1,normal.lambda,nl);axpy(1,normal.z,nz);axpy(1,normal.s,ns);
          bool positive=finite(nx)&&finite(nl)&&finite(nz)&&finite(ns);
          for(int i=0;i<w.mi;++i)if(w.active[i])positive=positive&&ns[i]>0&&nz[i]>0;
          if(!positive)break;
          auto point=dca_original(w,nx,nl,nz,&w.lambda,&w.z);
          if(precise_finish){
            auto snapshot=dca_precise_residual(w,stop_settings,nx,ns,nz,nl,outer_iteration,&w.lambda,&w.z);
            if(!snapshot.valid)break;
            point=snapshot.point;
            precise_primal_error=stop_settings.termination_test=="ipopt"?
                snapshot.certificate.termination_internal_primal_inf:snapshot.certificate.feasibility.total_inf;
          }
          if(!(point.primal<candidate.primal))break;
          if(w.st.live_display_level>=2)std::cout<<"[DCA-AGGREGATE-NORMAL] outer="<<outer_iteration
              <<" ls="<<ls<<" before="<<candidate.primal<<" after="<<point.primal
              <<" normal="<<inf_norm(normal.x)<<" factor_reused=1\n";
          cx.swap(nx);cl.swap(nl);cz.swap(nz);cs.swap(ns);candidate=point;normal_corrected=true;
        }
        if(precise_finish){
          auto snapshot=dca_precise_residual(w,stop_settings,cx,cs,cz,cl,outer_iteration,&w.lambda,&w.z);
          if(!snapshot.valid){last_rejection="precise-original-residual-invalid";return std::nullopt;}
          candidate=snapshot.point;
          if(!snapshot.strict && !(snapshot.certificate.scaled_nlp_error<outer.normalized_kkt*(1-1e-3*ap))){
            last_rejection="precise-error-no-contraction";return std::nullopt;
          }
        }
        stationarity_evidence.candidate(local_rho,outer.kkt,candidate.kkt);
        Vec dx=cx;axpy(-1,w.x,dx);
        double quadratic=0;
        long double shifted_quadratic=0,absolute_terms=0,retained_norm2=0;
        for(int k=0;k<(int)partial.size();++k){auto [a,b]=w.hpattern[k];
          quadratic+=(a==b?0.5:1)*partial[k]*dx[a]*dx[b];
          if(terminal_alternative){const long double term=(a==b?0.5L:1.L)*partial[k]*dx[a]*dx[b];
            shifted_quadratic+=term;absolute_terms+=std::abs(term);}
        }
        if(terminal_alternative)for(int j:w.retained)retained_norm2+=(long double)dx[j]*dx[j];
        const long double original_curvature=2*shifted_quadratic-local_rho*retained_norm2;
        const long double curvature_roundoff=128*std::numeric_limits<double>::epsilon()*
            (2*absolute_terms+local_rho*retained_norm2);
        const double predicted=-w.objective.directional(dx)-quadratic;
        const double objective_noise=32*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(outer.objective));
        // Small KKT plus feasibility does not imply a local minimizer. A
        // convexified DCA step can descend along negative ORIGINAL curvature
        // while stationarity initially grows (e.g. f=-x^2/2). Keep the actual
        // convexity certificate, original objective Armijo, raw feasibility,
        // persistent growth ceiling and ordinary filter HISTORY. Only the
        // terminal stationarity-monotonicity assumption is inapplicable.
        const bool curvature_descent=terminal_alternative && !outer.acceptable_certified && local_rho>0 &&
            original_curvature < -curvature_roundoff &&
            predicted>objective_noise && candidate.objective<=outer.objective-1e-4*predicted &&
            candidate.primal<=w.st.feasibility_tolerance &&
            detail::dca_kkt_growth_allowed(dca_kkt_guard_base(outer),candidate.kkt,
                w.st.dca_kkt_growth_max,10*w.st.acceptable_tolerance);
        if(w.st.live_display_level>=4 && terminal_alternative)std::cout<<"[DCA-CURVATURE-DESCENT-TEST] outer="<<outer_iteration
            <<" rho="<<local_rho<<" original_curvature="<<double(original_curvature)
            <<" curvature_roundoff="<<double(curvature_roundoff)<<" predicted="<<predicted
            <<" df="<<candidate.objective-outer.objective<<" kkt="<<candidate.kkt
            <<" primal="<<candidate.primal<<" eligible="<<curvature_descent<<"\n";
        if(candidate.kkt<last_candidate_kkt){
          last_candidate_kkt=candidate.kkt;
          last_candidate_primal=candidate.primal;
        }
        if(candidate_output && candidate.primal>w.st.feasibility_tolerance){
          last_rejection="rho-probe-primal-loss";
          dca_trace_filter(w,filter,outer,candidate,0.0,false,"aggregate",
                           candidate_stage,ls,ap,az,last_rejection,-1,trace_context);
          return std::nullopt;
        }
        if(candidate_output && !precise_finish && !curvature_descent && !detail::dca_rho_probe_candidate_allowed(
             outer.kkt,candidate.kkt,candidate.primal,
             w.st.feasibility_tolerance)){
          last_rejection="rho-probe-no-current-kkt-gain";
          dca_trace_filter(w,filter,outer,candidate,0.0,false,"aggregate",
                           candidate_stage,ls,ap,az,last_rejection,-1,trace_context);
          return std::nullopt;
        }
        double aggregate_gap=0.0;
        if(w.st.dca_dual_decomposition) {
          for(int j:w.retained)aggregate_gap=std::max(aggregate_gap,std::abs(local_rho*d.x[j]));
          aggregate_gap/=std::max(1.0,w.objective.scale);
          dca_trace_dual_decomposition(w,outer,candidate,w.x,w.lambda,w.z,d,
              ap,al,az,"aggregate",candidate_stage,d.residual,
              aggregate_gap,trace_context);
        }
        bool used_feasible_recovery=false;
        if(!curvature_descent && !dca_kkt_allowed(w,outer,candidate,&used_feasible_recovery)){
          last_rejection="insufficient-kkt-contraction";
          dca_trace_filter(w,filter,outer,candidate,predicted,false,"aggregate",candidate_stage,ls,ap,az,"kkt-growth",-1,trace_context);
          return std::nullopt;
        }
        const bool synchronized=std::abs(ap-al)<=1e-15 && std::abs(ap-az)<=1e-15;
        const double terminal_fraction=terminal_alternative && synchronized ? ap : 1.0;
        if(!filter.acceptable(outer,candidate,predicted,w.st.feasibility_tolerance,terminal_fraction,curvature_descent)){
          last_rejection="original-filter";
          dca_trace_filter(w,filter,outer,candidate,predicted,false,"aggregate",candidate_stage,ls,ap,az,nullptr,-1,trace_context);
          return std::nullopt;
        }
        if(curvature_descent && w.st.live_display_level>0)std::cout<<"[DCA-CURVATURE-DESCENT] outer="<<outer_iteration
            <<" rho="<<local_rho<<" original_curvature="<<double(original_curvature)
            <<" objective_change="<<candidate.objective-outer.objective<<" predicted="<<predicted
            <<" from_kkt="<<outer.kkt<<" candidate_kkt="<<candidate.kkt
            <<" primal="<<candidate.primal<<" filter_history=checked stage="<<candidate_stage<<"\n";
        if(used_feasible_recovery && w.st.live_display_level>0)
          std::cout<<"[DCA-KKT-RECOVERY] outer="<<outer_iteration
                   <<" from_kkt="<<outer.kkt
                   <<" candidate_kkt="<<candidate.kkt
                   <<" from_primal="<<outer.primal
                   <<" candidate_primal="<<candidate.primal
                   <<" rho="<<local_rho
                   <<" stage="<<candidate_stage
                   <<" persistent_reference="<<dca_kkt_guard_base(outer)
                   <<"\n";
        // Guard every inactive original row; a changed working set is rebuilt by general DCA.
        for(int i=0;i<w.mi;++i)if(!w.active[i]&&w.inequalities[i].original_at(cx)>w.st.feasibility_tolerance*w.inequalities[i].scale){
          last_rejection="inactive-original-row";
          dca_trace_filter(w,filter,outer,candidate,predicted,false,"aggregate",candidate_stage,ls,ap,az,last_rejection,i,trace_context);
          return std::nullopt;
        }
        if(!curvature_descent)dca_trace_filter(w,filter,outer,candidate,predicted,false,"aggregate",candidate_stage,ls,ap,az,nullptr,-1,trace_context);
        return Audited{std::move(cx),std::move(cs),std::move(cz),std::move(cl),candidate,ap,al,az,predicted,candidate_stage,curvature_descent,normal_corrected};
        };

        const char* stage=soc_trial?"soc":"trial";
        auto chosen=audit(x,s,z,lambda,primal_alpha,lambda_alpha,z_alpha,stage);
        const bool terminal_common=!chosen && detail::dca_stationary_phase(
            outer.primal,outer.normalized_kkt,w.st.feasibility_tolerance,w.st.stationarity_tolerance);
        if(!soc_trial && settings.dca_dual_step=="coupled" &&
           (terminal_common || (chosen && detail::compare_common_step(outer.primal,outer.dual,chosen->point.dual,
             settings.dca_common_step_primal_trigger,settings.dca_common_step_dual_growth)))) {
          const double common=detail::common_feasible_step(primal_alpha,lambda_alpha,z_alpha);
          if(common>0 && (std::abs(common-primal_alpha)>1e-15 ||
                          std::abs(common-lambda_alpha)>1e-15 ||
                          std::abs(common-z_alpha)>1e-15)) {
            Vec cx=w.x,cs=w.s,cz=w.z,cl=w.lambda;
            axpy(common,d.x,cx);axpy(common,d.lambda,cl);axpy(common,d.z,cz);
            for(int i=0;i<w.mi;++i)if(w.active[i])cs[i]=w.s[i]+common*d.s[i];
            auto synchronized=audit(std::move(cx),std::move(cs),std::move(cz),std::move(cl),
                                    common,common,common,"common-trial");
            if(synchronized && (!chosen || synchronized->point.kkt<chosen->point.kkt))
              chosen=std::move(synchronized);
          }
        }
        if(!chosen)return false;
        if(candidate_output){
          *candidate_output=Candidate{std::move(chosen->x),std::move(chosen->s),
              std::move(chosen->z),std::move(chosen->lambda),chosen->point,
              chosen->ap,chosen->al,chosen->az,local_rho,last_backward_error,
              stats.last_sigma,chosen->stage,soc_trial||chosen->normal_corrected,chosen->curvature_descent};
          last_rejection="none";
          return true;
        }
        if(chosen->point.theta>w.st.feasibility_tolerance)filter.insert(outer);
        w.x=std::move(chosen->x);w.lambda=std::move(chosen->lambda);w.z=std::move(chosen->z);
        if(w.st.live_display_level>=4)std::cout<<"[DCA-AGGREGATE-ACCEPT] rho="<<local_rho
          <<" alpha_pr="<<chosen->ap<<" alpha_z="<<chosen->az<<" alpha_y="<<chosen->al
          <<" ls="<<ls<<" mu="<<current_mu<<" kkt="<<chosen->point.kkt
          <<" selection="<<chosen->stage<<"\n";
        if(chosen->stage==std::string("common-trial") && w.st.live_display_level>0)
          std::cout<<"[DCA-STEP-SELECT] path=aggregate outer="<<outer_iteration
                   <<" mode=common-alpha alpha="<<chosen->ap<<" original_kkt="<<chosen->point.kkt<<"\n";
        w.s=std::move(chosen->s);
        stats.aggregate_max_accepted_residual=std::max(stats.aggregate_max_accepted_residual,last_backward_error);
        stats.aggregate_max_accepted_rho=std::max(stats.aggregate_max_accepted_rho,local_rho);
        accepted_rho=local_rho;stats.aggregate_last_rho=local_rho;
        if(chosen->curvature_descent)++stats.aggregate_curvature_descents;
        if(chosen->normal_corrected)++stats.aggregate_soc;
        dca_inherit_guard(w,outer,chosen->point);
        outer=chosen->point;++stats.aggregate_acceptances;last_rejection="none";return true;
      };
      if(accept())return true;
      if(ls==0){
        Vec eq(w.me),p(w.mi,0),c(w.mi,0),dual(w.n,0);
        for(int e=0;e<w.me;++e)eq[e]=w.equalities[e].original_at(x);
        if(precise_finish){
          auto rhs=dca_precise_residual(w,stop_settings,x,s,z,lambda,outer_iteration);
          if(!rhs.valid)continue;
          eq=std::move(rhs.equality);p=std::move(rhs.primal);
        }
        Direction soc;
        if(solve(dual,eq,p,c,soc)&&inf_norm(soc.x)<=std::max(1e-8,0.5*alpha*inf_norm(d.x))){
          axpy(1,soc.x,x);axpy(1,soc.lambda,lambda);axpy(1,soc.z,z);axpy(1,soc.s,s);
          if(accept(true)){if(!candidate_output)++stats.aggregate_soc;return true;}
        }
      }
    }
    }
    return reject(last_rejection);
  }
};
