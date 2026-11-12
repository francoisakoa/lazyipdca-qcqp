// Included inside lazy_newton.cpp's anonymous namespace, after Workspace.
// Diagnostics never change a coefficient, multiplier, acceptance gate or factor.
class LiveNewton {
  Workspace& w;
  LazyNewtonStatistics& stats;
  Clock::time_point start;
  int printed=0,last_outer=-1;
  LazyNewtonIteration pending;
  bool pending_valid=false;
  int retry_factors=0,retry_solves=0,retry_backtracks=0,retry_activations=0;

public:
  const bool enabled;
  const int level;
  LiveNewton(Workspace& workspace,Clock::time_point solver_start)
      :w(workspace),stats(w.stats),start(solver_start),
       enabled(w.st.live_iterations||w.st.live_curvature),level(w.st.live_display_level) {
    stats.live_iterations_enabled=enabled;stats.live_curvature_enabled=w.st.live_curvature;
    if(!enabled)return;
    CostScope display_cost(w.costs,NewtonCost::Diagnostics);const auto timer=Clock::now();
    std::ostringstream out;
    if(level<4) {
      out<<"\nLazyIPDCA  |  Newton\n";
      if(level>=2)out<<"Model: "<<w.original.n<<" variables, "<<w.me<<" equalities, "<<w.original.inequalities.size()<<" inequalities"
        <<"  |  Reduced KKT: "<<w.dim<<"\n";
      if(level>=2)out<<"Columns: wsI active user inequalities; +I activated; fac factors; sol linear solves.\n";
      if(level>=3)out<<"Lazy: a/r atoms/rank; F objective, I inequalities, E equalities, C completion; rR total rank; rk <= means rank bounds; reuse previous correction; traceR correction trace.\n";
      std::cout<<out.str()<<std::flush;stats.diagnostic_seconds+=seconds(timer);return;
    }
    out<<"[LAZY-NEWTON-MODE] signed Lagrangian + "<<(w.st.lazy_newton_atomic_regularization?"lazy sparse atoms":"uniform rho control")<<"; original Newton globalisation\n"
       <<"[LAZY-NEWTON-STRUCTURE] original_n="<<w.original.n<<" free_n="<<w.n
       <<" reduced_n="<<w.nr<<" reduced_eq="<<w.mr<<" kkt_dim="<<w.dim
       <<" kkt_upper_slots="<<w.kkt.size()<<" dense_borders="<<w.ndense<<"\n"
       <<"[LAZY-NEWTON-SOURCES] objective: atoms=0 correction_rank=0 Hessian_nnz="<<stats.objective_hessian_nnz
       <<"; inequalities: atoms=0 correction_rank=0 quadratic="<<stats.quadratic_inequalities
       <<" Hessian_nnz_sum="<<stats.inequality_hessian_nnz
       <<"; equalities: signed quadratic="<<stats.quadratic_equalities
       <<" Hessian_nnz_sum="<<stats.equality_hessian_nnz<<"\n"
       <<"[LAZY-NEWTON-LEGEND] STEP/WS+/LSFAIL: x_k BEFORE attempted step; INIT/STOP: current point. "
         "inf_* original units; comp_s=max(s*z)/objective_scale; lg(mu) internal scaling. "
         "rho: uniform control only; rR: active atomic numerical rank (upper bound if ranks_exact=0); rN: equivalent normal augmentation rank (not convexification).\n";
    if(w.st.live_curvature)
      out<<"[CURVATURE-LEGEND] aF/rF,aI/rI,aE/rE: active source atoms and reduced numerical ranks; ranks_exact=0 means rank upper bounds. "
           "cF,cI,cE: signed scaled Hessian contributions d'Hd/(d'd); bI,bB: barrier terms for inequalities/bounds; "
           "cR: actual atomic/uniform correction. These directional values are NOT eigenvalues or tangent-positivity certificates.\n";
    std::cout<<out.str()<<std::flush;
    stats.diagnostic_seconds+=seconds(timer);
  }

  void atomic_state(LazyNewtonIteration& r) {
    if(w.atoms) {
      r.objective_atoms=w.atoms->active_count(0);r.inequality_atoms=w.atoms->active_count(1);
      r.equality_atoms=w.atoms->active_count(2);r.completion_atoms=w.atoms->active_count(3);
      r.atomic_active=w.atoms->active_count();r.atomic_trace=w.atoms->trace();
      r.objective_correction_rank=w.atoms->rank(0,w.nr,r.atom_ranks_exact);
      r.inequality_correction_rank=w.atoms->rank(1,w.nr,r.atom_ranks_exact);
      r.equality_correction_rank=w.atoms->rank(2,w.nr,r.atom_ranks_exact);
      r.regularization_rank=w.atoms->rank(-1,w.nr,r.atom_ranks_exact);
    }
  }

  LazyNewtonIteration state(int iteration) {
    CostScope display_cost(w.costs,NewtonCost::Diagnostics);const auto timer=Clock::now();LazyNewtonIteration r;
    r.iteration=iteration;r.attempt=stats.numeric_updates;
    r.objective=w.objective.value/w.objective.scale;
    r.inf_du=inf_norm(w.rd)/w.objective.scale;
    r.total_inequalities=(int)w.original.inequalities.size();
    int active_count=0;
    for(int e=0;e<w.me;++e) {
      r.inf_eq=std::max(r.inf_eq,std::abs(w.h[e]/w.equalities[e].scale));
      if(w.lambda[e]!=0.0&&!w.equalities[e].hessian.empty())++r.weighted_quadratic_equalities;
    }
    for(int i=0;i<w.mi;++i) {
      const auto& q=w.inequalities[i];const double violation=std::max(0.0,q.value/q.scale);
      if(w.inequality_kind[i]==0)r.inf_ineq=std::max(r.inf_ineq,violation);
      else r.inf_bnd=std::max(r.inf_bnd,violation);
      if(!w.active[i])continue;
      ++active_count;r.mu+=w.s[i]*w.z[i];r.comp=std::max(r.comp,w.s[i]*w.z[i]/w.objective.scale);
      if(w.inequality_kind[i]==0)++r.active_inequalities;else ++r.active_bounds;
      if(!q.hessian.empty()) {
        ++r.active_quadratic_inequalities;
        if(w.z[i]!=0.0)++r.weighted_quadratic_inequalities;
      }
      if(w.jr[i].dense>=0)++r.active_dense_constraints;
    }
    if(active_count)r.mu/=active_count;
    // Store negative cumulative counters, completed at emit(). This also
    // accounts for unsuccessful factors, refinements and working-set retries.
    r.factor_attempts=-stats.factor_attempts;r.linear_solves=-stats.linear_solves;
    r.backtracks=-stats.line_search_backtracks;r.activations=-stats.constraint_activations;
    r.regularization_increases=-stats.regularization_increases;
    r.native_attempts=-stats.pivoted_fallbacks;r.recovery_attempts=-stats.normal_recovery_attempts;
    r.condensation_recoveries=-stats.condensation_recoveries;
    r.promoted_inequalities=-stats.promoted_inequalities;r.promoted_bounds=-stats.promoted_bounds;
    r.factor_seconds=-stats.factor_seconds;r.solve_seconds=-stats.solve_seconds;
    stats.diagnostic_seconds+=seconds(timer);return r;
  }

  void direction(LazyNewtonIteration& r,const Direction& d) {
    CostScope display_cost(w.costs,NewtonCost::Diagnostics);const auto timer=Clock::now();r.has_direction=true;r.inertia_satisfied=true;
    r.direction_inf=inf_norm(d.x);r.direction_residual=d.residual;
    r.regularization=w.regularization;r.regularization_rank=w.regularization>0.0?w.nr:0;
    if(level>=3)atomic_state(r);
    r.factor_backend=w.current_factor->name();
    if(w.recovery&&w.current_factor==w.recovery.get()) {
      r.normal_augmentation_rank=1;r.normal_augmentation_shift=w.recovery_shift;
    }
    if(w.st.live_curvature&&r.direction_inf>0.0&&std::isfinite(r.direction_inf)) {
      Vec u=d.x;double norm2=0.0;for(double& v:u){v/=r.direction_inf;norm2+=v*v;}
      auto quadratic=[&](const Quadratic& q) {
        q.note(&NewtonCalls::hessian_products);
        long double c=0.0;
        for(const auto& e:q.hessian)c+=(e.i==e.j?1.0L:2.0L)*e.v*u[e.i]*u[e.j];
        return (double)(q.scale*c/norm2);
      };
      r.objective_curvature=quadratic(w.objective);r.inequality_curvature=r.equality_curvature=0.0;
      r.inequality_barrier_curvature=r.bound_barrier_curvature=r.regularization_curvature=0.0;
      for(int e=0;e<w.me;++e)r.equality_curvature+=w.lambda[e]*quadratic(w.equalities[e]);
      for(int i=0;i<w.mi;++i)if(w.active[i]) {
        const auto& q=w.inequalities[i];r.inequality_curvature+=w.z[i]*quadratic(q);
        const double a=q.directional(u),b=(w.z[i]/w.s[i])*a*a/norm2;
        if(w.inequality_kind[i]==0)r.inequality_barrier_curvature+=b;else r.bound_barrier_curvature+=b;
      }
      if(w.atoms)r.regularization_curvature=w.atoms->curvature(u,w.retained)/norm2;
      else for(int j:w.retained)r.regularization_curvature+=w.regularization*u[j]*u[j]/norm2;
      r.lagrangian_curvature=r.objective_curvature+r.inequality_curvature+r.equality_curvature;
      r.barrier_newton_curvature=r.lagrangian_curvature+r.inequality_barrier_curvature+
          r.bound_barrier_curvature+r.regularization_curvature;
      r.curvature_available=std::isfinite(r.barrier_newton_curvature);
    }
    stats.diagnostic_seconds+=seconds(timer);
  }

  void begin_iteration(const LazyNewtonIteration& state) {
    if(level>=4)return;
    CostScope display_cost(w.costs,NewtonCost::Diagnostics);
    outer_row(state,false);
  }
  void work_header(std::ostream& out) const {
    if(level>=2)out<<" |"<<std::setw(7)<<"wsI"<<std::setw(5)<<"+I"<<std::setw(5)<<"fac"<<std::setw(5)<<"sol";
  }
  void work_columns(std::ostream& out,const LazyNewtonIteration& r,int active) const {
    if(level>=2)out<<" |"<<std::setw(7)<<active<<std::setw(5)<<r.activations<<std::setw(5)<<r.factor_attempts<<std::setw(5)<<r.linear_solves;
  }
  static void atom_header(std::ostream& out) {
    out<<" |"<<std::setw(6)<<"aF"<<std::setw(6)<<"rF"<<std::setw(6)<<"aI"<<std::setw(6)<<"rI"
       <<std::setw(6)<<"aE"<<std::setw(6)<<"rE"<<std::setw(6)<<"aC"<<std::setw(6)<<"rR"
       <<std::setw(4)<<"rk"<<std::setw(6)<<"reuse"<<std::setw(11)<<"traceR";
  }
  static void atom_columns(std::ostream& out,const LazyNewtonIteration& r) {
    out<<" |"<<std::setw(6)<<r.objective_atoms<<std::setw(6)<<r.objective_correction_rank
       <<std::setw(6)<<r.inequality_atoms<<std::setw(6)<<r.inequality_correction_rank
       <<std::setw(6)<<r.equality_atoms<<std::setw(6)<<r.equality_correction_rank
       <<std::setw(6)<<r.completion_atoms<<std::setw(6)<<r.regularization_rank
       <<std::setw(4)<<(r.atom_ranks_exact?"=":"<=")<<std::setw(6)<<(r.regularization_reused?"yes":"-")
       <<std::scientific<<std::setprecision(2)<<std::setw(11)<<r.atomic_trace;
  }
  void outer_row(const LazyNewtonIteration& r,bool force) {
    if(r.iteration==last_outer)return;
    if(!force&&r.iteration%std::max(1,w.st.live_iteration_every))return;
    last_outer=r.iteration;
    std::ostringstream out;
    if(printed++%std::max(1,w.st.live_iteration_header_every)==0) {
      out<<"\n iter      objective     inf_pr     inf_du   lg(mu)      ||d||   alpha_du   alpha_pr   ls";
      work_header(out);if(level>=3)atom_header(out);out<<"\n";
    }
    out<<std::setw(5)<<r.iteration<<std::scientific<<std::setprecision(7)<<std::setw(16)<<r.objective
       <<std::setprecision(2)<<std::setw(11)<<std::max({r.inf_eq,r.inf_ineq,r.inf_bnd})<<std::setw(11)<<r.inf_du
       <<std::fixed<<std::setprecision(1)<<std::setw(9)<<(r.mu>0?std::log10(r.mu):-std::numeric_limits<double>::infinity());
    if(pending_valid)out<<std::scientific<<std::setprecision(2)<<std::setw(11)<<pending.direction_inf
       <<std::setw(11)<<pending.alpha_dual<<std::setw(11)<<pending.alpha_primal<<std::setw(5)<<pending.backtracks;
    else out<<std::setw(11)<<"-"<<std::setw(11)<<"-"<<std::setw(11)<<"-"<<std::setw(5)<<"-";
    const auto& step=pending_valid?pending:r;
    work_columns(out,step,r.active_inequalities);
    if(level>=3)atom_columns(out,step);
    out<<"\n";
    std::cout<<out.str()<<std::flush;
  }

  void emit(LazyNewtonIteration r,const char* event,bool force=false) {
    CostScope display_cost(w.costs,NewtonCost::Diagnostics);const auto timer=Clock::now();r.event=event;r.attempt=stats.numeric_updates;
    if(!r.has_direction){r.regularization=w.regularization;if(level>=3)atomic_state(r);}
    r.factor_attempts+=stats.factor_attempts;r.linear_solves+=stats.linear_solves;
    r.backtracks+=stats.line_search_backtracks;r.activations+=stats.constraint_activations;
    r.regularization_increases+=stats.regularization_increases;
    r.native_attempts+=stats.pivoted_fallbacks;r.recovery_attempts+=stats.normal_recovery_attempts;
    r.condensation_recoveries+=stats.condensation_recoveries;r.kkt_dimension=w.dim;
    r.promoted_inequalities+=stats.promoted_inequalities;r.promoted_bounds+=stats.promoted_bounds;
    r.factor_seconds+=stats.factor_seconds;r.solve_seconds+=stats.solve_seconds;
    r.symbolic_analyses=w.factor->symbolic_analyses()+(w.recovery?w.recovery->symbolic_analyses():0);
    r.symbolic_reuses=w.factor->symbolic_reuses()+(w.recovery?w.recovery->symbolic_reuses():0);
    r.elapsed_seconds=seconds(start);if(level>=2)stats.iteration_history.push_back(r);
    if(level<4) {
      const std::string tag(event);
      if(tag=="WS+"){retry_factors+=r.factor_attempts;retry_solves+=r.linear_solves;retry_backtracks+=r.backtracks;retry_activations+=r.activations;}
      if(tag=="STEP") {
        pending=r;pending.factor_attempts+=retry_factors;pending.linear_solves+=retry_solves;
        pending.backtracks+=retry_backtracks;pending.activations+=retry_activations;
        retry_factors=retry_solves=retry_backtracks=retry_activations=0;pending_valid=true;
      }
      if(tag=="INIT"||tag=="SOLVED"||tag=="STOP")outer_row(r,true);
      stats.diagnostic_seconds+=seconds(timer);return;
    }
    if(force||r.native_attempts||r.recovery_attempts||r.regularization_increases||r.condensation_recoveries||
       r.iteration%std::max(1,w.st.live_iteration_every)==0) {
      std::ostringstream out;
      if(printed++%std::max(1,w.st.live_iteration_header_every)==0) {
        out<<"[ITER]    k     objective    inf_eq  inf_ineq   inf_bnd    inf_du    comp_s lg(mu)"
              "    ||d||     alpha_p   alpha_d       rho  rR rN    wsI/total   bnd  ls fac tag";
        atom_header(out);out<<"\n";
      }
      out<<"[ITER] "<<std::setw(4)<<r.iteration<<std::scientific<<std::setprecision(6)<<std::setw(14)<<r.objective
         <<std::setprecision(2)<<std::setw(10)<<r.inf_eq<<std::setw(10)<<r.inf_ineq<<std::setw(10)<<r.inf_bnd
         <<std::setw(10)<<r.inf_du<<std::setw(10)<<r.comp<<std::fixed<<std::setprecision(1)<<std::setw(7)
         <<(r.mu>0.0?std::log10(r.mu):-std::numeric_limits<double>::infinity())
         <<std::scientific<<std::setprecision(2)<<std::setw(10)<<r.direction_inf<<std::setw(11)<<r.alpha_primal
         <<std::setw(10)<<r.alpha_dual<<std::setw(10)<<r.regularization<<" "<<std::setw(4)<<r.regularization_rank
         <<std::setw(3)<<r.normal_augmentation_rank<<" "<<std::setw(6)<<r.active_inequalities<<"/"<<r.total_inequalities
         <<std::setw(6)<<r.active_bounds<<std::setw(4)<<r.backtracks<<std::setw(4)<<r.factor_attempts<<" "<<std::left<<std::setw(8)<<event<<std::right;
      atom_columns(out,r);out<<"\n";
      if(w.st.live_curvature&&r.has_direction) {
        out<<"[CURV] k="<<r.iteration<<" try="<<r.attempt<<" aF/rF="<<r.objective_atoms<<"/"<<r.objective_correction_rank
           <<" aI/rI="<<r.inequality_atoms<<"/"<<r.inequality_correction_rank
           <<" aE/rE="<<r.equality_atoms<<"/"<<r.equality_correction_rank
           <<" aN="<<r.completion_atoms<<" atoms="<<r.atomic_active<<" traceR="<<r.atomic_trace
           <<" ranks_exact="<<r.atom_ranks_exact
           <<" cF="<<r.objective_curvature<<" cI="<<r.inequality_curvature<<" cE="<<r.equality_curvature
           <<" bI="<<r.inequality_barrier_curvature<<" bB="<<r.bound_barrier_curvature<<" cR="<<r.regularization_curvature
           <<" cTotal="<<r.barrier_newton_curvature<<" qI="<<r.active_quadratic_inequalities<<"/"<<stats.quadratic_inequalities
           <<" R_reuse="<<r.regularization_reused<<" shift_up="<<r.regularization_increases
           <<" normal_rank="<<r.normal_augmentation_rank<<" normal_tau="<<r.normal_augmentation_shift<<"\n"
           <<"[KKT] k="<<r.iteration<<" try="<<r.attempt<<" backend="<<r.factor_backend
           <<" inertia_ok="<<r.inertia_satisfied<<" required="<<w.nr<<"/"<<w.dim-w.nr<<"/0"
           <<" fac="<<r.factor_attempts<<" solves="<<r.linear_solves<<" native="<<r.native_attempts
           <<" recovery="<<r.recovery_attempts<<" symbolic="<<r.symbolic_analyses<<" reuse="<<r.symbolic_reuses
           <<" dim="<<r.kkt_dimension<<" condensation="<<r.condensation_recoveries
           <<" promoted_I="<<r.promoted_inequalities<<" promoted_B="<<r.promoted_bounds
           <<" direction_residual="<<r.direction_residual<<" fac_ms="<<1e3*r.factor_seconds
           <<" solve_ms="<<1e3*r.solve_seconds<<" ws_added="<<r.activations<<" tag="<<event<<"\n";
      }
      std::cout<<out.str()<<std::flush;
    }
    stats.diagnostic_seconds+=seconds(timer);
  }
};
