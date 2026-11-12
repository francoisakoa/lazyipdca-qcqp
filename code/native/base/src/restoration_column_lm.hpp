// Bounded history of the ORIGINAL maximum violation, independent of the
// least-squares merit and the normal-model ratio. A stalled private method
// only releases its unused budget to another native restoration method.
struct ColumnLmPhysicalStall {
  std::vector<double> values;
  static constexpr std::size_t window=16;
  bool observe(double primal,double tolerance) {
    if(!std::isfinite(primal)||primal<0||!std::isfinite(tolerance)||!(tolerance>0))return false;
    values.push_back(primal);
    if(values.size()<=window)return false;
    const double reference=values.front();values.erase(values.begin());
    return primal>tolerance&&primal>std::max(tolerance,.9*reference);
  }
};

bool column_lm_finite_vector(const Vec& x) {
  for(double value:x)if(!std::isfinite(value))return false;
  return true;
}

// Evaluate EVERY raw original row before taking positive parts or maxima.
// An inactive inequality with NaN value must never disappear from the model.
bool column_lm_checked_values(const QCQPProblem& p,const Vec& x,
                              double& maximum,long double& merit) {
  maximum=0;merit=0;
  if(x.size()!=std::size_t(p.n)||p.lb.size()!=x.size()||p.ub.size()!=x.size()||!column_lm_finite_vector(x))return false;
  if(!std::isfinite(p.objective.value(x))||!column_lm_finite_vector(p.objective.gradient(x)))return false;
  LegacySparseGradientWorkspace workspace(p.n);
  auto finite_gradient=[&](const QuadraticFunction& q) {
    const auto row=workspace.gradient(q,x);
    for(const auto& e:row)if(e.first<0||e.first>=p.n||!std::isfinite(e.second))return false;
    return true;
  };
  auto add=[&](double residual,bool squared) {
    if(!std::isfinite(residual)||residual<0)return false;
    maximum=std::max(maximum,residual);
    if(squared)merit+=.5L*static_cast<long double>(residual)*residual;
    return std::isfinite(merit)&&std::isfinite(static_cast<double>(merit));
  };
  for(const auto& q:p.equalities) {
    const double value=q.value(x);if(!std::isfinite(value)||!finite_gradient(q)||!add(std::abs(value),true))return false;
  }
  for(const auto& q:p.inequalities) {
    const double value=q.value(x);if(!std::isfinite(value)||!finite_gradient(q)||!add(std::max(0.,value),true))return false;
  }
  for(int j=0;j<p.n;++j) {
    if(std::isnan(p.lb[j])||std::isnan(p.ub[j]))return false;
    if(std::isfinite(p.lb[j])&&!add(std::max(0.,p.lb[j]-x[j]),false))return false;
    if(std::isfinite(p.ub[j])&&!add(std::max(0.,x[j]-p.ub[j]),false))return false;
  }
  return true;
}

bool column_lm_checked_jacobian(const LegacyResidualJacobian& rj,int n) {
  if(!rj.sparse||rj.sparse_rows.size()!=rj.residual.size()||!column_lm_finite_vector(rj.residual))return false;
  for(const auto& row:rj.sparse_rows)for(const auto& e:row)
    if(e.first<0||e.first>=n||!std::isfinite(e.second))return false;
  return true;
}

// Column-equilibrated Levenberg--Marquardt restoration.
// Included inside restoration.cpp's anonymous namespace after the sparse
// residual/Jacobian helpers. All merit and stopping tests use ORIGINAL rows.
// D changes only the coordinates of the step, never constraint weights.
struct ColumnLmAdmission {
  bool allowed=false;
  std::vector<int> free;
  std::vector<int> index;
  double column_ratio=1;
  std::size_t estimated_triplets=0;
};

ColumnLmAdmission column_lm_admission(const QCQPProblem& p,const Vec& x,
                                    const Settings& st,bool mixed_startup_context=false) {
  ColumnLmAdmission a;
  const bool mixed_failure=st.restoration_failed_linearization_context&&
      st.restoration_shared_iteration_budget;
  const bool mixed_startup=mixed_startup_context&&st.restoration_shared_iteration_budget&&
      !st.restoration_failed_linearization_context&&!p.inequalities.empty();
  const bool mixed_allowed=mixed_failure||mixed_startup;
  if(st.dca_engine!="majorants" || p.equalities.empty() ||
     (!mixed_allowed&&!p.inequalities.empty()) || p.has_signed_square() ||
     x.size()!=static_cast<std::size_t>(p.n))return a;
  double original_maximum;long double original_merit;
  if(!column_lm_checked_values(p,x,original_maximum,original_merit))return a;
  a.index.assign(p.n,-1);
  for(int j=0;j<p.n;++j) {
    if(std::isfinite(p.lb[j])&&p.lb[j]==p.ub[j])continue;
    if(!mixed_allowed&&(std::isfinite(p.lb[j])||std::isfinite(p.ub[j])))return a;
    a.index[j]=static_cast<int>(a.free.size());a.free.push_back(j);
  }
  if(a.free.empty())return a;
  // Constant objective in the free coordinates, including fixed cross terms.
  if(!mixed_failure) {
  const auto og=p.objective.gradient(x);
  for(int j:a.free)if(!std::isfinite(og[j])||(!mixed_startup&&og[j]!=0.0))return a;
  for(const auto& e:p.objective.hessian.entries)
    if(e.v!=0&&a.index[e.i]>=0&&a.index[e.j]>=0)return a;
  }
  if(mixed_startup) {
    bool curved_infeasible=false;
    for(const auto& q:p.equalities) {
      const double value=q.value(x);if(!std::isfinite(value))return a;
      bool curved=false;for(const auto& h:q.hessian.entries)
        curved=curved||(h.v!=0&&a.index[h.i]>=0&&a.index[h.j]>=0);
      curved_infeasible=curved_infeasible||(curved&&std::abs(value)>st.feasibility_tolerance);
    }
    if(!curved_infeasible)return a;
  }
  // Bound triplet construction BEFORE forming row cliques. This estimates
  // assembly storage, not fill or total process RSS. The sparse factor keeps
  // the user's backend; no cross-backend recovery is permitted here.
  long double estimate=a.free.size();
  std::vector<int> support;
  std::vector<char> seen(p.n,0);
  auto count_row=[&](const auto& q) {
    support.clear();
    auto add=[&](int j){if(a.index[j]>=0&&!seen[j]){seen[j]=1;support.push_back(j);}};
    q.for_each_linear_nonzero([&](int j,double v){if(v!=0)add(j);});
    for(const auto& e:q.hessian.entries)if(e.v!=0){add(e.i);add(e.j);}
    const long double s=support.size();estimate+=s*(s+1)/2;
    for(int j:support)seen[j]=0;
    return estimate*sizeof(SymmetricTriplet)<=128.0L*1024*1024;
  };
  for(const auto& q:p.equalities)if(!count_row(q))return a;
  if(mixed_allowed)for(const auto& q:p.inequalities)if(!count_row(q))return a;
  a.estimated_triplets=static_cast<std::size_t>(estimate);
  const auto rj=legacy_residual_jacobian(p,x,true);
  if(!column_lm_checked_jacobian(rj,p.n))return a;
  Vec norms(p.n,0.0);
  for(const auto& row:rj.sparse_rows)for(auto e:row)if(a.index[e.first]>=0)
    norms[e.first]=std::max(norms[e.first],std::abs(e.second));
  double lo=std::numeric_limits<double>::infinity(),hi=0;
  for(int j:a.free)if(norms[j]>0){lo=std::min(lo,norms[j]);hi=std::max(hi,norms[j]);}
  if(!(lo>0)||!std::isfinite(lo)||!std::isfinite(hi))return a;
  a.column_ratio=hi/lo;
  a.allowed=std::isfinite(a.column_ratio)&&(mixed_allowed||a.column_ratio>=1e6);
  return a;
}

// The square-root augmented representation has eigenvalue scales of B,
// not B'B. Elimination gives EXACTLY the same damped normal equations:
// [sqrt(lambda) I  B'; B -sqrt(lambda) I] [y; eta] = [0; -r].
// A rejected normal factor may switch representations, never backends or
// acceptance thresholds. Rank deficiency is harmless for positive lambda.
std::vector<SymmetricTriplet> column_lm_augmented_entries(
    int n,const std::vector<LegacySparseRow>& B,double damping) {
  std::vector<SymmetricTriplet> entries;
  std::size_t nz=n+B.size();for(const auto& row:B)nz+=row.size();
  entries.reserve(nz);const double delta=std::sqrt(damping);
  for(int j=0;j<n;++j)entries.push_back({j,j,delta});
  for(std::size_t i=0;i<B.size();++i) {
    for(auto e:B[i])entries.push_back({e.first,n+static_cast<int>(i),e.second});
    entries.push_back({n+static_cast<int>(i),n+static_cast<int>(i),-delta});
  }
  return entries;
}

RestorationResult column_scaled_lm_restore(const QCQPProblem& p,const Vec& start,
                 const Settings& st,const ColumnLmAdmission& admission,
                 bool watch_physical_stall=false,NativeRestorationWork* work=nullptr,bool count_model_attempts=false) {
  RestorationResult out;out.x=start;out.effective_dimension=admission.free.size();
  double initial_maximum;long double initial_merit;
  const bool initial_finite=column_lm_checked_values(p,start,initial_maximum,initial_merit);
  out.initial_feasibility=initial_finite?initial_maximum:std::numeric_limits<double>::infinity();
  out.final_feasibility=out.initial_feasibility;
  const auto t0=std::chrono::steady_clock::now();
  auto finish=[&](const char* message) {
    out.message=message;out.total_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-t0).count();
    if(st.live_display_level||st.verbose)std::cout<<"[RESTORATION-COLUMN-LM] final=1 success="
      <<out.success<<" iterations="<<out.iterations<<" solves="<<out.linear_solves
      <<" primal="<<out.final_feasibility<<" seconds="<<out.total_seconds
      <<" original_merit=1 original_certificate_required=1"<<std::endl;
    return out;
  };
  if(!initial_finite)return finish("column-scaled LM: nonfinite original start or row");
  if(st.live_display_level||st.verbose)std::cout<<"[RESTORATION-COLUMN-LM] admitted=1 column_ratio="
    <<admission.column_ratio<<" free="<<admission.free.size()
    <<" estimated_triplets="<<admission.estimated_triplets
    <<" original_merit=1 backend="<<st.linear_backend<<" supplied_start_preserved=1"<<std::endl;
  const int n=static_cast<int>(admission.free.size());
  std::string message;
  // This routine already assembles its explicit LM diagonal. Suppress the
  // generic backend's additional fixed shift: an unadvertised 1e-12 shift
  // can dominate a valid 1e-15 normal model in weak directions, even when
  // its normwise backward error looks small. No tolerance is relaxed.
  Settings factor_settings=st;
  factor_settings.kkt_adaptive_regularization=true;
  factor_settings.kkt_primal_regularization=0;
  factor_settings.kkt_dual_regularization=0;
  auto factor=make_symmetric_factorization(factor_settings,message);
  if(!factor)return finish("column-scaled LM: selected backend unavailable");
  auto factor_call=[&](int dim,int primal,const std::vector<SymmetricTriplet>& entries,std::string& msg) {
    for(const auto& e:entries)if(!std::isfinite(e.value)){msg="nonfinite LM factor entry";return false;}
    if(!work||native_restoration_trace_active())return factor->factor(dim,primal,entries,msg);
    const auto begin=std::chrono::steady_clock::now();++work->factorization_calls;
    const bool ok=factor->factor(dim,primal,entries,msg);
    work->factorization_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    work->backend_regularization_attempts+=factor->last_regularization_attempts();return ok;
  };
  auto solve_call=[&](Vec& rhs,std::string& msg) {
    if(!column_lm_finite_vector(rhs)){msg="nonfinite LM solve right-hand side";return false;}
    ++out.linear_solves;
    if(!work||native_restoration_trace_active())return factor->solve(rhs,msg)&&column_lm_finite_vector(rhs);
    const auto begin=std::chrono::steady_clock::now();++work->linear_solve_calls;
    const bool ok=factor->solve(rhs,msg);
    work->linear_solve_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();return ok&&column_lm_finite_vector(rhs);
  };
  double damping=std::max(1e-15,st.restoration_lm_initial_damping);
  Vec x=start;bool augmented=false;
  // Under a shared budget every outer LM model is debited, including a
  // terminal rejected model. Keep count_model_attempts as the separate startup
  // context for physical-stall handoff below.
  const bool debit_models=count_model_attempts||st.restoration_shared_iteration_budget;
  ColumnLmPhysicalStall physical_stall;
  physical_stall.observe(out.initial_feasibility,st.feasibility_tolerance);
  Vec best_physical_x=x;double best_physical=out.initial_feasibility;
  for(int it=0;it<st.restoration_max_iterations&&!restoration_expired(st);++it) {
    double current_maximum;long double phi;
    if(!column_lm_checked_values(p,x,current_maximum,phi))
      return finish("column-scaled LM: nonfinite original iterate or row");
    auto rj=legacy_residual_jacobian(p,x,true);
    if(!column_lm_checked_jacobian(rj,p.n))return finish("column-scaled LM: nonfinite original Jacobian");
    out.final_merit=static_cast<double>(phi);
    out.final_feasibility=current_maximum;out.x=x;
    if(out.final_feasibility<=0.5*st.feasibility_tolerance) {
      out.success=true;return finish("column-scaled LM reached original feasibility");
    }
    if(!std::isfinite(phi))return finish("column-scaled LM: nonfinite merit");
    if(debit_models){++out.iterations;if(work)++work->outer_attempts;}
    Vec col(n,0),D(n),gradient(n,0);
    std::vector<long double> gl(n,0);
    for(const auto& row:rj.sparse_rows)for(auto e:row) {
      const int j=admission.index[e.first];if(j>=0)col[j]=std::max(col[j],std::abs(e.second));
    }
    for(int j=0;j<n;++j)D[j]=1/std::max(1e-6,col[j]);
    if(!column_lm_finite_vector(col)||!column_lm_finite_vector(D))return finish("column-scaled LM: nonfinite column scaling");
    std::vector<LegacySparseRow> B(rj.sparse_rows.size());
    std::vector<SymmetricTriplet> normal;normal.reserve(admission.estimated_triplets);
    for(std::size_t i=0;i<B.size();++i) {
      for(auto e:rj.sparse_rows[i]) {
        const int j=admission.index[e.first];if(j>=0&&e.second!=0) {
          const double scaled=e.second*D[j];if(!std::isfinite(scaled))return finish("column-scaled LM: nonfinite scaled Jacobian");
          B[i].push_back({j,scaled});
        }
      }
      const auto& row=B[i];
      for(std::size_t u=0;u<row.size();++u) {
        gl[row[u].first]+=static_cast<long double>(row[u].second)*rj.residual[i];
        for(std::size_t v=0;v<=u;++v)
          normal.push_back({row[v].first,row[u].first,row[v].second*row[u].second});
      }
    }
    for(int j=0;j<n;++j) {
      if(!std::isfinite(gl[j]))return finish("column-scaled LM: nonfinite normal gradient");
      gradient[j]=static_cast<double>(gl[j]);
    }
    if(!column_lm_finite_vector(gradient))return finish("column-scaled LM: normal gradient overflow");
    for(const auto& e:normal)if(!std::isfinite(e.value))return finish("column-scaled LM: nonfinite normal matrix");
    const std::size_t diagonal=normal.size();
    for(int j=0;j<n;++j)normal.push_back({j,j,damping});
    bool accepted=false;
    for(int attempt=0;attempt<30&&!restoration_expired(st);++attempt) {
      if(!(damping>0)||!std::isfinite(damping)||damping>st.restoration_lm_max_damping)break;
      for(int j=0;j<n;++j)normal[diagonal+j].value=damping;
      Vec y(n);for(int j=0;j<n;++j)y[j]=-gradient[j];
      bool ok=false;
      if(!augmented) {
        ok=factor_call(n,n,normal,message);
        if(!ok) {
          augmented=true;
          if(st.live_display_level||st.verbose)std::cout
            <<"[RESTORATION-COLUMN-LM] representation=augmented-sqrt same_model=1"
            <<" unchanged_pivot_threshold=1 reason="<<message<<std::endl;
        }
      }
      if(augmented) {
        const auto entries=column_lm_augmented_entries(n,B,damping);
        ok=factor_call(n+static_cast<int>(B.size()),n,entries,message);
        if(ok) {
          Vec rhs(n+B.size(),0);
          for(std::size_t i=0;i<B.size();++i)rhs[n+i]=-rj.residual[i];
          ok=solve_call(rhs,message);
          if(ok)std::copy(rhs.begin(),rhs.begin()+n,y.begin());
        }
      } else if(ok){ok=solve_call(y,message);}
      // Expanded normal-equation audit, without the backend's numerical
      // perturbations: (B'B + damping I)y = -B'r. D and damping are explicit
      // terms in this restoration model, NOT relaxations of outer KKT checks.
      double relative=std::numeric_limits<double>::infinity();
      for(int refine=0;ok&&refine<=5;++refine) {
        std::vector<long double> defect(n,0);
        for(int j=0;j<n;++j)defect[j]=gl[j]+static_cast<long double>(damping)*y[j];
        for(const auto& row:B) {
          long double by=0;for(auto e:row)by+=static_cast<long double>(e.second)*y[e.first];
          for(auto e:row)defect[e.first]+=static_cast<long double>(e.second)*by;
        }
        long double maxerr=0;
        for(auto r:defect) {
          if(!std::isfinite(r)){ok=false;break;}
          maxerr=std::max(maxerr,std::abs(r));
        }
        if(!ok)break;
        relative=static_cast<double>(maxerr)/std::max(1.0,inf_norm(gradient));
        if(std::isfinite(relative)&&relative<=st.kkt_backward_error_reject)break;
        if(refine==5){ok=false;break;}
        Vec correction(n);for(int j=0;j<n;++j)correction[j]=-static_cast<double>(defect[j]);
        if(augmented) {
          Vec rhs(n+B.size(),0);const double delta=std::sqrt(damping);
          for(int j=0;j<n;++j)rhs[j]=correction[j]/delta;
          ok=solve_call(rhs,message);
          if(ok)for(int j=0;j<n;++j)y[j]+=rhs[j];
        } else {
          ok=solve_call(correction,message);
          if(ok)axpy(1.0,correction,y);
        }
      }
      if(!ok||!column_lm_finite_vector(y)){damping*=10;continue;}
      // Project onto actual finite input bounds, then evaluate the normal
      // model on the PROJECTED displacement, not the unconstrained solve.
      // No finite box is invented for an absent bound side.
      Vec trial=x;for(int j=0;j<n;++j)trial[admission.free[j]]+=D[j]*y[j];
      // Test the unconstrained displacement BEFORE projection: an infinite
      // step must not become a finite bound endpoint through min/max.
      if(!column_lm_finite_vector(trial)){damping*=10;continue;}
      trial=clamp_to_bounds(p,std::move(trial));
      for(int j=0;j<n;++j)y[j]=(trial[admission.free[j]]-x[admission.free[j]])/D[j];
      if(!column_lm_finite_vector(y)){damping*=10;continue;}
      long double gy=0,by2=0;
      for(int j=0;j<n;++j)gy+=gl[j]*y[j];
      for(const auto& row:B){long double by=0;for(auto e:row)by+=static_cast<long double>(e.second)*y[e.first];by2+=by*by;}
      const long double predicted=-gy-0.5L*by2;
      if(!(predicted>0)||!(gy<0)||!std::isfinite(predicted)){damping*=10;continue;}
      double next_maximum;long double nextphi;
      if(!column_lm_checked_values(p,trial,next_maximum,nextphi)){damping*=10;continue;}
      const long double ratio=(phi-nextphi)/predicted;
      if(std::isfinite(ratio)&&ratio>1e-4L) {
        x=std::move(trial);accepted=true;
        if(!debit_models)++out.iterations;
        if(work)++work->accepted_steps;
        if(ratio>.75L)damping=std::max(1e-15,.25*damping);
        else if(ratio<.25L)damping*=4;
        if((st.live_display_level||st.verbose)&&(it%10==0))
          std::cout<<"[RESTORATION-COLUMN-LM] iter="<<out.iterations
            <<" primal="<<next_maximum<<" merit="<<static_cast<double>(nextphi)
            <<" damping="<<damping<<" model_residual="<<relative
            <<" ratio="<<static_cast<double>(ratio)<<std::endl;
        if(watch_physical_stall) {
          const double physical=next_maximum;
          if(physical<best_physical){best_physical=physical;best_physical_x=x;}
          if(physical_stall.observe(physical,st.feasibility_tolerance)&&
              (count_model_attempts||near_feasible_sparse_inequality_candidate(p,best_physical_x,st))) {
            // As in legacy_local_restore, the private method retains its
            // best ORIGINAL infinity-norm point. A lower squared merit need
            // not improve the physical maximum used by the outer Phase I.
            out.x=best_physical_x;out.final_feasibility=best_physical;
            out.final_merit=legacy_squared_merit_values(p,out.x,{});
            out.final_damping=damping;++out.stagnation_events;
            if(st.live_display_level||st.verbose)std::cout<<"[RESTORATION-PHYSICAL-STALL]"
              <<" iterations="<<out.iterations<<" observed_primal="<<physical
              <<" retained_primal="<<best_physical<<" window="<<ColumnLmPhysicalStall::window
              <<" original_maximum_residual=1"<<std::endl;
            return finish("column-scaled LM: original maximum residual stalled");
          }
        }
        break;
      }
      ++out.line_search_backtracks;damping*=4;
    }
    out.final_damping=damping;
    if(!accepted)return finish("column-scaled LM: trust model failed or deadline");
  }
  out.x=x;double final_maximum;long double final_merit;
  if(!column_lm_checked_values(p,x,final_maximum,final_merit))return finish("column-scaled LM: nonfinite final original point");
  out.final_feasibility=final_maximum;out.final_merit=static_cast<double>(final_merit);
  out.success=out.final_feasibility<=st.feasibility_tolerance;
  return finish(out.success?"column-scaled LM reached original feasibility":
      "column-scaled LM: iteration budget or deadline");
}
