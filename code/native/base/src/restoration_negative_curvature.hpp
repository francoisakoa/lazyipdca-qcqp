// Included inside restoration.cpp's anonymous namespace. A finite-budget
// negative-curvature step for pure equality feasibility. No geometry, target
// point, case name, or restart is used: the ORIGINAL merit at the current
// iterate supplies its Hessian and verifies the accepted candidate.
extern "C" void dsyev_(const char*,const char*,const int*,double*,const int*,
                       double*,double*,const int*,int*);

struct RestorationCurvatureStep {
  bool attempted=false,accepted=false;
  int matvecs=0,lanczos_steps=0;
  double seconds=0,rayleigh=0,merit_before=0,merit_after=0;
  double theta_before=0,theta_after=0,step_inf=0;
};

bool restoration_constant_objective(const QCQPProblem& p) {
  if(!p.objective.signed_square.empty())return false;
  bool constant=true;
  p.objective.for_each_linear_nonzero([&](int,double v){if(v!=0)constant=false;});
  for(const auto& e:p.objective.hessian.entries)if(e.v!=0)constant=false;
  return constant;
}

// Real roots of a derivative of degree at most three. Scale before forming
// the depressed cubic; evaluate each candidate with the original merit below.
std::vector<double> restoration_cubic_roots(long double a,long double b,
                                          long double c,long double d) {
  const long double norm=std::max({std::abs(a),std::abs(b),std::abs(c),std::abs(d)});
  std::vector<double> roots;
  if(!(norm>0)||!std::isfinite(norm))return roots;
  a/=norm;b/=norm;c/=norm;d/=norm;
  auto add=[&](long double t){if(std::isfinite(t)&&std::abs(t)<=std::numeric_limits<double>::max())roots.push_back(double(t));};
  if(a==0) {
    if(b==0){if(c!=0)add(-d/c);return roots;}
    const long double disc=c*c-4*b*d;
    if(disc<0)return roots;
    const long double q=-0.5L*(c+std::copysign(std::sqrt(disc),c));
    if(q!=0){add(q/b);add(d/q);}else add(-c/(2*b));
    return roots;
  }
  const long double B=b/a,C=c/a,D=d/a;
  const long double p=C-B*B/3,q=2*B*B*B/27-B*C/3+D;
  const long double disc=q*q/4+p*p*p/27;
  if(disc>=0) {
    const long double sd=std::sqrt(disc);
    const long double u=std::cbrt(-q/2+sd),v=std::cbrt(-q/2-sd);
    add(u+v-B/3);
    if(disc==0)add(-0.5L*(u+v)-B/3);
  } else if(p<0) {
    const long double radius=2*std::sqrt(-p/3);
    const long double arg=std::clamp(3*q/(p*radius),-1.0L,1.0L);
    const long double angle=std::acos(arg)/3,pi=std::acos(-1.0L);
    for(int k=0;k<3;++k)add(radius*std::cos(angle-2*pi*k/3)-B/3);
  }
  return roots;
}

RestorationCurvatureStep restoration_negative_curvature_step(
    const QCQPProblem& p,Vec& x,const Settings& st,
    bool stationary_merit_escape=false) {
  RestorationCurvatureStep out;
  auto skip=[&](const char* reason) {
    if(st.live_display_level>=2)
      std::cout<<"[PHASE-I-CURVATURE] attempted=0 reason="<<reason
        <<" n="<<p.n<<" equalities="<<p.equalities.size()
        <<" inequalities="<<p.inequalities.size()<<std::endl;
    return out;
  };
  // The degeneracy of a quadratic feasibility merit is independent of its
  // dimension. Preserve optimization/native-abs trajectories and the sparse
  // structural guard, but allow small systems to escape before their existing
  // dense LM continuation. The row-space factorization gate stays unchanged.
  if(!restoration_constant_objective(p) && !st.restoration_failed_linearization_context)
    return skip("nonconstant-objective");
  if(p.has_signed_square())return skip("signed-square");
  const bool bounded_failed_linearization=st.restoration_failed_linearization_context &&
      detail::failure_recovery_budget(p).allowed;
  if(!sparse_underdetermined_equality_candidate(p,1) && !bounded_failed_linearization)
    return skip("outside-sparse-underdetermined-equalities");
  if(restoration_expired(st))return skip("deadline");
  out.theta_before=feasibility(p,x).total_inf;out.theta_after=out.theta_before;
  // Failure recovery uses a COARSE stage target (for example theta/10),
  // not the final KKT tolerance. Multiplying that target by ten can classify
  // the unchanged, infeasible anchor as near-feasible and disable the only
  // escape from a zero-Jacobian equality. Preserve the historical guard
  // outside the explicit failed-linearization context.
  const double historical_near_threshold=std::max(1e-8,10*st.feasibility_tolerance);
  const bool coarse_target_exception=st.restoration_failed_linearization_context&&
      !(out.theta_before>historical_near_threshold);
  if(!(out.theta_before>std::max(1e-8,st.restoration_failed_linearization_context?
      st.feasibility_tolerance:10*st.feasibility_tolerance)))
    return skip("already-near-feasible");
  const auto fixed=fixed_mask(p,st.fixed_bound_tolerance);
  const int nf=std::count(fixed.begin(),fixed.end(),char(0));
  // A stationary infeasible merit can also have negative curvature when
  // the number of equations equals or exceeds the number of free variables.
  if(nf==0 || (!bounded_failed_linearization && nf<=int(p.equalities.size())))
    return skip("insufficient-free-variables");
  const auto started=std::chrono::steady_clock::now();out.attempted=true;
  auto finish=[&]() {
    out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    if(st.verbose||st.live_display_level>0)
      std::cout<<"[PHASE-I-CURVATURE] accepted="<<out.accepted
        <<" lanczos="<<out.lanczos_steps<<" matvecs="<<out.matvecs
        <<" rayleigh="<<out.rayleigh<<" merit="<<out.merit_before<<"->"<<out.merit_after
        <<" theta="<<out.theta_before<<"->"<<out.theta_after
        <<" step="<<out.step_inf<<" seconds="<<out.seconds
        <<" initial_point_preserved=1 exact_merit_guard=1 finite_bound_interval=1"<<std::endl;
    return out;
  };
  using SparseRow=std::vector<std::pair<int,double>>;
  std::vector<const std::decay_t<decltype(p.equalities)>::value_type*> functions;
  for(const auto& h:p.equalities)functions.push_back(&h);
  for(const auto& g:p.inequalities)if(g.value(x)>0)functions.push_back(&g);
  const int n=p.n,m=int(functions.size());
  std::vector<SparseRow> rows;rows.reserve(m);Vec residual(m,0),gradient(n,0);
  std::vector<int> touched;std::vector<char> used(n,0);
  long double phi=0;
  for(int i=0;i<m;++i) {
    if(restoration_expired(st))return finish();
    const auto& h=*functions[i];residual[i]=h.value(x);
    phi+=0.5L*residual[i]*residual[i];
    auto accumulate=[&](int j,double value){if(!used[j]){used[j]=1;touched.push_back(j);}gradient[j]+=value;};
    for(const auto& e:h.hessian.entries){accumulate(e.i,e.v*x[e.j]);if(e.i!=e.j)accumulate(e.j,e.v*x[e.i]);}
    h.for_each_linear_nonzero(accumulate);
    std::sort(touched.begin(),touched.end());SparseRow row;row.reserve(touched.size());
    for(int j:touched){if(!fixed[j]&&gradient[j]!=0)row.push_back({j,gradient[j]});gradient[j]=0;used[j]=0;}
    touched.clear();rows.push_back(std::move(row));
  }
  out.merit_before=out.merit_after=double(phi);
  // An unsuccessful column-LM solve may supply a separately checked nearly
  // stationary merit. Exact representable cancellation is not required for
  // this search policy: direct negative curvature and original merit decrease
  // remain mandatory, as does the outer original filter.
  const bool failed_stationary_escape=stationary_merit_escape&&
      st.restoration_failed_linearization_context&&st.restoration_shared_iteration_budget;
  if(coarse_target_exception&&!failed_stationary_escape) {
    // A coarse theta/10 target must not make curvature exploration eager
    // when a nonstationary first-order feasibility model can still progress.
    // The exception to the historical near-feasible guard is reserved for
    // a blind/stationary linearized merit model (notably zero Jacobian rows).
    // This relative eligibility test is a search policy, NEVER a PSD/KKT
    // certificate; every curvature trial still needs the original merit test.
    std::vector<long double> merit_gradient(n,0),absolute_contributions(n,0);
    for(int i=0;i<m;++i)for(const auto& e:rows[i]) {
      const long double term=static_cast<long double>(residual[i])*e.second;
      merit_gradient[e.first]+=term;absolute_contributions[e.first]+=std::abs(term);
    }
    int violated_blind_quadratic_rows=0;
    for(int i=0;i<m;++i)if(rows[i].empty()&&std::abs(residual[i])>st.feasibility_tolerance) {
      const auto& h=functions[i]->hessian.entries;
      if(std::any_of(h.begin(),h.end(),[&](const auto& e){return e.v!=0&&!fixed[e.i]&&!fixed[e.j];}))
        ++violated_blind_quadratic_rows;
    }
    long double gradient_inf=0,contribution_inf=0;
    for(int j=0;j<n;++j){gradient_inf=std::max(gradient_inf,std::abs(merit_gradient[j]));contribution_inf=std::max(contribution_inf,absolute_contributions[j]);}
    const long double relative_bound=64*std::numeric_limits<double>::epsilon()*contribution_inf;
    const bool blind=std::isfinite(gradient_inf)&&std::isfinite(contribution_inf)&&
        (gradient_inf==0||(contribution_inf>0&&gradient_inf<=relative_bound));
    if(st.live_display_level>=2)
      std::cout<<"[PHASE-I-FIRST-ORDER-GUARD] gradient_inf="<<gradient_inf
        <<" absolute_contribution_inf="<<contribution_inf<<" relative_bound="<<relative_bound
        <<" linear_model_blind="<<blind<<" violated_blind_quadratic_rows="<<violated_blind_quadratic_rows
        <<" coarse_target="<<st.feasibility_tolerance
        <<" historical_near_threshold="<<historical_near_threshold<<std::endl;
    if(!blind&&!violated_blind_quadratic_rows){out.attempted=false;return skip("coarse-target-first-order-model-not-blind");}
  }
  auto product=[&](const Vec& v,Vec& y) {
    y.assign(n,0);++out.matvecs;
    for(int i=0;i<m;++i) {
      double jv=0;for(const auto& e:rows[i])jv+=e.second*v[e.first];
      for(const auto& e:rows[i])y[e.first]+=jv*e.second;
      for(const auto& e:functions[i]->hessian.entries) {
        y[e.i]+=residual[i]*e.v*v[e.j];
        if(e.i!=e.j)y[e.j]+=residual[i]*e.v*v[e.i];
      }
    }
    for(int j=0;j<n;++j)if(fixed[j])y[j]=0;
  };
  auto dot_long=[](const Vec& a,const Vec& b){long double s=0;for(std::size_t j=0;j<a.size();++j)s+=static_cast<long double>(a[j])*b[j];return double(s);};
  std::mt19937 random(st.random_seed);std::normal_distribution<double> normal(0,1);
  Vec q(n,0);for(int j=0;j<n;++j)if(!fixed[j])q[j]=normal(random);
  double norm=std::sqrt(dot_long(q,q));if(!(norm>0))return finish();
  for(double& value:q)value/=norm;
  constexpr int budget=24;
  std::vector<Vec> basis;basis.reserve(budget);Vec diagonal,off,work;
  for(int k=0;k<budget;++k) {
    if(restoration_expired(st))return finish();
    basis.push_back(q);product(q,work);
    const double alpha=dot_long(q,work);if(!std::isfinite(alpha))return finish();
    diagonal.push_back(alpha);
    for(int j=0;j<n;++j)work[j]-=alpha*q[j];
    if(k)for(int j=0;j<n;++j)work[j]-=off.back()*basis[k-1][j];
    // Full reorthogonalization costs only O(24^2 n), with no n-by-n object.
    for(const auto& v:basis){const double projection=dot_long(v,work);for(int j=0;j<n;++j)work[j]-=projection*v[j];}
    const double beta=std::sqrt(dot_long(work,work));
    if(!std::isfinite(beta))return finish();
    if(beta<=1e-12*std::max(1.,std::abs(alpha))||k+1==budget)break;
    off.push_back(beta);q=work;for(double& value:q)value/=beta;
  }
  const int k=int(diagonal.size());out.lanczos_steps=k;
  if(k==0)return finish();
  Vec tridiagonal(k*k,0),eigenvalues(k),eigenwork(std::max(1,3*k));
  for(int i=0;i<k;++i){tridiagonal[i+k*i]=diagonal[i];if(i+1<k)tridiagonal[i+k*(i+1)]=tridiagonal[i+1+k*i]=off[i];}
  const char vectors='V',upper='U';const int lwork=int(eigenwork.size());int info=0;
  {NativeRestorationCall counted(NativeRestorationOperation::Spectral);
  dsyev_(&vectors,&upper,&k,tridiagonal.data(),&k,eigenvalues.data(),eigenwork.data(),&lwork,&info);
  }
  if(info||restoration_expired(st))return finish();
  Vec direction(n,0);
  for(int i=0;i<k;++i)for(int j=0;j<n;++j)direction[j]+=tridiagonal[i]*basis[i][j];
  const double scale=inf_norm(direction);if(!(scale>0)||!std::isfinite(scale))return finish();
  for(double& value:direction)value/=scale;
  product(direction,work);
  out.rayleigh=dot_long(direction,work)/dot_long(direction,direction);
  // Certify actual curvature of the full original merit Hessian, rather than
  // accepting an approximate Ritz value without a direct product.
  if(!(out.rayleigh < -1e-10*std::max(1.,inf_norm(work))))return finish();
  long double A=0,B=0,C=0,D=0;
  for(int i=0;i<m;++i) {
    long double linear=0,quadratic=0;
    for(const auto& e:rows[i])linear+=static_cast<long double>(e.second)*direction[e.first];
    for(const auto& e:functions[i]->hessian.entries)
      quadratic+=(e.i==e.j?0.5L:1.0L)*e.v*direction[e.i]*direction[e.j];
    A+=2*quadratic*quadratic;B+=3*quadratic*linear;
    C+=linear*linear+2*quadratic*residual[i];D+=linear*residual[i];
  }
  auto candidates=restoration_cubic_roots(A,B,C,D);
  double lo=-INFINITY,hi=INFINITY;
  for(int j=0;j<n;++j)if(direction[j]!=0) {
    // A nonfinite bound denotes an absent side throughout restoration.cpp.
    // Frozen V1 inputs can use +INF even on the lower side. Never subtract
    // that sentinel from x: it is not a constraint on the line parameter.
    if(std::isfinite(p.lb[j])) {
      const double t=(p.lb[j]-x[j])/direction[j];
      if(direction[j]>0)lo=std::max(lo,t);else hi=std::min(hi,t);
    }
    if(std::isfinite(p.ub[j])) {
      const double t=(p.ub[j]-x[j])/direction[j];
      if(direction[j]>0)hi=std::min(hi,t);else lo=std::max(lo,t);
    }
  }
  if(std::isfinite(lo))candidates.push_back(lo);
  if(std::isfinite(hi))candidates.push_back(hi);
  double best_merit=out.merit_before;Vec best;double best_alpha=0,best_theta=out.theta_before;
  for(double alpha:candidates) {
    if(restoration_expired(st))break;
    if(!std::isfinite(alpha)||alpha<lo||alpha>hi)continue;
    Vec trial=x;for(int j=0;j<n;++j)trial[j]+=alpha*direction[j];
    if(!std::isfinite(inf_norm(trial)))continue;
    long double trial_phi=0;double theta=0;bool finite=true;
    for(const auto& h:p.equalities){const double r=h.value(trial);finite=finite&&std::isfinite(r);theta=std::max(theta,std::abs(r));trial_phi+=0.5L*r*r;}
    for(const auto& g:p.inequalities){const double v=std::max(0.0,g.value(trial));finite=finite&&std::isfinite(v);theta=std::max(theta,v);trial_phi+=0.5L*v*v;}
    if(finite&&std::isfinite(trial_phi)&&trial_phi<best_merit) {
      best_merit=double(trial_phi);best=std::move(trial);best_alpha=alpha;best_theta=theta;
    }
  }
  if(best_merit<out.merit_before-1e-10*std::max(1.0,out.merit_before)&&!best.empty()) {
    out.accepted=true;out.merit_after=best_merit;out.theta_after=best_theta;
    out.step_inf=std::abs(best_alpha);x=std::move(best);
  }
  return finish();
}
