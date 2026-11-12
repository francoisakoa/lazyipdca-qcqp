// Included in the filter engine's anonymous namespace beside Workspace.
// Only inequality functions are replaced. Their constant QCQP curvature is
// certified once; reanchoring changes neither f nor the equality functions.
struct DcaInequalityModels {
  struct Gaps {
    double value_gap_inf = 0;
    double weighted_value_gap_inf = 0;
    double dual_gap_inf = 0;
  };
  struct Curvature {
    bool ready = false, model_mapped = false;
    bool half_spectral = false, positive_spectral = false;
    double scale = 1, rho = 0;
    double spectral_shift = 0;
    double native_shift = 0;
    std::vector<int> support;
    std::vector<SymTriplet> hessian;
  };
  struct ObjectiveAtom { int coordinate=-1; double beta=0; Vec direction; };
  double objective_floor=0, objective_excess=0, objective_rho=0;
  bool objective_ready=false, objective_nonconvex=false;
  int objective_admitted=0;
  std::vector<int> objective_support;
  std::vector<SymTriplet> objective_base;
  std::vector<ObjectiveAtom> objective_pending;
  std::map<int,double> objective_diagonal_atoms;

  void admit_objective_atoms(int budget) {
    const int stop=std::min({int(objective_pending.size()),w.st.objective_max_atoms,
                            objective_admitted+budget});
    for(;objective_admitted<stop;++objective_admitted) {
      const auto& a=objective_pending[objective_admitted];
      if(a.coordinate>=0)objective_diagonal_atoms[a.coordinate]+=a.beta;
      else {
        LowRankTerm atom;atom.beta=a.beta;atom.direction.assign(w.n,0);
        for(std::size_t j=0;j<objective_support.size();++j)
          atom.direction[objective_support[j]]=a.direction[j];
        w.objective.model_atoms.push_back(std::move(atom));
      }
    }
    objective_floor=objective_admitted<int(objective_pending.size())?
        objective_pending[objective_admitted].beta:0;
  }
  void prepare_objective() {
    if(objective_ready||!w.st.objective_lazy_convexification)return;
    objective_ready=true;
    const auto& q=w.objective;
    if(!q.signed_square.empty()) {
      // One fixed global upper Hessian across every sign, then a sufficient
      // PSD correction of that base. Native curvature is never frozen at a
      // local branch when objective convexification has been requested.
      std::map<Pair,double> cap;std::set<int> indices;
      for(auto e:q.base_hessian)cap[upper(e.i,e.j)]+=q.scale*e.v;
      for(auto e:q.signed_square)cap[upper(e.first,e.first)]+=2*q.scale*std::abs(e.second);
      std::map<int,double> diagonal,radius;
      for(auto e:cap){indices.insert(e.first.first);indices.insert(e.first.second);if(e.first.first==e.first.second)diagonal[e.first.first]+=e.second;else{radius[e.first.first]+=std::abs(e.second);radius[e.first.second]+=std::abs(e.second);}}
      double shift=0;for(int j:indices)shift=std::max(shift,radius[j]-diagonal[j]);
      for(int j:indices)cap[upper(j,j)]+=shift;
      for(auto e:cap)objective_base.push_back({e.first.first,e.first.second,e.second});
      objective_support.assign(indices.begin(),indices.end());objective_nonconvex=true;
      return;
    }
    std::map<Pair,double> canonical;
    for(const auto& e:q.hessian)canonical[upper(e.i,e.j)]+=q.scale*e.v;
    std::set<int> indices;
    for(const auto& e:canonical)if(e.second!=0){indices.insert(e.first.first);indices.insert(e.first.second);}
    objective_support.assign(indices.begin(),indices.end());
    const int n=int(objective_support.size());
    if(!n)return;
    auto local=[&](int j){return int(std::lower_bound(objective_support.begin(),objective_support.end(),j)-objective_support.begin());};
    Vec diagonal(n,0),radius(n,0);
    std::vector<SymmetricTriplet> matrix;
    for(const auto& e:canonical)if(e.second!=0) {
      const int i=local(e.first.first),j=local(e.first.second);const double v=e.second;
      if(!std::isfinite(v))throw std::runtime_error("nonfinite objective curvature");
      matrix.push_back({i,j,v});
      if(i==j)diagonal[i]+=v;else{radius[i]+=std::abs(v);radius[j]+=std::abs(v);}
    }
    double lower=INFINITY,norm=0;
    for(int i=0;i<n;++i){lower=std::min(lower,diagonal[i]-radius[i]);norm=std::max(norm,std::abs(diagonal[i])+radius[i]);}
    if(!std::isfinite(norm))throw std::runtime_error("objective curvature norm overflow");
    if(lower>=0)return;
    const double padding=64*std::numeric_limits<double>::epsilon()*std::max(1.0,norm);
    const bool diagonal_only=std::all_of(radius.begin(),radius.end(),[](double x){return x==0;});
    if(!diagonal_only&&n<=256) {
      Vec dense(n*n,0),values(n),work(std::max(1,3*n));
      for(const auto& e:matrix){dense[e.i+n*e.j]=e.value;dense[e.j+n*e.i]=e.value;}
      int info=0,lwork=int(work.size());const char vectors='V',up='U';
      dsyev_(&vectors,&up,&n,dense.data(),&n,values.data(),work.data(),&lwork,&info);
      if(info||!finite(values))throw std::runtime_error("objective spectrum failed");
      if(values.front()>=-padding)return;
      for(int k=0;k<n;++k)if(values[k]<padding) {
        ObjectiveAtom a;a.beta=padding-values[k];a.direction.resize(n);
        for(int j=0;j<n;++j)a.direction[j]=dense[j+n*k];
        objective_pending.push_back(std::move(a));
      }
    } else {
      if(!diagonal_only) {
        // One sparse PSD classification. A failed sufficient Gershgorin bound
        // alone never activates correction on a convex objective.
        Settings cert=w.factor_settings;
        cert.kkt_adaptive_regularization=true;cert.cholesk_adaptive_regularization=false;
        cert.kkt_primal_regularization=cert.kkt_dual_regularization=0;
        cert.cholesk_relative_pivot_tolerance=0;
        for(int i=0;i<n;++i)matrix.push_back({i,i,padding});
        std::string message;auto factor=make_symmetric_factorization(cert,message);
        if(!factor)throw std::runtime_error("objective PSD backend: "+message);
        const bool ok=factor->factor(n,n,matrix,message);
        if(ok&&factor->last_positive_pivots()==n&&factor->last_negative_pivots()==0)return;
        if(factor->last_negative_pivots()<=0)
          throw std::runtime_error("objective curvature undecided: "+message);
      }
      // Sparse coordinate atoms: each admitted atom repairs one row's
      // Gershgorin deficit. The largest uncorrected deficit certifies rho.
      // No dense n-by-n storage, Ritz loop or bisection for large objectives.
      for(int j=0;j<n;++j)if(diagonal[j]-radius[j]<padding)
        objective_pending.push_back({objective_support[j],padding+radius[j]-diagonal[j],{}});
    }
    objective_nonconvex=true;
    std::stable_sort(objective_pending.begin(),objective_pending.end(),
        [](const ObjectiveAtom& a,const ObjectiveAtom& b){return a.beta>b.beta;});
    const double initial=objective_pending.front().beta;
    admit_objective_atoms(w.st.objective_initial_atoms);
    objective_rho=std::max(objective_floor,initial*w.st.objective_rho_excess);
    objective_excess=objective_rho-objective_floor;
    if(!std::isfinite(objective_rho))throw std::runtime_error("objective rho overflow");
    for(const auto& e:canonical)objective_base.push_back({e.first.first,e.first.second,e.second});
  }
  void objective_accepted_step() {
    if(study_enabled()){study_accepted_step();return;}
    if(!objective_nonconvex)return;
    admit_objective_atoms(w.st.objective_atom_batch);
    objective_rho=std::max(objective_floor,objective_rho*w.st.objective_rho_decay);
    // rho tends to zero after C covers the deficits. Keep a positive finite
    // complement when G+C has affine null coordinates; do not force it to zero.
    objective_excess=objective_rho-objective_floor;
  }
  void install_objective() {
    if(!objective_nonconvex)return;
    auto h=objective_base;
    for(auto& e:h)if(e.i==e.j) {
      const auto a=objective_diagonal_atoms.find(e.i);
      e.v+=objective_rho+(a==objective_diagonal_atoms.end()?0:a->second);
    }
    w.objective.set_model(center,std::move(h),slots);
  }
  Workspace &w;
  DcaFilterStatistics &stats;
  Vec center;
  std::map<Pair, int> slots;
  std::vector<Curvature> inequality;
  std::vector<int> corrected_inequalities;
  Vec transported_slacks;
  bool installed = false, curvature_ready = false;
  int builds = 0, curvature_builds = 0, curvature_reuses = 0;
  int small_spectra = 0, analytic_spectra = 0, gershgorin_certificates = 0;
  int positive_spectral_count = 0;
  int transport_count = 0, curvature_threads = 1;
  std::vector<unsigned char> extended_small_spectrum;
  double rho_max = 0, curvature_seconds = 0;

#include "dca_convexification_study_models.hpp"

  ~DcaInequalityModels(){study_summary();}

  DcaInequalityModels(Workspace &work, DcaFilterStatistics &diagnostics)
      : w(work), stats(diagnostics), inequality(work.mi) {
    if (w.st.dca_engine != "filter" && w.st.dca_engine != "majorants")
      throw std::invalid_argument("inequality majorants require the filter Workspace layout");
    for (int k = 0; k < int(w.hpattern.size()); ++k) slots[w.hpattern[k]] = k;
    // A nearly PSD homogeneous block can have a very pessimistic
    // Gershgorin deficit. Extend exact small-block spectra only under a
    // deterministic total cubic-work cap, before any parallel preparation.
    extended_small_spectrum.assign(work.mi,0);
    std::size_t spectral_work=0,homogeneous_rows=0;
    if(w.st.convexification_study=="off")for(int i=0;i<work.mi;++i) {
      const auto& q=w.inequalities[i];
      if(q.constant!=0.||!q.signed_square.empty()||q.hessian.empty()||
          !std::all_of(q.linear.begin(),q.linear.end(),[](const auto& a){return a.second==0.;}))continue;
      std::set<int> support;for(const auto& h:q.hessian){support.insert(h.i);support.insert(h.j);}
      const std::size_t n=support.size();
      if(n>16&&n<=64&&spectral_work+n*n*n<=20000000) {
        extended_small_spectrum[i]=1;spectral_work+=n*n*n;++homogeneous_rows;
      }
    }
    // Dense bounded supports already occupy the fixed KKT envelope. Their
    // positive spectral part avoids injecting a uniform correction into the
    // many null directions of a low-rank negative quadratic. Bound both work
    // and storage before any spectral calculation; no coefficient signs or
    // dataset identities are assumed. The reconstruction-residual certificate
    // below remains the authority for each admitted model.
    constexpr std::size_t dense_spectral_work_cap=536870912;
    constexpr std::size_t dense_spectral_entries_cap=2000000;
    std::size_t dense_spectral_work=0,dense_spectral_entries=0,dense_rows=0;
    if(w.st.convexification_study=="off")for(int i=0;i<work.mi;++i) {
      const auto& q=w.inequalities[i];
      if(extended_small_spectrum[i]||!q.signed_square.empty()||q.hessian.empty())continue;
      std::set<int> support;
      for(const auto& h:q.hessian){support.insert(h.i);support.insert(h.j);}
      const std::size_t n=support.size();
      if(n<=16||n>256||n*n*n>dense_spectral_work_cap-dense_spectral_work||
          n*(n+1)/2>dense_spectral_entries_cap-dense_spectral_entries)continue;
      bool covered=true;
      for(int j:support)for(int k:support)if(j<=k&&!slots.count(upper(j,k)))covered=false;
      if(!covered)continue;
      // Sparse rows do not earn a dense eigendecomposition from unrelated
      // assembly fill: require at least half of the possible source entries.
      if(q.hessian.size()<n*(n+1)/4)continue;
      extended_small_spectrum[i]=1;dense_spectral_work+=n*n*n;
      dense_spectral_entries+=n*(n+1)/2;++dense_rows;
    }
    if(dense_spectral_work&&(w.st.verbose||w.st.live_display_level>0))
      std::cout<<"[BOUNDED-DENSE-SPECTRUM] rows="<<dense_rows<<" cubic_work="<<dense_spectral_work
          <<" cubic_work_cap="<<dense_spectral_work_cap<<" maximum_support=256"
          <<" model_entries="<<dense_spectral_entries<<" model_entries_cap="<<dense_spectral_entries_cap
          <<" fixed_envelope=1 original_coefficients_unchanged=1 certified_PSD_majorants=1"<<std::endl;
    if(spectral_work&&(w.st.verbose||w.st.live_display_level>0))
      std::cout<<"[HOMOGENEOUS-SMALL-SPECTRUM] rows="
          <<homogeneous_rows
          <<" cubic_work="<<spectral_work<<" cubic_work_cap=20000000 maximum_support=64"
          <<" original_coefficients_unchanged=1 certified_PSD_majorants=1"<<std::endl;
    if(study_enabled()&&w.st.convexification_schedule=="eager")study_prepare();
  }

  // Return a local event code; the parallel caller reduces all counters only
  // after joining. No evaluator, shared ledger or statistics is touched here.
  int prepare_curvature(int i) {
    auto &cache = inequality[i];
    const auto &q = w.inequalities[i];
    if (cache.ready) {
      if (cache.scale != q.scale)
        throw std::runtime_error("inequality scaling changed after its curvature certificate");
      return q.hessian.empty() ? 0 : 1;
    }
    cache.scale = q.scale;
    if (!(q.scale > 0) || !std::isfinite(q.scale))
      throw std::runtime_error("invalid inequality scale for convex majorization");
    if(!q.signed_square.empty()) {
      // A global Hessian upper bound across BOTH signs, not the local branch.
      std::map<Pair,double> cap;
      for(auto e:q.base_hessian)cap[upper(e.i,e.j)]+=q.scale*e.v;
      for(auto e:q.signed_square)cap[upper(e.first,e.first)]+=2*q.scale*std::abs(e.second);
      std::map<int,double> diag,radius;
      for(auto e:cap){if(e.first.first==e.first.second)diag[e.first.first]+=e.second;else{radius[e.first.first]+=std::abs(e.second);radius[e.first.second]+=std::abs(e.second);}}
      double shift=0;for(auto e:cap){for(int j:{e.first.first,e.first.second})shift=std::max(shift,radius[j]-diag[j]);}
      std::set<int> indices;for(auto e:cap){indices.insert(e.first.first);indices.insert(e.first.second);}
      cache.support.assign(indices.begin(),indices.end());
      cache.native_shift=shift;
      for(int j:cache.support)cap[upper(j,j)]+=shift;
      for(auto e:cap)cache.hessian.push_back({e.first.first,e.first.second,e.second});
      cache.rho=shift;for(auto e:q.signed_square)cache.rho=std::max(cache.rho,4*q.scale*std::abs(e.second)+shift);
      cache.ready=true;return 4;
    }
    if (q.hessian.empty()) { cache.ready = true; return 0; }
    std::set<int> indices;
    for (auto e : q.hessian) { indices.insert(e.i); indices.insert(e.j); }
    cache.support.assign(indices.begin(), indices.end());
    const int n = int(cache.support.size());
    auto local = [&](int j) {
      return int(std::lower_bound(cache.support.begin(), cache.support.end(), j) - cache.support.begin());
    };
    std::vector<SymTriplet> h;
    h.reserve(q.hessian.size());
    for (auto e : q.hessian) h.push_back({local(e.i), local(e.j), q.scale * e.v});
    h = detail::DcaCurvature::canonical(n, h);
    Vec diagonal(n, 0), radius(n, 0);
    for (auto e : h) {
      if (e.i == e.j) diagonal[e.i] = e.v;
      else { radius[e.i] += std::abs(e.v); radius[e.j] += std::abs(e.v); }
    }
    double lower = INFINITY, norm = 0;
    for (int j = 0; j < n; ++j) {
      lower = std::min(lower, diagonal[j] - radius[j]);
      norm = std::max(norm, std::abs(diagonal[j]) + radius[j]);
    }
    if (!std::isfinite(lower) || !std::isfinite(norm))
      throw std::runtime_error("nonfinite inequality curvature certificate");
    const double padding = 64 * std::numeric_limits<double>::epsilon() * std::max(1.0, norm);
    int event = 2;
    if (lower >= 0) {
      // Diagonal dominance gives a direct PSD certificate, including affine
      // and semidefinite rows. No factorization or spectral probe is needed.
      cache.rho = 0;
    } else if (std::all_of(radius.begin(), radius.end(), [](double value) { return value == 0; })) {
      // A diagonal spectrum is its diagonal. In particular -2*I needs no
      // LAPACK call, iterative probe or extra certificate.
      event = 4;
      cache.rho = std::max(0.0, -*std::min_element(diagonal.begin(), diagonal.end()) + padding);
    } else if (n == 2) {
      // The smaller eigenvalue of a symmetric 2x2 block, with hypot avoiding
      // squaring/overflow and half-scaled additions avoiding a+d overflow.
      double off_diagonal = 0;
      for (auto e : h) if (e.i != e.j) off_diagonal = e.v;
      const double smallest = 0.5 * diagonal[0] + 0.5 * diagonal[1] -
          std::hypot(0.5 * diagonal[0] - 0.5 * diagonal[1], off_diagonal);
      event = 4;
      cache.rho = std::max(0.0, -smallest + padding);
    } else if (n <= 16 || extended_small_spectrum[i]) {
      Vec dense(n * n, 0), values(n), work(std::max(1, 3 * n));
      for (auto e : h) { dense[e.i + n * e.j] = e.v; dense[e.j + n * e.i] = e.v; }
      int info = 0, lwork = int(work.size());
      const char no = 'N', upper_triangle = 'U';
      dsyev_(&no, &upper_triangle, &n, dense.data(), &n, values.data(), work.data(), &lwork, &info);
      event = 3;
      if (info != 0 || !finite(values) || !finite(dense))
        throw std::runtime_error("inequality exact small-block spectrum failed");
      cache.rho = std::max(0.0, -values.front() + padding);
      // OPF angle blocks have paired spectra (-sigma,+sigma). The scalar
      // shift doubles their positive curvature unnecessarily. Reuse the
      // spectrum already computed above, without vectors or another probe:
      //   Hbar = (H + eta I)/2, C = Hbar-H = (eta I-H)/2.
      // eta bounds both spectral extremes, so Hbar and C are PSD. Requiring
      // ||C|| <= rho_isotropic also guarantees a pointwise tighter majorant.
      bool paired = values.front() < -padding && values.back() > padding;
      for (int j = 0; j < n && paired; ++j)
        paired = std::abs(values[j] + values[n-1-j]) <= padding;
      const double eta = std::max(std::abs(values.front()), std::abs(values.back())) + padding;
      const double correction_norm = 0.5 * (eta - values.front());
      if (paired && std::isfinite(eta) && correction_norm <= cache.rho) {
        cache.half_spectral = true;
        cache.spectral_shift = 0.5 * eta;
        cache.rho = correction_norm;
      } else if(cache.rho>0) {
        // Use the positive spectral part only if its full small support is
        // already present in the fixed assembly envelope. Otherwise retain
        // the sparse isotropic certificate; never silently drop fill entries.
        bool covered=true;
        for(int j=0;j<n;++j)for(int k=j;k<n;++k)
          covered=covered&&slots.count(upper(cache.support[j],cache.support[k]));
        if(covered) {
          // Preserve the original eigenvalues-only path for paired blocks.
          // Only this newly admitted model needs the spectral eigenvectors.
          std::fill(dense.begin(),dense.end(),0.0);
          for(const auto& e:h)dense[e.i+n*e.j]=dense[e.j+n*e.i]=e.v;
          const char vectors='V';
          dsyev_(&vectors,&upper_triangle,&n,dense.data(),&n,values.data(),work.data(),&lwork,&info);
          if(info||!finite(values)||!finite(dense))
            throw std::runtime_error("inequality positive-part spectrum failed");
          // If E = U Lambda U' - H, then
          //   Hbar = U max(Lambda,0) U' + delta I,
          //   Hbar-H = U max(-Lambda,0) U' + E + delta I.
          // Both are PSD when delta bounds ||E||_2. The symmetric row-sum
          // bound below controls reconstruction error, and the existing
          // arithmetic padding covers the final rounded Gram coefficients.
          Vec original(n*n,0),row_error(n,0);
          for(const auto& e:h)original[e.i+n*e.j]=original[e.j+n*e.i]=e.v;
          for(int j=0;j<n;++j)for(int k=0;k<n;++k) {
            long double value=0;
            for(int e=0;e<n;++e)
              value+=static_cast<long double>(dense[j+n*e])*values[e]*dense[k+n*e];
            row_error[j]+=double(std::abs(value-original[j+n*k]));
          }
          const double delta=padding+*std::max_element(row_error.begin(),row_error.end());
          if(!std::isfinite(delta))throw std::runtime_error("nonfinite spectral majorant certificate");
          cache.rho=std::max(0.0,-values.front())+delta;
          cache.positive_spectral=true;
          for(int j=0;j<n;++j)for(int k=j;k<n;++k) {
            long double value=j==k?delta:0;
            for(int e=0;e<n;++e)
              value+=static_cast<long double>(dense[j+n*e])*std::max(0.0,values[e])*dense[k+n*e];
            if(!std::isfinite(value))throw std::runtime_error("nonfinite spectral majorant coefficient");
            cache.hessian.push_back({cache.support[j],cache.support[k],double(value)});
          }
          cache.ready=true;
          return event;
        }
      }
    } else {
      // Explicit conservative certificate for the uncommon larger block.
      // No Ritz iterations, repeated inertia tests or bisection are performed.
      cache.rho = -lower + padding;
    }
    if (!std::isfinite(cache.rho))
      throw std::runtime_error("nonfinite inequality majorant correction");
    if (cache.rho > 0) {
      cache.hessian.reserve(h.size());
      for (auto e : h) {
        if (cache.half_spectral) {
          e.v *= 0.5;
          if (e.i == e.j) e.v += cache.spectral_shift;
        } else if (e.i == e.j) e.v += cache.rho;
        if (!std::isfinite(e.v)) throw std::runtime_error("overflow in inequality majorant Hessian");
        cache.hessian.push_back({cache.support[e.i], cache.support[e.j], e.v});
      }
    }
    cache.ready = true;
    return event;
  }

  // Compute 1/2 d'C d and, optionally, accumulate z*C*d. The paired-spectrum
  // correction uses the ORIGINAL Hessian with its frozen scale. Long-double
  // accumulation preserves the small gap in a nearly uncorrected direction;
  // no subtraction of original/model function values is involved.
  long double correction_gap(int i,const Vec& x,
      std::vector<long double>* dual=nullptr,double multiplier=0) const {
    if(study_enabled())return study_gap(i,x,dual,multiplier);
    const auto& cache=inequality[i];
    const auto& q=w.inequalities[i];
    if(!q.signed_square.empty()) {
      // The base quadratic cancels exactly. For each native term the model
      // uses the global curvature cap 2*abs(c), valid across sign changes.
      // Evaluate its nonnegative gap without subtracting nearly equal values.
      long double gap=0;
      for(int j:cache.support) {
        const long double d=static_cast<long double>(x[j])-center[j];
        gap+=0.5L*cache.native_shift*d*d;
        if(dual)(*dual)[j]+=static_cast<long double>(multiplier)*cache.native_shift*d;
      }
      for(auto e:q.signed_square) {
        const long double coefficient=static_cast<long double>(q.scale)*e.second;
        if(coefficient==0)continue;
        const int sign=coefficient>0?1:-1;
        const long double a=std::abs(coefficient),u=sign*static_cast<long double>(center[e.first]),v=sign*static_cast<long double>(x[e.first]);
        long double dg=0;
        if(u>=0&&v>=0) {} // The native term is already convex on this branch.
        else if(u<=0&&v<=0) {const long double d=v-u;gap+=2*a*d*d;dg=4*a*d;}
        else if(u>=0) {gap+=2*a*v*v;dg=4*a*v;}
        else {gap+=2*a*u*(u-2*v);dg=-4*a*u;}
        if(dual)(*dual)[e.first]+=static_cast<long double>(multiplier)*sign*dg;
      }
      return gap;
    }
    if(cache.positive_spectral) {
      long double gap=0;
      auto accumulate=[&](const std::vector<SymTriplet>& matrix,long double weight) {
        for(const auto& e:matrix) {
          const long double v=weight*e.v;
          const long double di=static_cast<long double>(x[e.i])-center[e.i];
          const long double dj=static_cast<long double>(x[e.j])-center[e.j];
          gap+=(e.i==e.j?0.5L:1.0L)*v*di*dj;
          if(dual) {
            (*dual)[e.i]+=multiplier*v*dj;
            if(e.i!=e.j)(*dual)[e.j]+=multiplier*v*di;
          }
        }
      };
      accumulate(cache.hessian,1);accumulate(q.hessian,-cache.scale);
      return gap;
    }
    const long double diagonal=cache.half_spectral?cache.spectral_shift:cache.rho;
    long double gap=0;
    for(int j:cache.support) {
      const long double d=static_cast<long double>(x[j])-center[j];
      gap+=0.5L*diagonal*d*d;
      if(dual)(*dual)[j]+=static_cast<long double>(multiplier)*diagonal*d;
    }
    if(cache.half_spectral)for(const auto& e:q.hessian) {
      const long double hij=-0.5L*cache.scale*e.v;
      const long double di=static_cast<long double>(x[e.i])-center[e.i];
      const long double dj=static_cast<long double>(x[e.j])-center[e.j];
      gap+=(e.i==e.j?0.5L:1.0L)*hij*di*dj;
      if(dual) {
        (*dual)[e.i]+=static_cast<long double>(multiplier)*hij*dj;
        if(e.i!=e.j)(*dual)[e.j]+=static_cast<long double>(multiplier)*hij*di;
      }
    }
    return gap;
  }

  // Release the auxiliary model without touching the primal-dual state or
  // the function/gradient/residual cache. Do not early-return on !installed:
  // reanchoring can throw after installing only some corrected rows.
  void discard() noexcept {
    if(study_enabled()){w.objective.model_active=false;for(auto& q:w.equalities)q.model_active=false;for(auto& q:w.inequalities)q.model_active=false;installed=false;return;}
    for (int i : corrected_inequalities) w.inequalities[i].model_active = false;
    if(w.st.objective_lazy_convexification)w.objective.model_active=false;
    installed = false;
  }

  // Legacy compatibility only; the original-filter outer route uses discard.
  // Convert an auxiliary slack into the original slack at the CURRENT point:
  // g+s_original = g_tilde+s_auxiliary. This remains valid at infeasible
  // intermediate points and is deliberately independent of complementarity.
  // Turning the installed models off makes a second call a no-op.
  void transport_slacks() {
    if (!installed) return;
    // Stage only changed rows in a reusable buffer. No state is committed
    // until every correction has passed domain and finite-value checks.
    // Allocate only on this legacy path; R7 never needs a transport buffer.
    transported_slacks.resize(corrected_inequalities.size());
    for (std::size_t row = 0; row < corrected_inequalities.size(); ++row) {
      const int i = corrected_inequalities[row];
      auto &cache = inequality[i];
      auto &q = w.inequalities[i];
      if (cache.scale != q.scale)
        throw std::runtime_error("inequality scaling changed before slack transport");
      if (!q.model_active || q.model_center != &center)
        throw std::runtime_error("inequality majorant was removed before slack transport");
      const long double exact_gap = correction_gap(i,w.x);
      const double gap = static_cast<double>(exact_gap);
      if (!std::isfinite(gap) || gap < 0 ||
          !(w.s[i] > 0) || !std::isfinite(w.s[i]))
        throw std::runtime_error("invalid majorant gap or slack during transport");
      transported_slacks[row] = w.s[i] + gap;
      if (!(transported_slacks[row] > 0) || !std::isfinite(transported_slacks[row]))
        throw std::runtime_error("nonfinite transported original slack");
    }
    for (std::size_t row = 0; row < corrected_inequalities.size(); ++row) {
      const int i = corrected_inequalities[row];
      w.s[i] = transported_slacks[row];
      w.inequalities[i].model_active = false;
    }
    installed = false;
    ++transport_count;
    ++stats.majorant_slack_transports;
  }

  // Constant QCQP Hessians and frozen scaling make this a one-time operation.
  // It does not evaluate functions, move the centre, or transport any state.
  void prepare_curvature_once() {
    if(study_enabled()){study_prepare();return;}
    if (curvature_ready) return;
    const auto start = Clock::now();
    prepare_objective();
    rho_max = 0;
    std::vector<int> events(w.mi, 0);
    std::vector<std::exception_ptr> failures(w.mi);
    int current_threads = 1;
#if defined(LAZYIPDCA_HAVE_OPENMP)
#pragma omp parallel if(w.mi > 1)
    {
#pragma omp single
      current_threads = omp_get_num_threads();
#pragma omp for schedule(dynamic, 64)
      for (int i = 0; i < w.mi; ++i) {
        try { events[i] = prepare_curvature(i); }
        catch (...) { failures[i] = std::current_exception(); }
      }
    }
#else
    for (int i = 0; i < w.mi; ++i) {
      try { events[i] = prepare_curvature(i); }
      catch (...) { failures[i] = std::current_exception(); }
    }
#endif
    curvature_threads = std::max(curvature_threads, current_threads);
    for (int i = 0; i < w.mi; ++i) {
      if (failures[i]) std::rethrow_exception(failures[i]);
      if (events[i] == 1) ++curvature_reuses;
      else if (events[i] >= 2) {
        ++curvature_builds;
        if (events[i] == 3) ++small_spectra;
        else if (events[i] == 4) ++analytic_spectra;
        else ++gershgorin_certificates;
      }
      rho_max = std::max(rho_max, inequality[i].rho);
      if(inequality[i].positive_spectral)++positive_spectral_count;
      if (inequality[i].rho > 0) corrected_inequalities.push_back(i);
    }
    curvature_ready = true;
    curvature_seconds += seconds(start);
    stats.rho_lagrangian = 0;
    stats.rho_inequalities_max = rho_max;
    stats.atoms_lagrangian = stats.atoms_inequalities = 0;
    stats.curvature_reuses = curvature_reuses;
    stats.majorant_curvature_threads = curvature_threads;
    stats.majorant_curvature_certificates = curvature_builds;
    stats.majorant_curvature_reuses = curvature_reuses;
    stats.majorant_curvature_small_spectra = small_spectra;
    stats.majorant_curvature_analytic_spectra = analytic_spectra;
    stats.majorant_curvature_gershgorin = gershgorin_certificates;
    stats.majorant_curvature_seconds = curvature_seconds;
    stats.curvature_factors = 0;
    // The transport identity is tested independently. Production does not
    // claim to have measured it by recomputing an arithmetic identity.
    stats.majorant_slack_transport_error = std::numeric_limits<double>::quiet_NaN();
    stats.model_build_seconds += seconds(start);
  }

  // PRECONDITION: original values and (extended) gradients are cached at w.x,
  // the previous auxiliary has been discarded, and curvature was
  // prepared once. This method neither evaluates nor transports any state.
  void reanchor_from_cache() {
    if(study_enabled()){study_reanchor();return;}
    if (!curvature_ready)
      throw std::runtime_error("inequality curvature must be prepared before reanchoring");
    if (installed)
      throw std::runtime_error("discard the previous inequality model before reanchoring");
    const auto start = Clock::now();
    center = w.x;
    install_objective();
    for (int i : corrected_inequalities) {
      auto &q = w.inequalities[i];
      auto &cache = inequality[i];
      if (cache.scale != q.scale)
        throw std::runtime_error("inequality scaling changed after its curvature certificate");
      if (!cache.model_mapped) {
        q.set_model(center, cache.hessian, slots);
        cache.model_mapped = true;
        q.model_atoms.clear();
        q.model_atom_support.clear();
      }
      else {
        q.model_center = &center;
        q.model_constant = q.value;
        q.model_gradient = q.gradient;
        if (q.majorant_gradients_extended) q.model_gradient_extended = q.gradient_extended;
        q.model_active = true;
      }
    }
    installed = true;
    if (builds) curvature_reuses += curvature_builds;
    ++builds;
    ++stats.models;
    stats.curvature_reuses = curvature_reuses;
    stats.majorant_curvature_reuses = curvature_reuses;
    stats.model_build_seconds += seconds(start);
  }

  // Compatibility entry point for isolated model tests and the older engine.
  // The outer-only controller calls prepare, reanchor and discard explicitly.
  void build() {
    if (installed) transport_slacks();
    else dca_disable_models(w);
    w.evaluate();
    prepare_curvature_once();
    reanchor_from_cache();
  }

  // Positive value gaps are evaluated directly from their PSD correction,
  // avoiding cancellation between two large quadratic function values.
  // Values use original constraint units; multiplier-weighted gaps and
  // stationarity use original objective units.
  Gaps gaps(const Vec &x, const Vec &z) const {
    if (x.size() != std::size_t(w.n) || z.size() != std::size_t(w.mi))
      throw std::invalid_argument("inequality majorant gap dimension mismatch");
    Gaps result;
    if (center.empty()) return result;
    std::vector<long double> dual(w.n, 0);
    for (int i : corrected_inequalities) {
      const auto &cache = inequality[i];
      const double gap = static_cast<double>(correction_gap(i,x,&dual,z[i]));
      if(!std::isfinite(gap)||gap<0)
        throw std::runtime_error("invalid inequality majorant gap");
      result.value_gap_inf = std::max(result.value_gap_inf, gap / cache.scale);
      result.weighted_value_gap_inf = std::max(result.weighted_value_gap_inf,
          std::abs(z[i]) * gap / w.objective.scale);
    }
    long double dual_norm=0;
    for(long double value:dual) {
      if(!std::isfinite(value))throw std::runtime_error("nonfinite inequality majorant stationarity gap");
      dual_norm=std::max(dual_norm,std::abs(value));
    }
    result.dual_gap_inf = static_cast<double>(dual_norm / w.objective.scale);
    return result;
  }
};
