// Runtime MA27 adapter for the LP64, double precision HSL interface.
// Included inside linear_solver.cpp's anonymous namespace. No HSL solver code
// is distributed: the user supplies a shared library under its own licence.
#ifdef LAZYIPDCA_HAVE_HSL
class HslLibrary {
 public:
  explicit HslLibrary(const Settings& s) {
    std::string requested=s.hsl_library;
    if(requested.empty()) if(const char* p=std::getenv("HSL_LIBRARY")) requested=p;
    std::vector<std::string> candidates;
    auto add=[&](const std::filesystem::path& path){
      const auto value=path.string();
      if(!value.empty()&&std::find(candidates.begin(),candidates.end(),value)==candidates.end())
        candidates.push_back(value);
    };
    auto add_root=[&](const std::filesystem::path& root){
      for(const auto* name:{"libcoinhsl.so","libhsl.so"}){
        add(root/name);
        add(root/"lib"/name);
        add(root/"coinhsl"/"lib"/name);
      }
    };
    if(!requested.empty())add(requested);
    else {
      // Preserve the dynamic loader's standard search first (LD_LIBRARY_PATH,
      // rpath and system cache), then inspect deterministic installation roots.
      add("libcoinhsl.so");add("libhsl.so");
      for(const auto* variable:{"IPDCA_QCQP_BASE","LAZYIPDCA_HOME","COINHSL_ROOT","CONDA_PREFIX"})
        if(const char* value=std::getenv(variable);value&&*value)add_root(value);
#if defined(__linux__)
      std::error_code error;
      auto executable=std::filesystem::read_symlink("/proc/self/exe",error);
      if(!error){
        auto root=executable.parent_path();
        for(int level=0;level<6&&!root.empty();++level){
          add_root(root);
          const auto parent=root.parent_path();
          if(parent==root)break;
          root=parent;
        }
      }
#endif
    }
    std::string last_error;
    for(const auto& p:candidates) {
      handle_=dlopen(p.c_str(),RTLD_NOW|RTLD_LOCAL);
      if(handle_) {path_=p;return;}
      const char* error=dlerror();
      last_error=error?error:"dlopen failed";
    }
    std::string detail;
    if(!requested.empty())
      detail=" Failed to load '"+requested+"': "+last_error+".";
    else
      detail=" Automatic discovery checked "+std::to_string(candidates.size())+
        " loader/install locations (environment roots and executable parents).";
    throw std::runtime_error(
        "HSL unavailable for explicit MA27. Pass --hsl-library /path/libcoinhsl.so "
        "or set HSL_LIBRARY; otherwise select --linear-backend mumps."+detail);
  }
  ~HslLibrary(){if(handle_)dlclose(handle_);}
  HslLibrary(const HslLibrary&)=delete;
  HslLibrary& operator=(const HslLibrary&)=delete;
  template<class T> T symbol(const std::string& name,bool fortran=false) const {
    std::vector<std::string> names{name};
    if(fortran) {
      std::string upper=name;
      for(char& c:upper)c=static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      names={name+"_",name,upper,upper+"_"};
    }
    for(const auto& n:names)if(void* p=dlsym(handle_,n.c_str()))return reinterpret_cast<T>(p);
    throw std::runtime_error("HSL library '"+path_+"' does not provide "+name+
                             "; requested backend cannot be substituted");
  }
 private:
  void* handle_=nullptr;
  std::string path_;
};

int hsl_size(std::int64_t count) {
  if(count<1||count>std::numeric_limits<int>::max())
    throw std::runtime_error("HSL LP64 workspace/index size exceeds 32-bit range");
  return static_cast<int>(count);
}
int hsl_grow(int count,double factor) {
  return hsl_size(static_cast<std::int64_t>(std::ceil(std::max(1,count)*factor)));
}

class HslFactorBase : public SymmetricFactorization {
 public:
  explicit HslFactorBase(const Settings& s):st_(s),library_(s) {}
  bool factor(int n,int primal,const std::vector<SymmetricTriplet>& input,std::string& message) final {
    ready_=probe_ready_=false;defect_=0;reported_negative_=-1;
    try {
      if(n<1||primal<0||primal>n)throw std::runtime_error("invalid HSL dimensions");
      for(const auto& e:input)
        if(e.i<0||e.j<0||e.i>=n||e.j>=n||!std::isfinite(e.value))
          throw std::runtime_error("invalid HSL matrix coordinate or non-finite value");
      const bool cached=scatter_.assemble(n,input,st_.kkt_adaptive_regularization?-1:primal,
          st_.kkt_primal_regularization,st_.kkt_dual_regularization,true,matrix_);
      if(!cached)matrix_=st_.kkt_adaptive_regularization?aggregate_upper(n,input,-1,0,0,true)
        :aggregate_upper(n,input,primal,st_.kkt_primal_regularization,st_.kkt_dual_regularization,true);
      for(const auto& e:matrix_)if(!std::isfinite(e.value))throw std::runtime_error("HSL matrix accumulation overflow");
      norm_=symmetric_inf_norm(n,matrix_);
      const auto uncached_pattern=cached?std::vector<std::uint64_t>{}:entry_pattern(matrix_);
      const auto& current=cached?scatter_.pattern():uncached_pattern;
      const bool same=analyzed_&&n_==n;
      reused_=same&&pattern_contains(pattern_,current);
      if(!reused_) {
        auto envelope=same?pattern_union(pattern_,current):current;
        if(analyzed_)++invalidations_;
        clear_numeric();analyzed_=false;n_=n;pattern_=std::move(envelope);
        ne_=hsl_size(pattern_.size());
        rows_.resize(ne_);cols_.resize(ne_);values_.assign(ne_,0);
        for(int k=0;k<ne_;++k) {
          rows_[k]=upper_key_row(pattern_[k])+1;
          cols_[k]=upper_key_col(pattern_[k])+1;
        }
        if(!analyze(message))return false;
        analyzed_=true;++analyses_;
      }else ++reuses_;
      std::fill(values_.begin(),values_.end(),0.0);
      std::size_t k=0;
      for(std::size_t j=0;j<current.size();++j) {
        while(k<pattern_.size()&&pattern_[k]<current[j])++k;
        if(k==pattern_.size()||pattern_[k]!=current[j])throw std::runtime_error("HSL symbolic envelope mismatch");
        values_[k]=matrix_[j].value;
      }
      int negative=0;
      if(!numeric(negative,message))return false;
      if(negative<0||negative>n_)throw std::runtime_error("HSL returned invalid inertia");
      reported_negative_=negative;
      probe_ready_=true;defect_=std::max(0,negative-(n-primal));
      if(negative!=n-primal) {
        message=st_.linear_backend+" inertia mismatch: negative pivots="+std::to_string(negative)+
          ", expected="+std::to_string(n-primal);return false;
      }
      ready_=true;return true;
    }catch(const std::exception& e){message=e.what();return false;}
  }
  bool solve(Vec& rhs,std::string& message) const final {
    if(!ready_){message="HSL solve requires a successful inertia certificate";return false;}
    return audited_solve(rhs,message);
  }
  bool solve_curvature_probe(Vec& rhs,std::string& message) const final {
    if(!probe_ready_||defect_<=0){message="HSL curvature probe unavailable";return false;}
    return audited_solve(rhs,message);
  }
  int last_negative_pivots()const final{return reported_negative_;}
  int last_positive_pivots()const final{return reported_negative_>=0?n_-reported_negative_:-1;}
  int last_zero_pivots()const final{return reported_negative_>=0?0:-1;}
  int curvature_defect() const final{return defect_;}
  int symbolic_analyses()const final{return analyses_;}
  int symbolic_reuses()const final{return reuses_;}
  int symbolic_invalidations()const final{return invalidations_;}
  bool last_symbolic_reused()const final{return reused_;}
  bool has_symbolic_cache()const final{return analyzed_;}
  double symbolic_seconds()const final{return symbolic_seconds_;}
  double numeric_seconds()const final{return numeric_seconds_;}
  int numeric_scatter_builds()const final{return scatter_.builds();}
  int numeric_scatter_reuses()const final{return scatter_.reuses();}
  int numeric_scatter_invalidations()const final{return scatter_.invalidations();}
  bool last_numeric_scatter_reused()const final{return scatter_.last_reused();}
  double last_backward_error()const final{return berr_;}
  double last_initial_backward_error()const final{return initial_berr_;}
  double last_residual_inf()const final{return residual_;}
  int last_refinement_steps()const final{return refinements_;}
 protected:
  Settings st_;
  HslLibrary library_;
  int n_=0,ne_=0;
  std::vector<int> rows_,cols_;
  Vec values_;
  double symbolic_seconds_=0,numeric_seconds_=0;
  virtual bool analyze(std::string&)=0;
  virtual bool numeric(int&,std::string&)=0;
  virtual bool raw_solve(Vec&,std::string&)=0;
  virtual void clear_numeric()=0;
 private:
  bool audited_solve(Vec& rhs,std::string& message)const {
    if(static_cast<int>(rhs.size())!=n_){message="HSL RHS dimension mismatch";return false;}
    scratch_x_=rhs;Vec& x=scratch_x_;
    auto* self=const_cast<HslFactorBase*>(this);
    if(!self->raw_solve(x,message))return false;
    const double binf=inf_norm(rhs);
    berr_=initial_berr_=backward_error_into_known_binf(matrix_,norm_,binf,x,rhs,scratch_residual_,&residual_);refinements_=0;
    while(std::isfinite(berr_)&&berr_>st_.kkt_backward_error_tolerance&&
          refinements_<st_.kkt_iterative_refinement_steps) {
      if(!self->raw_solve(scratch_residual_,message))return false;
      axpy(1,scratch_residual_,x);++refinements_;
      berr_=backward_error_into_known_binf(matrix_,norm_,binf,x,rhs,scratch_residual_,&residual_);
    }
    for(double v:x)if(!std::isfinite(v)){message="HSL non-finite solution";return false;}
    if(!std::isfinite(berr_)||berr_>st_.kkt_backward_error_reject) {
      message=st_.linear_backend+" original matrix backward error rejected: "+std::to_string(berr_);return false;
    }
    rhs=x;return true;
  }
  bool ready_=false,probe_ready_=false,analyzed_=false,reused_=false;
  int defect_=0,reported_negative_=-1,analyses_=0,reuses_=0,invalidations_=0;
  std::vector<std::uint64_t> pattern_;
  EntryList matrix_;
  detail::StableUpperAssemblyCache scatter_;
  double norm_=0;
  mutable double berr_=0,initial_berr_=0,residual_=0;
  mutable int refinements_=0;
  mutable Vec scratch_x_,scratch_residual_;
};

class Ma27Factor final:public HslFactorBase {
  using Init=void(*)(int*,double*);
  using Analyze=void(*)(int*,int*,const int*,const int*,int*,int*,int*,int*,int*,int*,int*,double*,int*,double*);
  using Factor=void(*)(int*,int*,const int*,const int*,double*,int*,int*,int*,int*,int*,int*,int*,int*,double*,int*);
  using Solve=void(*)(int*,double*,int*,int*,int*,double*,int*,double*,int*,int*,int*,int*);
 public:
  explicit Ma27Factor(const Settings& s,int maximum_workspace_mb=0):HslFactorBase(s),workspace_cap_mb_(maximum_workspace_mb) {
    if(workspace_cap_mb_<0||workspace_cap_mb_>128)throw std::invalid_argument("MA27 bounded workspace cap must be in 1..128 MB or zero for the ordinary path");
    auto init=library_.symbol<Init>("ma27id",true);
    analyze_=library_.symbol<Analyze>("ma27ad",true);
    factor_=library_.symbol<Factor>("ma27bd",true);
    solve_=library_.symbol<Solve>("ma27cd",true);
    init(icntl_,cntl_);icntl_[0]=icntl_[1]=icntl_[2]=0;cntl_[0]=st_.hsl_pivot_tolerance;
  }
  const char* name()const final{return st_.linear_metis
    ?"ma27_metis_pivoted_persistent_symbolic_refined"
    :"ma27_internal_pivoted_persistent_symbolic_refined";}
 private:
  int workspace_cap_mb_=0;bool workspace_failed_=false;

  // Bounded restoration caps the numerical real/integer MA27 work arrays.
  // It does not claim to cap symbolic auxiliary arrays or total process RSS.
  // Integer growth preserves the analyzed factor data; real factor input is
  // reloaded from values_ before every numeric attempt, so it can be released.
  bool workspace_allowed(std::size_t real_count,std::size_t integer_count,
                         std::string& message) {
    if(workspace_cap_mb_<=0)return true;
    const long double bytes=sizeof(double)*static_cast<long double>(real_count)+
        sizeof(int)*static_cast<long double>(integer_count);
    if(bytes<=static_cast<long double>(workspace_cap_mb_)*1024*1024)return true;
    workspace_failed_=true;
    message="MA27 bounded numerical A/IW workspace exceeds "+std::to_string(workspace_cap_mb_)+" MB";
    return false;
  }
  bool resize_integer_workspace(int count,std::string& message) {
    if(workspace_cap_mb_<=0){iw_.resize(count);return true;}
    if(count<0){workspace_failed_=true;message="invalid MA27 bounded integer workspace";return false;}
    const auto wanted=static_cast<std::size_t>(count);
    if(wanted<=iw_.capacity()){iw_.resize(wanted);return true;}
    // During replacement the previous integer array remains live. Account
    // for BOTH arrays before allocating; free the obsolete real input first.
    Vec().swap(a_);
    if(!workspace_allowed(0,iw_.capacity()+wanted,message))return false;
    std::vector<int> next;
    try{next.resize(wanted);}catch(const std::exception& error){
      workspace_failed_=true;message=std::string("MA27 bounded integer allocation failed: ")+error.what();return false;
    }
    if(!workspace_allowed(0,iw_.capacity()+next.capacity(),message))return false;
    std::copy_n(iw_.begin(),std::min(iw_.size(),wanted),next.begin());
    iw_.swap(next);
    return true;
  }
  bool resize_real_workspace(int count,std::string& message) {
    if(workspace_cap_mb_<=0){a_.resize(count);return true;}
    if(count<0){workspace_failed_=true;message="invalid MA27 bounded real workspace";return false;}
    const auto wanted=static_cast<std::size_t>(count);
    if(wanted<=a_.capacity()){a_.resize(wanted);return true;}
    // No numeric factor input survives a growth retry: values_ is reloaded
    // before the next call to MA27BD. Releasing avoids an uncounted copy peak.
    Vec().swap(a_);
    if(!workspace_allowed(wanted,iw_.capacity(),message))return false;
    Vec next;
    try{next.resize(wanted);}catch(const std::exception& error){
      workspace_failed_=true;message=std::string("MA27 bounded real allocation failed: ")+error.what();return false;
    }
    if(!workspace_allowed(next.capacity(),iw_.capacity(),message))return false;
    a_.swap(next);
    return true;
  }

  bool metis_order(std::string& message) {
#ifndef LAZYIPDCA_HAVE_METIS
    message="MA27 METIS ordering requested but METIS is not compiled; rebuild with "
            "METIS or use --linear-metis 0 for MA27 internal ordering";
    return false;
#else
#ifdef METIS_NOPTIONS
    using MetisIndex=idx_t;
#else
    using MetisIndex=idxtype;
#endif
    // The symbolic envelope contains unique upper coordinates, including zeros.
    // METIS receives an undirected graph without self edges, in zero-based CSR.
    std::int64_t edge_count=0;
    for(int k=0;k<ne_;++k)if(rows_[k]!=cols_[k])edge_count+=2;
    if(edge_count>std::numeric_limits<MetisIndex>::max()||
       static_cast<std::int64_t>(n_)>std::numeric_limits<MetisIndex>::max()) {
      message="MA27 graph exceeds METIS index range";return false;
    }
    // Avoid calling METIS on a singleton or an entirely edgeless graph.
    if(n_==1||edge_count==0) {
      std::iota(keep_.begin(),keep_.begin()+n_,1);return true;
    }
    std::vector<MetisIndex> offsets(static_cast<std::size_t>(n_)+1,0);
    for(int k=0;k<ne_;++k)if(rows_[k]!=cols_[k]) {
      ++offsets[rows_[k]];++offsets[cols_[k]];
    }
    for(int j=0;j<n_;++j)offsets[j+1]+=offsets[j];
    std::vector<MetisIndex> edges(static_cast<std::size_t>(edge_count)),cursor=offsets;
    std::vector<MetisIndex> perm(n_,-1),inverse(n_,-1);
    for(int k=0;k<ne_;++k)if(rows_[k]!=cols_[k]) {
      const int i=rows_[k]-1,j=cols_[k]-1;
      edges[cursor[i]++]=j;edges[cursor[j]++]=i;
    }
#ifdef METIS_NOPTIONS
    MetisIndex n=n_,options[METIS_NOPTIONS];
    int rc=METIS_SetDefaultOptions(options);
    if(rc!=METIS_OK){message="MA27 METIS default options failed";return false;}
    options[METIS_OPTION_NUMBERING]=0;
    options[METIS_OPTION_SEED]=static_cast<MetisIndex>(st_.random_seed);
    {LinearPhaseTime phase(symbolic_seconds_);
      rc=METIS_NodeND(&n,offsets.data(),edges.data(),nullptr,options,perm.data(),inverse.data());}
    if(rc!=METIS_OK){message="MA27 METIS_NodeND failed, code="+std::to_string(rc);return false;}
#else
    int n=n_,numflag=0,options[8]{};
    {LinearPhaseTime phase(symbolic_seconds_);
      METIS_NodeND(&n,offsets.data(),edges.data(),&numflag,options,perm.data(),inverse.data());}
#endif
    // MA27 expects position-of-variable, NOT variable-at-position. For zero-
    // based METIS this is IKEEP(i+1)=iperm[i]+1 (MA27 specification, section 2.1).
    // Validate before crossing the Fortran ABI, including the METIS 4 void API.
    std::vector<unsigned char> seen(n_,0);
    for(int i=0;i<n_;++i) {
      const auto position=inverse[i];
      if(position<0||position>=n_||seen[position]++||perm[position]!=i) {
        message="MA27 METIS returned an invalid permutation";return false;
      }
      keep_[i]=static_cast<int>(position)+1;
    }
    return true;
#endif
  }
  bool analyze(std::string& message)final {
    liw_=hsl_size(2LL*(2LL*ne_+3LL*n_+1));if(!resize_integer_workspace(liw_,message))return false;
    keep_.resize(hsl_size(3LL*n_));iw1_.resize(hsl_size(2LL*n_));
    int flag=0;double ops=0;
    if(st_.linear_metis) {
      if(!metis_order(message))return false;
      flag=1;
    }
    {LinearPhaseTime phase(symbolic_seconds_);
      analyze_(&n_,&ne_,rows_.data(),cols_.data(),iw_.data(),&liw_,keep_.data(),iw1_.data(),
               &steps_,&flag,icntl_,cntl_,info_,&ops);}
    if(info_[0]!=0){message="MA27 analysis INFO(1)="+std::to_string(info_[0]);return false;}
    la_=hsl_grow(std::max(ne_,info_[4]),st_.hsl_memory_factor);
    liw_=hsl_grow(info_[5],st_.hsl_memory_factor);
    if(workspace_cap_mb_>0){
      if(!resize_integer_workspace(liw_,message)||!resize_real_workspace(la_,message))return false;
    }else{a_.resize(la_);iw_.resize(liw_);}
    return true;
  }
  bool numeric(int& negative,std::string& message)final {
    if(workspace_failed_){message="MA27 bounded workspace unavailable after failed allocation";return false;}
    for(int retry=0;retry<=4;++retry) {
      std::copy(values_.begin(),values_.end(),a_.begin());
      {LinearPhaseTime phase(numeric_seconds_);
        factor_(&n_,&ne_,rows_.data(),cols_.data(),a_.data(),&la_,iw_.data(),&liw_,keep_.data(),
                &steps_,&maxfront_,iw1_.data(),icntl_,cntl_,info_);}
      if(info_[0]==0){negative=info_[14];return true;}
      if(retry==4||(info_[0]!=-3&&info_[0]!=-4))break;
      if(info_[0]==-3){
        liw_=hsl_grow(std::max(liw_,info_[1]),st_.hsl_memory_factor);
        if(workspace_cap_mb_>0){
          if(!resize_integer_workspace(liw_,message)||!resize_real_workspace(la_,message))return false;
        }else iw_.resize(liw_);
      }else{
        la_=hsl_grow(std::max(la_,info_[1]),st_.hsl_memory_factor);
        if(workspace_cap_mb_>0){if(!resize_real_workspace(la_,message))return false;}
        else a_.resize(la_);
      }
    }
    message="MA27 factor INFO(1)="+std::to_string(info_[0])+" INFO(2)="+std::to_string(info_[1]);return false;
  }
  bool raw_solve(Vec& rhs,std::string& message)final {
    work_.resize(std::max(1,maxfront_));
    solve_(&n_,a_.data(),&la_,iw_.data(),&liw_,work_.data(),&maxfront_,rhs.data(),iw1_.data(),&steps_,icntl_,info_);
    if(info_[0]!=0){message="MA27 solve INFO(1)="+std::to_string(info_[0]);return false;}return true;
  }
  void clear_numeric()final{iw_.clear();keep_.clear();a_.clear();if(workspace_cap_mb_>0){std::vector<int>().swap(iw_);Vec().swap(a_);workspace_failed_=false;}}
  Analyze analyze_=nullptr;Factor factor_=nullptr;Solve solve_=nullptr;
  int icntl_[30]{},info_[20]{},la_=0,liw_=0,steps_=0,maxfront_=0;
  double cntl_[5]{};
  std::vector<int> iw_,keep_,iw1_;Vec a_,work_;
};

#endif // LAZYIPDCA_HAVE_HSL
