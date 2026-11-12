// Included before Workspace. No object, spectrum, scan or atom map is built
// until a Newton factor has failed its existing inertia/recovery gates.
struct AtomWitness {Vec direction;double curvature,initial;};
struct NewtonAtom {
  int family=0,origin=-1; // objective, inequality, signed equality, completion
  double eigenvalue=0.0,weight=0.0,source_negative=0.0;
  std::vector<int> original_support,support;
  Vec original_values,values;
  struct Entry {int i,j,slot;};
  std::vector<Entry> entries;
  detail::NewtonAtomProjection projection;
  long long projected_generation=-1;
  bool compiled=false,selected=false,coordinate=false;
};
struct NewtonAtoms {
  std::vector<NewtonAtom> pool;
  std::vector<int> active_ids;
  std::set<const Quadratic*> cached;
  std::map<Pair,int> slots;
  std::unordered_map<int,int> completion;
  bool slots_ready=false;
  long long tangent_generation=0;
  int projection_requests=0;
  struct ProductAtom {int begin,end;double weight;};
  std::vector<ProductAtom> product_atoms;
  std::vector<int> product_indices;
  Vec product_values;
  bool product_ready=false;
  static constexpr int block_limit=16, support_limit=64, rank_limit=64;

  static void eigensystem(Vec& matrix,int n,Vec& eigenvalues) {
    eigenvalues.resize(n);Vec work(std::max(1,8*n));int lwork=(int)work.size(),info=0;
    const char job='V',uplo='U';
    dsyev_(&job,&uplo,&n,matrix.data(),&n,eigenvalues.data(),work.data(),&lwork,&info);
    if(info||!finite(matrix)||!finite(eigenvalues))throw std::runtime_error("atomic source eigensystem failed");
  }
  void cache(const Quadratic& q,int family,int origin,double,LazyNewtonStatistics& stats,bool energy_sparsify) {
    if(!cached.insert(&q).second)return;
    ++stats.atom_source_analyses;
    // Connected components of the Hessian support (linear-only variables
    // are irrelevant). Small blocks use their exact signed eigendirections.
    std::map<int,int> parent;
    auto root=[&](int i) {int r=i;while(parent[r]!=r)r=parent[r];while(parent[i]!=i){int next=parent[i];parent[i]=r;i=next;}return r;};
    for(auto e:q.hessian){parent.emplace(e.i,e.i);parent.emplace(e.j,e.j);}
    for(auto e:q.hessian){int a=root(e.i),b=root(e.j);if(a!=b)parent[a]=b;}
    std::map<int,std::vector<int>> components;
    for(auto p:parent)components[root(p.first)].push_back(p.first);
    std::map<int,std::vector<SymTriplet>> entries;
    for(auto e:q.hessian)entries[root(e.i)].push_back(e);
    for(const auto& component:components) {
      const auto& support=component.second;const int m=(int)support.size();
      auto push=[&](double ev,const std::vector<int>& ids,const Vec& v) {
        if(ev==0.0||(family!=2&&ev>0.0))return;
        NewtonAtom a;a.family=family;a.origin=origin;a.eigenvalue=ev;
        a.original_support=ids;a.original_values=v;
        if(energy_sparsify&&ids.size()>1) {
          // QP C-AMRS: smallest top-|v| support retaining 0.999999 of
          // the energy, followed by an ORIGINAL SOURCE Rayleigh check.
          // Small source blocks bound this work; no global Lanczos scan.
          std::vector<int> order(ids.size());std::iota(order.begin(),order.end(),0);
          std::sort(order.begin(),order.end(),[&](int i,int j){return std::abs(v[i])!=std::abs(v[j])?
            std::abs(v[i])>std::abs(v[j]):i<j;});
          const double total=dot(v,v);double kept=0.0;std::size_t count=0;
          while(count<order.size()&&kept<0.999999*total){const double z=v[order[count++]];kept+=z*z;}
          if(count&&count<ids.size()&&kept>0.0) {
            order.resize(count);std::sort(order.begin(),order.end());
            a.original_support.clear();a.original_values.clear();
            for(int j:order){a.original_support.push_back(ids[j]);a.original_values.push_back(v[j]/std::sqrt(kept));}
            auto value=[&](int j) {auto it=std::lower_bound(a.original_support.begin(),a.original_support.end(),j);
              return it!=a.original_support.end()&&*it==j?a.original_values[it-a.original_support.begin()]:0.0;};
            double ray=0.0;for(auto e:entries[component.first])ray+=(e.i==e.j?1.0:2.0)*q.scale*e.v*value(e.i)*value(e.j);
            if(!std::isfinite(ray)||ray==0.0||std::signbit(ray)!=std::signbit(ev)){++stats.atom_sparse_sign_rejections;return;}
            a.eigenvalue=ray;++stats.atom_sparsifications;
            stats.atom_dropped_coordinates+=(int)(ids.size()-count);
          }
        }
        pool.push_back(std::move(a));
      };
      if(m<=block_limit) {
        Vec matrix(m*m,0.0),eval;
        for(auto e:entries[component.first]) {
          int i=(int)(std::lower_bound(support.begin(),support.end(),e.i)-support.begin());
          int j=(int)(std::lower_bound(support.begin(),support.end(),e.j)-support.begin());
          matrix[i+j*m]=matrix[j+i*m]=q.scale*e.v;
        }
        if(m==1)push(matrix[0],support,{1.0});
        else {
          eigensystem(matrix,m,eval);++stats.atom_spectral_blocks;
          for(int j=0;j<m;++j)push(eval[j],support,Vec(matrix.begin()+j*m,matrix.begin()+(j+1)*m));
        }
      } else {
        // Bounded work for large components: diagonal-dominance correction
        // is a sum of coordinate rank-one atoms. Both signs are cached for
        // equality multipliers. This is conservative, not an eigenspectrum.
        ++stats.atom_large_blocks;
        std::map<int,double> diagonal,radius;
        for(auto e:entries[component.first]) {
          if(e.i==e.j)diagonal[e.i]+=q.scale*e.v;
          else {radius[e.i]+=std::abs(q.scale*e.v);radius[e.j]+=std::abs(q.scale*e.v);}
        }
        for(int j:support) {
          const double negative=std::max(0.0,radius[j]-diagonal[j]);
          if(negative>0.0)push(-negative,{j},{1.0});
          if(family==2) {const double positive=std::max(0.0,radius[j]+diagonal[j]);if(positive>0.0)push(positive,{j},{1.0});}
        }
      }
    }
    stats.atom_cached_candidates=(int)pool.size();
  }
  void project_one(NewtonAtom& a,const std::vector<int>& begin,
      const std::vector<int>& column,const Vec& value) {
    if(a.coordinate)return;
    ++projection_requests;
    if(a.projected_generation==tangent_generation)return;
    a.projection.update(a.original_support,a.original_values,begin,column,value,a.support,a.values);
    a.projected_generation=tangent_generation;
    product_ready=false;
  }

  void compile(NewtonAtom& a,std::vector<SymmetricTriplet>& kkt,Vec& base) {
    if(a.compiled)return;
    if(!slots_ready){for(int k=0;k<(int)kkt.size();++k)slots[upper(kkt[k].i,kkt[k].j)]=k;slots_ready=true;}
    for(int i=0;i<(int)a.support.size();++i)for(int j=i;j<(int)a.support.size();++j) {
      auto key=upper(a.support[i],a.support[j]);auto it=slots.find(key);int slot;
      if(it==slots.end()){slot=(int)kkt.size();slots[key]=slot;kkt.push_back({key.first,key.second,0.0});base.push_back(0.0);}
      else slot=it->second;
      a.entries.push_back({i,j,slot});
    }
    a.compiled=true;
  }
  template<class Weight,class Probe> bool activate(Weight weight,const std::vector<int>& begin,
      const std::vector<int>& column,const Vec& value,const std::vector<int>&,
      std::vector<SymmetricTriplet>& kkt,Vec& base,double margin,int retry,LazyNewtonStatistics& stats,Probe probe,bool rayleigh_gate,std::vector<AtomWitness>* guides=nullptr) {
    struct Candidate {int index;double target,score,ray;};std::vector<Candidate> candidates;
    double worst_before=0.0;
    for(int i=0;i<(int)pool.size();++i) {
      auto& a=pool[i];if(a.family==3)continue;
      const double negative=-weight(a.family,a.origin)*a.eigenvalue;
      if(!(negative>0.0))continue;
      project_one(a,begin,column,value);
      if(a.support.size()>support_limit)continue; // sparse completion below
      const double norm=dot(a.values,a.values);if(!(norm>0.0)||!std::isfinite(norm))continue;
      double ray=-negative*norm,target=1.05*negative+margin/std::max(1.0,norm);
      if(rayleigh_gate) {
        const auto measured=probe(a);++stats.atom_rayleigh_probes;ray=measured.first;
        if(!std::isfinite(ray)||ray>=-measured.second){++stats.atom_covered_rejections;continue;}
        target=a.weight+(1.05*(-ray)+margin)/norm;
      }
      worst_before=std::min(worst_before,ray);
      if(target>a.weight*(1.0+1e-12)) {
        double score=(target-a.weight)*norm;
        if(guides&&!guides->empty()) {
          score=0.0;for(const auto& w:*guides) {
            double overlap=0.0;for(std::size_t k=0;k<a.support.size();++k)overlap+=a.values[k]*w.direction[a.support[k]];
            score+=std::min(std::max(0.0,-w.curvature),(target-a.weight)*overlap*overlap);
          }
          score/=1.0+.01*a.support.size()*a.support.size();
        }
        candidates.push_back({i,target,score,ray});
      }
    }
    if(candidates.empty())return false;
    const int batch=std::min((int)candidates.size(),32<<std::min(7,retry));
    std::partial_sort(candidates.begin(),candidates.begin()+batch,candidates.end(),
      [](const Candidate& a,const Candidate& b){return a.score!=b.score?a.score>b.score:a.index<b.index;});
    const auto old_slots=kkt.size();int committed=0;
    for(int j=0;j<batch;++j) {
      auto& a=pool[candidates[j].index];
      if(rayleigh_gate) {
        const auto measured=probe(a);++stats.atom_rayleigh_probes;
        if(!std::isfinite(measured.first)||measured.first>=-measured.second){++stats.atom_covered_rejections;continue;}
        candidates[j].target=a.weight+(1.05*(-measured.first)+margin)/dot(a.values,a.values);
      }
      compile(a,kkt,base);
      const double delta=candidates[j].target-a.weight;
      if(!a.selected){a.selected=true;++stats.atom_creations;}
      else ++stats.atom_reactivations;
      if(a.weight==0.0)active_ids.push_back(candidates[j].index);
      a.weight=candidates[j].target;a.source_negative=-weight(a.family,a.origin)*a.eigenvalue;++committed;
      product_ready=false;
      if(guides)for(auto& w:*guides) {
        double overlap=0.0;for(std::size_t k=0;k<a.support.size();++k)overlap+=a.values[k]*w.direction[a.support[k]];
        w.curvature+=delta*overlap*overlap;
      }
      // Transactional admission: rejected candidates do not change R.
      // Committed candidates update the probe matrix for the rest of the batch.
      if(rayleigh_gate)for(auto e:a.entries)kkt[e.slot].value+=delta*a.values[e.i]*a.values[e.j];
    }
    if(guides&&committed&&!guides->empty()) {
      ++stats.atom_guided_batches;double before=0.0,after=0.0;
      for(auto w:*guides){before+=std::max(0.0,-w.initial);after+=std::max(0.0,-w.curvature);}
      stats.atom_last_witness_coverage=before>0?std::clamp(1.0-after/before,0.0,1.0):0.0;
    }
    if(rayleigh_gate&&committed&&worst_before<0.0) {
      double worst_after=0.0;
      for(const auto& c:candidates){const auto value=probe(pool[c.index]);++stats.atom_rayleigh_probes;worst_after=std::min(worst_after,value.first);}
      stats.atom_last_probe_coverage=std::clamp(1.0-worst_after/worst_before,0.0,1.0);
      ++stats.atom_coverage_observations;
    }
    if(kkt.size()!=old_slots)++stats.atom_layout_updates;
    update_peak(stats);
    stats.kkt_structural_nnz=(int)kkt.size();return committed>0;
  }
  void project(const std::vector<int>& begin,const std::vector<int>& column,const Vec& value,
      const std::vector<int>&,std::vector<SymmetricTriplet>&,Vec&,LazyNewtonStatistics& stats) {
    ++tangent_generation;product_ready=false;
    if(active_ids.empty())return;
    const auto timer=Clock::now();
    for(int id:active_ids)project_one(pool[id],begin,column,value);
    const double elapsed=seconds(timer);stats.atom_update_seconds+=elapsed;stats.atom_projection_seconds+=elapsed;
  }
  bool complete(const Vec& projected,std::vector<SymmetricTriplet>& kkt,const std::vector<int>& retained,
                double margin,Vec& base,LazyNewtonStatistics& stats) {
    const int n=(int)retained.size();Vec diagonal(n,0.0),radius(n,0.0);
    for(int k=0;k<(int)projected.size();++k) {
      int i=kkt[k].i,j=kkt[k].j;if(i>=n||j>=n)continue;
      if(i==j)diagonal[i]+=projected[k];
      else {radius[i]+=std::abs(projected[k]);radius[j]+=std::abs(projected[k]);}
    }
    bool added=false;
    for(int j=0;j<n;++j) {
      const double floor=std::max(margin,64*std::numeric_limits<double>::epsilon()*std::max({1.0,radius[j],std::abs(diagonal[j])}));
      const double target=std::max(0.0,radius[j]-diagonal[j]+floor);
      if(!(target>0.0))continue;
      NewtonAtom* atom=nullptr;
      auto it=completion.find(j);if(it!=completion.end())atom=&pool[it->second];
      if(!atom) {
        NewtonAtom a;a.family=3;a.origin=j;a.coordinate=true;a.support={j};a.values={1.0};
        completion[j]=(int)pool.size();pool.push_back(std::move(a));atom=&pool.back();
      }
      const double next=atom->weight>=target?atom->weight*10.0:target;
      if(!std::isfinite(next)||next>1e12)continue;
      compile(*atom,kkt,base);
      if(!atom->selected){atom->selected=true;++stats.atom_creations;}
      else ++stats.atom_reactivations;
      if(atom->weight==0.0)active_ids.push_back((int)(atom-pool.data()));
      atom->weight=next;added=true;product_ready=false;
    }
    if(added){++stats.atom_completion_batches;update_peak(stats);stats.kkt_structural_nnz=(int)kkt.size();}
    return added;
  }
  void assemble(std::vector<SymmetricTriplet>& kkt,const std::vector<int>& retained) {
    product_atoms.clear();product_indices.clear();product_values.clear();
    for(int id:active_ids) {
      const auto& a=pool[id];for(auto e:a.entries)
        kkt[e.slot].value+=a.weight*a.values[e.i]*a.values[e.j];
      const int begin=(int)product_values.size();
      for(std::size_t i=0;i<a.support.size();++i){product_indices.push_back(retained[a.support[i]]);product_values.push_back(a.values[i]);}
      product_atoms.push_back({begin,(int)product_values.size(),a.weight});
    }
    product_ready=true;
  }
  void product(const Vec& d,const std::vector<int>& retained,Vec& result) const {
    // Pack the same rank-one actions in active-set order at assembly time.
    // No matrix aggregation or summation reordering: source atoms remain
    // independently auditable, with a direct path if weights have changed.
    if(product_ready) {
      for(auto a:product_atoms) {
        double v=0.0;for(int i=a.begin;i<a.end;++i)v+=product_values[i]*d[product_indices[i]];
        v*=a.weight;for(int i=a.begin;i<a.end;++i)result[product_indices[i]]+=v*product_values[i];
      }
      return;
    }
    for(int id:active_ids) {const auto& a=pool[id];
      double v=0.0;for(std::size_t i=0;i<a.support.size();++i)v+=a.values[i]*d[retained[a.support[i]]];
      v*=a.weight;for(std::size_t i=0;i<a.support.size();++i)result[retained[a.support[i]]]+=v*a.values[i];
    }
  }
  double curvature(const Vec& d,const std::vector<int>& retained) const {
    double sum=0.0;
    for(int id:active_ids) {const auto& a=pool[id];
      double v=0.0;for(std::size_t i=0;i<a.support.size();++i)v+=a.values[i]*d[retained[a.support[i]]];
      sum+=a.weight*v*v;
    }
    return sum;
  }
  int active_count(int family=-1) const {if(family<0)return (int)active_ids.size();int n=0;for(int id:active_ids)if(pool[id].family==family)++n;return n;}
  double trace() const {double t=0.0;for(int id:active_ids){const auto& a=pool[id];t+=a.weight*dot(a.values,a.values);}return t;}
  void update_peak(LazyNewtonStatistics& stats) const {
    stats.atom_peak_active=std::max(stats.atom_peak_active,active_count());
    stats.atom_maximum_trace=std::max(stats.atom_maximum_trace,trace());
    stats.atom_cached_candidates=(int)pool.size();
  }
  void clear_weights(){for(int id:active_ids)pool[id].weight=0.0;active_ids.clear();product_ready=false;}
  void decay(double factor,bool clear){
    product_ready=false;
    if(clear){clear_weights();return;}
    std::size_t keep=0;
    for(int id:active_ids){auto& a=pool[id];a.weight*=factor;if(a.weight<1e-10)a.weight=0.0;else active_ids[keep++]=id;}
    active_ids.resize(keep);
  }
  template<class Weight> void retire_changed_sources(Weight weight,bool transport,LazyNewtonStatistics& stats) {
    product_ready=false;
    std::size_t keep=0;
    for(int id:active_ids) {
      auto& a=pool[id];
      if(a.family<3) {
        const double negative=-weight(a.family,a.origin)*a.eigenvalue;
        if(!(negative>0.0)){a.weight=0.0;continue;}
        if(transport&&a.source_negative>0.0&&negative<a.source_negative) {
          a.weight*=negative/a.source_negative;++stats.atom_multiplier_transports;
          if(a.weight<1e-10){a.weight=0.0;a.source_negative=negative;continue;}
        }
        a.source_negative=negative;
      }
      active_ids[keep++]=id;
    }
    active_ids.resize(keep);
  }
  int rank(int family,int dimension,bool& exact) const {
    std::vector<const NewtonAtom*> rows;
    for(int id:active_ids){const auto& a=pool[id];if((family<0||a.family==family)&&dot(a.values,a.values)>0.0)rows.push_back(&a);}
    int m=(int)rows.size();if(!m)return 0;
    if(m>rank_limit){exact=false;return std::min(m,dimension);}
    Vec gram(m*m,0.0),eval;
    for(int i=0;i<m;++i)for(int j=i;j<m;++j) {
      const auto& a=*rows[i];const auto& b=*rows[j];std::size_t k=0,l=0;double d=0.0;
      while(k<a.support.size()&&l<b.support.size()) {
        if(a.support[k]<b.support[l])++k;else if(b.support[l]<a.support[k])++l;
        else {d+=a.values[k]*b.values[l];++k;++l;}
      }
      gram[i+j*m]=gram[j+i*m]=d/std::sqrt(dot(a.values,a.values)*dot(b.values,b.values));
    }
    eigensystem(gram,m,eval);int r=0;for(double x:eval)if(x>1e-10*std::max(1.0,eval.back()))++r;return r;
  }
  void diagnostics(LazyNewtonStatistics& stats,int dimension) const {
    stats.objective_atoms=stats.inequality_atoms=stats.equality_atoms=stats.completion_atoms=0;
    for(const auto& a:pool)if(a.selected) {
      if(a.family==0)++stats.objective_atoms;else if(a.family==1)++stats.inequality_atoms;
      else if(a.family==2)++stats.equality_atoms;else ++stats.completion_atoms;
    }
    stats.atom_projection_builds=stats.atom_projection_updates=0;
    stats.atom_projection_requests=projection_requests;
    for(const auto& a:pool){stats.atom_projection_builds+=a.projection.ready;stats.atom_projection_updates+=a.projection.updates;}
    stats.atom_active=active_count();stats.atom_final_trace=trace();stats.atom_ranks_exact=true;
    stats.objective_correction_rank=rank(0,dimension,stats.atom_ranks_exact);
    stats.inequality_correction_rank=rank(1,dimension,stats.atom_ranks_exact);
    stats.equality_correction_rank=rank(2,dimension,stats.atom_ranks_exact);
    stats.atom_final_rank=rank(-1,dimension,stats.atom_ranks_exact);
  }
};
