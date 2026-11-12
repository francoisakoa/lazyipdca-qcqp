// Included inside the filter engine's anonymous namespace, after DcaModels.
// One affine-manifold DCA transaction; a failure commits no primal-dual state.
struct DcaLocalActive {
  Settings settings;
  std::unique_ptr<SymmetricFactorization> certificate, factor;
  detail::SparsePairElimination raw_elimination,elimination;
  const char* last_rejection="not-attempted";
  double last_candidate_kkt=INFINITY,last_backward_error=INFINITY;
  explicit DcaLocalActive(const Settings& s):settings(s) {
    if(s.dca_local_linear_backend!="same")settings.linear_backend=s.dca_local_linear_backend;
    settings.kkt_adaptive_regularization=true;
    settings.kkt_primal_regularization=settings.kkt_dual_regularization=0;
    settings.kkt_interleaved_equality_ordering=true;
    settings.kkt_primal_first_ordering=false;
    settings.kkt_preserve_structural_zeros=true;
    std::string message;
    certificate=make_symmetric_factorization(settings,message);
    factor=make_symmetric_factorization(settings,message);
  }
  bool attempt(Workspace& w,DcaModels& models,DcaPoint& outer,detail::OriginalDcaFilter& filter, std::vector<bool> excluded={}, int revision=0) {
    last_rejection="none";last_candidate_kkt=last_backward_error=INFINITY;
    auto reject=[&](const char* reason){
      last_rejection=reason;
      if(w.st.live_display_level>=4)
        std::cout<<"[DCA-LOCAL-REJECT] reason="<<reason<<" original_kkt="<<outer.kkt
                 <<" candidate_kkt="<<last_candidate_kkt<<" backward_error="<<last_backward_error<<"\n";
      return false;
    };
    // Eligibility is owned by the outer controller. The transaction always
    // certifies its actual manifold and audits the original problem below.
    if(!certificate||!factor)return reject("backend-unavailable");
    auto& stats=models.stats;
    ++stats.local_trials;
    detail::DcaTimedScope timer(stats.local_seconds);
    CostScope cost(w.costs,NewtonCost::CoreOther);
    // Model caches (including spectral directions) are untouched by this raw test.
    dca_disable_models(w);w.evaluate();w.update_geometry();
    std::vector<int> strong;std::vector<bool> selected(w.mi,false);
    for(int i=0;i<w.mi;++i)if(w.active[i]) {
      const double g=w.inequalities[i].value;
      // Relative multiplier/slack separation excludes weakly active rows.
      if((excluded.empty()||!excluded[i]) &&
         (!w.st.dca_local_weak_affine || w.s[i]<=0.01*w.z[i]) &&
         w.z[i]>1e-7 && std::abs(g)<=1e-3*w.inequalities[i].scale &&
         std::abs(g)<=0.01*w.z[i]) {strong.push_back(i);selected[i]=true;}
      else if(!w.st.dca_local_weak_affine&&!w.inequalities[i].hessian.empty()&&!models.inequality[i])return reject("missing-weak-majorant");
    }
    if(w.st.live_display_level>=4)
      std::cout<<"[DCA-LOCAL-MODEL] strong="<<strong.size()<<" remaining="
               <<std::count(w.active.begin(),w.active.end(),1)-strong.size()<<" revision="<<revision
               <<" weak_model="<<(w.st.dca_local_weak_affine?"affine-lagrangian":"quadratic-majorant")<<"\n";
    // In the affine-weak model the existing equality manifold already
    // defines the certificate space. With no equalities either, this is a
    // full-space positive-definiteness test; every inequality keeps its
    // affine barrier. The original G7 model still requires selected rows.
    if(strong.empty()&&!w.st.dca_local_weak_affine)return reject("empty-manifold");
    stats.local_strong=std::max(stats.local_strong,int(strong.size()));
    Vec partial(w.hpattern.size(),0),full;
    auto add_original=[&](Vec& h,const Quadratic& q,double weight){
      for(int k=0;k<(int)q.hessian.size();++k)h[q.hglobal[k]]+=weight*q.scale*q.hessian[k].v;
    };
    add_original(partial,w.objective,1);
    for(int e=0;e<w.me;++e)add_original(partial,w.equalities[e],w.lambda[e]);
    for(int i=0;i<w.mi;++i)if(w.active[i]&&(selected[i]||w.st.dca_local_weak_affine))
      add_original(partial,w.inequalities[i],w.z[i]);
    full=partial;
    // The optional affine weak model moves every retained inequality's
    // ORIGINAL curvature into the local Lagrangian. Weak rows keep their
    // barrier and an affine constraint, so adding their old majorant Hessian
    // here would count curvature twice. Otherwise retain the G7 majorants.
    for(int i=0;i<w.mi;++i)if(w.active[i]&&!selected[i]&&!w.st.dca_local_weak_affine) {
      const auto& q=w.inequalities[i];
      for(int k=0;k<(int)q.model_hessian.size();++k)
        full[q.model_hglobal[k]]+=w.z[i]*q.model_hessian[k].v;
    }
    auto project=[&](const Vec& h){
      std::vector<SymmetricTriplet> out;Vec values(w.kkt.size(),0);
      for(auto m:w.hmap)values[m.k]+=m.multiplier*h[m.h]*w.tvalue[m.a]*w.tvalue[m.b];
      for(int k=0;k<(int)w.kkt.size();++k)if(w.kkt[k].i<w.nr&&w.kkt[k].j<w.nr)
        out.push_back({w.kkt[k].i,w.kkt[k].j,values[k]});
      return out;
    };
    const int equalities=w.mr+strong.size(),base_dim=w.nr+equalities;
    auto append_normals=[&](std::vector<SymmetricTriplet>& matrix){
      for(int e=0;e<equalities;++e){
        const auto& g=e<w.mr?w.ar[e]:w.jr[strong[e-w.mr]];
        matrix.push_back({w.nr+e,w.nr+e,0});
        for(int k=0;k<(int)g.support.size();++k)matrix.push_back({g.support[k],w.nr+e,g.values[k]});
      }
    };
    // Exact proportional duplicates (including zero rows) cannot establish a
    // full-rank active manifold, even if a roundoff-sized LDL pivot has a sign.
    std::set<std::vector<std::pair<int,double>>> independent_rows;
    for(int e=0;e<equalities;++e){
      const auto& g=e<w.mr?w.ar[e]:w.jr[strong[e-w.mr]];
      double pivot=0;for(double v:g.values)if(v!=0){pivot=v;break;}
      if(pivot==0)return reject("zero-manifold-row");
      std::vector<std::pair<int,double>> row;
      for(int k=0;k<(int)g.values.size();++k)if(g.values[k]!=0)row.push_back({g.support[k],g.values[k]/pivot});
      if(!independent_rows.insert(std::move(row)).second)return reject("dependent-manifold-row");
    }
    auto raw=project(partial);
    for(int j=0;j<w.nr;++j)raw.push_back({j,j,-1e-10});
    append_normals(raw);
    // Normalize only the certificate by a positive diagonal congruence.
    // The primal scale is common; each equality normal then has unit max
    // norm. Scaling the complete shifted matrix preserves both inertia and
    // the original -1e-10 margin. The Newton model/coordinates are unchanged.
    double hessian_norm=0;
    for(auto t:raw)if(t.i<w.nr&&t.j<w.nr)
      hessian_norm=std::max(hessian_norm,std::abs(t.value));
    const double primal_scale=1/std::sqrt(std::max(1e-12,hessian_norm));
    Vec certificate_scale(base_dim,primal_scale);
    for(int e=0;e<equalities;++e){
      const auto& g=e<w.mr?w.ar[e]:w.jr[strong[e-w.mr]];
      const double normal_norm=inf_norm(g.values);
      certificate_scale[w.nr+e]=1/(primal_scale*normal_norm);
      if(!(certificate_scale[w.nr+e]>0)||!std::isfinite(certificate_scale[w.nr+e]))
        return reject("certificate-scaling");
    }
    std::string message;
    // Test the uncorrected partial Lagrangian first on every transaction.
    // The affine-weak experiment can convexify its complete local Lagrangian
    // on a bounded larger ladder. Every level still needs the same actual
    // positive-inertia certificate; the G7 partial model keeps its old ladder.
    double local_rho=0;bool certified=false;
    const int certificate_trials=w.st.dca_local_weak_affine?8:4;
    for(int trial=0;trial<certificate_trials&&!certified;++trial){
      auto shifted=raw;
      if(local_rho>0)for(int j=0;j<w.nr;++j)shifted.push_back({j,j,local_rho});
      Vec shifted_scale=certificate_scale;
      const double rho_scale=std::sqrt(std::max(1e-12,hessian_norm)/std::max({1e-12,hessian_norm,local_rho}));
      for(int j=0;j<base_dim;++j)shifted_scale[j]*=j<w.nr?rho_scale:1/rho_scale;
      for(auto& t:shifted){
        const double scaled=static_cast<long double>(t.value)*shifted_scale[t.i]*shifted_scale[t.j];
        if(!std::isfinite(scaled)||(t.value!=0&&scaled==0))return reject("certificate-scaling");
        t.value=scaled;
      }
      auto raw_reduced=w.st.dca_pair_elimination?raw_elimination.reduce(base_dim,w.nr,shifted):shifted;
      ++stats.local_certificates;
      certified=certificate->factor(w.st.dca_pair_elimination?raw_elimination.dimension():base_dim,
                                    w.st.dca_pair_elimination?raw_elimination.reduced_primal:w.nr,
                                    raw_reduced,message);
      if(!certified)local_rho=local_rho==0?1e-6:100*local_rho;
    }
    if(!certified){
      if(w.st.live_display_level>=4)std::cout<<"[DCA-LOCAL-CERTIFICATE] strong="<<strong.size()<<" "<<message<<"\n";
      return reject("curvature-certificate");
    }
    for(int j:w.retained){int k=models.slots.at({j,j});partial[k]+=local_rho;full[k]+=local_rho;}
    auto matrix=project(full);append_normals(matrix);
    std::vector<int> weak_border(w.mi,-1);int dimension=base_dim;
    for(int i=0;i<w.mi;++i)if(w.active[i]&&!selected[i]) {
      const auto& g=w.jr[i];
      if(g.dense>=0 || w.z[i]/w.s[i]*dot(g.values,g.values)>1e4){
        int row=dimension++;weak_border[i]=row;
        matrix.push_back({row,row,-w.s[i]/w.z[i]});
        for(int k=0;k<(int)g.support.size();++k)matrix.push_back({g.support[k],row,g.values[k]});
      }else for(int a=0;a<(int)g.support.size();++a)for(int b=a;b<(int)g.support.size();++b)
        matrix.push_back({g.support[a],g.support[b],w.z[i]/w.s[i]*g.values[a]*g.values[b]});
    }
    const bool eliminate=w.st.dca_pair_elimination;
    auto reduced=eliminate?elimination.reduce(dimension,w.nr,matrix):matrix;
    stats.local_pairs+=elimination.pairs;stats.local_fill+=elimination.fill;
    if(!factor->factor(eliminate?elimination.dimension():dimension,
                       eliminate?elimination.reduced_primal:w.nr,reduced,message))return reject("newton-factorization");
    auto product=[&](const Vec& x){Vec y(w.n,0);for(int k=0;k<(int)full.size();++k){auto [a,b]=w.hpattern[k];y[a]+=full[k]*x[b];if(a!=b)y[b]+=full[k]*x[a];}return y;};
    auto solve=[&](const Vec& dual,const Vec& eq,const Vec& primal,const Vec& comp,Direction& d){
      d.x.assign(w.n,0);d.lambda.assign(w.me,0);d.s.assign(w.mi,0);d.z.assign(w.mi,0);
      Vec offset(w.n,0);
      for(int e=0;e<w.me;++e)if(w.pivot_variable[e]>=0)offset[w.pivot_variable[e]]=-eq[e]/w.pivot_coefficient[e];
      Vec hd=product(offset),rhs(dimension,0);
      for(int j=0;j<w.n;++j)for(int t=w.tbegin[j];t<w.tbegin[j+1];++t)
        rhs[w.tcolumn[t]]-=w.tvalue[t]*(hd[j]+dual[j]);
      for(int e=0;e<w.mr;++e)rhs[w.nr+e]=-eq[w.eq_retained[e]]-w.equalities[w.eq_retained[e]].directional(offset);
      for(int e=0;e<(int)strong.size();++e){int i=strong[e];rhs[w.nr+w.mr+e]=-primal[i]-w.inequalities[i].directional(offset);}
      for(int i=0;i<w.mi;++i)if(w.active[i]&&!selected[i]){
        const double r=primal[i]+w.inequalities[i].directional(offset)-comp[i]/w.z[i];
        if(weak_border[i]>=0)rhs[weak_border[i]]=-r;
        else for(int k=0;k<(int)w.jr[i].support.size();++k)rhs[w.jr[i].support[k]]-=w.jr[i].values[k]*w.z[i]/w.s[i]*r;
      }
      const Vec original_rhs=rhs;
      if(!(eliminate?elimination.solve(*factor,rhs,message):factor->solve(rhs,message))||!finite(rhs))return false;
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
        if(!std::isfinite(error))return false;
        if(error<best_error){best_error=error;best_rhs=rhs;}
        else break;
        if(error<=1e-12 || refinement==5)break;
        if(!(eliminate?elimination.solve(*factor,residual,message):factor->solve(residual,message)))return false;
        axpy(1,residual,rhs);
      }
      rhs=std::move(best_rhs);last_backward_error=best_error;
      if(best_error>w.st.kkt_backward_error_reject)return false;
      for(int j=0;j<w.n;++j){d.x[j]=offset[j];for(int t=w.tbegin[j];t<w.tbegin[j+1];++t)d.x[j]+=w.tvalue[t]*rhs[w.tcolumn[t]];}
      for(int e=0;e<w.mr;++e)d.lambda[w.eq_retained[e]]=rhs[w.nr+e];
      for(int e=0;e<(int)strong.size();++e)d.z[strong[e]]=rhs[w.nr+w.mr+e];
      for(int i=0;i<w.mi;++i)if(w.active[i]&&!selected[i]){
        d.s[i]=-primal[i]-w.inequalities[i].directional(d.x);
        d.z[i]=weak_border[i]>=0?rhs[weak_border[i]]:(-comp[i]-w.z[i]*d.s[i])/w.s[i];
      }
      hd=product(d.x);for(int j=0;j<w.n;++j)hd[j]+=dual[j];
      for(int i=0;i<w.mi;++i)if(w.active[i])w.inequalities[i].add_gradient(d.z[i],hd);
      for(int e:w.eq_retained)w.equalities[e].add_gradient(d.lambda[e],hd);
      for(int e=0;e<w.me;++e)if(w.pivot_variable[e]>=0)d.lambda[e]=-hd[w.pivot_variable[e]]/w.pivot_coefficient[e];
      for(int e=0;e<w.me;++e)if(w.pivot_variable[e]>=0)w.equalities[e].add_gradient(d.lambda[e],hd);
      double expanded=inf_norm(hd),scale=std::max(1.0,inf_norm(dual));
      for(int e=0;e<w.me;++e){expanded=std::max(expanded,std::abs(eq[e]+w.equalities[e].directional(d.x)));scale=std::max(scale,std::abs(eq[e]));}
      for(int i=0;i<w.mi;++i)if(w.active[i]){
        expanded=std::max(expanded,std::abs(primal[i]+w.inequalities[i].directional(d.x)+d.s[i]));
        if(!selected[i])expanded=std::max(expanded,std::abs(comp[i]+w.z[i]*d.s[i]+w.s[i]*d.z[i]));
        scale=std::max({scale,std::abs(primal[i]),std::abs(comp[i])});
      }
      d.residual=expanded/scale;
      last_backward_error=std::max(last_backward_error,d.residual);
      stats.local_residual=std::max(stats.local_residual,d.residual);
      return finite(d.x)&&finite(d.lambda)&&finite(d.z)&&std::isfinite(d.residual)&&d.residual<=w.st.kkt_backward_error_reject;
    };
    Vec primal=w.rp,comp(w.mi,0);
    for(int i=0;i<w.mi;++i)if(w.active[i]){
      if(selected[i])primal[i]=w.inequalities[i].value;
      else comp[i]=w.s[i]*w.z[i];
    }
    Direction d;if(!solve(w.rd,w.h,primal,comp,d)){
      if(w.st.live_display_level>=4)std::cout<<"[DCA-LOCAL-DIRECTION] "<<message<<" residual="<<d.residual<<"\n";
      return reject("newton-direction");
    }
    if(w.st.dca_local_weak_affine){
      // Predictor/corrector applies only to the remaining affine barrier
      // rows. Strong rows have no slack complementarity equation, but their
      // multiplier positivity still limits the affine dual prediction.
      const Direction affine=d;
      double affine_primal=1,affine_dual=1,mu=0;int weak_count=0;
      for(int i=0;i<w.mi;++i)if(w.active[i]){
        if(affine.z[i]<0)affine_dual=std::min(affine_dual,-w.z[i]/affine.z[i]);
        if(!selected[i]){
          if(affine.s[i]<0)affine_primal=std::min(affine_primal,-w.s[i]/affine.s[i]);
          mu+=w.s[i]*w.z[i];++weak_count;
        }
      }
      if(weak_count){
        mu/=weak_count;double affine_mu=0;
        for(int i=0;i<w.mi;++i)if(w.active[i]&&!selected[i])
          affine_mu+=(w.s[i]+affine_primal*affine.s[i])*(w.z[i]+affine_dual*affine.z[i]);
        affine_mu/=weak_count;
        const double sigma=std::clamp(std::pow(std::max(0.0,affine_mu)/std::max(mu,1e-300),3),1e-8,0.95);
        const double target=std::max(0.001*w.objective.scale*
            std::min(w.st.complementarity_tolerance,w.st.stationarity_tolerance),sigma*mu);
        for(int i=0;i<w.mi;++i)if(w.active[i]&&!selected[i])
          comp[i]=w.s[i]*w.z[i]-target+affine.s[i]*affine.z[i];
        ++stats.corrector_steps;stats.last_sigma=sigma;
        if(w.st.live_display_level>=4)
          std::cout<<"[DCA-LOCAL-CORRECTOR] weak="<<weak_count<<" mu="<<mu
                   <<" affine_mu="<<affine_mu<<" sigma="<<sigma<<" target="<<target<<"\n";
        // Reuse the unchanged factor and audit the corrected original bordered
        // equations, exactly as for the predictor. No second factorization.
        if(!solve(w.rd,w.h,primal,comp,d))return reject("newton-corrector");
      }
    }
    double alpha=1;
    for(int i=0;i<w.mi;++i)if(w.active[i]){
      if(d.z[i]<0)alpha=std::min(alpha,-0.995*w.z[i]/d.z[i]);
      if(!selected[i]&&d.s[i]<0)alpha=std::min(alpha,-0.995*w.s[i]/d.s[i]);
    }
    if(alpha<0.2 && revision<2){
      if(excluded.empty())excluded.assign(w.mi,false);
      bool changed=false;
      for(int i:strong)if(w.z[i]+d.z[i]<0.1*w.z[i]){excluded[i]=true;changed=true;}
      if(changed){
        const double before=stats.local_seconds;
        const bool accepted=attempt(w,models,outer,filter,std::move(excluded),revision+1);
        stats.local_seconds=before; // outer timer already includes the recursive transaction
        return accepted;
      }
    }
    if(alpha<1e-3)return reject("multiplier-or-slack-step");
    for(int ls=0;ls<8;++ls,alpha*=0.5){
      Vec x=w.x,z=w.z,lambda=w.lambda,s=w.s;
      axpy(alpha,d.x,x);axpy(alpha,d.z,z);axpy(alpha,d.lambda,lambda);
      for(int i=0;i<w.mi;++i)if(w.active[i])s[i]=selected[i]?0:w.s[i]+alpha*d.s[i];
      auto accept=[&](){
        for(int i=0;i<w.mi;++i)if(w.active[i]&&(!(z[i]>0)||(!selected[i]&&!(s[i]>0)))){last_rejection="nonpositive-local-state";return false;}
        auto candidate=dca_original(w,x,lambda,z,&w.lambda,&w.z);
        last_candidate_kkt=std::min(last_candidate_kkt,candidate.kkt);
        Vec dx=x;axpy(-1,w.x,dx);
        double quadratic=0;for(int k=0;k<(int)partial.size();++k){auto [a,b]=w.hpattern[k];quadratic+=(a==b?0.5:1)*partial[k]*dx[a]*dx[b];}
        const double predicted=-w.objective.directional(dx)-quadratic;
        if(candidate.kkt>0.9*outer.kkt){last_rejection="insufficient-kkt-contraction";return false;}
        if(!filter.acceptable(outer,candidate,predicted,w.st.feasibility_tolerance)){
          last_rejection="original-filter";return false;
        }
        // Guard every inactive original row; a changed working set is rebuilt by general DCA.
        for(int i=0;i<w.mi;++i)if(!w.active[i]&&w.inequalities[i].original_at(x)>w.st.feasibility_tolerance*w.inequalities[i].scale){last_rejection="inactive-original-row";return false;}
        if(candidate.theta>w.st.feasibility_tolerance)filter.insert(outer);
        w.x=x;w.lambda=lambda;w.z=z;
        if(w.st.live_display_level>=4)std::cout<<"[DCA-LOCAL-ACCEPT] strong="<<strong.size()<<" rho="<<local_rho<<" alpha="<<alpha<<" kkt="<<candidate.kkt<<"\n";
        // Return original positive slack coordinates to the general controller.
        for(int i=0;i<w.mi;++i){
          if(!w.active[i]||selected[i])w.s[i]=std::max(1e-14,-w.inequalities[i].original_at(x));
          else {
            const auto& q=w.inequalities[i];double model=q.value+q.directional(dx);
            if(!w.st.dca_local_weak_affine)
              for(auto t:q.model_hessian)model+=(t.i==t.j?0.5:1)*t.v*dx[t.i]*dx[t.j];
            // An affine local model is not an original-constraint majorant:
            // its displacement has either sign and must be transported with
            // that sign. The floor preserves admissible barrier coordinates.
            const double displacement=model-q.original_at(x);
            w.s[i]=w.st.dca_local_weak_affine?std::max(1e-14,s[i]+displacement):
                       s[i]+std::max(0.0,displacement);
          }
        }
        outer=candidate;++stats.local_acceptances;last_rejection="none";return true;
      };
      if(accept())return true;
      if(ls==0){
        Vec eq(w.me),p(w.mi,0),c(w.mi,0),dual(w.n,0);
        for(int e=0;e<w.me;++e)eq[e]=w.equalities[e].original_at(x);
        for(int i:strong)p[i]=w.inequalities[i].original_at(x);
        Direction soc;
        if(solve(dual,eq,p,c,soc)&&inf_norm(soc.x)<=std::max(1e-8,0.5*alpha*inf_norm(d.x))){
          axpy(1,soc.x,x);axpy(1,soc.lambda,lambda);axpy(1,soc.z,z);axpy(1,soc.s,s);
          if(accept()){++stats.local_soc;return true;}
        }
      }
    }
    return reject(last_rejection);
  }
};
