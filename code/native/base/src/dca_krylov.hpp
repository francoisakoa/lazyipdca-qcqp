// Included in the shared Workspace translation unit. The operator below uses
// ORIGINAL quadratic derivatives. The existing convex-majorant factor is only
// a preconditioner; neither its matrix nor the public iterate is changed by a
// rejected transaction. Slacks are transported to original coordinates.
struct DcaKrylov {
  int next_iteration = 0;
  double accepted_dual_alpha = 0;
  int accepted_line_steps = 0;
  // Last attempted solve, including failures: a rejection reason alone does
  // not distinguish an exhausted Krylov budget from a reconstruction defect.
  int last_dimension=0,last_lifted_rows=0,last_iterations=0;
  double last_residual=INFINITY,last_dual=INFINITY,last_equality=INFINITY;
  double last_primal=INFINITY,last_complementarity=INFINITY,last_reduced_residual=INFINITY;
  std::string last_linear_message;

  struct Image {Vec x,lambda,z,s;DcaPoint point;} proposal;
  struct Row {
    const Quadratic* q;
    Vec gradient;
    Row(const Quadratic& function,const Vec& x):q(&function),gradient(q->support.size(),0) {
      q->note(&NewtonCalls::gradients);
      for(std::size_t k=0;k<q->linear.size();++k)gradient[q->linear_pos[k]]+=q->linear[k].second;
      for(std::size_t k=0;k<q->hessian.size();++k){auto e=q->hessian[k];
        gradient[q->hi[k]]+=e.v*x[e.j];if(e.i!=e.j)gradient[q->hj[k]]+=e.v*x[e.i];}
      for(double& a:gradient)a*=q->scale;
    }
    double product(const Vec& x)const {
      double v=0;for(std::size_t k=0;k<gradient.size();++k)v+=gradient[k]*x[q->support[k]];return v;
    }
    void transpose(double a,Vec& x)const {
      for(std::size_t k=0;k<gradient.size();++k)x[q->support[k]]+=a*gradient[k];
    }
  };
  bool attempt(Workspace& w,DcaFilterStatistics& stats,DcaPoint& outer,
               detail::OriginalDcaFilter& filter,const Direction& model_affine,
               const Direction& model_corrector,double model_target,
               int iteration,Clock::time_point start,double& accepted_alpha,
               const DcaPoint& reference,bool forced=false, DcaAggregate* aggregate_factor=nullptr) {
    if((!w.st.dca_krylov&&!forced)||w.st.dca_krylov_max_iterations==0)return false;
    // A forced retry is supplied with a newly rebuilt preconditioner. The
    // optional accelerator cooldown must not hide it at the same outer index.
    if(iteration<next_iteration && !forced){++stats.krylov_skips;return false;}
    if(aggregate_factor){if(!aggregate_factor->root_factor_compatible(w))return false;}
    else if(!w.current_factor)return false;
    ++stats.krylov_trials;
    last_iterations=last_lifted_rows=last_dimension=0;
    last_residual=last_dual=last_equality=last_primal=last_complementarity=last_reduced_residual=INFINITY;
    last_linear_message="not-run";

    detail::DcaTimedScope timer(stats.krylov_seconds);
    CostScope scope(w.costs,NewtonCost::CoreOther);
    const std::string saved_message=w.message;
    auto record_linear=[&](){
      stats.krylov_last_dimension=last_dimension;stats.krylov_last_lifted_rows=last_lifted_rows;
      stats.krylov_last_iterations=last_iterations;stats.krylov_last_original_residual=last_residual;
      stats.krylov_last_reduced_residual=last_reduced_residual;stats.krylov_last_dual_residual=last_dual;
      stats.krylov_last_equality_residual=last_equality;stats.krylov_last_primal_residual=last_primal;
      stats.krylov_last_complementarity_residual=last_complementarity;
      stats.krylov_last_linear_message=last_linear_message;
    };
    auto reject=[&](const std::string& reason,bool linear=false){
      record_linear();
      stats.krylov_last_rejection=reason;
      if(linear)++stats.krylov_linear_rejections;else ++stats.krylov_filter_rejections;
      next_iteration=iteration+1+w.st.dca_krylov_failure_cooldown;
      w.message=saved_message;
      if(w.st.live_display_level>=4)std::cout<<"[DCA-KRYLOV-REJECT] iter="<<iteration<<" reason="<<reason<<"\n";
      return false;
    };
    try {
      auto deadline=[&](){if(seconds(start)>=w.st.time_limit_seconds)throw std::runtime_error("time-budget");};
      std::vector<int> active;for(int i=0;i<w.mi;++i)if(w.active[i])active.push_back(i);
      const int ni=active.size();
      std::vector<Row> a,j;a.reserve(w.me);j.reserve(ni);
      Vec eh(w.me),primal(ni),slack(ni),comp(ni),dual(w.n,0);
      w.objective.original_gradient_at(w.x,1,dual);
      for(int e=0;e<w.me;++e){a.emplace_back(w.equalities[e],w.x);eh[e]=w.equalities[e].original_at(w.x);a.back().transpose(w.lambda[e],dual);}
      double mu=0;
      for(int k=0;k<ni;++k){int i=active[k];const auto& q=w.inequalities[i];
        j.emplace_back(q,w.x);const double value=q.original_at(w.x);
        slack[k]=w.s[i]+std::max(0.0,q.at(w.x)-value);
        if(!(slack[k]>0)||!(w.z[i]>0))return reject("nonpositive-anchor");
        primal[k]=value+slack[k];comp[k]=slack[k]*w.z[i];mu+=comp[k];
        j.back().transpose(w.z[i],dual);
      }
      mu=ni?mu/ni:0;
      // One sparse Hessian assembly, retaining the original objective and ALL
      // multiplier-weighted constraint curvature, including equality terms.
      Vec hv(w.hpattern.size(),0);
      auto append=[&](const Quadratic& q,double weight){
        for(std::size_t k=0;k<q.hessian.size();++k)hv[q.hglobal[k]]+=weight*q.scale*q.hessian[k].v;
      };
      append(w.objective,1);
      for(int e=0;e<w.me;++e)append(w.equalities[e],w.lambda[e]);
      for(int i:active)append(w.inequalities[i],w.z[i]);
      // Condense ORIGINAL equations on the SAME retained coordinate indices
      // as the DCA factor. Nonlinear eliminated equalities require their own
      // current T, never the frozen-majorant tangent.
      Vec tv=w.tvalue;
      for(int p=0;p<w.n;++p)if(w.pivot_equation[p]>=0)
        for(int t=w.tbegin[p];t<w.tbegin[p+1];++t){auto [e,k]=w.tgradient[t];tv[t]=-a[e].gradient[k]/w.pivot_coefficient[e];}
      const int base_reduced=w.nr+w.mr+w.ndense;
      int reduced=base_reduced;
      std::vector<int> border(ni,-1),lifted;
      std::vector<SymmetricTriplet> extra;

      const std::size_t entries=w.has_model_borders?w.atom_border_begin:w.kkt.size();
      Vec kv(entries,0);
      for(auto m:w.hmap)kv[m.k]+=m.multiplier*hv[m.h]*tv[m.a]*tv[m.b];
      std::vector<Vec> gj(ni),ga(w.mr);
      for(int k=0;k<ni;++k){int i=active[k];const auto& g=w.jr[i];gj[k].assign(g.support.size(),0);
        for(auto m:g.map)gj[k][m.output]+=j[k].gradient[m.gradient]*tv[m.t];
      }
      // Inactive dense borders remain independent -1 pivots, as in DCA.
      for(int i=0;i<w.mi;++i)if(w.jr[i].dense>=0)kv[w.dense_diagonal[w.jr[i].dense]]=-1;
      for(int k=0;k<ni;++k){int i=active[k];const auto& g=w.jr[i];
        long double gram=0;
        for(double value:gj[k])gram+=(long double)value*value;
        gram*=((long double)w.z[i]/slack[k]);
        if(g.dense>=0){
          border[k]=w.nr+w.mr+g.dense;
          for(std::size_t p=0;p<g.support.size();++p)kv[g.border[p]]=gj[k][p];
          kv[w.dense_diagonal[g.dense]]=-slack[k]/w.z[i];
        }else if(gram>1e4){
          // The aggregate route already lifts these rows. Re-forming their
          // z/s Gram here can erase the ORIGINAL tangent curvature in double
          // precision. Keep the row in the target operator, irrespective of
          // its structural support length. The old factor is ONLY a preconditioner.
          border[k]=reduced++;lifted.push_back(k);
          extra.push_back({border[k],border[k],-slack[k]/w.z[i]});
          for(std::size_t p=0;p<g.support.size();++p)
            extra.push_back({g.support[p],border[k],gj[k][p]});
        }else for(auto t:g.gram)kv[t.k]+=(w.z[i]/slack[k])*gj[k][t.a]*gj[k][t.b];
      }
      for(int e=0;e<w.mr;++e){const auto& g=w.ar[e];ga[e].assign(g.support.size(),0);
        for(auto m:g.map)ga[e][m.output]+=a[w.eq_retained[e]].gradient[m.gradient]*tv[m.t];
        for(std::size_t p=0;p<g.support.size();++p)kv[g.border[p]]=ga[e][p];
      }
      last_dimension=reduced;last_lifted_rows=static_cast<int>(lifted.size());
      auto hproduct=[&](const Vec& v){
        Vec out(w.n,0);for(std::size_t k=0;k<hv.size();++k){auto [p,q]=w.hpattern[k];out[p]+=hv[k]*v[q];if(p!=q)out[q]+=hv[k]*v[p];}return out;
      };
      auto apply=[&](const Vec& v,Vec& out){
        deadline();++stats.krylov_operator_products;
        std::vector<long double> accumulated(reduced,0);
        for(std::size_t k=0;k<entries;++k){const auto& e=w.kkt[k];
          accumulated[e.i]+=(long double)kv[k]*v[e.j];
          if(e.i!=e.j)accumulated[e.j]+=(long double)kv[k]*v[e.i];}
        for(const auto& e:extra){accumulated[e.i]+=(long double)e.value*v[e.j];
          if(e.i!=e.j)accumulated[e.j]+=(long double)e.value*v[e.i];}
        out.assign(accumulated.begin(),accumulated.end());
      };
      auto precondition=[&](const Vec& b,Vec& out,std::string& message){
        deadline();++stats.krylov_preconditioner_solves;++w.stats.linear_solves;
        if(aggregate_factor){
          CostScope backend(w.costs,NewtonCost::LinearBackend);
          return aggregate_factor->root_precondition(w,active,border,b,out,message);
        }
        // Lift the inverse of the CURRENT certified factor to the added
        // border coordinates by Schur substitution. No new factorization,
        // no use of the aggregate factor that failed its inertia certificate.
        out.assign(w.dim,0);std::copy_n(b.begin(),base_reduced,out.begin());
        std::vector<long double> lifted_rhs(out.begin(),out.end());
        for(int k:lifted){const int i=active[k];const auto& g=w.jr[i];
          const long double weight=(long double)w.z[i]/slack[k]*b[border[k]];
          for(std::size_t p=0;p<g.support.size();++p)lifted_rhs[g.support[p]]+=weight*gj[k][p];}
        out.assign(lifted_rhs.begin(),lifted_rhs.end());
        if(!finite(out)){message="nonfinite-lifted-preconditioner-rhs";return false;}
        if(w.current_factor==w.recovery.get()){
          const auto& row=w.ar[w.recovery_equation];
          const double v=out[w.nr+w.recovery_equation]/w.recovery_scale*w.recovery_shift;
          for(std::size_t k=0;k<row.support.size();++k)out[row.support[k]]+=v*w.recovery_normal[k];
        }
        // Border RHSs for the convexifying atoms are zero. The existing
        // factor implicitly applies their Schur complement. No new factor.
        bool ok;
        {CostScope backend(w.costs,NewtonCost::LinearBackend);ok=w.current_factor->solve(out,message);}
        if(!ok||!finite(out))return false;
        // Strip any convexifying-atom coordinates before inserting the new
        // original inequality borders: the two layouts need not have the same size.
        out.resize(base_reduced);out.resize(reduced,0);
        for(int k:lifted){const int i=active[k];const auto& g=w.jr[i];
          long double value=-b[border[k]];
          for(std::size_t p=0;p<g.support.size();++p)value+=(long double)gj[k][p]*out[g.support[p]];
          out[border[k]]=(double)(value*((long double)w.z[i]/slack[k]));}
        return finite(out);
      };
      double largest_residual=0;
      auto solve=[&](const Vec& rc,const Direction& seed,Direction& d,
                     const Vec* rhs_dual=nullptr,const Vec* rhs_eq=nullptr,
                     const Vec* rhs_primal=nullptr,const char* label="main"){
        const Vec& input_d=rhs_dual?*rhs_dual:dual;
        const Vec& input_h=rhs_eq?*rhs_eq:eh;
        const Vec& input_p=rhs_primal?*rhs_primal:primal;
        ++stats.krylov_solves;
        Vec residual_d=input_d,residual_h=input_h,residual_p=input_p,residual_c=rc;
        int used=0;
        d.x.assign(w.n,0);d.lambda.assign(w.me,0);d.s.assign(w.mi,0);d.z.assign(w.mi,0);
        double scale=std::max(1.0,inf_norm(input_d));
        for(double v:input_h)scale=std::max(scale,std::abs(v));
        for(int k=0;k<ni;++k)scale=std::max({scale,std::abs(input_p[k]),std::abs(rc[k])});
        for(int refinement=0;refinement<4;++refinement){
          Vec normal(w.n,0);for(int e=0;e<w.me;++e)if(w.pivot_variable[e]>=0)normal[w.pivot_variable[e]]=-residual_h[e]/w.pivot_coefficient[e];
          Vec b=hproduct(normal);axpy(1,residual_d,b);
          Vec rhs(reduced,0),v(reduced,0),applied,defect,correction;
          for(int p=0;p<w.n;++p)for(int t=w.tbegin[p];t<w.tbegin[p+1];++t)rhs[w.tcolumn[t]]-=tv[t]*b[p];
          if(refinement==0)for(int p=0;p<w.nr;++p)v[p]=seed.x[w.retained[p]];
          for(int e=0;e<w.mr;++e){int original=w.eq_retained[e];rhs[w.nr+e]=-residual_h[original]-a[original].product(normal);if(refinement==0)v[w.nr+e]=seed.lambda[original];}
          for(int k=0;k<ni;++k){int i=active[k];const auto& g=w.jr[i];double jo=j[k].product(normal);
            if(border[k]>=0){rhs[border[k]]=residual_c[k]/w.z[i]-residual_p[k]-jo;if(refinement==0)v[border[k]]=seed.z[i];}
            else {double q=(w.z[i]*(residual_p[k]+jo)-residual_c[k])/slack[k];
              for(std::size_t p=0;p<g.support.size();++p)rhs[g.support[p]]-=gj[k][p]*q;}
          }
          apply(v,applied);defect=rhs;axpy(-1,applied,defect);
          const double tolerance=std::max(1e-15,0.001*w.st.kkt_backward_error_reject*scale);
          const int budget=w.st.dca_krylov_max_iterations-used;
          if(budget<=0)break;
          auto gmres=original_kkt_right_preconditioned_gmres(apply,precondition,defect,correction,budget,tolerance);
          used+=gmres.iterations;stats.krylov_iterations+=gmres.iterations;axpy(1,correction,v);
          last_iterations=used;last_reduced_residual=gmres.residual_2;
          last_linear_message=gmres.message;
          if(!finite(v))return false;
          Direction step;step.x=normal;step.lambda.assign(w.me,0);step.s.assign(w.mi,0);step.z.assign(w.mi,0);
          for(int p=0;p<w.n;++p)for(int t=w.tbegin[p];t<w.tbegin[p+1];++t)step.x[p]+=tv[t]*v[w.tcolumn[t]];
          for(int e=0;e<w.mr;++e)step.lambda[w.eq_retained[e]]=v[w.nr+e];
          for(int k=0;k<ni;++k){int i=active[k];step.s[i]=-residual_p[k]-j[k].product(step.x);
            step.z[i]=border[k]>=0?v[border[k]]:(-residual_c[k]-w.z[i]*step.s[i])/slack[k];}
          Vec stationarity=hproduct(step.x);axpy(1,residual_d,stationarity);
          for(int k=0;k<ni;++k)j[k].transpose(step.z[active[k]],stationarity);
          for(int e:w.eq_retained)a[e].transpose(step.lambda[e],stationarity);
          for(int e=0;e<w.me;++e)if(w.pivot_variable[e]>=0)step.lambda[e]=-stationarity[w.pivot_variable[e]]/w.pivot_coefficient[e];
          axpy(1,step.x,d.x);axpy(1,step.lambda,d.lambda);axpy(1,step.s,d.s);axpy(1,step.z,d.z);
          // Compute the full Newton residual after substitution in extended
          // precision. Refinement solves these original defects; it does not
          // repeatedly subtract two ill-conditioned condensed products.
          std::vector<long double> station(w.n,0);
          for(int p=0;p<w.n;++p)station[p]=input_d[p];
          for(std::size_t k=0;k<hv.size();++k){auto [p,q]=w.hpattern[k];station[p]+=(long double)hv[k]*d.x[q];if(p!=q)station[q]+=(long double)hv[k]*d.x[p];}
          auto transpose=[&](const Row& row,double weight){for(std::size_t k=0;k<row.gradient.size();++k)station[row.q->support[k]]+=(long double)row.gradient[k]*weight;};
          auto product=[&](const Row& row){long double value=0;for(std::size_t k=0;k<row.gradient.size();++k)value+=(long double)row.gradient[k]*d.x[row.q->support[k]];return value;};
          for(int k=0;k<ni;++k)transpose(j[k],d.z[active[k]]);
          for(int e=0;e<w.me;++e)transpose(a[e],d.lambda[e]);
          for(int p=0;p<w.n;++p)residual_d[p]=(double)station[p];
          double err=inf_norm(residual_d);
          last_dual=err;last_equality=last_primal=last_complementarity=0;
          for(int e=0;e<w.me;++e){residual_h[e]=(double)((long double)input_h[e]+product(a[e]));err=std::max(err,std::abs(residual_h[e]));last_equality=std::max(last_equality,std::abs(residual_h[e]));}
          for(int k=0;k<ni;++k){int i=active[k];
            residual_p[k]=(double)((long double)input_p[k]+product(j[k])+d.s[i]);
            residual_c[k]=(double)((long double)rc[k]+(long double)w.z[i]*d.s[i]+(long double)slack[k]*d.z[i]);
            err=std::max({err,std::abs(residual_p[k]),std::abs(residual_c[k])});
            last_primal=std::max(last_primal,std::abs(residual_p[k]));
            last_complementarity=std::max(last_complementarity,std::abs(residual_c[k]));
          }
          d.residual=err/scale;last_residual=d.residual;
          if(w.st.live_display_level>=4 || (forced && w.st.live_display_level>0)){
            std::ostringstream trace;trace<<std::scientific<<std::setprecision(8)
              <<"[DCA-KRYLOV-LINEAR] outer="<<iteration+1<<" rhs="<<label<<" dim="<<reduced
              <<" lifted="<<lifted.size()<<" gmres="<<used<<"/"<<w.st.dca_krylov_max_iterations
              <<" refine="<<refinement<<" original_residual="<<d.residual
              <<" threshold="<<w.st.kkt_backward_error_reject<<" dual="<<last_dual
              <<" equality="<<last_equality<<" primal="<<last_primal
              <<" complementarity="<<last_complementarity<<" scale="<<scale
              <<" reduced_residual2="<<gmres.residual_2<<" detail=\""<<gmres.message<<"\"\n";
            std::cout<<trace.str();
          }
          if(finite(d.x)&&finite(d.lambda)&&finite(d.s)&&finite(d.z)&&std::isfinite(d.residual)&&d.residual<=w.st.kkt_backward_error_reject){largest_residual=std::max(largest_residual,d.residual);return true;}
          if(!std::isfinite(d.residual))break;
        }
        return false;
      };
      Direction affine,d;
      double target=std::min(model_target,0.95*mu);
      if(w.st.dca_krylov_predictor_corrector){
        if(!solve(comp,model_affine,affine,nullptr,nullptr,nullptr,"affine"))return reject("affine-original-residual",true);
        double ap=1,ad=1;
        for(int k=0;k<ni;++k){int i=active[k];if(affine.s[i]<0)ap=std::min(ap,-slack[k]/affine.s[i]);if(affine.z[i]<0)ad=std::min(ad,-w.z[i]/affine.z[i]);}
        double amu=0;for(int k=0;k<ni;++k){int i=active[k];amu+=(slack[k]+ap*affine.s[i])*(w.z[i]+ad*affine.z[i]);}
        amu=ni?amu/ni:0;
        const double sigma=std::clamp(std::pow(std::max(0.0,amu)/std::max(mu,1e-300),3),1e-8,0.95);
        target=std::max(0.001*w.objective.scale*std::min(w.st.complementarity_tolerance,w.st.stationarity_tolerance),sigma*mu);
      }
      Vec rc=comp;
      for(int k=0;k<ni;++k)rc[k]+= -target+(w.st.dca_krylov_predictor_corrector?affine.s[active[k]]*affine.z[active[k]]:0);
      if(!solve(rc,model_corrector,d,nullptr,nullptr,nullptr,"corrector"))return reject("corrector-original-residual",true);
      double ap=1,ad=1;
      for(int k=0;k<ni;++k){int i=active[k];if(d.s[i]<0)ap=std::min(ap,-w.st.lazy_newton_fraction_to_boundary*slack[k]/d.s[i]);if(d.z[i]<0)ad=std::min(ad,-w.st.lazy_newton_fraction_to_boundary*w.z[i]/d.z[i]);}
      const bool terminal_root=forced && detail::dca_stationary_phase(
          outer.primal,outer.normalized_kkt,w.st.feasibility_tolerance,
          w.st.stationarity_tolerance);
      if(terminal_root && w.st.dca_krylov_predictor_corrector){
        double affine_ap=1,affine_ad=1;
        for(int k=0;k<ni;++k){const int i=active[k];
          if(affine.s[i]<0)affine_ap=std::min(affine_ap,-w.st.lazy_newton_fraction_to_boundary*slack[k]/affine.s[i]);
          if(affine.z[i]<0)affine_ad=std::min(affine_ad,-w.st.lazy_newton_fraction_to_boundary*w.z[i]/affine.z[i]);}
        auto predicted_stationarity=[&](const Direction& step,double alpha){
          Vec value=dual;const auto hd=hproduct(step.x);axpy(alpha,hd,value);
          for(int e=0;e<w.me;++e)a[e].transpose(alpha*step.lambda[e],value);
          for(int k=0;k<ni;++k)j[k].transpose(alpha*step.z[active[k]],value);
          return inf_norm(value);
        };
        const double affine_pred=predicted_stationarity(affine,std::min(affine_ap,affine_ad));
        const double corrector_pred=predicted_stationarity(d,std::min(ap,ad));
        // Both directions have passed the SAME expanded original KKT audit.
        // A second-order centering term can destroy the feasible fraction.
        // Prefer the already available affine direction only when it predicts
        // better original stationarity; no new solve or factor is required.
        if(affine_pred<corrector_pred){
          if(w.st.live_display_level>=2)std::cout<<"[DCA-KRYLOV-SELECT] outer="<<iteration+1
              <<" direction=affine affine_fraction="<<std::min(affine_ap,affine_ad)
              <<" corrector_fraction="<<std::min(ap,ad)<<" affine_prediction="<<affine_pred
              <<" corrector_prediction="<<corrector_pred<<"\n";
          d=affine;ap=affine_ap;ad=affine_ad;target=0;
        }
      }
      // This is a root correction of ONE original primal-dual KKT system.
      // A split step adds (alpha_du-alpha_pr)*J' dz to its linear residual;
      // it is not a damped root direction, even if both positivity tests pass.
      // The common fraction preserves every positivity boundary AND the
      // cancellation proved by the expanded original-system residual audit.
      if(terminal_root)ap=ad=std::min(ap,ad);
      if(w.st.live_display_level>=4){
        double bound=1;int blocking=-1;
        for(int k=0;k<ni;++k)if(d.s[active[k]]<0){const double f=-w.st.lazy_newton_fraction_to_boundary*slack[k]/d.s[active[k]];if(f<bound){bound=f;blocking=k;}}
        if(blocking>=0){const int i=active[blocking];std::cout<<"[DCA-KRYLOV-BOUND] row="<<i
          <<" kind="<<w.inequality_kind[i]<<" origin="<<w.inequality_origin[i]
          <<" s="<<slack[blocking]<<" z="<<w.z[i]<<" ds="<<d.s[i]
          <<" primal_equation="<<primal[blocking]<<" fraction="<<bound<<"\n";}
      }
      std::string reason="original-filter";
      int soc_solves=0;
      for(int ls=0;ls<8;++ls){
        deadline();const double tau=std::pow(0.5,ls),alpha=ap*tau,alphad=ad*tau;
        Vec x=w.x,lambda=w.lambda,z=w.z,s=w.s;
        const double alphal=detail::dca_equality_step(
            outer.primal,w.st.feasibility_tolerance,alpha,alphad);
        axpy(alpha,d.x,x);axpy(alphal,d.lambda,lambda);axpy(alphad,d.z,z);
        bool positive=finite(x)&&finite(lambda)&&finite(z);
        for(int k=0;k<ni;++k){int i=active[k];s[i]=slack[k]+alpha*d.s[i];positive=positive&&s[i]>0&&z[i]>0;}
        if(!positive)continue;
        auto candidate=dca_original(w,x,lambda,z,&w.lambda,&w.z);
        if(w.st.live_display_level>=4){
          Vec predicted_dual=dual;auto hd=hproduct(d.x);axpy(alpha,hd,predicted_dual);
          for(int e=0;e<w.me;++e)a[e].transpose(alphal*d.lambda[e],predicted_dual);
          for(int k=0;k<ni;++k)j[k].transpose(alphad*d.z[active[k]],predicted_dual);
          std::cout<<"[DCA-KRYLOV-STEP] ls="<<ls<<" ap="<<alpha<<" ad="<<alphad
                   <<" dx="<<inf_norm(d.x)<<" dl="<<inf_norm(d.lambda)<<" dz="<<inf_norm(d.z)
                   <<" dual_linear="<<inf_norm(predicted_dual)/w.objective.scale
                   <<" dual_nonlinear="<<candidate.dual<<" primal="<<candidate.primal
                   <<" comp="<<candidate.comp<<" target="<<target<<"\n";
        }
        bool used_soc=false;
        // A tangent root step leaves a quadratic equality defect. Re-use the
        // SAME original Jacobian and certified preconditioner for at most two
        // second-order normal corrections, instead of relaxing feasibility.
        // Each correction has its OWN expanded original linear residual audit;
        // only a positive, bounded and original-filter-admissible endpoint commits.
        while(terminal_root && (candidate.primal>w.st.feasibility_tolerance || (used_soc && candidate.primal>0)) &&
              detail::dca_terminal_contracts(reference.kkt,candidate.kkt,alpha) && soc_solves<2){
          Vec zero_dual(w.n,0),zero_comp(ni,0),soc_eq(w.me),soc_primal(ni);
          for(int e=0;e<w.me;++e)soc_eq[e]=w.equalities[e].original_at(x);
          for(int k=0;k<ni;++k)soc_primal[k]=w.inequalities[active[k]].original_at(x)+s[active[k]];
          Direction zero,normal;zero.x.assign(w.n,0);zero.lambda.assign(w.me,0);
          zero.s.assign(w.mi,0);zero.z.assign(w.mi,0);
          ++soc_solves;++stats.soc_attempts;
          const bool solved=solve(zero_comp,zero,normal,&zero_dual,&soc_eq,&soc_primal,"soc");
          if(!solved)break;
          const double radius=std::max(1e-8,0.5*alpha*inf_norm(d.x));
          if(inf_norm(normal.x)>radius)break;
          Vec cx=x,cl=lambda,cz=z,cs=s;
          axpy(1,normal.x,cx);axpy(1,normal.lambda,cl);axpy(1,normal.z,cz);axpy(1,normal.s,cs);
          bool admissible=finite(cx)&&finite(cl)&&finite(cz)&&finite(cs);
          for(int i:active)admissible=admissible&&cs[i]>0&&cz[i]>0;
          if(!admissible)break;
          auto corrected=dca_original(w,cx,cl,cz,&w.lambda,&w.z);
          if(w.st.live_display_level>=4)std::cout<<"[DCA-KRYLOV-SOC] ls="<<ls
              <<" normal="<<inf_norm(normal.x)<<" radius="<<radius
              <<" primal_before="<<candidate.primal<<" primal_after="<<corrected.primal
              <<" kkt_after="<<corrected.kkt<<"\n";
          if(!(corrected.primal<candidate.primal) ||
              !detail::dca_terminal_contracts(reference.kkt,corrected.kkt,alpha))break;
          x.swap(cx);lambda.swap(cl);z.swap(cz);s.swap(cs);candidate=corrected;used_soc=true;
        }
        const double local_growth=std::min(1.1,w.st.dca_kkt_growth_max);
        if(!detail::dca_kkt_growth_allowed(
               outer.kkt,candidate.kkt,local_growth,
               10.0*w.st.acceptable_tolerance)){
          reason="original-kkt-growth";continue;
        }
        // The accelerated endpoint must outperform the already accepted
        // genuine DCA endpoint. A filter-admissible tiny Newton step alone
        // is not sufficient reason to replace that endpoint.
        const bool terminal_rescue=forced && detail::dca_stationary_phase(
            outer.primal,outer.normalized_kkt,w.st.feasibility_tolerance,
            w.st.stationarity_tolerance);
        if(terminal_rescue){
          if(candidate.primal>w.st.feasibility_tolerance ||
             !detail::dca_terminal_contracts(reference.kkt,candidate.kkt,alpha) ||
             !dca_kkt_allowed(w,outer,candidate)){
            reason=candidate.primal>w.st.feasibility_tolerance ? "terminal-primal" :
                !detail::dca_terminal_contracts(reference.kkt,candidate.kkt,alpha) ? "terminal-contraction" : "terminal-persistent-guard";
            if(w.st.live_display_level>=4)std::cout<<"[DCA-KRYLOV-GATE] ls="<<ls<<" reason="<<reason
                <<" from_kkt="<<outer.kkt<<" reference="<<reference.kkt<<" persistent="<<outer.guard_reference_kkt
                <<" candidate="<<candidate.kkt<<" alpha="<<alpha<<" primal="<<candidate.primal<<"\n";
            continue;
          }
        }else if(candidate.kkt>0.9*reference.kkt ||
                 candidate.theta>std::max(reference.theta,0.9*outer.theta)){
          // Keep the historical theta guard for the opt-in accelerator.
          reason="no-gain-over-dca";continue;
        }
        double quadratic=0;Vec dx=x;axpy(-1,w.x,dx);
        for(std::size_t k=0;k<hv.size();++k){auto [p,q]=w.hpattern[k];quadratic+=(p==q?0.5:1)*hv[k]*dx[p]*dx[q];}
        Vec gradient(w.n,0);w.objective.original_gradient_at(w.x,1,gradient);
        double predicted=-dot(gradient,dx)-quadratic;
        const auto filter_decision=filter.diagnose(outer,candidate,predicted,w.st.feasibility_tolerance,terminal_root?alpha:1.0);
        if(!filter_decision.accepted){
          reason=std::string("original-filter:")+filter_decision.reason;
          if(w.st.live_display_level>=4)std::cout<<"[DCA-KRYLOV-FILTER] ls="<<ls<<" reason="<<reason
              <<" stationary="<<filter_decision.stationary_phase<<" df="<<candidate.objective-outer.objective
              <<" dlag="<<candidate.lagrangian-outer.lagrangian<<" noise="<<filter_decision.noise
              <<" primal="<<candidate.primal<<" from_kkt="<<outer.kkt<<" kkt="<<candidate.kkt
              <<" terminal_limit="<<filter_decision.terminal_kkt_limit<<"\n";
          continue;
        }
        bool violated=false;for(int i=0;i<w.mi;++i)if(!w.active[i]&&w.inequalities[i].original_at(x)>w.st.feasibility_tolerance*w.inequalities[i].scale){violated=true;break;}
        if(violated){reason="inactive-original-row";continue;}
        proposal={std::move(x),std::move(lambda),std::move(z),std::move(s),candidate};
        record_linear();
        ++stats.krylov_acceptances;if(used_soc)++stats.soc_acceptances;
        stats.krylov_last_rejection.clear();
        stats.krylov_max_accepted_residual=std::max(stats.krylov_max_accepted_residual,largest_residual);
        if(stats.krylov_first_accept_iteration<0){stats.krylov_first_accept_iteration=iteration+1;stats.krylov_first_accept_seconds=seconds(start);}
        accepted_alpha=alpha;accepted_dual_alpha=alphad;accepted_line_steps=ls;
        next_iteration=iteration+1;w.message=saved_message;
        if(w.st.live_display_level>=2)std::cout<<"  KRYLOV-DCA iter="<<iteration+1<<" original_kkt="<<candidate.kkt<<" original_primal="<<candidate.primal<<" residual="<<largest_residual<<" time="<<seconds(start)<<"\n";
        return true;
      }
      return reject(reason);
    }catch(const std::runtime_error& e){return reject(e.what(),true);}
  }
};
