// Internal recovery for a failed square quadratic feasibility trajectory.
// Not a model rewrite: coefficients, bounds, Workspace and its accepted state
// are read-only here. With y=x-origin, follow
//       F(t,y)=h(origin)+J(origin)y+t/2 [y' H_e y]_e = 0.
// Only t==1 is the original system. Intermediate roots are NEVER exported as
// feasible points. The caller must check the ORIGINAL filter and original KKT.
// No problem-name, physical parameter, alternative backend or stored solution
// enters this continuation. Included beside Workspace in atomic_newton.cpp.
struct DcaSquareContinuation {
  struct Row {
    double value=0, scale=1;
    std::vector<long double> gradient;
  };
  struct Outcome {
    bool success=false,limited=false;
    Vec x;
    int iterations=0,stages=0,trials=0,factors=0,solves=0,evaluations=0,gradient_evaluations=0,precision_rejections=0,krylov_iterations=0,reused_directions=0;
    double parameter=0,linear_error=0,attempted_linear_error=0,seconds=0;
    std::string reason;
  };
  const Workspace& w;
  Vec origin;
  Clock::time_point start;
  double deadline;
  int limit,level;
  std::vector<Row> rows;
  std::vector<std::vector<long double>> gradients;
  std::unique_ptr<SymmetricFactorization> factor;
  std::vector<SymmetricTriplet> matrix;
  std::vector<int> offsets;
  Outcome out;
  bool factor_ready=false;
  const bool reuse_preconditioner;
  bool expired() const {return seconds(start)>=deadline;}
  DcaSquareContinuation(const Workspace& work,const Vec& anchor,
      Clock::time_point begin,double seconds_limit,int iteration_limit,int display)
      :w(work),origin(anchor),start(begin),deadline(seconds_limit),
       limit(iteration_limit),level(display),
       reuse_preconditioner(work.st.linear_backend=="cholesk"&&!work.st.linear_metis) {
    if(w.n<=0||w.me!=w.n||w.mi||origin.size()!=std::size_t(w.n))
      throw std::invalid_argument("square continuation dimensions");
    rows.resize(w.me);gradients.resize(w.me);offsets.resize(w.me);
    // Pin the exact same requested backend/ordering. Do not reuse a factor
    // built for a different Jacobian, nor add a cross-backend fallback.
    std::string message;
    factor=make_symmetric_factorization(w.factor_settings,message);
    if(!factor)throw std::runtime_error("square continuation backend: "+message);
    for(int j=0;j<w.n;++j)matrix.push_back({j,j,1.0});
    for(int e=0;e<w.me;++e) {
      const auto& q=w.equalities[e];auto& row=rows[e];
      if(!q.signed_square.empty()||!(q.scale>0)||!std::isfinite(q.scale))
        throw std::invalid_argument("square continuation requires fixed quadratic rows");
      row.scale=q.scale;row.gradient.assign(q.support.size(),0);
      long double value=q.constant;
      for(std::size_t k=0;k<q.linear.size();++k) {
        const auto term=q.linear[k];value+=static_cast<long double>(term.second)*origin[term.first];
        row.gradient[q.linear_pos[k]]+=term.second;
      }
      for(std::size_t k=0;k<q.hessian.size();++k) {
        const auto t=q.hessian[k];const long double v=t.v;
        value+=(t.i==t.j?0.5L:1.L)*v*origin[t.i]*origin[t.j];
        row.gradient[q.hi[k]]+=v*origin[t.j];
        if(t.i!=t.j)row.gradient[q.hj[k]]+=v*origin[t.i];
      }
      row.value=double(value*q.scale);
      for(auto& v:row.gradient)v*=q.scale;
      gradients[e]=row.gradient;offsets[e]=int(matrix.size());
      for(int j:q.support)matrix.push_back({j,w.n+e,0.0});
      matrix.push_back({w.n+e,w.n+e,0.0});
    }
  }
  // Full stage polynomial, evaluated by extended sums with unchanged scales.
  Vec evaluate(const Vec& y,double t,bool with_gradients) {
    CostScope cost(w.costs,NewtonCost::Evaluation);
    ++out.evaluations;if(with_gradients)++out.gradient_evaluations;
    Vec r(w.me);
    for(int e=0;e<w.me;++e) {
      const auto& q=w.equalities[e];const auto& row=rows[e];
      long double value=row.value;
      for(std::size_t k=0;k<q.support.size();++k)value+=row.gradient[k]*y[q.support[k]];
      if(with_gradients)gradients[e]=row.gradient;
      for(std::size_t k=0;k<q.hessian.size();++k) {
        const auto z=q.hessian[k];const long double v=static_cast<long double>(t)*row.scale*z.v;
        value+=(z.i==z.j?0.5L:1.L)*v*y[z.i]*y[z.j];
        if(with_gradients) {
          gradients[e][q.hi[k]]+=v*y[z.j];
          if(z.i!=z.j)gradients[e][q.hj[k]]+=v*y[z.i];
        }
      }
      r[e]=double(value);
    }
    if(!finite(r))throw std::runtime_error("nonfinite square continuation residual");
    return r;
  }
  long double merit(const Vec& r) const {
    long double sum=0;for(double v:r)sum+=static_cast<long double>(v)*v;return sum;
  }
  double residual_original_units(const Vec& r) const {
    double v=0;for(int e=0;e<w.me;++e)v=std::max(v,std::abs(r[e])/rows[e].scale);return v;
  }
  bool direction(const Vec& residual,Vec& d) {
    if(expired()||out.iterations>=limit){out.limited=true;out.reason="time-or-iteration-budget";return false;}
    ++out.iterations;
    for(int e=0;e<w.me;++e)for(std::size_t k=0;k<gradients[e].size();++k) {
      const double v=double(gradients[e][k]);
      if(!std::isfinite(v))throw std::runtime_error("nonfinite square continuation Jacobian");
      matrix[offsets[e]+k].value=v;
    }
    std::string message;bool ok;
    if(reuse_preconditioner&&factor_ready) {
      // This is a linear solve of the STAGE Jacobian, not a reactivation of
      // DCA Krylov/atoms. Avoid repeated expensive natural-order Cholesk
      // factors; certify the true stage residual, otherwise refactor below.
      Vec right=residual;for(auto& v:right)v=-v;
      auto apply=[&](const Vec& x,Vec& result) {
        result.assign(w.me,0.0);
        for(int e=0;e<w.me;++e) {
          long double v=0;const auto& q=w.equalities[e];
          for(std::size_t k=0;k<q.support.size();++k)v+=gradients[e][k]*x[q.support[k]];
          result[e]=double(v);
        }
      };
      auto precondition=[&](const Vec& b,Vec& x,std::string& why) {
        if(expired()){out.limited=true;why="time-budget";return false;}
        Vec full(2*w.n,0.0);std::copy(b.begin(),b.end(),full.begin()+w.n);
        CostScope cost(w.costs,NewtonCost::LinearBackend);++out.solves;
        if(!factor->solve(full,why)||!finite(full))return false;
        x.assign(full.begin(),full.begin()+w.n);return true;
      };
      const double tolerance=std::min(0.1*w.st.kkt_backward_error_reject*
          std::max(1.0,inf_norm(residual)),0.001*std::sqrt(double(merit(residual))));
      const auto gmres=original_kkt_right_preconditioned_gmres(apply,precondition,right,d,80,
          std::max(std::numeric_limits<double>::min(),tolerance));
      out.krylov_iterations+=gmres.iterations;
      if(out.limited)return false;
      if(gmres.converged) {
        Vec actual;apply(d,actual);long double error=0;
        for(int e=0;e<w.me;++e)error=std::max(error,std::abs(
            static_cast<long double>(actual[e])+residual[e]));
        const double scaled=double(error)/std::max(1.0,inf_norm(residual));
        out.attempted_linear_error=std::max(out.attempted_linear_error,scaled);
        if(std::isfinite(scaled)&&scaled<=w.st.kkt_backward_error_reject) {
          out.linear_error=std::max(out.linear_error,scaled);++out.reused_directions;return true;
        }
        ++out.precision_rejections;
      }
    }
    if(expired()){out.limited=true;out.reason="time-budget";return false;}
    {CostScope cost(w.costs,NewtonCost::Factorization);
      ++out.factors;ok=factor->factor(2*w.n,w.n,matrix,message);}
    factor_ready=ok;
    if(!ok){out.reason="stage-factorization: "+message;return false;}
    Vec rhs(2*w.n,0.0);for(int e=0;e<w.me;++e)rhs[w.n+e]=-residual[e];
    {CostScope cost(w.costs,NewtonCost::LinearBackend);
      ++out.solves;ok=factor->solve(rhs,message);}
    if(!ok||!finite(rhs)){out.reason="stage-solve: "+message;return false;}
    // Check the UNPERTURBED stage equations, including the identity metric.
    // Backend-specific hidden shifts are never accepted on their own residual.
    long double error=0;std::vector<long double> upper(rhs.begin(),rhs.begin()+w.n);
    for(int e=0;e<w.me;++e) {
      long double lower=residual[e];const auto& q=w.equalities[e];
      for(std::size_t k=0;k<q.support.size();++k) {
        const int j=q.support[k];const auto v=gradients[e][k];
        lower+=v*rhs[j];upper[j]+=v*rhs[w.n+e];
      }
      error=std::max(error,std::abs(lower));
    }
    for(auto v:upper)error=std::max(error,std::abs(v));
    const double scaled=double(error)/std::max(1.0,inf_norm(residual));
    out.attempted_linear_error=std::max(out.attempted_linear_error,scaled);
    if(!std::isfinite(scaled)||scaled>w.st.kkt_backward_error_reject) {
      ++out.precision_rejections;out.reason="stage-linear-residual";return false;
    }
    out.linear_error=std::max(out.linear_error,scaled);
    d.assign(rhs.begin(),rhs.begin()+w.n);return true;
  }
  bool correct(Vec& y,double t,int& used) {
    const int before=out.iterations;
    // Stage precision is stronger than the final requested feasibility. It is
    // not used to label t<1 roots as original feasible.
    const double tolerance=0.01*w.st.feasibility_tolerance;
    for(int k=0;k<30;++k) {
      if(expired()){out.limited=true;out.reason="time-budget";return false;}
      const Vec r=evaluate(y,t,true);
      if(residual_original_units(r)<=tolerance){used=out.iterations-before;return true;}
      Vec d;if(!direction(r,d)){used=out.iterations-before;return false;}
      const long double old=merit(r);bool accepted=false;
      for(int ls=0;ls<w.st.line_search_max_steps;++ls) {
        if(expired()){out.limited=true;out.reason="time-budget";return false;}
        const double a=std::pow(w.st.line_search_contraction,ls);
        Vec trial=y;axpy(a,d,trial);if(!finite(trial))continue;
        const Vec next=evaluate(trial,t,false);
        if(merit(next)<(1-static_cast<long double>(w.st.line_search_armijo)*a)*old) {y.swap(trial);accepted=true;break;}
      }
      if(!accepted){out.reason="stage-line-search";used=out.iterations-before;return false;}
    }
    out.reason="stage-iteration-limit";used=out.iterations-before;return false;
  }
  Outcome run() {
    const auto began=Clock::now();Vec y(w.n,0.0);
    try {
      int used=0;
      if(!correct(y,0,used)){out.seconds=seconds(began);return out;}
      double t=0.0,step=0.1;
      for(int stage=0;stage<128 && t<1;++stage) {
        if(expired()){out.limited=true;out.reason="time-budget";break;}
        const double target=std::min(1.0,t+step);Vec trial=y;used=0;
        ++out.trials;const bool ok=correct(trial,target,used);
        if(level>=2)std::cout<<"[SQUARE-CONTINUATION-STAGE] parameter="<<target
            <<" accepted="<<ok<<" steps="<<used
            <<" stage_primal="<<residual_original_units(evaluate(trial,target,false))
            <<" original_primal="<<residual_original_units(evaluate(trial,1.0,false))
            <<" original_success_not_implied=1"<<std::endl;
        if(ok){y.swap(trial);t=target;out.parameter=t;++out.stages;
          if(used<=6)step=std::min(0.25,step*1.5);
        } else {if(out.limited)break;step*=0.5;}
        if(step<std::ldexp(1.0,-24)){out.reason="parameter-step-too-small";break;}
      }
      if(out.parameter==1.0) {
        // Reassemble coordinates and evaluate original rows directly. Avoid
        // relying solely on cancellation in the expanded homotopy polynomial.
        out.x=origin;axpy(1.0,y,out.x);double primal=0;
        for(const auto& q:w.equalities)primal=std::max(primal,std::abs(q.original_at(out.x))/q.scale);
        out.success=finite(out.x)&&std::isfinite(primal)&&primal<=w.st.feasibility_tolerance;
        out.reason=out.success?"original-root":"original-certificate-residual";
      }
    } catch(const std::exception& error) {out.reason=error.what();out.success=false;}
    out.seconds=seconds(began);return out;
  }
};
