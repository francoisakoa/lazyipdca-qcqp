// Exact ORIGINAL-QCQP barrier/filter polynomials. Include beside Workspace and
// Direction. An auxiliary KKT supplies the direction; its inequality curvature
// is never used to evaluate candidates. Legacy slacks follow s+alpha*ds.
// Hybrid outer globalization can select a geometric slack curve with the SAME
// binary endpoints; its derivative and actual rounded slacks are explicit.
// Tested slacks must be kept unchanged when that candidate is accepted.
//
// PRECONDITION for construction/current: values and (extended) gradients are
// cached at w.x and equal those of the original QCQP there. Active models are
// allowed only immediately after tangent reanchoring at this same w.x; caches
// evaluated away from their model centre are not valid original caches. The
// outer controller establishes this once, without per-row rechecks here.
// Workspace and Direction remain unchanged while this path is used. mu is
// frozen during line search; a filter compares points only at the same mu.
// Preparation visits each original Hessian once; trials perform no function,
// gradient, Hessian, majorant-gap or KKT evaluation.
#include "bound_product_path.hpp"
#include "consistent_bound_slack_path.hpp"

struct DcaMajorantBarrierPoint {
  double objective=0,phi=0,phi_change=0,theta=0,primal=0,primal_internal=0;
  // Absolute active-slack log sum permits exact fixed-mu rebasing of a
  // retained ORIGINAL filter entry after private model continuation.
  long double slack_logs=0;
  double omitted_bound_max=0,omitted_bound_theta=0;
  int omitted_bound_sides=0;
  bool positive=true,finite=true;
};

struct DcaMajorantBarrierPath {
  using Acc=long double;
  struct Polynomial {
    Acc constant=0,linear=0,quadratic=0;
    struct SignedSquare {Acc x,d,c;};
    std::vector<SignedSquare> signed_square;
    Acc change(Acc alpha) const {
      Acc value=alpha*(linear+alpha*quadratic);
      for(auto e:signed_square) {
        const Acc step=alpha*e.d,y=e.x+step;
        if(e.x==0)value+=e.c*step*std::abs(step);
        else if((e.x>0&&y>=0)||(e.x<0&&y<=0))value+=e.c*(e.x>0?1:-1)*step*step;
        else value+=e.c*(y*std::abs(y)-e.x*std::abs(e.x)-2*std::abs(e.x)*step);
      }
      return value;
    }
    // Exact second-order remainder, without subtracting two first-order
    // quantities. Native terms retain their correct branch when crossing 0.
    Acc nonlinear_change(Acc alpha) const {
      Acc value=alpha*alpha*quadratic;
      for(auto e:signed_square) {
        const Acc step=alpha*e.d,y=e.x+step;
        if(e.x==0)value+=e.c*step*std::abs(step);
        else if((e.x>0&&y>=0)||(e.x<0&&y<=0))value+=e.c*(e.x>0?1:-1)*step*step;
        else value+=e.c*(y*std::abs(y)-e.x*std::abs(e.x)-2*std::abs(e.x)*step);
      }
      return value;
    }
    Acc at(Acc alpha) const {return constant+change(alpha);}
  };
  const Workspace& w;
  const Direction& direction;
  const Direction* second_order_correction=nullptr;
  Polynomial objective;
  std::vector<Polynomial> equalities,inequalities;
  Acc slack_log_slope=0,initial_slack_logs=0;
  bool geometric_slacks=false;
  bool bound_product_duals=false;
  bool consistent_bound_slacks=false;
  bool rounding_safe_bound_duals=false;
  std::vector<hybrid_slack::Curve> slack_curves;
  const char* slack_path_name() const {return consistent_bound_slacks?"original-bound-consistent":(geometric_slacks?"geometric":(second_order_correction?"quadratic-SOC":"affine"));}
  bool has_second_order() const {return second_order_correction!=nullptr;}

  // q.value/q.gradient are original caches at the path origin (or identical
  // tangent-model caches). Stored original Hessians are unscaled; model
  // Hessians/atoms already include scale but must not enter this polynomial.
  static Polynomial prepare_original(const Quadratic& q,const Vec& dx,const Vec& x) {
    Polynomial p;
    p.constant=q.value;
    for(std::size_t k=0;k<q.support.size();++k) {
      const int j=q.support[k];
      const Acc gradient=q.majorant_gradients_extended&&
          q.gradient_extended.size()==q.support.size()?q.gradient_extended[k]:Acc(q.gradient[k]);
      p.linear+=gradient*dx[j];
    }
    const Acc scale=q.scale;
    for(const auto& e:(q.signed_square.empty()?q.hessian:q.base_hessian))
      p.quadratic+=(e.i==e.j?Acc(0.5):Acc(1))*scale*e.v*dx[e.i]*dx[e.j];
    for(auto e:q.signed_square)p.signed_square.push_back({x[e.first],dx[e.first],scale*e.second});
    return p;
  }

  DcaMajorantBarrierPath(const Workspace& work,const Direction& step,bool request_geometric_slacks=false,
      bool request_bound_product_duals=false,const Direction* correction=nullptr,
      bool request_consistent_bound_slacks=false,bool request_rounding_safe_bound_duals=false)
      :w(work),direction(step),second_order_correction(correction),equalities(w.me),inequalities(w.mi) {
    consistent_bound_slacks=request_consistent_bound_slacks;
    rounding_safe_bound_duals=request_rounding_safe_bound_duals&&consistent_bound_slacks;
    if(consistent_bound_slacks) {
      if(request_geometric_slacks||correction||!request_bound_product_duals)
        throw std::invalid_argument("consistent original bound slacks require the native affine primal path and bound product duals");
      for(int i=0;i<w.mi;++i)if(w.active[i]&&w.inequality_kind[i]!=0) {
        const auto& q=w.inequalities[i];
        if(!q.signed_square.empty()||std::any_of(q.hessian.begin(),q.hessian.end(),[](auto h){return h.v!=0;}))
          throw std::invalid_argument("consistent original bound slack row must be affine");
        if(!std::isfinite(DcaConsistentBoundSlack::origin_link(q.value,w.s[i])))
          throw std::invalid_argument("invalid original bound slack origin");
      }
    }
    if(second_order_correction) {
      if(request_geometric_slacks||second_order_correction->x.size()!=w.x.size()||
          second_order_correction->s.size()!=w.s.size()||second_order_correction->z.size()!=w.z.size()||
          second_order_correction->lambda.size()!=w.lambda.size())
        throw std::invalid_argument("invalid quadratic SOC path dimensions or geometric slacks");
      if(!finite(second_order_correction->x)||!finite(second_order_correction->s)||
          !finite(second_order_correction->z)||!finite(second_order_correction->lambda))
        throw std::invalid_argument("nonfinite quadratic SOC direction");
    }
    bound_product_duals=request_bound_product_duals;
    if(bound_product_duals&&request_geometric_slacks)
      throw std::invalid_argument("bound product dual path requires affine slacks");
    geometric_slacks=request_geometric_slacks;
    if(geometric_slacks)for(int i=0;i<w.mi;++i)if(w.active[i]) {
      const double endpoint=w.s[i]+direction.s[i];
      if(!(w.s[i]>0&&endpoint>0)||!std::isfinite(w.s[i])||!std::isfinite(endpoint))
        {geometric_slacks=false;break;}
    }
    if(geometric_slacks) {
      slack_curves.resize(w.mi);
      for(int i=0;i<w.mi;++i)if(w.active[i])slack_curves[i]=
          hybrid_slack::Curve(w.s[i],w.s[i]+direction.s[i]);
    }
    if(second_order_correction) {
      // Validate before the parallel preparation loop: exceptions must never
      // escape an OpenMP worker. This geometry excludes all a^3/a^4 terms.
      auto validate=[&](const Quadratic& q) {
        if(!q.signed_square.empty())throw std::invalid_argument("SOC path requires polynomial QCQP rows");
        for(auto h:q.hessian)if(h.v!=0&&(second_order_correction->x[h.i]!=0||second_order_correction->x[h.j]!=0))
          throw std::invalid_argument("SOC correction touches an original Hessian column");
      };
      validate(w.objective);for(const auto& q:w.equalities)validate(q);for(const auto& q:w.inequalities)validate(q);
    }
    auto prepare=[&](const Quadratic& q) {
      auto p=prepare_original(q,direction.x,w.x);
      if(second_order_correction)p.quadratic+=prepare_original(q,second_order_correction->x,w.x).linear;
      return p;
    };
    objective=prepare(w.objective);
    // Rows read immutable caches, preserve their own addition order, and
    // write disjoint polynomial slots. No evaluator or shared ledger is used.
#if defined(LAZYIPDCA_HAVE_OPENMP)
#pragma omp parallel for schedule(static) if(w.me+w.mi>=1024 && omp_get_max_threads()>1 && !omp_in_parallel())
#endif
    for(int row=0;row<w.me+w.mi;++row) {
      if(row<w.me)equalities[row]=prepare(w.equalities[row]);
      else {
        const int i=row-w.me;
        inequalities[i]=prepare(w.inequalities[i]);
      }
    }
    for(int i=0;i<w.mi;++i)if(w.active[i]) {
      if(!(w.s[i]>0)||!std::isfinite(w.s[i])){slack_log_slope=INFINITY;break;}
      slack_log_slope+=geometric_slacks?slack_curves[i].rate:slack_slope(i)/w.s[i];
      initial_slack_logs+=std::log(w.s[i]);
    }
  }

  double objective_slope() const {return double(objective.linear);}
  double slope(double mu) const {
    if(!(mu>=0)||!std::isfinite(mu))return INFINITY;
    return double(mu==0?objective.linear:objective.linear-Acc(mu)*slack_log_slope);
  }

  // Canonical DOUBLE candidate writers shared by merit evaluation and commit.
  // The optional quadratic term is added after the identical affine update.
  double primal(int j,double alpha) const {
    double value=w.x[j]+alpha*direction.x[j];
    if(second_order_correction)value+=(alpha*alpha)*second_order_correction->x[j];
    return value;
  }
  void write_primal(double alpha,Vec& destination) const {
    destination.resize(w.n);for(int j=0;j<w.n;++j)destination[j]=primal(j,alpha);
  }
  double equality_dual(int e,double alpha) const {
    double value=w.lambda[e]+alpha*direction.lambda[e];
    if(second_order_correction)value+=(alpha*alpha)*second_order_correction->lambda[e];
    return value;
  }
  void write_equality_duals(double alpha,Vec& destination) const {
    destination.resize(w.me);for(int e=0;e<w.me;++e)destination[e]=equality_dual(e,alpha);
  }
  Acc bound_link0(int i) const {return DcaConsistentBoundSlack::origin_link(w.inequalities[i].value,w.s[i]);}
  Acc slack_slope(int i) const {
    if(consistent_bound_slacks&&w.active[i]&&w.inequality_kind[i]!=0)
      return DcaConsistentBoundSlack::slope(w.inequalities[i].value,w.s[i],inequalities[i].linear);
    return geometric_slacks&&w.active[i]?Acc(w.s[i])*slack_curves[i].rate:Acc(direction.s[i]);
  }
  Acc bound_link_slope(int i) const {
    return consistent_bound_slacks&&w.active[i]&&w.inequality_kind[i]!=0?
        -bound_link0(i):inequalities[i].linear+slack_slope(i);
  }
  double slack(int i,double alpha,const Vec* actual_primal=nullptr) const {
    if(consistent_bound_slacks&&w.active[i]&&w.inequality_kind[i]!=0) {
      if(alpha==0)return w.s[i];
      Vec point;if(!actual_primal){write_primal(alpha,point);actual_primal=&point;}
      return DcaConsistentBoundSlack::value(w.inequalities[i].value,w.s[i],
          w.inequalities[i].original_at(*actual_primal),alpha);
    }
    if(geometric_slacks&&w.active[i])return slack_curves[i].value(alpha);
    double value=w.s[i]+alpha*direction.s[i];
    if(second_order_correction)value+=(alpha*alpha)*second_order_correction->s[i];
    return value;
  }
  void write_slacks(double alpha,Vec& destination,const Vec* actual_primal=nullptr) const {
    Vec point;
    if(consistent_bound_slacks&&!actual_primal){write_primal(alpha,point);actual_primal=&point;}
    destination.resize(w.mi);
    for(int i=0;i<w.mi;++i)destination[i]=slack(i,alpha,actual_primal);
  }
  double dual(int i,double alpha,double mu,const Vec* actual_slacks=nullptr) const {
    if(bound_product_duals&&w.active[i]&&w.inequality_kind[i]!=0){
      const double actual_s=actual_slacks?actual_slacks->at(i):slack(i,alpha);
      const double product_dual=DcaBoundProductPath::dual(w.s[i],w.z[i],direction.s[i],alpha,mu,actual_s);
      const double affine=w.z[i]+alpha*direction.z[i];
      if(rounding_safe_bound_duals&&affine>0&&std::isfinite(affine)&&std::isfinite(product_dual)){
        long double floor=0;
        const auto& q=w.inequalities[i];
        for(std::size_t k=0;k<q.support.size();++k){
          int j=q.support[k];double point=w.x[j]+alpha*direction.x[j];
          double ulp=std::abs(std::nextafter(point,INFINITY)-point);
          floor+=std::abs(static_cast<long double>(q.gradient[k]))*ulp;
        }
        const long double allowance=2*floor*std::max(std::abs(static_cast<long double>(w.z[i])),std::abs(static_cast<long double>(affine)));
        const auto target=DcaBoundProductPath::target_product(w.s[i],w.z[i],alpha,mu);
        const long double nominal_s=std::fma(static_cast<long double>(alpha),static_cast<long double>(direction.s[i]),static_cast<long double>(w.s[i]));
        const long double nominal_error=std::abs(nominal_s*affine-target);
        const long double actual_error=std::abs(static_cast<long double>(actual_s)*affine-target);
        // Only replace the positive-product division when the discrepancy
        // fits the representability floor of the ACTUAL bound slack, and
        // the audited affine Newton product fits that same floor. Positivity,
        // actual merit/Armijo and full original KKT checks remain mandatory.
        if(floor>0&&nominal_error<=allowance&&actual_error<=allowance)return affine;
      }
      return product_dual;
    }
    double value=w.z[i]+alpha*direction.z[i];
    if(second_order_correction)value+=(alpha*alpha)*second_order_correction->z[i];
    return value;
  }
  void write_duals(double alpha,double mu,Vec& destination,const Vec* actual_slacks=nullptr) const {
    Vec staged_slacks;
    if(consistent_bound_slacks&&!actual_slacks){write_slacks(alpha,staged_slacks);actual_slacks=&staged_slacks;}
    destination.resize(w.mi);
    for(int i=0;i<w.mi;++i)destination[i]=dual(i,alpha,mu,actual_slacks);
  }

  template<class EqualityValue,class InequalityValue,class SlackValue,class DualValue,class LogValue>
  static DcaMajorantBarrierPoint reduce(const Workspace& w,Acc objective,double mu,
      EqualityValue equality_value,InequalityValue inequality_value,
      SlackValue slack_value,DualValue dual_value,LogValue log_value,
      Acc initial_logs=0,Acc objective_change=0,bool trial=false) {
    DcaMajorantBarrierPoint p;
    p.objective=double(objective);
    p.finite=std::isfinite(objective)&&mu>=0&&std::isfinite(mu);
    Acc theta=0,primal=0,primal_internal=0,logs=0;
    for(int e=0;e<w.me;++e) {
      const Acc h=equality_value(e);
      p.finite=p.finite&&std::isfinite(h);
      const Acc residual=std::abs(h);
      theta+=residual;
      primal_internal=std::max(primal_internal,residual);
      primal=std::max(primal,residual/w.equalities[e].scale);
    }
    for(int i=0;i<w.mi;++i) {
      const Acc g=inequality_value(i);
      p.finite=p.finite&&std::isfinite(g);
      Acc residual=std::max(Acc(0),g);
      if(w.active[i]) {
        const Acc s=slack_value(i),z=dual_value(i);
        const bool positive=s>0&&std::isfinite(s)&&std::isfinite(z)&&
            (!w.requires_positive_dual(i)||z>0);
        p.positive=p.positive&&positive;
        p.finite=p.finite&&std::isfinite(s)&&std::isfinite(z);
        residual=std::abs(g+s);
        if(positive)logs+=log_value(i,s);
      }
      theta+=residual;
      primal_internal=std::max(primal_internal,residual);
      primal=std::max(primal,residual/w.inequalities[i].scale);
    }
    p.theta=double(theta);
    p.primal=double(primal);
    p.primal_internal=double(primal_internal);
    p.slack_logs=initial_logs+logs;
    p.phi=double(objective-Acc(mu)*p.slack_logs);
    p.phi_change=trial?double(objective_change-Acc(mu)*logs):0;
    p.finite=p.finite&&std::isfinite(p.objective)&&std::isfinite(p.phi)&&
        std::isfinite(p.phi_change)&&std::isfinite(p.theta)&&
        std::isfinite(p.primal)&&std::isfinite(p.primal_internal);
    if(!p.positive||!p.finite) {
      p.phi=INFINITY;
      p.finite=false;
    }
    return p;
  }

  static void add_omitted_original_bounds(DcaMajorantBarrierPoint& p,
      const Workspace& w,const Vec& point) {
    const auto omitted=w.omitted_original_bounds(point);
    if(!omitted.sides&&omitted.maximum==0&&omitted.sum==0)return;
    const double scale=w.omitted_original_bound_scale();
    p.omitted_bound_max=omitted.maximum;
    p.omitted_bound_theta=static_cast<double>(scale*omitted.sum);
    p.omitted_bound_sides=omitted.sides;
    p.theta=static_cast<double>(static_cast<Acc>(p.theta)+scale*omitted.sum);
    p.primal=std::max(p.primal,omitted.maximum);
    p.primal_internal=std::max(p.primal_internal,scale*omitted.maximum);
    p.finite=p.finite&&std::isfinite(omitted.maximum)&&std::isfinite(omitted.sum)&&
        std::isfinite(p.theta)&&std::isfinite(p.primal)&&std::isfinite(p.primal_internal);
  }
  DcaMajorantBarrierPoint point(double alpha,double mu,const Vec* actual_slacks=nullptr) const {
    if(consistent_bound_slacks)return rounded_point(alpha,mu,actual_slacks);
    const Acc a=alpha;
    // Evaluate affine s/z in double, exactly as the eventual Workspace step.
    auto p=reduce(w,objective.at(a),mu,
        [&](int e){return equalities[e].at(a);},
        [&](int i){return inequalities[i].at(a);},
        [&](int i){return Acc(actual_slacks?actual_slacks->at(i):slack(i,alpha));},
        [&](int i){return Acc(dual(i,alpha,mu,actual_slacks));},
        [&](int i,Acc candidate_slack) {
          // Measure the actual rounded candidate. A sub-ulp affine update
          // must not invent barrier progress that the committed state lacks.
          if(geometric_slacks||actual_slacks)return hybrid_slack::log_ratio(w.s[i],static_cast<double>(candidate_slack));
          const Acc relative=(candidate_slack-Acc(w.s[i]))/w.s[i];
          return std::log1p(relative);
        },initial_slack_logs,objective.change(a),true);
    if(w.st.rank_presolve_context&&w.st.rank_presolve_context->has_omitted_equality_bounds()) {
      Vec candidate;write_primal(alpha,candidate);
      add_omitted_original_bounds(p,w,candidate);
    }
    return p;
  }

  // SOC is judged at the rounded point that Workspace will commit. Evaluate
  // every original row there; models remain installed and untouched. Retain
  // a compensated objective difference to avoid subtracting two large values.
  DcaMajorantBarrierPoint rounded_point(double alpha,double mu,const Vec* actual_slacks=nullptr) const {
    Vec point=w.x,displacement(w.n);
    for(int j=0;j<w.n;++j) {
      point[j]=primal(j,alpha);
      displacement[j]=point[j]-w.x[j];
    }
    Vec staged_slacks;
    if(consistent_bound_slacks&&!actual_slacks){write_slacks(alpha,staged_slacks,&point);actual_slacks=&staged_slacks;}
    const auto rounded_objective=prepare_original(w.objective,displacement,w.x);
    auto p=reduce(w,w.objective.original_at(point),mu,
        [&](int e){return Acc(w.equalities[e].original_at(point));},
        [&](int i){return Acc(w.inequalities[i].original_at(point));},
        [&](int i){return Acc(actual_slacks?actual_slacks->at(i):slack(i,alpha));},
        [&](int i){return Acc(dual(i,alpha,mu,actual_slacks));},
        [&](int i,Acc candidate_slack) {
          return (geometric_slacks||actual_slacks)?hybrid_slack::log_ratio(w.s[i],static_cast<double>(candidate_slack)):
              std::log1p((candidate_slack-Acc(w.s[i]))/w.s[i]);
        },initial_slack_logs,rounded_objective.change(1),true);
    add_omitted_original_bounds(p,w,point);return p;
  }

  // Evaluate the exact four-vector candidate staged by globalization. Bound
  // product duals have already used THESE rounded slacks; commit must move the
  // same arrays, without reconstructing the quadratic curve afterwards.
  DcaMajorantBarrierPoint staged_point(const Vec& point,const Vec& slacks,const Vec& duals,double mu) const {
    if(point.size()!=w.x.size()||slacks.size()!=w.s.size()||duals.size()!=w.z.size())
      throw std::invalid_argument("staged SOC path dimensions");
    if(!finite(point)||!finite(slacks)||!finite(duals)) {
      DcaMajorantBarrierPoint rejected;rejected.finite=false;rejected.positive=false;
      rejected.phi=INFINITY;return rejected;
    }
    Vec displacement(w.n);for(int j=0;j<w.n;++j)displacement[j]=point[j]-w.x[j];
    const auto actual_objective=prepare_original(w.objective,displacement,w.x);
    auto p=reduce(w,w.objective.original_at(point),mu,
        [&](int e){return Acc(w.equalities[e].original_at(point));},
        [&](int i){return Acc(w.inequalities[i].original_at(point));},
        [&](int i){return Acc(slacks[i]);},[&](int i){return Acc(duals[i]);},
        [&](int i,Acc candidate_slack){return hybrid_slack::log_ratio(w.s[i],static_cast<double>(candidate_slack));},
        initial_slack_logs,actual_objective.change(1),true);
    add_omitted_original_bounds(p,w,point);return p;
  }

  static DcaMajorantBarrierPoint current(const Workspace& w,double mu) {
    // Same original-cache/tangent-centre precondition as construction. No
    // model removal or reevaluation is needed at the current anchor.
    auto p=reduce(w,w.objective.value,mu,
        [&](int e){return Acc(w.equalities[e].value);},
        [&](int i){return Acc(w.inequalities[i].value);},
        [&](int i){return Acc(w.s[i]);},
        [&](int i){return Acc(w.z[i]);},
        [&](int,Acc s){return Acc(std::log(double(s)));});
    add_omitted_original_bounds(p,w,w.x);return p;
  }
};
