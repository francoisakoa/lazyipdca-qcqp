// The public reader accepts same-sign infinite bound sentinels. They remain
// free in BOTH native certificate paths, including when the gradient is nonzero.
#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main() {
  try {
    int checks=0;
    auto require=[&](bool value,const char* why){++checks;if(!value)throw std::runtime_error(why);};
    Settings st;st.feasibility_tolerance=st.dual_infeasibility_tolerance=
        st.complementarity_tolerance=st.stationarity_tolerance=1e-6;
    for(const auto& sides:std::vector<std::pair<double,double>>{
        {-INFINITY,INFINITY},{INFINITY,INFINITY},{-INFINITY,-INFINITY},{0,0}}) {
      const bool fixed=std::isfinite(sides.first)&&sides.first==sides.second;
      QCQPProblem p;p.n=1;p.name="nonzero_gradient_free_domain";p.x0={0};
      p.lb={sides.first};p.ub={sides.second};p.objective.linear={1};p.objective.hessian.n=1;
      p.validate();study::validate_h4_domain(p);
      for(bool indexed:{false,true}) {
        Result r;r.x={0};r.certificate_lower_multipliers={0};r.certificate_upper_multipliers={0};
        const bool certified=indexed?certify_indexed_original(p,st,r,nullptr):certify_dense_original(p,st,r);
        // Independent exact gradient of f(x)=x is 1; no finite bound and no
        // constraint can cancel it. A finite fixed coordinate is the control.
        require(certified==fixed,"native certificate discarded a nonzero gradient on a free variable");
        require(r.dual_infeasibility_inf==(fixed?0.0:1.0),"native certificate stationarity disagrees with exact gradient");
        if(!fixed) {
          // A genuine unbounded-domain optimum must still be certifiable.
          p.objective.linear={0};p.objective.hessian.entries={{0,0,2}};
          Result optimum;optimum.x={0};optimum.certificate_lower_multipliers={0};optimum.certificate_upper_multipliers={0};
          const bool valid=indexed?certify_indexed_original(p,st,optimum,nullptr):certify_dense_original(p,st,optimum);
          require(valid&&optimum.dual_infeasibility_inf==0,"true free-domain optimum was rejected");
          p.objective.linear={1};p.objective.hessian.entries.clear();
        }
      }
    }
    // More retained equality rows than coordinates is legal redundancy,
    // not evidence of inconsistency. Keep the nonlinear zero-normal row too.
    QCQPProblem redundant;redundant.name="anonymous_redundant_quadratics";redundant.n=1;
    redundant.x0={1};redundant.lb={-INFINITY};redundant.ub={INFINITY};
    redundant.objective.linear={0};redundant.objective.hessian.n=1;
    QuadraticFunction sphere;sphere.constant=-1;sphere.linear={0};sphere.hessian.n=1;
    sphere.hessian.entries={{0,0,2}};redundant.equalities={sphere,sphere};
    QuadraticFunction zero;zero.constant=1;zero.linear={-2};zero.hessian.n=1;
    zero.hessian.entries={{0,0,2}};redundant.equalities.push_back(zero);
    st.convexification_study="HH";st.dca_engine="majorants";
    LazyNewtonStatistics diagnostics;detail::NewtonCostLedger costs(diagnostics,Clock::now());
    Workspace workspace(redundant,st,diagnostics,costs);
    require(workspace.me==3&&workspace.mr==3&&workspace.nr==1,
        "redundant or zero-normal quadratic equality was discarded");
    for(const auto& q:workspace.equalities)require(q.value==0,"feasible original equality changed");
    std::cout<<"h4_domain_certificate PASS checks="<<checks<<" dense=1 indexed=1 nonzero_gradient_rejected=1 true_optimum=1 finite_fixed_control=1\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
