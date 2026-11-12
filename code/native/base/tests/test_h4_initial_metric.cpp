// Exercise the numerical model factory, not only its lower-level curvature
// constructor: a selection safety factor must not inflate an installed rho.
#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main() {
  int checks=0;
  auto require=[&](bool value,const char* message){++checks;if(!value)throw std::runtime_error(message);};
  try {
    for(const std::string mode:{"HH","HS","SH"})for(double safety:{1.,1.1,2.}) {
      Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
      st.linear_backend="mumps";st.linear_metis=true;st.dca_pair_elimination=false;
      st.convexification_study=mode;st.convexification_schedule="lazy";
      st.study_hybrid_scalar_safety=safety;
      QCQPProblem p;p.name="certified_initial_metric";p.n=3;p.x0={1,1,.1};
      p.lb={1,1,-INFINITY};p.ub={INFINITY,INFINITY,INFINITY};
      p.objective.linear={1,1,0};p.objective.hessian.n=3;
      QuadraticFunction g;g.linear={0,0,0};g.hessian.n=3;
      g.hessian.entries={{0,0,2},{0,1,2},{1,1,2},{2,2,-2}};
      p.inequalities={g};p.validate();
      LazyNewtonStatistics native_stats;DcaFilterStatistics model_stats;
      detail::NewtonCostLedger ledger(native_stats,std::chrono::steady_clock::now());
      Workspace w(p,st,native_stats,ledger);DcaInequalityModels models(w,model_stats);
      models.study_prepare();
      for(bool objective:{false,true}) {
        auto c=models.study_construct(w.inequalities[0],objective);
        require(c.rho==2,"exact initial scalar was inflated by selection safety");
        require(c.hybrid_atoms.empty()&&c.atoms==0,"initial metric constructed atoms eagerly");
        require(study::hybrid_certificate(c,c.rho),"initial model or correction is not PSD");
      }
      st.study_hybrid_margin=.5;
      Quadratic convex=w.objective;
      convex.hessian={{0,0,1},{0,1,2},{1,1,4}}; // rank-one PSD, not affine
      for(bool objective:{false,true})for(const Quadratic* q:{&w.objective,&convex}) {
        auto c=models.study_construct(*q,objective);
        require(c.original_convex&&c.rho==0&&c.atoms==0,"convex function received initial curvature");
        require(study::hybrid_certificate(c,0),"uncorrected convex source lost its PSD certificate");
      }
    }
    // A non-DD but globally SPD objective must supply its own certified
    // model margin in every H/S mixed role, without changing C_f or C_g.
    for(const std::string mode:{"HH","HS","SH"}) {
      Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
      st.linear_backend="mumps";st.linear_metis=true;st.dca_pair_elimination=false;
      st.convexification_study=mode;st.convexification_schedule="lazy";
      QCQPProblem p;p.name="installed_nonDD_margin";p.n=2;p.x0={.1,.2};p.lb={-5,-5};p.ub={5,5};
      p.objective.linear={0,0};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,1},{0,1,2},{1,1,5}};
      p.inequalities={p.objective};p.validate();LazyNewtonStatistics native_stats;DcaFilterStatistics model_stats;
      detail::NewtonCostLedger ledger(native_stats,std::chrono::steady_clock::now());Workspace w(p,st,native_stats,ledger);
      DcaInequalityModels models(w,model_stats);models.study_prepare();models.study_reanchor();
      study::Correction installed;installed.original=w.objective.model_hessian;installed.support={0,1};
      auto proof=study::certified_objective_implicit_lower(installed,w.hybrid_objective_margin_certificate);
      require(proof->lower>.17*w.objective.scale&&proof->interval_certified,"installed objective did not provide globally proved non-DD margin");
      require(models.study_objective.rho==0&&models.study_objective.hybrid_atoms.empty(),"margin proof changed convex source objective correction");
      require(models.study_inequalities[0].rho==0&&models.study_inequalities[0].hybrid_atoms.empty(),"margin proof changed convex source inequality correction");
      w.hybrid_objective_base_lower=proof->lower;require(w.hybrid_objective_margin(0)==proof->lower,"retained-coordinate proof lost ambient objective margin");
    }
    // A failed first convex model at an infeasible ORIGINAL anchor demands
    // a metric proposal immediately, before any accepted outer step. The
    // temporary batch remains rollback-able and never changes S into H.
    for(const std::string mode:{"HH","HS","SH"})for(bool feasible:{false,true}) {
      Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
      st.linear_backend="mumps";st.linear_metis=true;st.dca_pair_elimination=false;
      st.convexification_study=mode;st.convexification_schedule="lazy";
      QCQPProblem p;p.name="unaccepted_model_recovery";p.n=4;p.x0={1,1,feasible?10.:.1,feasible?10.:.1};
      p.lb={1,1,-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY,INFINITY,INFINITY};
      p.objective.linear={1,1,0,0};p.objective.hessian.n=4;
      QuadraticFunction g;g.linear={0,0,0,0};g.hessian.n=4;
      g.hessian.entries={{0,0,2},{0,1,2},{1,1,2},{2,2,-2},{3,3,-2}};
      p.inequalities={g};p.validate();
      LazyNewtonStatistics native_stats;DcaFilterStatistics model_stats;
      detail::NewtonCostLedger ledger(native_stats,std::chrono::steady_clock::now());
      Workspace w(p,st,native_stats,ledger);DcaInequalityModels models(w,model_stats);
      models.study_prepare();const auto accepted=models.study_metric_digest();const auto anchor=w.x;
      models.study_note_rejection("inner-criterion");
      const bool proposed=models.study_begin_trial();
      require(proposed==(!feasible&&mode!="HS"),"first-model recovery event or role routing incorrect");
      require(models.study_accepted==0&&models.study_commits==0&&w.x==anchor,"metric recovery claimed or changed an accepted outer point");
      require(models.study_objective.rho==0&&models.study_objective.hybrid_atoms.empty(),"recovery corrected a convex objective");
      if(proposed) {
        require(models.study_inequalities[0].hybrid_atoms.size()==2,"model recovery omitted the repeated-eigenvalue batch");
        require(study::hybrid_certificate(models.study_inequalities[0],models.study_inequalities[0].rho),"recovery metric certificate failed");
        models.study_rollback_trial();
      }
      require(!models.study_trial_pending&&models.study_metric_digest()==accepted,"unaccepted recovery metric was not fully rolled back");
    }
    // Ordinary periodic events also need a joint lift when the negative
    // minimum is repeated, even at an entirely feasible original anchor.
    // Exercise both objective and inequality routing in every mixed mode.
    for(const std::string mode:{"HH","HS","SH"})for(bool isotropic:{true,false}) {
      Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
      st.linear_backend="mumps";st.linear_metis=true;st.dca_pair_elimination=false;
      st.convexification_study=mode;st.convexification_schedule="lazy";
      QCQPProblem p;p.name="ordinary_cluster_event";p.n=isotropic?2:3;p.x0=Vec(p.n,0);p.x0[0]=p.x0[1]=1;
      p.lb=Vec(p.n,-5);p.ub=Vec(p.n,5);p.objective.linear=Vec(p.n,0);p.objective.hessian.n=p.n;
      p.objective.hessian.entries={{0,0,-2},{1,1,-2}};
      if(!isotropic)p.objective.hessian.entries.push_back({2,2,1});
      QuadraticFunction g=p.objective;
      p.inequalities={g};p.validate();
      LazyNewtonStatistics native_stats;DcaFilterStatistics model_stats;
      detail::NewtonCostLedger ledger(native_stats,std::chrono::steady_clock::now());
      Workspace w(p,st,native_stats,ledger);DcaInequalityModels models(w,model_stats);
      models.study_prepare();const auto accepted=models.study_metric_digest();const auto anchor=w.x;
      models.study_hybrid_due=true;
      const bool proposed=models.study_begin_trial("normal-periodic-test");
      require(proposed==!isotropic,"same-matrix isotropic exchange or genuine rank-two update misclassified");
      if(isotropic) {
        require(models.study_metric_digest()==accepted&&!models.study_trial_pending,
            "minimal scalar correction consumed an atom trial");
        require(models.study_objective.hybrid_atoms.empty()&&models.study_inequalities[0].hybrid_atoms.empty(),
            "unnecessary isotropic atoms consumed the global budget");
        require(w.x==anchor&&models.study_commits==0,"rejected identical correction moved the original point");
        continue;
      }
      for(bool objective:{true,false}) {
        const auto& c=objective?models.study_objective:models.study_inequalities[0];
        const bool hybrid=study::correction_family(mode,objective)=="hybrid";
        require(c.hybrid_atoms.size()==(hybrid?2:0),"ordinary event violated f/g H/S routing");
        require(study::hybrid_certificate(c,c.rho),"ordinary cluster model is not PSD certified");
      }
      require(w.x==anchor&&models.study_accepted==0&&models.study_commits==0,"ordinary cluster proposal changed the accepted outer point");
      models.study_rollback_trial();
      require(models.study_metric_digest()==accepted&&!models.study_trial_pending,"ordinary unaccepted cluster was not rolled back");
    }
    // Cross-start diversification in the >32-coordinate Lanczos route must
    // benefit every H role while S roles remain scalar. This exact dyadic
    // matrix has two eigenvalues -5 and 62 eigenvalues +1.
    for(const std::string mode:{"HH","HS","SH"}) {
      Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
      st.linear_backend="mumps";st.linear_metis=true;st.dca_pair_elimination=false;
      st.convexification_study=mode;st.convexification_schedule="lazy";st.study_hybrid_atom_support=128;
      QCQPProblem p;p.name="large_ordinary_cluster_event";p.n=64;p.x0=Vec(64,1);
      p.lb=Vec(64,-5);p.ub=Vec(64,5);p.objective.linear=Vec(64,0);p.objective.hessian.n=64;
      for(int i=0;i<64;++i)for(int j=i;j<64;++j) {
        const double u=.125,v_i=i<32?.125:-.125,v_j=j<32?.125:-.125;
        const double q=(i==j?1.:0)-6*(u*u+v_i*v_j);if(q)p.objective.hessian.entries.push_back({i,j,q});
      }
      p.inequalities={p.objective};p.validate();
      LazyNewtonStatistics native_stats;DcaFilterStatistics model_stats;
      detail::NewtonCostLedger ledger(native_stats,std::chrono::steady_clock::now());
      Workspace w(p,st,native_stats,ledger);DcaInequalityModels models(w,model_stats);
      models.study_prepare();const auto accepted=models.study_metric_digest();const auto anchor=w.x;
      models.study_hybrid_due=true;require(models.study_begin_trial("large-cluster-diversified"),"large periodic cluster had no proposal");
      for(bool objective:{true,false}) {
        const auto& c=objective?models.study_objective:models.study_inequalities[0];
        const bool hybrid=study::correction_family(mode,objective)=="hybrid";
        require(c.hybrid_atoms.size()==(hybrid?2:0),"large cluster violated H/S f/g routing");
        require(study::hybrid_certificate(c,c.rho),"large diversified cluster is not PSD certified");
        if(hybrid)require(c.rho<1e-9,"large rank-two batch retained excess scalar after diversification");
        models.study_trace("large_cluster_diversified",objective?"objective":"inequality",objective?-1:0,c);
      }
      require(w.x==anchor&&models.study_accepted==0&&models.study_commits==0,"large metric proposal claimed an accepted point");
      models.study_rollback_trial();require(models.study_metric_digest()==accepted,"large diversified trial failed complete rollback");
    }
    // Both H and support-S roles refine a global scalar certificate beyond
    // Gershgorin on a genuine 20-coordinate indefinite, non-DD function.
    for(const std::string mode:{"HH","HS","SH"}) {
      Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
      st.linear_backend="mumps";st.linear_metis=true;st.dca_pair_elimination=false;
      st.convexification_study=mode;st.convexification_schedule="lazy";
      QCQPProblem p;p.name="medium_support_scalar_refinement";p.n=20;p.x0=Vec(20,.1);
      p.lb=Vec(20,-5);p.ub=Vec(20,5);p.objective.linear=Vec(20,0);p.objective.hessian.n=20;
      for(int i=0;i<20;++i){const double d=i%2?1:2;p.objective.hessian.entries.push_back({i,i,d*d+(i?2.25:0)-2});if(i+1<20)p.objective.hessian.entries.push_back({i,i+1,1.5*d});}
      p.inequalities={p.objective};p.validate();
      LazyNewtonStatistics native_stats;DcaFilterStatistics model_stats;
      detail::NewtonCostLedger ledger(native_stats,std::chrono::steady_clock::now());
      Workspace w(p,st,native_stats,ledger);DcaInequalityModels models(w,model_stats);models.study_prepare();
      for(bool objective:{true,false}) {
        auto c=objective?models.study_objective:models.study_inequalities[0];const double old=c.rho;
        c.rho=study::hybrid_certified_residual(c,c.rho);
        require(c.rho<old&&c.rho>0,"medium support f/g scalar certificate was not refined");
        require(study::hybrid_certificate(c,c.rho),"medium support refined f/g model lost certification");
        require(c.hybrid_atoms.empty(),"scalar refinement constructed an unrequested atom");
        require(c.family==study::correction_family(mode,objective),"medium support refinement changed f/g role family");
        require(study::exact_psd_shift(study::dense(c.original,c.support),20,c.rho),"actual f/g scalar certificate disagrees with exact Bareiss");
        models.study_trace("medium_interval_refine",objective?"objective":"inequality",objective?-1:0,c);
      }
    }
    // A failed first lazy construction must keep its original error. Its
    // destructor must not index a correction vector which was never built.
    for(const std::string mode:{"HH","HS","SH"}) {
      bool exact_error=false;
      {
        Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
        st.linear_backend="mumps";st.linear_metis=true;st.dca_pair_elimination=false;
        st.convexification_study=mode;st.convexification_schedule="lazy";
        QCQPProblem p;p.name="first_construction_error";p.n=1;p.x0={0};
        p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={0};p.objective.hessian.n=1;
        QuadraticFunction g;g.linear={0};g.hessian.n=1;g.hessian.entries={{0,0,-2}};
        p.inequalities={g};p.validate();
        LazyNewtonStatistics native_stats;DcaFilterStatistics model_stats;
        detail::NewtonCostLedger ledger(native_stats,std::chrono::steady_clock::now());
        Workspace w(p,st,native_stats,ledger);DcaInequalityModels models(w,model_stats);
        Quadratic invalid=w.objective;invalid.hessian={{0,0,NAN}};
        try {models.study_construct(invalid,true);}
        catch(const std::runtime_error& error) {
          exact_error=std::string(error.what())=="study nonfinite curvature";
        }
      }
      require(exact_error,"diagnostic destructor obscured the initial curvature error");
    }
    // Exact geometric counterexample: radii r,t >= 1, free x, anchor
    // (1,1,.1), source g=(r+t)^2-x^2. Inflating the exact rho=2 to 2.2
    // creates an impossible model, since its global lower bound is positive.
    using R=study::Rational;const R a=R(.1),inflated=R(2.2);
    const R lower=4-inflated/(inflated-2)*a*a;
    require(lower>0,"counterexample does not prove artificial model infeasibility");
    // With the certified rho=2 the model is affine in x. The point (1,1,30)
    // satisfies it, while both Q+2I and 2I are PSD by the factory checks above.
    const R trial=4-R(30)*30+(R(30)-a)*(R(30)-a);
    require(trial<0,"minimal certified correction did not restore model feasibility");
    std::cout<<"h4_initial_metric PASS checks="<<checks
      <<" actual_factory=1 HH_HS_SH=1 objective_and_inequality=1 PSD_and_C0=1 no_safety_inflation=1 exact_infeasibility_counterexample=1 first_error_preserved=1 first_model_event_and_rollback=1 normal_cluster_f_g=1 medium_support_f_g=1\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
