#include "lazyipdca_qcqp/linear_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lazyipdca::qcqp;

namespace {

void require(bool condition,const std::string& message) {
  if(!condition) throw std::runtime_error(message);
}

double residual_inf(int n,const std::vector<SymmetricTriplet>& entries,
                    const Vec& x,const Vec& rhs) {
  Vec residual=rhs;
  for(const auto& entry:entries) {
    require(entry.i>=0 && entry.i<n && entry.j>=0 && entry.j<n,
            "invalid test triplet");
    residual[static_cast<std::size_t>(entry.i)] -=
        entry.value*x[static_cast<std::size_t>(entry.j)];
    if(entry.i!=entry.j)
      residual[static_cast<std::size_t>(entry.j)] -=
          entry.value*x[static_cast<std::size_t>(entry.i)];
  }
  double value=0.0;
  for(double component:residual) value=std::max(value,std::abs(component));
  return value;
}

void factor_and_check(SymmetricFactorization& factor,int n,int primal,
                      const std::vector<SymmetricTriplet>& entries,
                      const std::vector<SymmetricTriplet>* effective=nullptr) {
  std::string message;
  require(factor.factor(n,primal,entries,message),
          "LDL factorization failed: "+message);
  Vec rhs(static_cast<std::size_t>(n),0.5);
  if(n>0) rhs[0]=1.0;
  if(n>1) rhs[1]=-2.0;
  Vec solution=rhs;
  require(factor.solve(solution,message),"LDL solve failed: "+message);
  require(residual_inf(n,effective ? *effective : entries,solution,rhs)<=1e-10,
          "cached scatter changed the numerical matrix");
}

} // namespace

int main() {
  try {
    if(!linear_backend_compiled("ldl")) {
      std::cout << "[SKIP] SuiteSparse LDL+METIS not compiled\n";
      return 0;
    }

    Settings settings;
    settings.linear_backend="ldl";
    settings.linear_metis=true;
    // The test supplies every numerical regularization entry explicitly.
    settings.kkt_adaptive_regularization=true;
    settings.kkt_iterative_refinement_steps=2;

    std::string message;
    auto factor=make_symmetric_factorization(settings,message);
    require(static_cast<bool>(factor),"LDL factory failed: "+message);

    // Deliberately unsorted and duplicated.  The two (0,0) contributions
    // verify that the positional scatter retains aggregate_upper's stable
    // floating-point accumulation order.
    const std::vector<SymmetricTriplet> first={
        {2,2,-2.0},{0,1,0.5},{0,0,1.5},{0,2,1.0},
        {1,1,3.0},{0,0,2.5}};
    factor_and_check(*factor,3,2,first);
    require(factor->numeric_scatter_builds()==1,
            "first factorization did not build one positional scatter");
    require(factor->numeric_scatter_reuses()==0 &&
            factor->numeric_scatter_invalidations()==0,
            "first factorization reported spurious scatter reuse/invalidation");

    auto same_layout=first;
    same_layout[0].value=-1.5;
    same_layout[2].value=2.0;
    same_layout[4].value=4.0;
    same_layout[5].value=3.0;
    factor_and_check(*factor,3,2,same_layout);
    require(factor->last_numeric_scatter_reused() &&
            factor->numeric_scatter_builds()==1 &&
            factor->numeric_scatter_reuses()==1 &&
            factor->numeric_scatter_invalidations()==0,
            "unchanged triplet layout did not hit the positional scatter");

    // A temporary numerical zero must remain a value-only update.  It must
    // neither evict the symbolic envelope nor the positional scatter.
    auto temporary_zero=same_layout;
    temporary_zero[1].value=0.0;
    factor_and_check(*factor,3,2,temporary_zero);
    require(factor->last_numeric_scatter_reused() &&
            factor->numeric_scatter_reuses()==2 &&
            factor->symbolic_invalidations()==0,
            "temporary zero invalidated a structural cache");

    auto restored=temporary_zero;
    restored[1].value=-0.25;
    factor_and_check(*factor,3,2,restored);
    require(factor->last_numeric_scatter_reused() &&
            factor->numeric_scatter_reuses()==3,
            "reactivated value missed the cached scatter");

    // A genuinely new graph entry must fail closed: rebuild the scatter and
    // invalidate/reanalyse LDL's symbolic graph exactly once.
    auto grown=restored;
    grown.push_back({1,2,0.2});
    const int symbolic_before=factor->symbolic_analyses();
    factor_and_check(*factor,3,2,grown);
    require(!factor->last_numeric_scatter_reused() &&
            factor->numeric_scatter_builds()==2 &&
            factor->numeric_scatter_invalidations()==1,
            "new structural entry did not rebuild the positional scatter");
    require(factor->symbolic_analyses()==symbolic_before+1 &&
            factor->symbolic_invalidations()==1,
            "new structural entry did not invalidate LDL symbolic analysis");

    auto grown_values=grown;
    grown_values.back().value=-0.1;
    factor_and_check(*factor,3,2,grown_values);
    require(factor->last_numeric_scatter_reused() &&
            factor->numeric_scatter_reuses()==4,
            "grown graph did not reuse its rebuilt scatter");

    // Reordering the same graph changes only the caller layout.  The scatter
    // is rebuilt, but the monotone matrix envelope and symbolic LDL analysis
    // remain valid.
    auto reordered=grown_values;
    std::swap(reordered[0],reordered[1]);
    const int analyses_before_reorder=factor->symbolic_analyses();
    const int symbolic_invalidations_before_reorder=
        factor->symbolic_invalidations();
    factor_and_check(*factor,3,2,reordered);
    require(factor->numeric_scatter_builds()==3 &&
            factor->numeric_scatter_invalidations()==2,
            "raw-layout reorder did not rebuild only the scatter");
    require(factor->symbolic_analyses()==analyses_before_reorder &&
            factor->symbolic_invalidations()==
                symbolic_invalidations_before_reorder,
            "raw-layout reorder unnecessarily invalidated symbolic LDL");

    // The non-adaptive backend appends fixed primal/dual diagonal shifts
    // internally.  Their precomputed scatter must reproduce that effective
    // matrix on every value-only update as well.
    Settings fixed_settings=settings;
    fixed_settings.kkt_adaptive_regularization=false;
    fixed_settings.kkt_primal_regularization=0.25;
    fixed_settings.kkt_dual_regularization=0.5;
    auto fixed=make_symmetric_factorization(fixed_settings,message);
    require(static_cast<bool>(fixed),"fixed-shift LDL factory failed: "+message);
    const std::vector<SymmetricTriplet> fixed_input={
        {0,0,2.0},{0,1,1.0},{1,1,-1.0}};
    auto fixed_effective=fixed_input;
    fixed_effective.push_back({0,0,0.25});
    fixed_effective.push_back({1,1,-0.5});
    factor_and_check(*fixed,2,1,fixed_input,&fixed_effective);
    auto fixed_update=fixed_input;
    fixed_update[0].value=3.0;
    fixed_update[2].value=-2.0;
    auto fixed_update_effective=fixed_update;
    fixed_update_effective.push_back({0,0,0.25});
    fixed_update_effective.push_back({1,1,-0.5});
    factor_and_check(*fixed,2,1,fixed_update,&fixed_update_effective);
    require(fixed->numeric_scatter_builds()==1 &&
            fixed->numeric_scatter_reuses()==1 &&
            fixed->numeric_scatter_invalidations()==0,
            "fixed regularization did not use its diagonal scatter");

    std::cout << "[PASS] LDL positional numerical scatter"
              << " builds="<<factor->numeric_scatter_builds()
              << " reuses="<<factor->numeric_scatter_reuses()
              << " invalidations="<<factor->numeric_scatter_invalidations()
              << " symbolic="<<factor->symbolic_analyses()<<"/"
              << factor->symbolic_reuses()<<"/"
              << factor->symbolic_invalidations()
              << " fixed_reuses="<<fixed->numeric_scatter_reuses()<<"\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "[FAIL] "<<error.what()<<"\n";
    return 1;
  }
}
