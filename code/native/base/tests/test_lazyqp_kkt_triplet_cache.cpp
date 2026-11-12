#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

ConvexQuadraticModel linear_row(double constant,double a0,double a1) {
  ConvexQuadraticModel q;
  q.constant=constant;
  q.linear={a0,a1};
  q.base_hessian.n=2;
  return q;
}

ConvexSubproblem make_problem(bool changed_structure=false) {
  ConvexSubproblem p;
  p.nvar=2;
  p.original_dimension=2;
  p.prefer_feasible_centered_start=true;
  p.objective.linear={-0.20,-0.10};
  p.objective.base_hessian.n=2;
  p.objective.base_hessian.entries={{0,0,1.0},{1,1,1.0}};
  if(changed_structure)
    p.objective.base_hessian.entries.push_back({0,1,0.10});
  // Exercise the case118 representation: inequality-reduced with an exact
  // low-rank border instead of dense condensation.
  p.objective.low_rank.push_back({0.25,{1.0,0.0}});
  p.equalities.nvar=2;
  p.inequalities={linear_row(-1.0,+1.0,0.0),
                  linear_row(-1.0,-1.0,0.0),
                  linear_row(-1.0,0.0,+1.0),
                  linear_row(-1.0,0.0,-1.0)};
  return p;
}

Settings settings(const std::string& kkt_form,
                  const std::string& low_rank_form) {
  Settings st;
  st.verbose=false;
  st.max_inner_iterations=100;
  st.inner_tolerance=1e-9;
  st.inner_iteration_limit_mu_fallback=1e-8;
  st.lazyqp_low_rank_form=low_rank_form;
  st.lazyqp_kkt_form=kkt_form;
  st.lazyqp_numeric_kkt_reuse=false;
  st.kkt_adaptive_regularization=true;
  return st;
}

void require_same_solution(const InnerResult& left,const InnerResult& right) {
  if(left.y.size()!=right.y.size())
    throw std::runtime_error("cached/reference solution dimension mismatch");
  for(std::size_t i=0;i<left.y.size();++i)
    if(std::abs(left.y[i]-right.y[i])>2e-8)
      throw std::runtime_error("triplet cache changed the minimizer");
  if(std::abs(left.dual_inf-right.dual_inf)>2e-8 ||
     std::abs(left.primal_residual_inf-right.primal_residual_inf)>2e-8 ||
     std::abs(left.complementarity-right.complementarity)>2e-8)
    throw std::runtime_error("triplet cache changed final KKT residuals");
}

void exercise_form(const std::string& kkt_form,
                   const std::string& low_rank_form) {
  Settings cached_settings=settings(kkt_form,low_rank_form);
  ConvexQCQPLazyQPKernel cached(cached_settings);
  const std::string label=kkt_form+"/"+low_rank_form;

  const InnerResult first=cached.solve(make_problem(false));
  const InnerResult second=cached.solve(make_problem(false));
  if(!first.success || !second.success)
    throw std::runtime_error(label+" cache regression solve failed");
  if(first.lazyqp_kkt_triplet_layout_builds!=1 ||
     first.lazyqp_kkt_triplet_layout_reuses<1 ||
     first.lazyqp_kkt_triplet_layout_invalidations!=0)
    throw std::runtime_error(
        label+" first solve did not build once then update values in-place");
  if(second.lazyqp_kkt_triplet_layout_builds!=0 ||
     second.lazyqp_kkt_triplet_layout_reuses<1 ||
     second.lazyqp_kkt_triplet_layout_invalidations!=0)
    throw std::runtime_error(label+" layout did not persist across solves");
  require_same_solution(first,second);

  // Add one same-dimension PSD Hessian coordinate.  The old cache must be
  // invalidated before the new layout is built and used.
  const ConvexSubproblem changed=make_problem(true);
  const InnerResult rebuilt=cached.solve(changed);
  Settings reference_settings=settings(kkt_form,low_rank_form);
  ConvexQCQPLazyQPKernel reference(reference_settings);
  const InnerResult fresh=reference.solve(changed);
  if(!rebuilt.success || !fresh.success)
    throw std::runtime_error(label+" structural-change reference solve failed");
  if(rebuilt.lazyqp_kkt_triplet_layout_invalidations!=1 ||
     rebuilt.lazyqp_kkt_triplet_layout_builds!=1 ||
     rebuilt.lazyqp_kkt_triplet_layout_reuses<1)
    throw std::runtime_error(
        label+" structural change was not rebuilt exactly once");
  require_same_solution(rebuilt,fresh);

  cached.clear_factorization_cache();
  const InnerResult after_clear=cached.solve(changed);
  if(!after_clear.success ||
     after_clear.lazyqp_kkt_triplet_layout_builds!=1 ||
     after_clear.lazyqp_kkt_triplet_layout_invalidations!=0)
    throw std::runtime_error(label+" explicit clear did not force one build");
}

ConvexSubproblem make_affine_problem() {
  ConvexSubproblem p;
  p.nvar=3;
  p.original_dimension=3;
  p.prefer_feasible_centered_start=true;
  p.objective.linear={-0.20,-0.10,0.0};
  p.objective.base_hessian.n=3;
  p.objective.base_hessian.entries={{0,0,1.0},{1,1,1.0},{2,2,1.0}};
  p.objective.low_rank.push_back({0.25,{1.0,0.0,0.0}});
  p.equalities.nvar=3;
  p.equalities.rows.push_back({{{0,1.0},{2,1.0}}});
  p.equalities.rhs.push_back(0.0);
  auto row3=[](double constant,double a0,double a1) {
    ConvexQuadraticModel row;
    row.constant=constant;
    row.linear={a0,a1,0.0};
    row.base_hessian.n=3;
    return row;
  };
  p.inequalities={row3(-1.0,+1.0,0.0),row3(-1.0,-1.0,0.0),
                  row3(-1.0,0.0,+1.0),row3(-1.0,0.0,-1.0)};
  p.exact_affine_eliminations={{0,2}};
  return p;
}

void exercise_affine_reduced_and_full() {
  Settings st=settings("full","condensed");
  st.lazyqp_flow_lift_elimination=true;
  ConvexQCQPLazyQPKernel kernel(st);
  const ConvexSubproblem affine=make_affine_problem();
  const InnerResult reduced_first=kernel.solve(affine);
  const InnerResult reduced_second=kernel.solve(affine);
  if(!reduced_first.success || !reduced_second.success ||
     reduced_first.lazyqp_affine_elimination_acceptances!=1 ||
     reduced_second.lazyqp_affine_elimination_acceptances!=1 ||
     reduced_first.lazyqp_kkt_triplet_layout_builds!=1 ||
     reduced_first.lazyqp_kkt_triplet_layout_reuses<1 ||
     reduced_second.lazyqp_kkt_triplet_layout_builds!=0 ||
     reduced_second.lazyqp_kkt_triplet_layout_reuses<1)
    throw std::runtime_error(
        "affine-reduced full-KKT layout was not persisted");
  require_same_solution(reduced_first,reduced_second);

  ConvexSubproblem full=affine;
  full.exact_affine_eliminations.clear();
  const InnerResult full_result=kernel.solve(full);
  if(!full_result.success ||
     full_result.lazyqp_affine_elimination_attempts!=0 ||
     full_result.lazyqp_kkt_triplet_layout_invalidations!=1 ||
     full_result.lazyqp_kkt_triplet_layout_builds!=1 ||
     full_result.lazyqp_kkt_triplet_layout_reuses<1)
    throw std::runtime_error(
        "transition from affine-reduced to full layout did not rebuild safely");
}

} // namespace

int main() {
  try {
    exercise_form("reduced","bordered");
    exercise_form("reduced","condensed");
    exercise_form("full","bordered");
    exercise_form("full","condensed");
    exercise_affine_reduced_and_full();
    std::cout << "[PASS] persistent KKT triplet layouts"
              << " forms=full/reduced x bordered/condensed"
              << " affine=reduced/full\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "[FAIL] "<<error.what()<<"\n";
    return 1;
  }
}
