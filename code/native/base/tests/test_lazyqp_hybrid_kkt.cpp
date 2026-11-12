#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

constexpr int kVariables=12;
constexpr int kEqualities=1;
constexpr int kDenseCandidates=3;
constexpr int kRetainedRows=2;
constexpr int kSparseRows=2*kVariables;
constexpr int kConstantRows=8;
constexpr int kInequalities=kDenseCandidates+kSparseRows+kConstantRows;
constexpr int kLowRankBorders=1;

ConvexQuadraticModel linear_row(double constant,const Vec& linear) {
  ConvexQuadraticModel row;
  row.constant=constant;
  row.linear=linear;
  row.base_hessian.n=kVariables;
  return row;
}

ConvexSubproblem make_problem() {
  ConvexSubproblem p;
  p.nvar=kVariables;
  p.original_dimension=kVariables;
  p.prefer_feasible_centered_start=true;

  p.objective.constant=0.0;
  p.objective.linear.resize(kVariables);
  p.objective.base_hessian.n=kVariables;
  for(int j=0;j<kVariables;++j) {
    p.objective.linear[static_cast<std::size_t>(j)]=
        -0.04*static_cast<double>(j+1);
    p.objective.base_hessian.entries.push_back({j,j,1.0+0.02*j});
  }
  // Keep one dense positive-semidefinite term as an exact KKT border.  The
  // hybrid retained-inequality columns must be inserted before this border,
  // without changing its algebra or offset.
  Vec border_direction(kVariables,0.0);
  for(int j=0;j<kVariables;++j)
    border_direction[static_cast<std::size_t>(j)]=
        (j%2==0 ? 1.0 : -0.5)/std::sqrt(static_cast<double>(kVariables));
  p.objective.low_rank.push_back({0.35,std::move(border_direction)});

  p.equalities.nvar=kVariables;
  p.equalities.rows.push_back({{{0,1.0},{1,1.0}}});
  p.equalities.rhs.push_back(0.0);

  // Three rows are eligible, but only the two full-support rows maximize the
  // structural saving.  The third has eight entries and proves that selection
  // is top-k rather than first-k.  All are inactive at the minimizer.
  Vec dense_a(kVariables,0.0),dense_b(kVariables,0.0),dense_c(kVariables,0.0);
  for(int j=0;j<kVariables;++j) {
    dense_a[static_cast<std::size_t>(j)]=0.025*(j+1);
    dense_b[static_cast<std::size_t>(j)]=
        (j%2==0 ? 0.035 : -0.030)*(j+1);
    if(j<8) dense_c[static_cast<std::size_t>(j)]=0.02*(j+1);
  }
  p.inequalities.push_back(linear_row(-3.0,dense_a));
  p.inequalities.push_back(linear_row(-3.0,dense_c));
  p.inequalities.push_back(linear_row(-3.0,dense_b));

  // Sparse box rows exercise the eliminated Schur-complement path in the
  // same factorization as the two retained rows.
  for(int j=0;j<kVariables;++j) {
    Vec upper(kVariables,0.0),lower(kVariables,0.0);
    upper[static_cast<std::size_t>(j)]=1.0;
    lower[static_cast<std::size_t>(j)]=-1.0;
    p.inequalities.push_back(linear_row(-1.5,upper));
    p.inequalities.push_back(linear_row(-1.5,lower));
  }

  // Zero-gradient, strictly inactive rows make the full/reduced dimension
  // contrast material while remaining ineligible for hybrid retention.
  for(int i=0;i<kConstantRows;++i)
    p.inequalities.push_back(linear_row(-2.0,Vec(kVariables,0.0)));

  if(static_cast<int>(p.inequalities.size())!=kInequalities)
    throw std::runtime_error("synthetic inequality count is inconsistent");
  return p;
}

Settings settings(const std::string& form) {
  Settings st;
  st.verbose=false;
  st.max_inner_iterations=300;
  st.inner_tolerance=1e-9;
  st.inner_iteration_limit_mu_fallback=1e-8;
  st.lazyqp_kkt_form=form;
  st.lazyqp_low_rank_form="bordered";
  st.lazyqp_numeric_kkt_reuse=false;
  st.lazyqp_hybrid_kkt_max_retained_inequalities=kRetainedRows;
  st.lazyqp_hybrid_kkt_min_row_nnz=8;
  return st;
}

void require_certified(const std::string& label,const InnerResult& result) {
  if(!result.success)
    throw std::runtime_error(label+" solve failed: "+result.message);
  if(result.dual_inf>2e-7 || result.equality_inf>2e-8 ||
     result.inequality_inf>2e-8 || result.primal_residual_inf>2e-8 ||
     result.complementarity>2e-7) {
    std::ostringstream out;
    out << label << " residual certificate is too large: dual="
        << result.dual_inf << " eq=" << result.equality_inf
        << " ineq=" << result.inequality_inf
        << " pri=" << result.primal_residual_inf
        << " comp=" << result.complementarity;
    throw std::runtime_error(out.str());
  }
}

void require_equivalent(const ConvexSubproblem& problem,
                        const std::string& label,
                        const InnerResult& reference,
                        const InnerResult& candidate) {
  if(reference.y.size()!=candidate.y.size())
    throw std::runtime_error(label+" solution dimension mismatch");
  double distance=0.0;
  for(std::size_t j=0;j<reference.y.size();++j)
    distance=std::max(distance,std::abs(reference.y[j]-candidate.y[j]));
  const double reference_objective=problem.objective.value(reference.y);
  const double candidate_objective=problem.objective.value(candidate.y);
  const double objective_error=std::abs(reference_objective-candidate_objective);
  const double residual_error=std::max({
      std::abs(reference.dual_inf-candidate.dual_inf),
      std::abs(reference.equality_inf-candidate.equality_inf),
      std::abs(reference.inequality_inf-candidate.inequality_inf),
      std::abs(reference.primal_residual_inf-candidate.primal_residual_inf),
      std::abs(reference.complementarity-candidate.complementarity)});
  if(distance>3e-7 || objective_error>2e-8 || residual_error>2e-7) {
    std::ostringstream out;
    out << label << " is not algebraically equivalent to full KKT: dy="
        << distance << " dobj=" << objective_error
        << " dres=" << residual_error;
    throw std::runtime_error(out.str());
  }
}

bool has_compiled_backend(const std::string& requested) {
  std::istringstream backends(compiled_linear_backends());
  std::string backend;
  while(std::getline(backends,backend,','))
    if(backend==requested) return true;
  return false;
}

} // namespace

int main() {
  try {
    const ConvexSubproblem problem=make_problem();
    const InnerResult full=
        ConvexQCQPLazyQPKernel(settings("full")).solve(problem);
    const InnerResult reduced=
        ConvexQCQPLazyQPKernel(settings("reduced")).solve(problem);
    Settings hybrid_settings=settings("hybrid");
    hybrid_settings.lazyqp_model_support_fastpath=true;
    const InnerResult hybrid=
        ConvexQCQPLazyQPKernel(hybrid_settings).solve(problem);

    require_certified("full",full);
    require_certified("reduced",reduced);
    require_certified("hybrid",hybrid);
    require_equivalent(problem,"reduced",full,reduced);
    require_equivalent(problem,"hybrid",full,hybrid);

    const int expected_full_dimension=
        kVariables+kEqualities+kInequalities+kLowRankBorders;
    const int expected_reduced_dimension=
        kVariables+kEqualities+kLowRankBorders;
    const int expected_hybrid_dimension=
        expected_reduced_dimension+kRetainedRows;
    if(full.lazyqp_kkt_dimension!=expected_full_dimension ||
       full.lazyqp_full_kkt_dimension!=expected_full_dimension ||
       full.lazyqp_full_kkt_factorizations<1 ||
       full.lazyqp_reduced_kkt_factorizations!=0 ||
       full.lazyqp_hybrid_kkt_factorizations!=0)
      throw std::runtime_error("forced full KKT telemetry/dimension mismatch");
    if(reduced.lazyqp_kkt_dimension!=expected_reduced_dimension ||
       reduced.lazyqp_full_kkt_dimension!=expected_full_dimension ||
       reduced.lazyqp_reduced_kkt_factorizations<1 ||
       reduced.lazyqp_full_kkt_factorizations!=0 ||
       reduced.lazyqp_reduced_kkt_fallbacks!=0 ||
       reduced.lazyqp_hybrid_kkt_factorizations!=0)
      throw std::runtime_error("forced reduced KKT telemetry/dimension mismatch");
    if(hybrid.lazyqp_kkt_dimension!=expected_hybrid_dimension ||
       hybrid.lazyqp_full_kkt_dimension!=expected_full_dimension ||
       hybrid.lazyqp_reduced_kkt_factorizations<1 ||
       hybrid.lazyqp_full_kkt_factorizations!=0 ||
       hybrid.lazyqp_reduced_kkt_fallbacks!=0 ||
       hybrid.lazyqp_hybrid_kkt_fallbacks!=0 ||
       hybrid.lazyqp_hybrid_kkt_attempts<1 ||
       hybrid.lazyqp_hybrid_kkt_factorizations!=
           hybrid.lazyqp_hybrid_kkt_attempts ||
       hybrid.lazyqp_hybrid_kkt_retained_rows_max!=kRetainedRows ||
       hybrid.lazyqp_hybrid_kkt_retained_rows_total!=
           kRetainedRows*hybrid.lazyqp_hybrid_kkt_attempts)
      throw std::runtime_error("forced hybrid KKT telemetry/dimension mismatch");

    if(hybrid.lazyqp_model_support_fastpath_requests!=1 ||
       hybrid.lazyqp_model_support_fastpath_effective_subproblems!=1 ||
       hybrid.lazyqp_model_support_value_evaluations<=0 ||
       hybrid.lazyqp_model_support_gradient_evaluations<=0)
      throw std::runtime_error("hybrid/model-support interaction was vacuous");

    const long long dense_outer_products_per_factor=
        kRetainedRows*kVariables*(kVariables+1LL)/2LL;
    const long long retained_border_triplets_per_factor=
        kRetainedRows*(kVariables+1LL);
    if(hybrid.lazyqp_hybrid_kkt_outer_product_triplets_avoided!=
           dense_outer_products_per_factor*
               hybrid.lazyqp_hybrid_kkt_attempts ||
       hybrid.lazyqp_hybrid_kkt_border_triplets_emitted!=
           retained_border_triplets_per_factor*
               hybrid.lazyqp_hybrid_kkt_attempts)
      throw std::runtime_error("hybrid triplet-accounting counters are not exact");

    Settings no_candidate_settings=settings("hybrid");
    no_candidate_settings.lazyqp_hybrid_kkt_min_row_nnz=kVariables+1;
    const InnerResult no_candidate=
        ConvexQCQPLazyQPKernel(no_candidate_settings).solve(problem);
    require_certified("hybrid no-candidate fallback",no_candidate);
    require_equivalent(problem,"hybrid no-candidate fallback",full,no_candidate);
    if(no_candidate.lazyqp_kkt_dimension!=expected_reduced_dimension ||
       no_candidate.lazyqp_hybrid_kkt_attempts!=0 ||
       no_candidate.lazyqp_hybrid_kkt_factorizations!=0 ||
       no_candidate.lazyqp_hybrid_kkt_fallbacks!=1)
      throw std::runtime_error("hybrid no-candidate fallback telemetry mismatch");

    // Do not use substring matching here: "qdldl" contains "ldl", but it
    // does not make the historical SuiteSparse LDL backend available.
    if(has_compiled_backend("ldl")) {
      Settings ldl_settings=settings("hybrid");
      ldl_settings.linear_backend="ldl";
      ldl_settings.kkt_adaptive_regularization=true;
      const InnerResult ldl=
          ConvexQCQPLazyQPKernel(ldl_settings).solve(problem);
      require_certified("hybrid adaptive LDL",ldl);
      require_equivalent(problem,"hybrid adaptive LDL",full,ldl);
      if(ldl.lazyqp_hybrid_kkt_factorizations<1 ||
         ldl.lazyqp_hybrid_kkt_fallbacks!=0 ||
         ldl.lazyqp_kkt_dimension!=expected_hybrid_dimension)
        throw std::runtime_error("hybrid adaptive LDL telemetry mismatch");
    }

    std::cout << "[PASS] exact hybrid inequality KKT matches full/reduced"
              << " dimensions=" << full.lazyqp_kkt_dimension << "/"
              << reduced.lazyqp_kkt_dimension << "/"
              << hybrid.lazyqp_kkt_dimension
              << " factors=" << full.factorizations << "/"
              << reduced.factorizations << "/" << hybrid.factorizations
              << " retained=" << hybrid.lazyqp_hybrid_kkt_retained_rows_max
              << " avoided="
              << hybrid.lazyqp_hybrid_kkt_outer_product_triplets_avoided
              << " emitted="
              << hybrid.lazyqp_hybrid_kkt_border_triplets_emitted << "\n";
    return 0;
  } catch(const std::exception& exception) {
    std::cerr << "[FAIL] " << exception.what() << "\n";
    return 1;
  }
}
