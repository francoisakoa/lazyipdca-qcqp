#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/version.hpp"
#include "qcq_bound_sentinels.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace lazyipdca::qcqp {
namespace {

void expect(std::istream& in, const std::string& wanted) {
  std::string got;
  if (!(in >> got) || got != wanted) {
    throw std::runtime_error("QCQ parse error: expected '" + wanted + "', got '" + got + "'");
  }
}

double parse_double_token(const std::string& s) {
  if (s == "INF" || s == "+INF" || s == "inf" || s == "+inf")
    return std::numeric_limits<double>::infinity();
  if (s == "-INF" || s == "-inf") return -std::numeric_limits<double>::infinity();
  std::size_t pos = 0;
  const double v = std::stod(s, &pos);
  if (pos != s.size()) throw std::runtime_error("QCQ parse error: bad floating token '" + s + "'");
  return v;
}

double read_double(std::istream& in) {
  std::string s;
  if (!(in >> s)) throw std::runtime_error("QCQ parse error: missing floating value");
  return parse_double_token(s);
}

Vec read_vec(std::istream& in, int n) {
  Vec x(n, 0.0);
  for (int i = 0; i < n; ++i) x[i] = read_double(in);
  return x;
}

QuadraticFunction read_quadratic(std::istream& in, int n) {
  expect(in, "quadratic");
  QuadraticFunction q;
  expect(in, "name");
  in >> q.name;
  expect(in, "constant");
  q.constant = read_double(in);
  expect(in, "linear");
  q.linear = read_vec(in, n);
  expect(in, "hessian_nnz");
  int nnz = 0;
  in >> nnz;
  if (nnz < 0) throw std::runtime_error("QCQ parse error: negative Hessian nnz");
  q.hessian.n = n;
  q.hessian.entries.reserve(nnz);
  for (int k = 0; k < nnz; ++k) {
    int i, j;
    std::string sv;
    if (!(in >> i >> j >> sv)) throw std::runtime_error("QCQ parse error: truncated Hessian");
    if (i > j) std::swap(i, j);
    q.hessian.entries.push_back({i, j, parse_double_token(sv)});
  }
  expect(in, "end_quadratic");
  q.validate(n);
  return q;
}

std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
      case '"': o << "\\\""; break;
      case '\\': o << "\\\\"; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default: o << c;
    }
  }
  return o.str();
}

void json_number(std::ostringstream& o, double v) {
  if (std::isfinite(v)) o << std::setprecision(17) << v;
  else o << "null";
}

} // namespace

QCQPProblem read_qcq(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open QCQ file: " + path);
  std::string magic;
  if (!(in >> magic)) throw std::runtime_error("QCQ parse error: missing format identifier");
  // V2 stores sparse linear coefficients. Keep one public reader contract for
  // every client (notably the direct Ipopt executable) and delegate V2 to the
  // mmap/scanner implementation that already owns that format.
  if (magic == "LAZYIPDCA_QCQP_V2") return read_qcq_compact(path);
  if (magic != "LAZYIPDCA_QCQP_V1")
    throw std::runtime_error("QCQ parse error: expected V1 or V2, got '" + magic + "'");
  QCQPProblem p;
  expect(in, "name"); in >> p.name;
  expect(in, "n"); in >> p.n;
  if (p.n <= 0) throw std::runtime_error("QCQ parse error: n must be positive");
  expect(in, "x0"); p.x0 = read_vec(in, p.n);
  expect(in, "lb"); p.lb = read_vec(in, p.n);
  expect(in, "ub"); p.ub = read_vec(in, p.n);
  detail::canonicalize_qcq_absent_bound_sentinels(p);
  expect(in, "objective");
  p.objective = read_quadratic(in, p.n);
  expect(in, "equalities");
  int me = 0; in >> me;
  if (me < 0) throw std::runtime_error("QCQ parse error: negative equality count");
  for (int i = 0; i < me; ++i) p.equalities.push_back(read_quadratic(in, p.n));
  expect(in, "inequalities");
  int mi = 0; in >> mi;
  if (mi < 0) throw std::runtime_error("QCQ parse error: negative inequality count");
  for (int i = 0; i < mi; ++i) p.inequalities.push_back(read_quadratic(in, p.n));
  expect(in, "end_problem");
  p.validate();
  return p;
}

std::string result_json(const Result& r, const std::string& problem_name) {
  std::ostringstream o;
  o << "{\n";
  {
    const auto& a=r.native_al_fallback;
    o << "  \"solution_origin\": \"" << json_escape(a.endpoint_origin) << "\",\n";
    o << "  \"native_al_fallback\": {";
    o << "\"policy\":";
    o << "\"" << json_escape(a.policy) << "\"";
    o << ",\"reason\":";
    o << "\"" << json_escape(a.reason) << "\"";
    o << ",\"endpoint_origin\":";
    o << "\"" << json_escape(a.endpoint_origin) << "\"";
    o << ",\"enabled\":";
    o << (a.enabled?"true":"false");
    o << ",\"coefficient_eligible\":";
    o << (a.coefficient_eligible?"true":"false");
    o << ",\"attempted\":";
    o << (a.attempted?"true":"false");
    o << ",\"accepted\":";
    o << (a.accepted?"true":"false");
    o << ",\"primary_strict\":";
    o << (a.primary_strict?"true":"false");
    o << ",\"fallback_strict\":";
    o << (a.fallback_strict?"true":"false");
    o << ",\"within_global_budget\":";
    o << (a.within_global_budget?"true":"false");
    o << ",\"original_input_preserved\":";
    o << (a.original_input_preserved?"true":"false");
    o << ",\"original_start_preserved\":";
    o << (a.original_start_preserved?"true":"false");
    o << ",\"internal_product_elimination_authorized\":";
    o << (a.internal_product_elimination_authorized?"true":"false");
    o << ",\"internal_dyadic_scaling_authorized\":";
    o << (a.internal_dyadic_scaling_authorized?"true":"false");
    o << ",\"internal_ray_initialization_authorized\":";
    o << (a.internal_ray_initialization_authorized?"true":"false");
    o << ",\"original_variables\":";
    o << a.original_variables;
    o << ",\"preflight_private_variables\":";
    o << a.preflight_private_variables;
    o << ",\"dense_workspace_estimate_bytes\":";
    json_number(o,a.dense_workspace_estimate_bytes);
    o << ",\"dense_workspace_limit_bytes\":";
    json_number(o,a.dense_workspace_limit_bytes);
    o << ",\"global_budget_seconds\":";
    json_number(o,a.global_budget_seconds);
    o << ",\"primary_seconds\":";
    json_number(o,a.primary_seconds);
    o << ",\"primary_audit_seconds\":";
    json_number(o,a.primary_audit_seconds);
    o << ",\"preflight_seconds\":";
    json_number(o,a.preflight_seconds);
    o << ",\"fallback_budget_seconds\":";
    json_number(o,a.fallback_budget_seconds);
    o << ",\"fallback_seconds\":";
    json_number(o,a.fallback_seconds);
    o << ",\"final_audit_seconds\":";
    json_number(o,a.final_audit_seconds);
    o << ",\"total_seconds\":";
    json_number(o,a.total_seconds);
    o << ",\"primary_outer_iterations\":";
    o << a.primary_outer_iterations;
    o << ",\"primary_inner_iterations\":";
    o << a.primary_inner_iterations;
    o << ",\"primary_factorizations\":";
    o << a.primary_factorizations;
    o << ",\"fallback_outer_stages\":";
    o << a.fallback_outer_stages;
    o << ",\"fallback_accepted_newton_steps\":";
    o << a.fallback_accepted_newton_steps;
    o << ",\"fallback_factorizations\":";
    o << a.fallback_factorizations;
    o << ",\"fallback_linear_solves\":";
    o << a.fallback_linear_solves;
    o << ",\"total_factorizations\":";
    o << a.total_factorizations;
    o << ",\"primary_iteration_budget\":";
    o << a.primary_iteration_budget;
    o << ",\"fallback_iteration_budget\":";
    o << a.fallback_iteration_budget;
    o << ",\"primary_iteration_unit\":";
    o << "\"" << json_escape(a.primary_iteration_unit) << "\"";
    o << ",\"fallback_iteration_unit\":";
    o << "\"" << json_escape(a.fallback_iteration_unit) << "\"";
    o << ",\"primary_status\":";
    o << "\"" << json_escape(a.primary_status) << "\"";
    o << ",\"fallback_status\":";
    o << "\"" << json_escape(a.fallback_status) << "\"";
    o << ",\"primary_message\":";
    o << "\"" << json_escape(a.primary_message) << "\"";
    o << ",\"fallback_message\":";
    o << "\"" << json_escape(a.fallback_message) << "\"";
    o << ",\"primary_primal\":";
    json_number(o,a.primary_primal);
    o << ",\"primary_dual\":";
    json_number(o,a.primary_dual);
    o << ",\"primary_complementarity\":";
    json_number(o,a.primary_complementarity);
    o << ",\"fallback_primal\":";
    json_number(o,a.fallback_primal);
    o << ",\"fallback_dual\":";
    json_number(o,a.fallback_dual);
    o << ",\"fallback_complementarity\":";
    json_number(o,a.fallback_complementarity);
    o << ",\"fallback_admitted\":";
    o << (a.fallback_admitted?"true":"false");
    o << ",\"fallback_initial_projection\":";
    o << (a.fallback_initial_projection?"true":"false");
    o << ",\"fallback_reduced_variables\":";
    o << a.fallback_reduced_variables;
    o << ",\"fallback_initial_projection_inf\":";
    json_number(o,a.fallback_initial_projection_inf);
    o << ",\"fallback_preparation_seconds\":";
    json_number(o,a.fallback_preparation_seconds);
    o << ",\"fallback_core_seconds\":";
    json_number(o,a.fallback_core_seconds);
    o << ",\"fallback_certificate_seconds\":";
    json_number(o,a.fallback_certificate_seconds);
    o << ",\"fallback_reason\":";
    o << "\"" << json_escape(a.fallback_reason) << "\"";
    o << ",\"fallback_seed_policy\":";
    o << "\"" << json_escape(a.fallback_seed_policy) << "\"";
    o << ",\"preflight_reason\":";
    o << "\"" << json_escape(a.preflight_reason) << "\"";
    o << ",\"primary_standard_auto_enabled\":";
    o << (a.primary_standard_auto_enabled?"true":"false");
    o << ",\"primary_standard_auto_exact_global_faces\":";
    o << (a.primary_standard_auto_exact_global_faces?"true":"false");
    o << ",\"primary_standard_auto_exact_global_faces_applied\":";
    o << (a.primary_standard_auto_exact_global_faces_applied?"true":"false");
    o << ",\"primary_standard_auto_checkpoint_requested\":";
    o << (a.primary_standard_auto_checkpoint_requested?"true":"false");
    o << ",\"primary_standard_auto_checkpoint_effective\":";
    o << (a.primary_standard_auto_checkpoint_effective?"true":"false");
    o << ",\"primary_standard_auto_builtin_checkpoint_skipped\":";
    o << (a.primary_standard_auto_builtin_checkpoint_skipped?"true":"false");
    o << ",\"primary_standard_auto_rank_presolve_requested\":";
    o << (a.primary_standard_auto_rank_presolve_requested?"true":"false");
    o << ",\"primary_standard_auto_rank_presolve_effective\":";
    o << (a.primary_standard_auto_rank_presolve_effective?"true":"false");
    o << ",\"primary_standard_auto_free_epigraph_requested\":";
    o << (a.primary_standard_auto_free_epigraph_requested?"true":"false");
    o << ",\"primary_standard_auto_free_epigraph_effective\":";
    o << (a.primary_standard_auto_free_epigraph_effective?"true":"false");
    o << ",\"primary_standard_auto_mode\":";
    o << "\"" << json_escape(a.primary_standard_auto_mode) << "\"";
    o << ",\"primary_standard_auto_reason\":";
    o << "\"" << json_escape(a.primary_standard_auto_reason) << "\"";
    o << ",\"primary_standard_auto_requested_engine\":";
    o << "\"" << json_escape(a.primary_standard_auto_requested_engine) << "\"";
    o << ",\"primary_standard_auto_effective_engine\":";
    o << "\"" << json_escape(a.primary_standard_auto_effective_engine) << "\"";
    o << ",\"primary_standard_auto_configuration\":";
    o << "\"" << json_escape(a.primary_standard_auto_configuration) << "\"";
    o << ",\"primary_effective_outer_algorithm\":";
    o << "\"" << json_escape(a.primary_effective_outer_algorithm) << "\"";
    o << ",\"primary_standard_auto_requested_startup\":";
    o << "\"" << json_escape(a.primary_standard_auto_requested_startup) << "\"";
    o << ",\"primary_standard_auto_effective_startup\":";
    o << "\"" << json_escape(a.primary_standard_auto_effective_startup) << "\"";
    o << ",\"primary_standard_auto_configured_initial_mu\":";
    json_number(o,a.primary_standard_auto_configured_initial_mu);
    o << ",\"primary_gr_adjustments_active\":";
    o << (a.primary_gr_adjustments_active?"true":"false");
    o << ",\"primary_gr_coefficient_constraint_increments\":";
    o << (a.primary_gr_coefficient_constraint_increments?"true":"false");
    o << ",\"primary_gr_residual_components\":";
    o << (a.primary_gr_residual_components?"true":"false");
    o << ",\"primary_gr_proximal_policy\":";
    o << "\"" << json_escape(a.primary_gr_proximal_policy) << "\"";
    o << ",\"primary_gr_fixed_centering_steps\":";
    o << a.primary_gr_fixed_centering_steps;
    o << ",\"dense_workspace_scope\":\"private dense Hessian assembly estimate only; excludes MUMPS, rational rewrite, sparse copies and ray workspace\"},\n";
  }
  {
    const auto& a=r.native_augmented;
    o << "  \"native_augmented_repair\": {\"enabled\":" << (a.enabled?"true":"false")
      << ",\"admitted\":" << (a.admitted?"true":"false")
      << ",\"initial_projection\":" << (a.initial_projection?"true":"false")
      << ",\"original_variables\":" << a.original_variables
      << ",\"reduced_variables\":" << a.reduced_variables
      << ",\"stages\":" << a.stages << ",\"accepted_steps\":" << a.accepted_steps
      << ",\"factorizations\":" << a.factorizations << ",\"linear_solves\":" << a.linear_solves
      << ",\"preparation_seconds\":" << a.preparation_seconds
      << ",\"core_seconds\":" << a.core_seconds
      << ",\"certificate_seconds\":" << a.certificate_seconds
      << ",\"initial_projection_inf\":" << a.initial_projection_inf
      << ",\"reason\":\"" << json_escape(a.reason)
      << "\",\"seed_policy\":\"" << json_escape(a.seed_policy)
      << "\",\"certificate_scope\":\"full original native QCQP in physical units\"},\n";
  }
  {
    o << "  \"coefficient_graph_certificate\": " << (r.coefficient_graph_certificate?"true":"false") << ",\n";
    o << "  \"coefficient_graph_vertices\": " << r.coefficient_graph_vertices << ",\n";
    o << "  \"coefficient_graph_edges\": " << r.coefficient_graph_edges << ",\n";
    o << "  \"coefficient_graph_components\": " << r.coefficient_graph_components << ",\n";
    o << "  \"coefficient_graph_independent_vertices\": " << r.coefficient_graph_independent << ",\n";
    o << "  \"common_factor_substitution\": " << (r.common_factor_substitution?"true":"false") << ",\n";
    o << "  \"affine_infeasibility_certificate\": " << (r.affine_infeasibility_certificate?"true":"false") << ",\n";
    o << "  \"convex_infeasibility_certificate\": " << (r.convex_infeasibility_certificate?"true":"false") << ",\n";
    o << "  \"convex_infeasibility_equality_sign\": " << r.convex_infeasibility_equality_sign << ",\n";
    o << "  \"convex_infeasibility_bound_variable\": " << r.convex_infeasibility_bound_variable << ",\n";
    o << "  \"convex_infeasibility_lower_side\": " << (r.convex_infeasibility_lower_side?"true":"false") << ",\n";
    o << "  \"convex_infeasibility_bound_weight\": \"" << json_escape(r.convex_infeasibility_bound_weight) << "\",\n";
    o << "  \"convex_infeasibility_minimum\": \"" << json_escape(r.convex_infeasibility_minimum) << "\",\n";
    o << "  \"convex_infeasibility_weight_norm\": \"" << json_escape(r.convex_infeasibility_weight_norm) << "\",\n";
    o << "  \"convex_infeasibility_lower_bound\": \"" << json_escape(r.convex_infeasibility_lower_bound) << "\",\n";
    o << "  \"convex_infeasibility_rows\": [";
    for(std::size_t k=0;k<r.convex_infeasibility_rows.size();++k){if(k)o<<',';o<<r.convex_infeasibility_rows[k];}o<<"],\n";
    o << "  \"convex_infeasibility_weights\": [";
    for(std::size_t k=0;k<r.convex_infeasibility_weights.size();++k){if(k)o<<',';o<<'\"'<<json_escape(r.convex_infeasibility_weights[k])<<'\"';}o<<"],\n";
    o << "  \"affine_infeasibility_rows\": [";
    for(std::size_t k=0;k<r.affine_infeasibility_rows.size();++k){if(k)o<<',';o<<r.affine_infeasibility_rows[k];}o<<"],\n";
    o << "  \"affine_infeasibility_weights\": [";
    for(std::size_t k=0;k<r.affine_infeasibility_weights.size();++k){if(k)o<<',';o<<'\"'<<json_escape(r.affine_infeasibility_weights[k])<<'\"';}o<<"],\n";
    o << "  \"affine_infeasibility_constant\": \"" << json_escape(r.affine_infeasibility_constant) << "\",\n";
    o << "  \"affine_infeasibility_primal_lower_bound\": \"" << json_escape(r.affine_infeasibility_primal_lower_bound) << "\",\n";
    o << "  \"common_factor_variable\": " << r.common_factor_variable << ",\n";
    o << "  \"positive_square_epigraph_substitution\": " << (r.positive_square_epigraph_substitution?"true":"false") << ",\n";
    o << "  \"positive_square_epigraph_variable\": " << r.positive_square_epigraph_variable << ",\n";
    const auto& a=r.adaptive_equality_probe;
    o << "  \"adaptive_equality_probe\": {\"eligible\":" << (a.eligible?"true":"false")
      << ",\"attempted\":" << (a.attempted?"true":"false")
      << ",\"transitioned\":" << (a.transitioned?"true":"false")
      << ",\"probe_strict\":" << (a.probe_strict?"true":"false")
      << ",\"stabilized_strict\":" << (a.stabilized_strict?"true":"false")
      << ",\"within_global_budget\":" << (a.within_global_budget?"true":"false")
      << ",\"probe_outer_budget\":" << a.probe_outer_budget
      << ",\"stabilized_outer_budget\":" << a.stabilized_outer_budget
      << ",\"global_outer_budget\":" << a.global_outer_budget
      << ",\"probe_outer\":" << a.probe_outer
      << ",\"stabilized_outer\":" << a.stabilized_outer
      << ",\"probe_restoration\":" << a.probe_restoration
      << ",\"stabilized_restoration\":" << a.stabilized_restoration
      << ",\"global_restoration_budget\":" << a.global_restoration_budget
      << ",\"stabilized_restoration_budget\":" << a.stabilized_restoration_budget
      << ",\"probe_factorizations\":" << a.probe_factorizations
      << ",\"stabilized_factorizations\":" << a.stabilized_factorizations;
    o << ",\"global_seconds\":";json_number(o,a.global_seconds);
    o << ",\"probe_seconds\":";json_number(o,a.probe_seconds);
    o << ",\"stabilized_seconds\":";json_number(o,a.stabilized_seconds);
    o << ",\"total_seconds\":";json_number(o,a.total_seconds);
    o << ",\"selection_audit_seconds\":";json_number(o,a.selection_audit_seconds);
    o << ",\"probe_raw_residual\":";json_number(o,a.probe_raw_residual);
    o << ",\"stabilized_raw_residual\":";json_number(o,a.stabilized_raw_residual);
    o << ",\"probe_status\":\"" << json_escape(a.probe_status) << "\""
      << ",\"stabilized_status\":\"" << json_escape(a.stabilized_status) << "\""
      << ",\"selected_phase\":\"" << json_escape(a.selected_phase) << "\""
      << ",\"reason\":\"" << json_escape(a.reason) << "\""
      << ",\"original_frozen_start_each_phase\":true,\"same_configuration\":true,\"external_optimizer\":false},\n";
  }
  {
    const auto& a=r.native_barrier_retry;
    o << "  \"native_barrier_retry\": {\"attempted\":" << (a.attempted?"true":"false")
      << ",\"accepted\":" << (a.accepted?"true":"false")
      << ",\"primary_status\":\"" << json_escape(a.primary_status)
      << "\",\"retry_status\":\"" << json_escape(a.retry_status)
      << "\",\"statistics_scope\":\"all phase costs and designated work counters cumulative; retained endpoint and certificate unchanged\""
      << ",\"primary_seconds\":" << a.primary_seconds << ",\"retry_seconds\":" << a.retry_seconds
      << ",\"primary_outer_iterations\":" << a.primary_outer_iterations << ",\"retry_outer_iterations\":" << a.retry_outer_iterations
      << ",\"primary_restoration_iterations\":" << a.primary_restoration_iterations
      << ",\"retry_restoration_iterations\":" << a.retry_restoration_iterations
      << ",\"primary_factor_attempts\":" << a.primary_factor_attempts << ",\"retry_factor_attempts\":" << a.retry_factor_attempts
      << ",\"primary_linear_solves\":" << a.primary_linear_solves << ",\"retry_linear_solves\":" << a.retry_linear_solves;
    o << ",\"primary_cost_seconds\":[";
    for(std::size_t k=0;k<a.primary_cost_seconds.size();++k){if(k)o<<',';o<<a.primary_cost_seconds[k];}
    o << "],\"retry_cost_seconds\":[";
    for(std::size_t k=0;k<a.retry_cost_seconds.size();++k){if(k)o<<',';o<<a.retry_cost_seconds[k];}
    o << "]},\n";
  }
  {
    const auto& a=r.free_epigraph;
    o << "  \"free_epigraph_enabled\": " << (a.enabled?"true":"false") << ",\n"
      << "  \"free_epigraph_applied\": " << (a.applied?"true":"false") << ",\n"
      << "  \"free_epigraph_initial_point_preserved\": " << (a.initial_point_preserved?"true":"false") << ",\n"
      << "  \"free_epigraph_postsolve_valid\": " << (a.postsolve_valid?"true":"false") << ",\n"
      << "  \"free_epigraph_original_variables\": " << a.original_variables << ",\n"
      << "  \"free_epigraph_reduced_variables\": " << a.reduced_variables << ",\n"
      << "  \"free_epigraph_original_inequalities\": " << a.original_inequalities << ",\n"
      << "  \"free_epigraph_reduced_inequalities\": " << a.reduced_inequalities << ",\n"
      << "  \"free_epigraph_candidates\": " << a.candidates << ",\n"
      << "  \"free_epigraph_selected\": " << a.selected << ",\n"
      << "  \"free_epigraph_bounds_rejected\": " << a.bounds_rejected << ",\n"
      << "  \"free_epigraph_start_rejected\": " << a.start_rejected << ",\n"
      << "  \"free_epigraph_ambiguous_rejected\": " << a.ambiguous_rejected << ",\n"
      << "  \"free_epigraph_checkpoint_lifts\": " << a.checkpoint_lifts << ",\n"
      << "  \"free_epigraph_preparation_seconds\": " << a.preparation_seconds << ",\n"
      << "  \"free_epigraph_postsolve_seconds\": " << a.postsolve_seconds << ",\n"
      << "  \"free_epigraph_checkpoint_seconds\": " << a.checkpoint_seconds << ",\n"
      << "  \"free_epigraph_reason\": \"" << json_escape(a.reason) << "\",\n";
  }
  o << "  \"rank_presolve_enabled\": " << (r.rank_presolve.enabled?"true":"false") << ",\n";
  o << "  \"rank_presolve_applied\": " << (r.rank_presolve.applied?"true":"false") << ",\n";
  o << "  \"rank_presolve_initial_point_preserved\": " << (r.rank_presolve.initial_point_preserved?"true":"false") << ",\n";
  o << "  \"rank_presolve_postsolve_valid\": " << (r.rank_presolve.postsolve_valid?"true":"false") << ",\n";
  o << "  \"rank_presolve_original_variables\": " << r.rank_presolve.original_variables << ",\n";
  o << "  \"rank_presolve_reduced_variables\": " << r.rank_presolve.reduced_variables << ",\n";
  o << "  \"rank_presolve_original_equalities\": " << r.rank_presolve.original_equalities << ",\n";
  o << "  \"rank_presolve_reduced_equalities\": " << r.rank_presolve.reduced_equalities << ",\n";
  o << "  \"rank_presolve_original_inequalities\": " << r.rank_presolve.original_inequalities << ",\n";
  o << "  \"rank_presolve_reduced_inequalities\": " << r.rank_presolve.reduced_inequalities << ",\n";
  o << "  \"rank_presolve_leaf_candidates\": " << r.rank_presolve.leaf_candidates << ",\n";
  o << "  \"rank_presolve_leaf_pairs\": " << r.rank_presolve.leaf_pairs << ",\n";
  o << "  \"rank_presolve_nonzero_rejected\": " << r.rank_presolve.nonzero_rejected << ",\n";
  o << "  \"rank_presolve_start_rejected\": " << r.rank_presolve.start_rejected << ",\n";
  o << "  \"rank_presolve_arithmetic_rejected\": " << r.rank_presolve.arithmetic_rejected << ",\n";
  o << "  \"rank_presolve_duplicate_equalities\": " << r.rank_presolve.duplicate_equalities << ",\n";
  o << "  \"rank_presolve_zero_equalities\": " << r.rank_presolve.zero_equalities << ",\n";
  o << "  \"rank_presolve_duplicate_inequalities\": " << r.rank_presolve.duplicate_inequalities << ",\n";
  o << "  \"rank_presolve_constant_inequalities\": " << r.rank_presolve.constant_inequalities << ",\n";
  o << "  \"rank_presolve_duplicate_bounds\": " << r.rank_presolve.duplicate_bounds << ",\n";
  o << "  \"rank_presolve_implied_bound_sides\": " << r.rank_presolve.implied_bound_sides << ",\n";
  o << "  \"rank_presolve_preparation_seconds\": " << r.rank_presolve.preparation_seconds << ",\n";
  o << "  \"rank_presolve_postsolve_seconds\": " << r.rank_presolve.postsolve_seconds << ",\n";
  o << "  \"rank_presolve_initial_lift_error\": " << r.rank_presolve.initial_lift_error << ",\n";
  {
    const auto& a=r.rank_presolve;
    o << "  \"rank_exact_fixed_retry\": {\"attempted\":" << (a.exact_fixed_retry_attempted?"true":"false")
      << ",\"primary_status\":\"" << json_escape(a.exact_fixed_primary_status)
      << "\",\"retry_status\":\"" << json_escape(a.exact_fixed_retry_status)
      << "\",\"statistics_scope\":\"costs, function calls and designated work counters cumulative; dimensions, ranks and controller end state from final attempt\""
      << ",\"primary_seconds\":" << a.exact_fixed_primary_seconds
      << ",\"retry_seconds\":" << a.exact_fixed_retry_seconds
      << ",\"primary_core_seconds\":" << a.exact_fixed_primary_core_seconds
      << ",\"primary_restoration_seconds\":" << a.exact_fixed_primary_restoration_seconds
      << ",\"primary_outer_iterations\":" << a.exact_fixed_primary_outer_iterations
      << ",\"retry_outer_iterations\":" << a.exact_fixed_retry_outer_iterations
      << ",\"primary_restoration_iterations\":" << a.exact_fixed_primary_restoration_iterations
      << ",\"retry_restoration_iterations\":" << a.exact_fixed_retry_restoration_iterations
      << ",\"primary_factor_attempts\":" << a.exact_fixed_primary_factor_attempts
      << ",\"retry_factor_attempts\":" << a.exact_fixed_retry_factor_attempts
      << ",\"primary_linear_solves\":" << a.exact_fixed_primary_linear_solves
      << ",\"retry_linear_solves\":" << a.exact_fixed_retry_linear_solves
      << ",\"primary_history_records\":" << a.exact_fixed_primary_history_records
      << ",\"primary_cost_seconds\":{";
    for(std::size_t k=0;k<a.exact_fixed_primary_cost_seconds.size();++k) {
      if(k)o << ',';
      o << '"' << newton_cost_names[k] << "\":";json_number(o,a.exact_fixed_primary_cost_seconds[k]);
    }
    o << "}},\n";
  }
  o << "  \"rank_audit_mode\": \"" << json_escape(r.rank_presolve.audit_mode) << "\",\n";
  o << "  \"rank_audit_duplicate_rows_covered\": " << r.rank_presolve.audit_duplicate_rows_covered << ",\n";
  o << "  \"rank_audit_unique_terms\": " << r.rank_presolve.audit_unique_terms << ",\n";
  o << "  \"rank_audit_ordered_sums\": " << r.rank_presolve.audit_ordered_sums << ",\n";
  o << "  \"rank_audit_term_evaluations\": " << r.rank_presolve.audit_term_evaluations << ",\n";
  o << "  \"rank_audit_term_reuses\": " << r.rank_presolve.audit_term_reuses << ",\n";
  o << "  \"rank_audit_shared_values\": " << r.rank_presolve.audit_shared_values << ",\n";
  o << "  \"rank_audit_ordered_variant_rows\": " << r.rank_presolve.audit_recipe_exceptions << ",\n";
  o << "  \"rank_audit_calls\": " << r.rank_presolve.audit_calls << ",\n";
  o << "  \"rank_audit_mapped_calls\": " << r.rank_presolve.audit_mapped_calls << ",\n";
  o << "  \"rank_audit_full_calls\": " << r.rank_presolve.audit_full_calls << ",\n";
  o << "  \"rank_audit_check_calls\": " << r.rank_presolve.audit_check_calls << ",\n";
  o << "  \"rank_audit_value_evaluations\": " << r.rank_presolve.audit_value_evaluations << ",\n";
  o << "  \"rank_audit_value_reuses\": " << r.rank_presolve.audit_value_reuses << ",\n";
  o << "  \"rank_audit_gradient_evaluations\": " << r.rank_presolve.audit_gradient_evaluations << ",\n";
  o << "  \"rank_audit_result_copies\": " << r.rank_presolve.audit_result_copies << ",\n";
  o << "  \"rank_audit_coordinate_expansions\": " << r.rank_presolve.audit_coordinate_expansions << ",\n";
  o << "  \"rank_audit_lift_calls\": " << r.rank_presolve.audit_lift_calls << ",\n";
  o << "  \"rank_audit_final_full_calls\": " << r.rank_presolve.audit_final_full_calls << ",\n";
  o << "  \"rank_audit_seconds\": " << r.rank_presolve.audit_seconds << ",\n";
  o << "  \"schema\": \"lazyipdca-qcqp-result-v2\",\n";
  o << "  \"solver\": \"" << kSolverName << "\",\n";
  o << "  \"version\": \"" << kVersion << "\",\n";
  o << "  \"api_version\": " << kApiVersion << ",\n";
  o << "  \"build_id\": \"" << kBuildId << "\",\n";
  o << "  \"linear_solver_backend\": \"" << json_escape(r.linear_solver_backend) << "\",\n";
  o << "  \"effective_outer_algorithm\": \"" << json_escape(r.effective_outer_algorithm) << "\",\n";
  o << "  \"standard_auto_enabled\": " << (r.standard_auto_enabled?"true":"false") << ",\n";
  o << "  \"gr_coefficient_constraint_increments\": " << (r.gr_coefficient_constraint_increments?"true":"false") << ",\n";
  o << "  \"gr_adjustments_active\": " << (r.gr_adjustments_active?"true":"false") << ",\n";
  o << "  \"gr_proximal_policy\": \"" << json_escape(r.gr_proximal_policy) << "\",\n";
  o << "  \"gr_fixed_centering_steps\": " << r.gr_fixed_centering_steps << ",\n";
  o << "  \"gr_residual_components\": " << (r.gr_residual_components?"true":"false") << ",\n";
  o << "  \"standard_auto_rank_presolve_requested\": " << (r.standard_auto_rank_presolve_requested?"true":"false") << ",\n";
  o << "  \"standard_auto_rank_presolve_effective\": " << (r.standard_auto_rank_presolve_effective?"true":"false") << ",\n";
  o << "  \"standard_auto_free_epigraph_requested\": " << (r.standard_auto_free_epigraph_requested?"true":"false") << ",\n";
  o << "  \"standard_auto_free_epigraph_effective\": " << (r.standard_auto_free_epigraph_effective?"true":"false") << ",\n";
  o << "  \"standard_auto_rank_presolve_skipped\": " << (r.standard_auto_rank_presolve_skipped?"true":"false") << ",\n";
  o << "  \"standard_auto_free_epigraph_presolve_skipped\": " << (r.standard_auto_free_epigraph_presolve_skipped?"true":"false") << ",\n";
  o << "  \"standard_auto_checkpoint_requested\": " << (r.standard_auto_checkpoint_requested?"true":"false") << ",\n";
  o << "  \"standard_auto_checkpoint_effective\": " << (r.standard_auto_checkpoint_effective?"true":"false") << ",\n";
  o << "  \"standard_auto_builtin_checkpoint_skipped\": " << (r.standard_auto_builtin_checkpoint_skipped?"true":"false") << ",\n";
  o << "  \"standard_auto_requested_startup\": \"" << json_escape(r.standard_auto_requested_startup) << "\",\n";
  o << "  \"standard_auto_effective_startup\": \"" << json_escape(r.standard_auto_effective_startup) << "\",\n";
  o << "  \"standard_auto_configured_initial_mu\": "; json_number(o,r.standard_auto_configured_initial_mu); o << ",\n";
  o << "  \"standard_auto_mode\": \"" << json_escape(r.standard_auto_mode) << "\",\n";
  o << "  \"standard_auto_reason\": \"" << json_escape(r.standard_auto_reason) << "\",\n";
  o << "  \"standard_auto_requested_engine\": \"" << json_escape(r.standard_auto_requested_engine) << "\",\n";
  o << "  \"standard_auto_effective_engine\": \"" << json_escape(r.standard_auto_effective_engine) << "\",\n";
  o << "  \"standard_auto_configuration\": \"" << json_escape(r.standard_auto_configuration) << "\",\n";
  o << "  \"standard_auto_exact_global_faces\": " << (r.standard_auto_exact_global_faces?"true":"false") << ",\n";
  o << "  \"standard_auto_exact_global_faces_applied\": " << (r.standard_auto_exact_global_faces_applied?"true":"false") << ",\n";
  o << "  \"standard_auto_exact_global_faces_reason\": \"" << json_escape(r.standard_auto_exact_global_faces_reason) << "\",\n";
  o << "  \"standard_auto_exact_face_start_projection_inf\": "; json_number(o,r.standard_auto_exact_face_start_projection_inf); o << ",\n";
  o << "  \"dca_dual_step\": \"" << json_escape(r.dca_dual_step) << "\",\n";
  o << "  \"problem\": \"" << json_escape(problem_name) << "\",\n";
  o << "  \"status\": \"" << status_string(r.status) << "\",\n";
  o << "  \"message\": \"" << json_escape(r.message) << "\",\n";
  o << "  \"objective\": "; json_number(o, r.objective); o << ",\n";
  o << "  \"feasibility\": "; json_number(o, r.feasibility.total_inf); o << ",\n";
  o << "  \"equality_violation\": "; json_number(o, r.feasibility.equality_inf); o << ",\n";
  o << "  \"inequality_violation\": "; json_number(o, r.feasibility.inequality_inf); o << ",\n";
  o << "  \"bound_violation\": "; json_number(o, r.feasibility.bound_inf); o << ",\n";
  o << "  \"model_stationarity_inf\": "; json_number(o, r.model_stationarity_inf); o << ",\n";
  o << "  \"dual_infeasibility_inf\": "; json_number(o, r.dual_infeasibility_inf); o << ",\n";
  o << "  \"complementarity_inf\": "; json_number(o, r.complementarity_inf); o << ",\n";
  o << "  \"termination_test\": \"" << json_escape(r.termination_test) << "\",\n";
  o << "  \"termination_internal_dual_inf\": "; json_number(o, r.termination_internal_dual_inf); o << ",\n";
  o << "  \"termination_internal_primal_inf\": "; json_number(o, r.termination_internal_primal_inf); o << ",\n";
  o << "  \"termination_internal_compl_inf\": "; json_number(o, r.termination_internal_compl_inf); o << ",\n";
  o << "  \"termination_dual_inf\": "; json_number(o, r.termination_dual_inf); o << ",\n";
  o << "  \"termination_compl_inf\": "; json_number(o, r.termination_compl_inf); o << ",\n";
  o << "  \"original_normalized_nlp_error\": "; json_number(o, r.original_normalized_nlp_error); o << ",\n";
  o << "  \"termination_objective_scale\": "; json_number(o, r.termination_objective_scale); o << ",\n";
  o << "  \"termination_previous_objective\": "; json_number(o, r.termination_previous_objective); o << ",\n";
  o << "  \"termination_current_objective\": "; json_number(o, r.termination_current_objective); o << ",\n";
  o << "  \"termination_objective_change\": "; json_number(o, r.termination_objective_change); o << ",\n";
  o << "  \"termination_equality_scales\": ["; for(std::size_t i=0;i<r.termination_equality_scales.size();++i){if(i)o<<",";json_number(o,r.termination_equality_scales[i]);} o << "],\n";
  o << "  \"termination_inequality_scales\": ["; for(std::size_t i=0;i<r.termination_inequality_scales.size();++i){if(i)o<<",";json_number(o,r.termination_inequality_scales[i]);} o << "],\n";
  o << "  \"termination_inequality_slacks\": ["; for(std::size_t i=0;i<r.termination_inequality_slacks.size();++i){if(i)o<<",";json_number(o,r.termination_inequality_slacks[i]);} o << "],\n";
  o << "  \"termination_square_problem\": " << (r.termination_square_problem ? "true" : "false") << ",\n";
  o << "  \"raw_kkt_error\": "; json_number(o, r.raw_kkt_error); o << ",\n";
  o << "  \"acceptable_error\": "; json_number(o, r.acceptable_error); o << ",\n";
  o << "  \"termination_sd\": "; json_number(o, r.termination_sd); o << ",\n";
  o << "  \"termination_sc\": "; json_number(o, r.termination_sc); o << ",\n";
  o << "  \"acceptable_convergence\": " << (r.acceptable_convergence ? "true" : "false") << ",\n";
  o << "  \"acceptable_point\": " << (r.acceptable_point ? "true" : "false") << ",\n";
  o << "  \"acceptable_on_failure\": " << (r.acceptable_on_failure ? "true" : "false") << ",\n";
  o << "  \"acceptable_trigger\": \"" << json_escape(r.acceptable_trigger) << "\",\n";
  o << "  \"original_failure_status\": \"" << json_escape(r.original_failure_status) << "\",\n";
  o << "  \"original_failure_message\": \"" << json_escape(r.original_failure_message) << "\",\n";
  o << "  \"termination_audit_valid\": " << (r.termination_audit_valid ? "true" : "false") << ",\n";
  o << "  \"acceptable_streak\": " << r.acceptable_streak << ",\n";
  o << "  \"acceptable_required\": " << r.acceptable_required << ",\n";
  o << "  \"fixed_multipliers_reconstructed\": " << r.fixed_multipliers_reconstructed << ",\n";
  o << "  \"scaled_nlp_error\": "; json_number(o, r.scaled_nlp_error); o << ",\n";
  o << "  \"dual_error_scaling\": "; json_number(o, r.dual_error_scaling); o << ",\n";
  o << "  \"complementarity_error_scaling\": "; json_number(o, r.complementarity_error_scaling); o << ",\n";
  o << "  \"constraint_multiplier_l1\": "; json_number(o, r.constraint_multiplier_l1); o << ",\n";
  o << "  \"bound_slack_multiplier_l1\": "; json_number(o, r.bound_slack_multiplier_l1); o << ",\n";
  o << "  \"ipopt_dual_multiplier_dimension\": " << r.ipopt_dual_multiplier_dimension << ",\n";
  o << "  \"ipopt_complementarity_multiplier_dimension\": " << r.ipopt_complementarity_multiplier_dimension << ",\n";
  o << "  \"ipopt_y_c_dimension\": " << r.ipopt_y_c_dimension << ",\n";
  o << "  \"ipopt_y_d_dimension\": " << r.ipopt_y_d_dimension << ",\n";
  o << "  \"ipopt_z_L_dimension\": " << r.ipopt_z_L_dimension << ",\n";
  o << "  \"ipopt_z_U_dimension\": " << r.ipopt_z_U_dimension << ",\n";
  o << "  \"ipopt_v_L_dimension\": " << r.ipopt_v_L_dimension << ",\n";
  o << "  \"ipopt_v_U_dimension\": " << r.ipopt_v_U_dimension << ",\n";
  o << "  \"equality_multiplier_inf\": "; json_number(o, r.equality_multiplier_inf); o << ",\n";
  o << "  \"inequality_multiplier_inf\": "; json_number(o, r.inequality_multiplier_inf); o << ",\n";
  o << "  \"bound_multiplier_inf\": "; json_number(o, r.bound_multiplier_inf); o << ",\n";
  o << "  \"ipopt_desired_convergence\": " << (r.ipopt_desired_convergence ? "true" : "false") << ",\n";
  o << "  \"termination_multiplier_source\": \"" << json_escape(r.termination_multiplier_source) << "\",\n";
  o << "  \"termination_multiplier_fit_converged\": " << (r.termination_multiplier_fit_converged ? "true" : "false") << ",\n";
  o << "  \"termination_multiplier_fit_sweeps\": " << r.termination_multiplier_fit_sweeps << ",\n";
  o << "  \"termination_multiplier_fit_projected_gradient_inf\": "; json_number(o, r.termination_multiplier_fit_projected_gradient_inf); o << ",\n";
  o << "  \"termination_multiplier_fit_stationarity_inf\": "; json_number(o, r.termination_multiplier_fit_stationarity_inf); o << ",\n";
  o << "  \"termination_multiplier_refinement_attempts\": " << r.termination_multiplier_refinement_attempts << ",\n";
  o << "  \"termination_multiplier_regularization\": "; json_number(o, r.termination_multiplier_regularization); o << ",\n";
  o << "  \"termination_certificate_rescued\": " << (r.termination_certificate_rescued ? "true" : "false") << ",\n";
  o << "  \"multiplier_refit_attempts\": " << r.multiplier_refit_attempts << ",\n";
  o << "  \"multiplier_refit_successes\": " << r.multiplier_refit_successes << ",\n";
  o << "  \"multiplier_refit_fallback_reuses\": "
    << r.multiplier_refit_fallback_reuses << ",\n";
  o << "  \"multiplier_row_space_factorizations\": " << r.multiplier_row_space_factorizations << ",\n";
  o << "  \"multiplier_row_space_solves\": " << r.multiplier_row_space_solves << ",\n";
  o << "  \"multiplier_row_space_symbolic_analyses\": " << r.multiplier_row_space_symbolic_analyses << ",\n";
  o << "  \"multiplier_row_space_symbolic_reuses\": " << r.multiplier_row_space_symbolic_reuses << ",\n";
  o << "  \"multiplier_row_space_failures\": " << r.multiplier_row_space_failures << ",\n";
  o << "  \"last_multiplier_refit_stationarity_inf\": "; json_number(o, r.last_multiplier_refit_stationarity_inf); o << ",\n";
  o << "  \"dca_single_loop_refits_skipped\": " << r.dca_single_loop_refits_skipped << ",\n";
  o << "  \"dca_single_loop_refit_fallbacks\": " << r.dca_single_loop_refit_fallbacks << ",\n";
  o << "  \"dca_single_loop_bootstrap_full_fits\": "
    << r.dca_single_loop_bootstrap_full_fits << ",\n";
  o << "  \"dca_single_loop_full_refit_calls\": " << r.dca_single_loop_full_refit_calls << ",\n";
  o << "  \"dca_single_loop_retractions_skipped\": " << r.dca_single_loop_retractions_skipped << ",\n";
  o << "  \"dca_single_loop_retraction_fallbacks\": " << r.dca_single_loop_retraction_fallbacks << ",\n";
  o << "  \"step_inf\": "; json_number(o, r.step_inf); o << ",\n";
  o << "  \"last_inner_dual_inf\": "; json_number(o, r.last_inner_dual_inf); o << ",\n";
  o << "  \"last_inner_equality_inf\": "; json_number(o, r.last_inner_equality_inf); o << ",\n";
  o << "  \"last_inner_inequality_inf\": "; json_number(o, r.last_inner_inequality_inf); o << ",\n";
  o << "  \"last_inner_primal_residual_inf\": "; json_number(o, r.last_inner_primal_residual_inf); o << ",\n";
  o << "  \"last_inner_complementarity\": "; json_number(o, r.last_inner_complementarity); o << ",\n";
  o << "  \"last_inner_scaled_model_residual_inf\": "; json_number(o, r.last_inner_scaled_model_residual_inf); o << ",\n";
  o << "  \"last_inner_used_iteration_limit_fallback\": " << (r.last_inner_used_iteration_limit_fallback ? "true" : "false") << ",\n";
  o << "  \"final_kkt_rho_x\": "; json_number(o, r.final_kkt_rho_x); o << ",\n";
  o << "  \"max_kkt_rho_x\": "; json_number(o, r.max_kkt_rho_x); o << ",\n";
  o << "  \"final_kkt_delta_d\": "; json_number(o, r.final_kkt_delta_d); o << ",\n";
  o << "  \"max_kkt_delta_d\": "; json_number(o, r.max_kkt_delta_d); o << ",\n";
  o << "  \"max_kkt_backward_error\": "; json_number(o, r.max_kkt_backward_error); o << ",\n";
  o << "  \"max_kkt_residual_inf\": "; json_number(o, r.max_kkt_residual_inf); o << ",\n";
  o << "  \"kkt_regularization_retries\": " << r.kkt_regularization_retries << ",\n";
  o << "  \"kkt_refinement_steps\": " << r.kkt_refinement_steps << ",\n";
  o << "  \"equality_qp_original_kkt_gmres_attempts\": "
    << r.equality_qp_original_kkt_gmres_attempts << ",\n";
  o << "  \"equality_qp_original_kkt_gmres_acceptances\": "
    << r.equality_qp_original_kkt_gmres_acceptances << ",\n";
  o << "  \"equality_qp_original_kkt_gmres_iterations\": "
    << r.equality_qp_original_kkt_gmres_iterations << ",\n";
  o << "  \"inner_dual_warm_start_acceptances\": " << r.inner_dual_warm_start_acceptances << ",\n";
  o << "  \"inner_full_dual_warm_start_acceptances\": " << r.inner_full_dual_warm_start_acceptances << ",\n";
  o << "  \"inner_dual_warm_start_rejections\": " << r.inner_dual_warm_start_rejections << ",\n";
  o << "  \"lazyqp_zero_dual_continuation_attempts\": " << r.lazyqp_zero_dual_continuation_attempts << ",\n";
  o << "  \"lazyqp_zero_dual_continuation_mapped_attempts\": " << r.lazyqp_zero_dual_continuation_mapped_attempts << ",\n";
  o << "  \"lazyqp_zero_dual_continuation_direct_attempts\": " << r.lazyqp_zero_dual_continuation_direct_attempts << ",\n";
  o << "  \"lazyqp_zero_dual_continuation_acceptances\": " << r.lazyqp_zero_dual_continuation_acceptances << ",\n";
  o << "  \"lazyqp_zero_dual_continuation_fallbacks\": " << r.lazyqp_zero_dual_continuation_fallbacks << ",\n";
  o << "  \"lazyqp_zero_dual_continuation_repairs\": " << r.lazyqp_zero_dual_continuation_repairs << ",\n";
  o << "  \"working_set_primal_transports\": " << r.working_set_primal_transports << ",\n";
  o << "  \"lazyqp_reduced_kkt_factorizations\": " << r.lazyqp_reduced_kkt_factorizations << ",\n";
  o << "  \"lazyqp_full_kkt_factorizations\": " << r.lazyqp_full_kkt_factorizations << ",\n";
  o << "  \"lazyqp_reduced_kkt_fallbacks\": " << r.lazyqp_reduced_kkt_fallbacks << ",\n";
  o << "  \"lazyqp_hybrid_kkt_attempts\": " << r.lazyqp_hybrid_kkt_attempts << ",\n";
  o << "  \"lazyqp_hybrid_kkt_factorizations\": " << r.lazyqp_hybrid_kkt_factorizations << ",\n";
  o << "  \"lazyqp_hybrid_kkt_fallbacks\": " << r.lazyqp_hybrid_kkt_fallbacks << ",\n";
  o << "  \"lazyqp_hybrid_kkt_retained_rows_total\": " << r.lazyqp_hybrid_kkt_retained_rows_total << ",\n";
  o << "  \"lazyqp_hybrid_kkt_retained_rows_max\": " << r.lazyqp_hybrid_kkt_retained_rows_max << ",\n";
  o << "  \"lazyqp_hybrid_kkt_outer_product_triplets_avoided\": "
    << r.lazyqp_hybrid_kkt_outer_product_triplets_avoided << ",\n";
  o << "  \"lazyqp_hybrid_kkt_border_triplets_emitted\": "
    << r.lazyqp_hybrid_kkt_border_triplets_emitted << ",\n";
  o << "  \"socp_formulation_attempts\": " << r.socp_formulation_attempts << ",\n";
  o << "  \"socp_formulation_builds\": " << r.socp_formulation_builds << ",\n";
  o << "  \"socp_formulation_rejections\": " << r.socp_formulation_rejections << ",\n";
  o << "  \"socp_formulation_audit_failures\": " << r.socp_formulation_audit_failures << ",\n";
  o << "  \"socp_objective_rotated_cones\": " << r.socp_objective_rotated_cones << ",\n";
  o << "  \"socp_inequality_rotated_cones\": " << r.socp_inequality_rotated_cones << ",\n";
  o << "  \"socp_linear_nonnegative_cones\": " << r.socp_linear_nonnegative_cones << ",\n";
  o << "  \"socp_max_affine_equalities\": " << r.socp_max_affine_equalities << ",\n";
  o << "  \"socp_max_explicit_lift_cone_dimension\": "
    << r.socp_max_explicit_lift_cone_dimension << ",\n";
  o << "  \"socp_condensed_newton_iterations\": "
    << r.socp_condensed_newton_iterations << ",\n";
  o << "  \"socp_max_rotated_cone_violation\": ";
  json_number(o, r.socp_max_rotated_cone_violation); o << ",\n";
  o << "  \"lazyqp_kkt_triplet_layout_builds\": "
    << r.lazyqp_kkt_triplet_layout_builds << ",\n";
  o << "  \"lazyqp_kkt_triplet_layout_reuses\": "
    << r.lazyqp_kkt_triplet_layout_reuses << ",\n";
  o << "  \"lazyqp_kkt_triplet_layout_invalidations\": "
    << r.lazyqp_kkt_triplet_layout_invalidations << ",\n";
  o << "  \"lazyqp_numeric_kkt_reuse_attempts\": " << r.lazyqp_numeric_kkt_reuse_attempts << ",\n";
  o << "  \"lazyqp_numeric_kkt_reuse_acceptances\": " << r.lazyqp_numeric_kkt_reuse_acceptances << ",\n";
  o << "  \"lazyqp_numeric_kkt_reuse_fallbacks\": " << r.lazyqp_numeric_kkt_reuse_fallbacks << ",\n";
  o << "  \"lazyqp_numeric_kkt_reuse_iterations\": " << r.lazyqp_numeric_kkt_reuse_iterations << ",\n";
  o << "  \"lazyqp_max_kkt_dimension\": " << r.lazyqp_max_kkt_dimension << ",\n";
  o << "  \"lazyqp_max_full_kkt_dimension\": " << r.lazyqp_max_full_kkt_dimension << ",\n";
  o << "  \"lazyqp_affine_elimination_attempts\": " << r.lazyqp_affine_elimination_attempts << ",\n";
  o << "  \"lazyqp_affine_elimination_acceptances\": " << r.lazyqp_affine_elimination_acceptances << ",\n";
  o << "  \"lazyqp_affine_elimination_fallbacks\": " << r.lazyqp_affine_elimination_fallbacks << ",\n";
  o << "  \"lazyqp_affine_eliminated_variables\": " << r.lazyqp_affine_eliminated_variables << ",\n";
  o << "  \"lazyqp_affine_eliminated_equalities\": " << r.lazyqp_affine_eliminated_equalities << ",\n";
  o << "  \"lazyqp_affine_reduced_variables\": " << r.lazyqp_affine_reduced_variables << ",\n";
  o << "  \"lazyqp_affine_reduced_equalities\": " << r.lazyqp_affine_reduced_equalities << ",\n";
  o << "  \"restoration_iterations\": " << r.restoration_iterations << ",\n";
  o << "  \"restoration_restarts\": " << r.restoration_restarts << ",\n";
  o << "  \"restoration_linear_solves\": " << r.restoration_linear_solves << ",\n";
  o << "  \"restoration_exact_hessian_solves\": " << r.restoration_exact_hessian_solves << ",\n";
  o << "  \"restoration_matrix_free_solves\": " << r.restoration_matrix_free_solves << ",\n";
  o << "  \"restoration_row_space_solves\": " << r.restoration_row_space_solves << ",\n";
  o << "  \"restoration_harmonic_attempts\": " << r.restoration_harmonic_attempts << ",\n";
  o << "  \"restoration_harmonic_acceptances\": " << r.restoration_harmonic_acceptances << ",\n";
  o << "  \"restoration_harmonic_factorizations\": " << r.restoration_harmonic_factorizations << ",\n";
  o << "  \"restoration_tangent_activation_attempts\": " << r.restoration_tangent_activation_attempts << ",\n";
  o << "  \"restoration_tangent_activation_acceptances\": " << r.restoration_tangent_activation_acceptances << ",\n";
  o << "  \"restoration_tangent_activation_components\": " << r.restoration_tangent_activation_components << ",\n";
  o << "  \"restoration_cg_iterations\": " << r.restoration_cg_iterations << ",\n";
  o << "  \"restoration_exact_workspace_growths\": "
    << r.restoration_exact_workspace_growths << ",\n";
  o << "  \"restoration_exact_matvecs\": "
    << r.restoration_exact_matvecs << ",\n";
  o << "  \"restoration_exact_parallel_row_dot_batches\": "
    << r.restoration_exact_parallel_row_dot_batches << ",\n";
  o << "  \"restoration_exact_parallel_teams\": "
    << r.restoration_exact_parallel_teams << ",\n";
  o << "  \"restoration_exact_sparse_row_dot_batches\": "
    << r.restoration_exact_sparse_row_dot_batches << ",\n";
  o << "  \"restoration_exact_sparse_row_dot_verifications\": "
    << r.restoration_exact_sparse_row_dot_verifications << ",\n";
  o << "  \"restoration_exact_sparse_row_dot_fallbacks\": "
    << r.restoration_exact_sparse_row_dot_fallbacks << ",\n";
  o << "  \"restoration_line_search_backtracks\": " << r.restoration_line_search_backtracks << ",\n";
  o << "  \"restoration_stagnation_events\": " << r.restoration_stagnation_events << ",\n";
  o << "  \"restoration_minimax_qp_solves\": " << r.restoration_minimax_qp_solves << ",\n";
  o << "  \"restoration_minimax_inner_iterations\": " << r.restoration_minimax_inner_iterations << ",\n";
  o << "  \"restoration_minimax_factorizations\": " << r.restoration_minimax_factorizations << ",\n";
  o << "  \"restoration_minimax_accepted_steps\": " << r.restoration_minimax_accepted_steps << ",\n";
  o << "  \"restoration_minimax_line_search_backtracks\": " << r.restoration_minimax_line_search_backtracks << ",\n";
  o << "  \"restoration_structured_abs_links\": " << r.restoration_structured_abs_links << ",\n";
  o << "  \"restoration_effective_dimension\": " << r.restoration_effective_dimension << ",\n";
  o << "  \"restoration_initial_feasibility\": "; json_number(o, r.restoration_initial_feasibility); o << ",\n";
  o << "  \"restoration_harmonic_feasibility\": "; json_number(o, r.restoration_harmonic_feasibility); o << ",\n";
  o << "  \"restoration_tangent_activation_feasibility\": "; json_number(o, r.restoration_tangent_activation_feasibility); o << ",\n";
  o << "  \"restoration_tangent_activation_step_inf\": "; json_number(o, r.restoration_tangent_activation_step_inf); o << ",\n";
  o << "  \"restoration_harmonic_seconds\": "; json_number(o, r.restoration_harmonic_seconds); o << ",\n";
  o << "  \"restoration_tangent_activation_seconds\": "; json_number(o, r.restoration_tangent_activation_seconds); o << ",\n";
  o << "  \"restoration_legacy_gate_seconds\": "; json_number(o, r.restoration_legacy_gate_seconds); o << ",\n";
  o << "  \"restoration_exact_seconds\": "; json_number(o, r.restoration_exact_seconds); o << ",\n";
  o << "  \"restoration_minimax_seconds\": "; json_number(o, r.restoration_minimax_seconds); o << ",\n";
  o << "  \"restoration_restart_seconds\": "; json_number(o, r.restoration_restart_seconds); o << ",\n";
  o << "  \"restoration_total_seconds\": "; json_number(o, r.restoration_total_seconds); o << ",\n";
  o << "  \"restoration_tangent_activation_gradient_rows\": " << r.restoration_tangent_activation_gradient_rows << ",\n";
  o << "  \"restoration_tangent_activation_line_quadratics\": " << r.restoration_tangent_activation_line_quadratics << ",\n";
  o << "  \"restoration_tangent_activation_candidates\": " << r.restoration_tangent_activation_candidates << ",\n";
  o << "  \"restoration_harmonic_unknowns\": " << r.restoration_harmonic_unknowns << ",\n";
  o << "  \"restoration_harmonic_graph_edges\": " << r.restoration_harmonic_graph_edges << ",\n";
  o << "  \"restoration_final_feasibility\": "; json_number(o, r.restoration_final_feasibility); o << ",\n";
  o << "  \"final_proximal_regularization\": "; json_number(o, r.final_proximal_regularization); o << ",\n";
  o << "  \"outer_iterations\": " << r.outer_iterations << ",\n";
  o << "  \"inner_iterations\": " << r.inner_iterations << ",\n";
  o << "  \"kkt_factorizations\": " << r.kkt_factorizations << ",\n";
  o << "  \"kkt_symbolic_analyses\": " << r.kkt_symbolic_analyses << ",\n";
  o << "  \"kkt_symbolic_reuses\": " << r.kkt_symbolic_reuses << ",\n";
  o << "  \"kkt_symbolic_invalidations\": " << r.kkt_symbolic_invalidations << ",\n";
  o << "  \"kkt_numeric_scatter_builds\": " << r.kkt_numeric_scatter_builds << ",\n";
  o << "  \"kkt_numeric_scatter_reuses\": " << r.kkt_numeric_scatter_reuses << ",\n";
  o << "  \"kkt_numeric_scatter_invalidations\": "
    << r.kkt_numeric_scatter_invalidations << ",\n";
  o << "  \"inner_iteration_limit_fallback_acceptances\": " << r.inner_iteration_limit_fallback_acceptances << ",\n";
  o << "  \"lazyqp_inexact_forcing_attempts\": " << r.lazyqp_inexact_forcing_attempts << ",\n";
  o << "  \"lazyqp_inexact_forcing_acceptances\": " << r.lazyqp_inexact_forcing_acceptances << ",\n";
  o << "  \"lazyqp_inexact_forcing_strict_fallbacks\": " << r.lazyqp_inexact_forcing_strict_fallbacks << ",\n";
  o << "  \"lazyqp_inexact_forcing_outer_retries\": " << r.lazyqp_inexact_forcing_outer_retries << ",\n";
  o << "  \"lazyqp_inexact_forcing_model_evaluations\": "
    << r.lazyqp_inexact_forcing_model_evaluations << ",\n";
  o << "  \"lazyqp_inexact_forcing_max_target\": "; json_number(o, r.lazyqp_inexact_forcing_max_target); o << ",\n";
  o << "  \"lazyqp_model_support_fastpath_requests\": "
    << r.lazyqp_model_support_fastpath_requests << ",\n";
  o << "  \"lazyqp_model_support_fastpath_effective_subproblems\": "
    << r.lazyqp_model_support_fastpath_effective_subproblems << ",\n";
  o << "  \"lazyqp_model_support_builds\": "
    << r.lazyqp_model_support_builds << ",\n";
  o << "  \"lazyqp_model_support_eligible_models\": "
    << r.lazyqp_model_support_eligible_models << ",\n";
  o << "  \"lazyqp_model_support_generic_fallback_models\": "
    << r.lazyqp_model_support_generic_fallback_models << ",\n";
  o << "  \"lazyqp_model_support_value_evaluations\": "
    << r.lazyqp_model_support_value_evaluations << ",\n";
  o << "  \"lazyqp_model_support_gradient_evaluations\": "
    << r.lazyqp_model_support_gradient_evaluations << ",\n";
  o << "  \"dca_single_loop_attempts\": " << r.dca_single_loop_attempts << ",\n";
  o << "  \"dca_single_loop_acceptances\": " << r.dca_single_loop_acceptances << ",\n";
  o << "  \"dca_single_loop_strict_fallbacks\": " << r.dca_single_loop_strict_fallbacks << ",\n";
  o << "  \"dca_single_loop_outer_retries\": " << r.dca_single_loop_outer_retries << ",\n";
  o << "  \"dca_single_loop_state_transport_acceptances\": " << r.dca_single_loop_state_transport_acceptances << ",\n";
  o << "  \"dca_single_loop_state_transport_rejections\": " << r.dca_single_loop_state_transport_rejections << ",\n";
  o << "  \"dca_single_loop_reduced_bootstrap_attempts\": "
    << r.dca_single_loop_reduced_bootstrap_attempts << ",\n";
  o << "  \"dca_single_loop_reduced_bootstrap_acceptances\": "
    << r.dca_single_loop_reduced_bootstrap_acceptances << ",\n";
  o << "  \"dca_single_loop_keyed_transport_attempts\": " << r.dca_single_loop_keyed_transport_attempts << ",\n";
  o << "  \"dca_single_loop_keyed_transport_acceptances\": " << r.dca_single_loop_keyed_transport_acceptances << ",\n";
  o << "  \"dca_single_loop_keyed_transport_rejections\": " << r.dca_single_loop_keyed_transport_rejections << ",\n";
  o << "  \"dca_single_loop_keyed_transport_churn_acceptances\": " << r.dca_single_loop_keyed_transport_churn_acceptances << ",\n";
  o << "  \"dca_single_loop_keyed_transport_matched_rows\": " << r.dca_single_loop_keyed_transport_matched_rows << ",\n";
  o << "  \"dca_single_loop_keyed_transport_new_rows\": " << r.dca_single_loop_keyed_transport_new_rows << ",\n";
  o << "  \"dca_single_loop_keyed_transport_dropped_rows\": " << r.dca_single_loop_keyed_transport_dropped_rows << ",\n";
  o << "  \"dca_single_loop_keyed_transport_centrality_caps\": " << r.dca_single_loop_keyed_transport_centrality_caps << ",\n";
  o << "  \"dca_single_loop_inner_steps\": " << r.dca_single_loop_inner_steps << ",\n";
  o << "  \"dca_single_loop_factorizations\": " << r.dca_single_loop_factorizations << ",\n";
  o << "  \"dca_single_loop_max_initial_residual\": "; json_number(o, r.dca_single_loop_max_initial_residual); o << ",\n";
  o << "  \"dca_single_loop_max_final_residual\": "; json_number(o, r.dca_single_loop_max_final_residual); o << ",\n";
  o << "  \"dca_single_loop_baseline_proximal_regularization\": ";
  json_number(o, r.dca_single_loop_baseline_proximal_regularization); o << ",\n";
  o << "  \"dca_single_loop_direct_start_attempts\": " << r.dca_single_loop_direct_start_attempts << ",\n";
  o << "  \"dca_single_loop_direct_start_acceptances\": " << r.dca_single_loop_direct_start_acceptances << ",\n";
  o << "  \"dca_single_loop_direct_start_rejections_nonfinite\": " << r.dca_single_loop_direct_start_rejections_nonfinite << ",\n";
  o << "  \"dca_single_loop_direct_start_rejections_bounds\": " << r.dca_single_loop_direct_start_rejections_bounds << ",\n";
  o << "  \"dca_single_loop_direct_start_rejections_equalities\": " << r.dca_single_loop_direct_start_rejections_equalities << ",\n";
  o << "  \"dca_single_loop_direct_start_rejections_theta\": " << r.dca_single_loop_direct_start_rejections_theta << ",\n";
  o << "  \"dca_single_loop_direct_start_violated_inequalities\": " << r.dca_single_loop_direct_start_violated_inequalities << ",\n";
  o << "  \"dca_single_loop_direct_start_working_set_rows\": " << r.dca_single_loop_direct_start_working_set_rows << ",\n";
  o << "  \"dca_single_loop_direct_start_filter_initializations\": " << r.dca_single_loop_direct_start_filter_initializations << ",\n";
  o << "  \"dca_single_loop_direct_start_initial_theta\": "; json_number(o, r.dca_single_loop_direct_start_initial_theta); o << ",\n";
  o << "  \"dca_single_loop_direct_start_recovery_attempts\": " << r.dca_single_loop_direct_start_recovery_attempts << ",\n";
  o << "  \"dca_single_loop_direct_start_recovery_acceptances\": " << r.dca_single_loop_direct_start_recovery_acceptances << ",\n";
  o << "  \"dca_single_loop_direct_start_recovery_rejections_nonfinite\": " << r.dca_single_loop_direct_start_recovery_rejections_nonfinite << ",\n";
  o << "  \"dca_single_loop_direct_start_recovery_rejections_bounds\": " << r.dca_single_loop_direct_start_recovery_rejections_bounds << ",\n";
  o << "  \"dca_single_loop_direct_start_recovery_rejections_cap\": " << r.dca_single_loop_direct_start_recovery_rejections_cap << ",\n";
  o << "  \"dca_single_loop_direct_start_recovery_rejections_contraction\": " << r.dca_single_loop_direct_start_recovery_rejections_contraction << ",\n";
  o << "  \"dca_single_loop_direct_start_recovery_rejections_unmodeled\": " << r.dca_single_loop_direct_start_recovery_rejections_unmodeled << ",\n";
  o << "  \"dca_single_loop_direct_start_recovery_working_set_additions\": " << r.dca_single_loop_direct_start_recovery_working_set_additions << ",\n";
  o << "  \"dca_single_loop_direct_start_strict_latches\": " << r.dca_single_loop_direct_start_strict_latches << ",\n";
  o << "  \"dca_single_loop_direct_start_recovery_max_theta\": "; json_number(o, r.dca_single_loop_direct_start_recovery_max_theta); o << ",\n";
  o << "  \"line_search_backtracks\": " << r.line_search_backtracks << ",\n";
  o << "  \"armijo_rejections\": " << r.armijo_rejections << ",\n";
  o << "  \"lazy_dca_equality_model\": \"" << json_escape(r.lazy_dca_equality_model) << "\",\n";
  o << "  \"lazy_dca_elastic_subproblems\": " << r.lazy_dca_elastic_subproblems << ",\n";
  o << "  \"lazy_dca_elastic_accepted_steps\": " << r.lazy_dca_elastic_accepted_steps << ",\n";
  o << "  \"lazy_dca_elastic_last_eta\": "; json_number(o, r.lazy_dca_elastic_last_eta); o << ",\n";
  o << "  \"lazy_dca_linearized_equality_steps\": " << r.lazy_dca_linearized_equality_steps << ",\n";
  o << "  \"lazy_dca_last_equality_step_alpha\": "; json_number(o, r.lazy_dca_last_equality_step_alpha); o << ",\n";
  o << "  \"lazy_dca_equality_objective_steps\": " << r.lazy_dca_equality_objective_steps << ",\n";
  o << "  \"lazy_dca_equality_feasibility_steps\": " << r.lazy_dca_equality_feasibility_steps << ",\n";
  o << "  \"lazy_dca_final_equality_funnel\": "; json_number(o, r.lazy_dca_final_equality_funnel); o << ",\n";
  o << "  \"lazy_dca_equality_retraction_attempts\": " << r.lazy_dca_equality_retraction_attempts << ",\n";
  o << "  \"lazy_dca_equality_retraction_acceptances\": " << r.lazy_dca_equality_retraction_acceptances << ",\n";
  o << "  \"lazy_dca_equality_retraction_rejections\": " << r.lazy_dca_equality_retraction_rejections << ",\n";
  o << "  \"lazy_dca_max_equality_retraction_inf\": "; json_number(o, r.lazy_dca_max_equality_retraction_inf); o << ",\n";
  o << "  \"lazy_dca_condensed_factorizations\": " << r.lazy_dca_condensed_factorizations << ",\n";
  o << "  \"lazy_dca_condensed_bound_face_factorizations\": "
    << r.lazy_dca_condensed_bound_face_factorizations << ",\n";
  o << "  \"lazy_dca_condensed_max_active_bound_coordinates\": "
    << r.lazy_dca_condensed_max_active_bound_coordinates << ",\n";
  o << "  \"lazy_dca_condensed_normal_solves\": " << r.lazy_dca_condensed_normal_solves << ",\n";
  o << "  \"lazy_dca_condensed_geometry_normal_solves\": " << r.lazy_dca_condensed_geometry_normal_solves << ",\n";
  o << "  \"lazy_dca_condensed_retraction_normal_solves\": " << r.lazy_dca_condensed_retraction_normal_solves << ",\n";
  o << "  \"lazy_dca_condensed_symbolic_analyses\": " << r.lazy_dca_condensed_symbolic_analyses << ",\n";
  o << "  \"lazy_dca_condensed_symbolic_reuses\": " << r.lazy_dca_condensed_symbolic_reuses << ",\n";
  o << "  \"lazy_dca_condensed_symbolic_invalidations\": " << r.lazy_dca_condensed_symbolic_invalidations << ",\n";
  o << "  \"lazy_dca_condensed_retraction_attempts\": " << r.lazy_dca_condensed_retraction_attempts << ",\n";
  o << "  \"lazy_dca_condensed_retraction_acceptances\": " << r.lazy_dca_condensed_retraction_acceptances << ",\n";
  o << "  \"lazy_dca_condensed_accepted_steps\": " << r.lazy_dca_condensed_accepted_steps << ",\n";
  o << "  \"lazy_dca_condensed_retraction_rejections\": " << r.lazy_dca_condensed_retraction_rejections << ",\n";
  o << "  \"lazy_dca_condensed_continuation_attempts\": " << r.lazy_dca_condensed_continuation_attempts << ",\n";
  o << "  \"lazy_dca_condensed_continuation_acceptances\": " << r.lazy_dca_condensed_continuation_acceptances << ",\n";
  o << "  \"lazy_dca_condensed_newton_iterations\": " << r.lazy_dca_condensed_newton_iterations << ",\n";
  o << "  \"lazy_dca_condensed_krylov_iterations\": " << r.lazy_dca_condensed_krylov_iterations << ",\n";
  o << "  \"lazy_dca_condensed_backtracks\": " << r.lazy_dca_condensed_backtracks << ",\n";
  o << "  \"lazy_dca_condensed_affine_normal_map_reuses\": " << r.lazy_dca_condensed_affine_normal_map_reuses << ",\n";
  o << "  \"lazy_dca_condensed_newton_endpoint_reuses\": " << r.lazy_dca_condensed_newton_endpoint_reuses << ",\n";
  o << "  \"lazy_dca_condensed_quadratic_linesearch_reuses\": " << r.lazy_dca_condensed_quadratic_linesearch_reuses << ",\n";
  o << "  \"lazy_dca_condensed_quadratic_linesearch_predictor_builds\": " << r.lazy_dca_condensed_quadratic_linesearch_predictor_builds << ",\n";
  o << "  \"lazy_dca_condensed_quadratic_linesearch_exact_checks\": " << r.lazy_dca_condensed_quadratic_linesearch_exact_checks << ",\n";
  o << "  \"lazy_dca_condensed_quadratic_linesearch_early_rejections\": " << r.lazy_dca_condensed_quadratic_linesearch_early_rejections << ",\n";
  o << "  \"lazy_dca_condensed_quadratic_linesearch_rows_skipped\": " << r.lazy_dca_condensed_quadratic_linesearch_rows_skipped << ",\n";
  o << "  \"lazy_dca_condensed_cached_regularization_attempts\": " << r.lazy_dca_condensed_cached_regularization_attempts << ",\n";
  o << "  \"lazy_dca_condensed_cached_regularization_hits\": " << r.lazy_dca_condensed_cached_regularization_hits << ",\n";
  o << "  \"lazy_dca_condensed_sparse_equality_evaluations\": " << r.lazy_dca_condensed_sparse_equality_evaluations << ",\n";
  o << "  \"lazy_dca_condensed_parallel_equality_batches\": " << r.lazy_dca_condensed_parallel_equality_batches << ",\n";
  o << "  \"lazy_dca_condensed_normal_refinement_steps\": " << r.lazy_dca_condensed_normal_refinement_steps << ",\n";
  o << "  \"lazy_dca_condensed_geometry_normal_refinement_steps\": " << r.lazy_dca_condensed_geometry_normal_refinement_steps << ",\n";
  o << "  \"lazy_dca_condensed_retraction_normal_refinement_steps\": " << r.lazy_dca_condensed_retraction_normal_refinement_steps << ",\n";
  o << "  \"lazy_dca_condensed_normal_refinement_zero_solves\": " << r.lazy_dca_condensed_normal_refinement_zero_solves << ",\n";
  o << "  \"lazy_dca_condensed_normal_refinement_one_solves\": " << r.lazy_dca_condensed_normal_refinement_one_solves << ",\n";
  o << "  \"lazy_dca_condensed_normal_refinement_two_solves\": " << r.lazy_dca_condensed_normal_refinement_two_solves << ",\n";
  o << "  \"lazy_dca_condensed_normal_refinement_three_plus_solves\": " << r.lazy_dca_condensed_normal_refinement_three_plus_solves << ",\n";
  o << "  \"lazy_dca_condensed_relaxed_normal_solves\": " << r.lazy_dca_condensed_relaxed_normal_solves << ",\n";
  o << "  \"lazy_dca_condensed_strict_normal_solves\": " << r.lazy_dca_condensed_strict_normal_solves << ",\n";
  o << "  \"lazy_dca_condensed_relaxed_accepts_above_strict_tolerance\": " << r.lazy_dca_condensed_relaxed_accepts_above_strict_tolerance << ",\n";
  o << "  \"lazy_dca_condensed_strict_retry_retractions\": " << r.lazy_dca_condensed_strict_retry_retractions << ",\n";
  o << "  \"lazy_dca_condensed_max_normal_initial_backward_error\": "; json_number(o, r.lazy_dca_condensed_max_normal_initial_backward_error); o << ",\n";
  o << "  \"lazy_dca_condensed_mean_normal_initial_backward_error\": "; json_number(o, r.lazy_dca_condensed_mean_normal_initial_backward_error); o << ",\n";
  o << "  \"lazy_dca_condensed_mean_normal_final_backward_error\": "; json_number(o, r.lazy_dca_condensed_mean_normal_final_backward_error); o << ",\n";
  o << "  \"lazy_dca_condensed_max_normal_backward_error\": "; json_number(o, r.lazy_dca_condensed_max_normal_backward_error); o << ",\n";
  o << "  \"lazy_dca_condensed_gmres_workspace_growths\": " << r.lazy_dca_condensed_gmres_workspace_growths << ",\n";
  o << "  \"lazy_dca_condensed_zero_rhs_solve_skips\": " << r.lazy_dca_condensed_zero_rhs_solve_skips << ",\n";
  o << "  \"lazy_dca_condensed_alpha_predictor_uses\": " << r.lazy_dca_condensed_alpha_predictor_uses << ",\n";
  o << "  \"lazy_dca_condensed_alpha_predictor_full_step_resets\": " << r.lazy_dca_condensed_alpha_predictor_full_step_resets << ",\n";
  o << "  \"lazy_dca_condensed_alpha_predictor_leading_retractions_skipped\": " << r.lazy_dca_condensed_alpha_predictor_leading_retractions_skipped << ",\n";
  o << "  \"lazy_dca_condensed_alpha_predictor_historical_retries\": " << r.lazy_dca_condensed_alpha_predictor_historical_retries << ",\n";
  o << "  \"lazy_dca_condensed_outer_extrapolation_attempts\": " << r.lazy_dca_condensed_outer_extrapolation_attempts << ",\n";
  o << "  \"lazy_dca_condensed_outer_extrapolation_acceptances\": " << r.lazy_dca_condensed_outer_extrapolation_acceptances << ",\n";
  o << "  \"lazy_dca_condensed_outer_extrapolation_historical_fallbacks\": " << r.lazy_dca_condensed_outer_extrapolation_historical_fallbacks << ",\n";
  o << "  \"lazy_dca_condensed_outer_extrapolation_full_step_streak_max\": " << r.lazy_dca_condensed_outer_extrapolation_full_step_streak_max << ",\n";
  o << "  \"lazy_dca_condensed_outer_extrapolation_alpha_max\": "; json_number(o, r.lazy_dca_condensed_outer_extrapolation_alpha_max); o << ",\n";
  o << "  \"lazy_dca_condensed_outer_reject_bound\": " << r.lazy_dca_condensed_outer_reject_bound << ",\n";
  o << "  \"lazy_dca_condensed_outer_reject_side\": " << r.lazy_dca_condensed_outer_reject_side << ",\n";
  o << "  \"lazy_dca_condensed_outer_reject_funnel\": " << r.lazy_dca_condensed_outer_reject_funnel << ",\n";
  o << "  \"lazy_dca_condensed_outer_reject_filter\": " << r.lazy_dca_condensed_outer_reject_filter << ",\n";
  o << "  \"lazy_dca_condensed_outer_reject_objective\": " << r.lazy_dca_condensed_outer_reject_objective << ",\n";
  o << "  \"lazy_dca_condensed_post_retraction_modeled_offenders\": " << r.lazy_dca_condensed_post_retraction_modeled_offenders << ",\n";
  o << "  \"lazy_dca_condensed_post_retraction_omitted_offenders\": " << r.lazy_dca_condensed_post_retraction_omitted_offenders << ",\n";
  o << "  \"lazy_dca_condensed_post_retraction_working_set_expansions\": " << r.lazy_dca_condensed_post_retraction_working_set_expansions << ",\n";
  o << "  \"lazy_dca_condensed_post_retraction_expansion_fallbacks\": " << r.lazy_dca_condensed_post_retraction_expansion_fallbacks << ",\n";
  o << "  \"lazy_dca_condensed_modeled_side_filter_attempts\": " << r.lazy_dca_condensed_modeled_side_filter_attempts << ",\n";
  o << "  \"lazy_dca_condensed_modeled_side_filter_acceptances\": " << r.lazy_dca_condensed_modeled_side_filter_acceptances << ",\n";
  o << "  \"lazy_dca_condensed_modeled_side_filter_recovery_acceptances\": " << r.lazy_dca_condensed_modeled_side_filter_recovery_acceptances << ",\n";
  o << "  \"lazy_dca_condensed_modeled_side_filter_fallbacks\": " << r.lazy_dca_condensed_modeled_side_filter_fallbacks << ",\n";
  o << "  \"lazy_dca_condensed_modeled_side_filter_recovery_margin_rows\": " << r.lazy_dca_condensed_modeled_side_filter_recovery_margin_rows << ",\n";
  o << "  \"lazy_dca_condensed_modeled_side_filter_max_violation\": " << r.lazy_dca_condensed_modeled_side_filter_max_violation << ",\n";
  o << "  \"lazy_dca_condensed_flow_lift_schur_detected_variables\": " << r.lazy_dca_condensed_flow_lift_schur_detected_variables << ",\n";
  o << "  \"lazy_dca_condensed_flow_lift_schur_factor_dimension\": " << r.lazy_dca_condensed_flow_lift_schur_factor_dimension << ",\n";
  o << "  \"lazy_dca_condensed_flow_lift_schur_factorizations\": " << r.lazy_dca_condensed_flow_lift_schur_factorizations << ",\n";
  o << "  \"lazy_dca_condensed_flow_lift_schur_fallbacks\": " << r.lazy_dca_condensed_flow_lift_schur_fallbacks << ",\n";
  o << "  \"lazy_dca_condensed_tangent_curvature_calls\": " << r.lazy_dca_condensed_tangent_curvature_calls << ",\n";
  o << "  \"lazy_dca_condensed_block_exact_attempts\": " << r.lazy_dca_condensed_block_exact_attempts << ",\n";
  o << "  \"lazy_dca_condensed_block_exact_acceptances\": " << r.lazy_dca_condensed_block_exact_acceptances << ",\n";
  o << "  \"lazy_dca_condensed_block_exact_blocks_max\": " << r.lazy_dca_condensed_block_exact_blocks_max << ",\n";
  o << "  \"lazy_dca_condensed_block_exact_active_dimension_max\": " << r.lazy_dca_condensed_block_exact_active_dimension_max << ",\n";
  o << "  \"lazy_dca_condensed_block_exact_max_block_dimension\": " << r.lazy_dca_condensed_block_exact_max_block_dimension << ",\n";
  o << "  \"lazy_dca_condensed_block_partial_spectrum_attempts\": " << r.lazy_dca_condensed_block_partial_spectrum_attempts << ",\n";
  o << "  \"lazy_dca_condensed_block_partial_spectrum_acceptances\": " << r.lazy_dca_condensed_block_partial_spectrum_acceptances << ",\n";
  o << "  \"lazy_dca_condensed_block_partial_spectrum_fallbacks\": " << r.lazy_dca_condensed_block_partial_spectrum_fallbacks << ",\n";
  o << "  \"lazy_dca_condensed_projected_multi_ritz_batches\": " << r.lazy_dca_condensed_projected_multi_ritz_batches << ",\n";
  o << "  \"lazy_dca_condensed_projected_multi_ritz_fallbacks\": " << r.lazy_dca_condensed_projected_multi_ritz_fallbacks << ",\n";
  o << "  \"lazy_dca_condensed_tangent_inertia_attempts\": " << r.lazy_dca_condensed_tangent_inertia_attempts << ",\n";
  o << "  \"lazy_dca_condensed_tangent_inertia_acceptances\": " << r.lazy_dca_condensed_tangent_inertia_acceptances << ",\n";
  o << "  \"lazy_dca_condensed_tangent_inertia_shift_updates\": " << r.lazy_dca_condensed_tangent_inertia_shift_updates << ",\n";
  o << "  \"lazy_dca_condensed_tangent_inertia_shift_max\": "; json_number(o, r.lazy_dca_condensed_tangent_inertia_shift_max); o << ",\n";
  o << "  \"lazy_dca_condensed_tangent_rank_max\": " << r.lazy_dca_condensed_tangent_rank_max << ",\n";
  o << "  \"lazy_dca_condensed_scalar_safeguard_updates_max\": " << r.lazy_dca_condensed_scalar_safeguard_updates_max << ",\n";
  o << "  \"lazy_dca_condensed_tangent_shift_max\": "; json_number(o, r.lazy_dca_condensed_tangent_shift_max); o << ",\n";
  o << "  \"lazy_dca_condensed_initial_tangent_min_eigenvalue\": "; json_number(o, r.lazy_dca_condensed_initial_tangent_min_eigenvalue); o << ",\n";
  o << "  \"lazy_dca_condensed_final_tangent_min_eigenvalue\": "; json_number(o, r.lazy_dca_condensed_final_tangent_min_eigenvalue); o << ",\n";
  o << "  \"lazy_dca_condensed_normal_regularization\": "; json_number(o, r.lazy_dca_condensed_normal_regularization); o << ",\n";
  o << "  \"lazy_dca_condensed_last_residual_inf\": "; json_number(o, r.lazy_dca_condensed_last_residual_inf); o << ",\n";
  o << "  \"lazy_dca_condensed_max_correction_inf\": "; json_number(o, r.lazy_dca_condensed_max_correction_inf); o << ",\n";
  o << "  \"lazy_dca_condensed_max_correction_ratio\": "; json_number(o, r.lazy_dca_condensed_max_correction_ratio); o << ",\n";
  o << "  \"lazy_dca_corrector_attempts\": " << r.lazy_dca_corrector_attempts << ",\n";
  o << "  \"lazy_dca_corrector_acceptances\": " << r.lazy_dca_corrector_acceptances << ",\n";
  o << "  \"lazy_dca_corrector_rejections\": " << r.lazy_dca_corrector_rejections << ",\n";
  o << "  \"lazy_dca_corrector_inner_iterations\": " << r.lazy_dca_corrector_inner_iterations << ",\n";
  o << "  \"lazy_dca_corrector_cold_retries\": " << r.lazy_dca_corrector_cold_retries << ",\n";
  o << "  \"lazy_dca_iterated_soc_attempts\": " << r.lazy_dca_iterated_soc_attempts << ",\n";
  o << "  \"lazy_dca_iterated_soc_acceptances\": " << r.lazy_dca_iterated_soc_acceptances << ",\n";
  o << "  \"lazy_dca_iterated_soc_rejections\": " << r.lazy_dca_iterated_soc_rejections << ",\n";
  o << "  \"lazy_dca_inner_retries\": " << r.lazy_dca_inner_retries << ",\n";
  o << "  \"lazy_dca_cold_retries\": " << r.lazy_dca_cold_retries << ",\n";
  o << "  \"lazy_dca_proximal_retries\": " << r.lazy_dca_proximal_retries << ",\n";
  o << "  \"lazy_dca_step_control_predictions\": " << r.lazy_dca_step_control_predictions << ",\n";
  o << "  \"lazy_dca_step_control_retries\": " << r.lazy_dca_step_control_retries << ",\n";
  o << "  \"lazy_dca_warm_start_suppressions\": " << r.lazy_dca_warm_start_suppressions << ",\n";
  o << "  \"lazy_dca_warm_start_skips\": " << r.lazy_dca_warm_start_skips << ",\n";
  o << "  \"lazy_dca_max_raw_direction_inf\": "; json_number(o, r.lazy_dca_max_raw_direction_inf); o << ",\n";
  o << "  \"lazy_dca_last_raw_direction_inf\": "; json_number(o, r.lazy_dca_last_raw_direction_inf); o << ",\n";
  o << "  \"lazy_dca_last_step_target_inf\": "; json_number(o, r.lazy_dca_last_step_target_inf); o << ",\n";
  o << "  \"lazy_dca_max_model_proximal\": "; json_number(o, r.lazy_dca_max_model_proximal); o << ",\n";
  o << "  \"trust_region_subproblems\": " << r.trust_region_subproblems << ",\n";
  o << "  \"trust_region_accepted_steps\": " << r.trust_region_accepted_steps << ",\n";
  o << "  \"trust_region_rejected_steps\": " << r.trust_region_rejected_steps << ",\n";
  o << "  \"trust_region_contractions\": " << r.trust_region_contractions << ",\n";
  o << "  \"trust_region_expansions\": " << r.trust_region_expansions << ",\n";
  o << "  \"trust_region_feasibility_steps\": " << r.trust_region_feasibility_steps << ",\n";
  o << "  \"final_trust_region_radius\": "; json_number(o, r.final_trust_region_radius); o << ",\n";
  o << "  \"last_trust_region_ratio\": "; json_number(o, r.last_trust_region_ratio); o << ",\n";
  o << "  \"last_trust_region_predicted_reduction\": "; json_number(o, r.last_trust_region_predicted_reduction); o << ",\n";
  o << "  \"last_trust_region_actual_reduction\": "; json_number(o, r.last_trust_region_actual_reduction); o << ",\n";
  o << "  \"last_trust_region_penalty\": "; json_number(o, r.last_trust_region_penalty); o << ",\n";
  o << "  \"last_trust_region_merit_predicted_reduction\": "; json_number(o, r.last_trust_region_merit_predicted_reduction); o << ",\n";
  o << "  \"last_trust_region_merit_actual_reduction\": "; json_number(o, r.last_trust_region_merit_actual_reduction); o << ",\n";
  o << "  \"soc_attempts\": " << r.soc_attempts << ",\n";
  o << "  \"soc_accepted_steps\": " << r.soc_accepted_steps << ",\n";
  o << "  \"soc_rejected_steps\": " << r.soc_rejected_steps << ",\n";
  o << "  \"soc_inner_iterations\": " << r.soc_inner_iterations << ",\n";
  o << "  \"soc_factorizations\": " << r.soc_factorizations << ",\n";
  o << "  \"curvature_calls\": " << r.curvature_calls << ",\n";
  o << "  \"constraint_curvature_seconds\": " << r.constraint_curvature_seconds << ",\n";
  o << "  \"equality_model_seconds\": " << r.equality_model_seconds << ",\n";
  o << "  \"primary_model_build_seconds\": " << r.primary_model_build_seconds << ",\n";
  o << "  \"primary_inner_seconds\": " << r.primary_inner_seconds << ",\n";
  o << "  \"equality_tangent_inertia_attempts\": " << r.equality_tangent_inertia_attempts << ",\n";
  o << "  \"equality_tangent_inertia_acceptances\": " << r.equality_tangent_inertia_acceptances << ",\n";
  o << "  \"equality_tangent_stall_certificates\": " << r.equality_tangent_stall_certificates << ",\n";
  o << "  \"lagrangian_correction_rank\": " << r.objective_rank << ",\n";
  o << "  \"objective_rank\": " << r.objective_rank << ",\n";
  o << "  \"inequality_total_rank\": " << r.inequality_total_rank << ",\n";
  o << "  \"active_inequalities\": " << r.active_inequalities << ",\n";
  o << "  \"working_set_additions\": " << r.working_set_additions << ",\n";
  o << "  \"working_set_drops\": " << r.working_set_drops << ",\n";
  o << "  \"active_jacobian_rank\": " << r.active_jacobian_rank << ",\n";
  o << "  \"tangent_dimension\": " << r.tangent_dimension << ",\n";
  o << "  \"reduced_negative_eigenvalues\": " << r.reduced_negative_eigenvalues << ",\n";
  o << "  \"reduced_min_eigenvalue\": "; json_number(o, r.reduced_min_eigenvalue); o << ",\n";
  o << "  \"terminal_kkt_iterations\": " << r.terminal_kkt_iterations << ",\n";
  o << "  \"terminal_kkt_accepted_steps\": " << r.terminal_kkt_accepted_steps << ",\n";
  o << "  \"terminal_kkt_active_set_drops\": " << r.terminal_kkt_active_set_drops << ",\n";
  o << "  \"terminal_kkt_attempts\": " << r.terminal_kkt_attempts << ",\n";
  o << "  \"terminal_kkt_rejected_steps\": " << r.terminal_kkt_rejected_steps << ",\n";
  o << "  \"terminal_kkt_skipped_unstable\": " << r.terminal_kkt_skipped_unstable << ",\n";
  o << "  \"terminal_kkt_active_stability\": " << r.terminal_kkt_active_stability << ",\n";
  o << "  \"terminal_kkt_multiplier_cutoff\": "; json_number(o, r.terminal_kkt_multiplier_cutoff); o << ",\n";
  if(r.atomic_engine_used||r.dca_filter.used) {
  o << "  \"lazy_newton_dual_only_acceptances\": " << r.lazy_newton.dual_only_acceptances << ",\n";
  o << "  \"lazy_newton_structure_builds\": " << r.lazy_newton.structure_builds << ",\n";
  o << "  \"lazy_newton_reduced_variables\": " << r.lazy_newton.reduced_variables << ",\n";
  o << "  \"lazy_newton_reduced_equalities\": " << r.lazy_newton.reduced_equalities << ",\n";
  o << "  \"lazy_newton_eliminated_variables\": " << r.lazy_newton.eliminated_variables << ",\n";
  o << "  \"lazy_newton_fixed_variables\": " << r.lazy_newton.fixed_variables << ",\n";
  o << "  \"lazy_newton_numeric_updates\": " << r.lazy_newton.numeric_updates << ",\n";
  o << "  \"lazy_newton_factor_attempts\": " << r.lazy_newton.factor_attempts << ",\n";
  o << "  \"lazy_newton_inertia_rejections\": " << r.lazy_newton.inertia_rejections << ",\n";
  o << "  \"lazy_newton_regularization_increases\": " << r.lazy_newton.regularization_increases << ",\n";
  o << "  \"lazy_newton_regularization_decreases\": " << r.lazy_newton.regularization_decreases << ",\n";
  o << "  \"lazy_newton_backend_regularization_attempts\": " << r.lazy_newton.backend_regularization_attempts << ",\n";
  o << "  \"lazy_newton_backend_factor_failures\": " << r.lazy_newton.backend_factor_failures << ",\n";
  o << "  \"lazy_newton_backend_regularized_factor_failures\": " << r.lazy_newton.backend_regularized_factor_failures << ",\n";
  o << "  \"lazy_newton_dual_regularized_factors\": " << r.lazy_newton.dual_regularized_factors << ",\n";
  o << "  \"lazy_newton_curvature_reuses\": " << r.lazy_newton.curvature_reuses << ",\n";
  o << "  \"lazy_newton_unshifted_factors\": " << r.lazy_newton.unshifted_factors << ",\n";
  o << "  \"lazy_newton_unshifted_accepted_steps\": " << r.lazy_newton.unshifted_accepted_steps << ",\n";
  o << "  \"lazy_newton_pivoted_fallbacks\": " << r.lazy_newton.pivoted_fallbacks << ",\n";
  o << "  \"lazy_newton_native_factor_successes\": " << r.lazy_newton.native_factor_successes << ",\n";
  o << "  \"lazy_newton_native_linear_solves\": " << r.lazy_newton.native_linear_solves << ",\n";
  o << "  \"lazy_newton_sparse_numeric_failures\": " << r.lazy_newton.sparse_numeric_failures << ",\n";
  o << "  \"lazy_newton_normal_recovery_attempts\": " << r.lazy_newton.normal_recovery_attempts << ",\n";
  o << "  \"lazy_newton_normal_recovery_successes\": " << r.lazy_newton.normal_recovery_successes << ",\n";
  o << "  \"lazy_newton_normal_recovery_linear_solves\": " << r.lazy_newton.normal_recovery_linear_solves << ",\n";
  o << "  \"lazy_newton_normal_recovery_symbolic_analyses\": " << r.lazy_newton.normal_recovery_symbolic_analyses << ",\n";
  o << "  \"lazy_newton_normal_recovery_symbolic_reuses\": " << r.lazy_newton.normal_recovery_symbolic_reuses << ",\n";
  o << "  \"lazy_newton_first_sparse_failure_iteration\": " << r.lazy_newton.first_sparse_failure_iteration << ",\n";
  o << "  \"lazy_newton_last_sparse_failure_coordinate\": " << r.lazy_newton.last_sparse_failure_coordinate << ",\n";
  o << "  \"lazy_newton_last_native_iteration\": " << r.lazy_newton.last_native_iteration << ",\n";
  o << "  \"lazy_newton_last_normal_recovery_equation\": " << r.lazy_newton.last_normal_recovery_equation << ",\n";
  o << "  \"lazy_newton_symbolic_analyses\": " << r.lazy_newton.symbolic_analyses << ",\n";
  o << "  \"lazy_newton_symbolic_reuses\": " << r.lazy_newton.symbolic_reuses << ",\n";
  o << "  \"lazy_newton_scatter_builds\": " << r.lazy_newton.scatter_builds << ",\n";
  o << "  \"lazy_newton_scatter_reuses\": " << r.lazy_newton.scatter_reuses << ",\n";
  o << "  \"lazy_newton_initial_active_constraints\": " << r.lazy_newton.initial_active_constraints << ",\n";
  o << "  \"lazy_newton_final_active_constraints\": " << r.lazy_newton.final_active_constraints << ",\n";
  o << "  \"lazy_newton_constraint_activations\": " << r.lazy_newton.constraint_activations << ",\n";
  o << "  \"lazy_newton_working_set_retries\": " << r.lazy_newton.working_set_retries << ",\n";
  o << "  \"lazy_newton_line_search_backtracks\": " << r.lazy_newton.line_search_backtracks << ",\n";
  o << "  \"lazy_newton_accepted_steps\": " << r.lazy_newton.accepted_steps << ",\n";
  o << "  \"lazy_newton_full_steps\": " << r.lazy_newton.full_steps << ",\n";
  o << "  \"lazy_newton_direction_residual_evaluations\": " << r.lazy_newton.direction_residual_evaluations << ",\n";
  o << "  \"lazy_newton_direction_corrections\": " << r.lazy_newton.direction_corrections << ",\n";
  o << "  \"lazy_newton_precision_condensation_recoveries\": " << r.lazy_newton.precision_condensation_recoveries << ",\n";
  o << "  \"lazy_newton_precision_condensation_rows\": " << r.lazy_newton.precision_condensation_rows << ",\n";
  o << "  \"lazy_newton_atom_projection_updates\": " << r.lazy_newton.atom_projection_updates << ",\n";
  o << "  \"lazy_newton_atom_projection_builds\": " << r.lazy_newton.atom_projection_builds << ",\n";
  o << "  \"lazy_newton_atom_product_seconds\": " << r.lazy_newton.atom_product_seconds << ",\n";
  o << "  \"lazy_newton_atom_assembly_seconds\": " << r.lazy_newton.atom_assembly_seconds << ",\n";
  o << "  \"lazy_newton_atom_projection_seconds\": " << r.lazy_newton.atom_projection_seconds << ",\n";
  o << "  \"lazy_newton_atom_projection_requests\": " << r.lazy_newton.atom_projection_requests << ",\n";
  o << "  \"lazy_newton_predictor_corrector_fallbacks\": " << r.lazy_newton.predictor_corrector_fallbacks << ",\n";
  o << "  \"lazy_newton_packed_gradients_enabled\": " << (r.lazy_newton.packed_gradients_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_predictor_corrector_enabled\": " << (r.lazy_newton.predictor_corrector_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_predictor_corrector_requested\": " << (r.lazy_newton.predictor_corrector_requested?"true":"false") << ",\n";
  o << "  \"lazy_newton_predictor_corrector_initialization_guard\": " << (r.lazy_newton.predictor_corrector_initialization_guard?"true":"false") << ",\n";
  o << "  \"lazy_newton_linear_solves\": " << r.lazy_newton.linear_solves << ",\n";
  o << "  \"lazy_newton_original_certificates\": " << r.lazy_newton.original_certificates << ",\n";
  o << "  \"lazy_newton_objective_scale\": "; json_number(o,r.lazy_newton.objective_scale); o << ",\n";
  o << "  \"lazy_newton_maximum_regularization\": "; json_number(o,r.lazy_newton.maximum_regularization); o << ",\n";
  o << "  \"lazy_newton_final_regularization\": "; json_number(o,r.lazy_newton.final_regularization); o << ",\n";
  o << "  \"lazy_newton_maximum_backend_primal_regularization\": "; json_number(o,r.lazy_newton.maximum_backend_primal_regularization); o << ",\n";
  o << "  \"lazy_newton_maximum_backend_dual_regularization\": "; json_number(o,r.lazy_newton.maximum_backend_dual_regularization); o << ",\n";
  o << "  \"lazy_newton_maximum_failed_backend_primal_regularization\": "; json_number(o,r.lazy_newton.maximum_failed_backend_primal_regularization); o << ",\n";
  o << "  \"lazy_newton_maximum_failed_backend_dual_regularization\": "; json_number(o,r.lazy_newton.maximum_failed_backend_dual_regularization); o << ",\n";
  o << "  \"lazy_newton_final_backend_primal_regularization\": "; json_number(o,r.lazy_newton.final_backend_primal_regularization); o << ",\n";
  o << "  \"lazy_newton_final_backend_dual_regularization\": "; json_number(o,r.lazy_newton.final_backend_dual_regularization); o << ",\n";
  o << "  \"lazy_newton_minimum_step\": "; json_number(o,r.lazy_newton.minimum_step); o << ",\n";
  o << "  \"lazy_newton_maximum_direction_residual\": "; json_number(o,r.lazy_newton.maximum_direction_residual); o << ",\n";
  o << "  \"lazy_newton_initial_scaled_violation\": "; json_number(o,r.lazy_newton.initial_scaled_violation); o << ",\n";
  o << "  \"lazy_newton_initial_barrier\": "; json_number(o,r.lazy_newton.initial_barrier); o << ",\n";
  o << "  \"lazy_newton_native_factor_seconds\": "; json_number(o,r.lazy_newton.native_factor_seconds); o << ",\n";
  o << "  \"lazy_newton_native_solve_seconds\": "; json_number(o,r.lazy_newton.native_solve_seconds); o << ",\n";
  o << "  \"lazy_newton_normal_recovery_factor_seconds\": "; json_number(o,r.lazy_newton.normal_recovery_factor_seconds); o << ",\n";
  o << "  \"lazy_newton_normal_recovery_solve_seconds\": "; json_number(o,r.lazy_newton.normal_recovery_solve_seconds); o << ",\n";
  o << "  \"lazy_newton_normal_recovery_maximum_shift\": "; json_number(o,r.lazy_newton.normal_recovery_maximum_shift); o << ",\n";
  o << "  \"lazy_newton_curvature_model\": "
    << (r.effective_outer_algorithm=="lazy-newton"?(r.lazy_newton.atomic_regularization_enabled?"\"signed-lagrangian-lazy-source-atoms\"":"\"signed-lagrangian-with-global-inertia-correction\""):"null") << ",\n";
  o << "  \"lazy_newton_equality_atoms\": " << r.lazy_newton.equality_atoms << ",\n";
  o << "  \"lazy_newton_equality_correction_rank\": " << r.lazy_newton.equality_correction_rank << ",\n";
  o << "  \"lazy_newton_completion_atoms\": " << r.lazy_newton.completion_atoms << ",\n";
  o << "  \"lazy_newton_atom_source_analyses\": " << r.lazy_newton.atom_source_analyses << ",\n";
  o << "  \"lazy_newton_atom_spectral_blocks\": " << r.lazy_newton.atom_spectral_blocks << ",\n";
  o << "  \"lazy_newton_atom_large_blocks\": " << r.lazy_newton.atom_large_blocks << ",\n";
  o << "  \"lazy_newton_atom_cached_candidates\": " << r.lazy_newton.atom_cached_candidates << ",\n";
  o << "  \"lazy_newton_atom_creations\": " << r.lazy_newton.atom_creations << ",\n";
  o << "  \"lazy_newton_atom_reactivations\": " << r.lazy_newton.atom_reactivations << ",\n";
  o << "  \"lazy_newton_atom_activation_batches\": " << r.lazy_newton.atom_activation_batches << ",\n";
  o << "  \"lazy_newton_atom_completion_batches\": " << r.lazy_newton.atom_completion_batches << ",\n";
  o << "  \"lazy_newton_atom_layout_updates\": " << r.lazy_newton.atom_layout_updates << ",\n";
  o << "  \"lazy_newton_atom_peak_active\": " << r.lazy_newton.atom_peak_active << ",\n";
  o << "  \"lazy_newton_atom_active\": " << r.lazy_newton.atom_active << ",\n";
  o << "  \"lazy_newton_atom_final_rank\": " << r.lazy_newton.atom_final_rank << ",\n";
  o << "  \"lazy_newton_atom_oracle_calls\": " << r.lazy_newton.atom_oracle_calls << ",\n";
  o << "  \"lazy_newton_atom_oracle_solves\": " << r.lazy_newton.atom_oracle_solves << ",\n";
  o << "  \"lazy_newton_atom_oracle_failures\": " << r.lazy_newton.atom_oracle_failures << ",\n";
  o << "  \"lazy_newton_atom_tangent_witnesses\": " << r.lazy_newton.atom_tangent_witnesses << ",\n";
  o << "  \"lazy_newton_atom_oracle_seconds\": " << r.lazy_newton.atom_oracle_seconds << ",\n";
  o << "  \"lazy_newton_atom_maximum_tangent_error\": " << r.lazy_newton.atom_maximum_tangent_error << ",\n";
  o << "  \"lazy_newton_atom_last_witness_coverage\": " << r.lazy_newton.atom_last_witness_coverage << ",\n";
  o << "  \"lazy_newton_atom_multiplier_transports\": " << r.lazy_newton.atom_multiplier_transports << ",\n";
  o << "  \"lazy_newton_atom_step_decay_enabled\": " << (r.lazy_newton.atom_step_decay_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_atom_transport_enabled\": " << (r.lazy_newton.atom_transport_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_atom_guided_batches\": " << r.lazy_newton.atom_guided_batches << ",\n";
  o << "  \"lazy_newton_sparse_certificate_enabled\": " << (r.lazy_newton.sparse_certificate_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_atom_adaptation_activated\": " << (r.lazy_newton.atom_adaptation_activated?"true":"false") << ",\n";
  o << "  \"lazy_newton_atom_plateau_events\": " << r.lazy_newton.atom_plateau_events << ",\n";
  o << "  \"lazy_newton_atom_guided_after\": " << r.lazy_newton.atom_guided_after << ",\n";
  o << "  \"lazy_newton_atom_guided_enabled\": " << (r.lazy_newton.atom_guided_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_atom_energy_sparsify_enabled\": " << (r.lazy_newton.atom_energy_sparsify_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_atom_rayleigh_gate_enabled\": " << (r.lazy_newton.atom_rayleigh_gate_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_atom_sparsifications\": " << r.lazy_newton.atom_sparsifications << ",\n";
  o << "  \"lazy_newton_atom_dropped_coordinates\": " << r.lazy_newton.atom_dropped_coordinates << ",\n";
  o << "  \"lazy_newton_atom_sparse_sign_rejections\": " << r.lazy_newton.atom_sparse_sign_rejections << ",\n";
  o << "  \"lazy_newton_atom_rayleigh_probes\": " << r.lazy_newton.atom_rayleigh_probes << ",\n";
  o << "  \"lazy_newton_atom_covered_rejections\": " << r.lazy_newton.atom_covered_rejections << ",\n";
  o << "  \"lazy_newton_atom_coverage_observations\": " << r.lazy_newton.atom_coverage_observations << ",\n";
  o << "  \"lazy_newton_atom_last_probe_coverage\": "; json_number(o,r.lazy_newton.atom_last_probe_coverage); o << ",\n";
  o << "  \"lazy_newton_atom_decay\": "; json_number(o,r.lazy_newton.atom_decay); o << ",\n";
  o << "  \"lazy_newton_atom_analysis_seconds\": "; json_number(o,r.lazy_newton.atom_analysis_seconds); o << ",\n";
  o << "  \"lazy_newton_atom_update_seconds\": "; json_number(o,r.lazy_newton.atom_update_seconds); o << ",\n";
  o << "  \"lazy_newton_atom_maximum_trace\": "; json_number(o,r.lazy_newton.atom_maximum_trace); o << ",\n";
  o << "  \"lazy_newton_atom_final_trace\": "; json_number(o,r.lazy_newton.atom_final_trace); o << ",\n";
  o << "  \"lazy_newton_atomic_regularization_enabled\": " << (r.lazy_newton.atomic_regularization_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_atom_ranks_exact\": " << (r.lazy_newton.atom_ranks_exact?"true":"false") << ",\n";
  o << "  \"lazy_newton_objective_atoms\": " << r.lazy_newton.objective_atoms << ",\n";
  o << "  \"lazy_newton_objective_correction_rank\": " << r.lazy_newton.objective_correction_rank << ",\n";
  o << "  \"lazy_newton_inequality_atoms\": " << r.lazy_newton.inequality_atoms << ",\n";
  o << "  \"lazy_newton_inequality_correction_rank\": " << r.lazy_newton.inequality_correction_rank << ",\n";
  o << "  \"lazy_newton_objective_hessian_nnz\": " << r.lazy_newton.objective_hessian_nnz << ",\n";
  o << "  \"lazy_newton_inequality_hessian_nnz\": " << r.lazy_newton.inequality_hessian_nnz << ",\n";
  o << "  \"lazy_newton_equality_hessian_nnz\": " << r.lazy_newton.equality_hessian_nnz << ",\n";
  o << "  \"lazy_newton_quadratic_inequalities\": " << r.lazy_newton.quadratic_inequalities << ",\n";
  o << "  \"lazy_newton_quadratic_equalities\": " << r.lazy_newton.quadratic_equalities << ",\n";
  o << "  \"lazy_newton_kkt_dimension\": " << r.lazy_newton.kkt_dimension << ",\n";
  o << "  \"lazy_newton_kkt_structural_nnz\": " << r.lazy_newton.kkt_structural_nnz << ",\n";
  o << "  \"lazy_newton_dense_constraint_borders\": " << r.lazy_newton.dense_constraint_borders << ",\n";
  o << "  \"lazy_newton_condensation_recoveries\": " << r.lazy_newton.condensation_recoveries << ",\n";
  o << "  \"lazy_newton_promoted_inequalities\": " << r.lazy_newton.promoted_inequalities << ",\n";
  o << "  \"lazy_newton_promoted_bounds\": " << r.lazy_newton.promoted_bounds << ",\n";
  o << "  \"lazy_newton_maximum_condensation_ratio\": "; json_number(o,r.lazy_newton.maximum_condensation_ratio); o << ",\n";
  o << "  \"lazy_newton_diagnostic_seconds\": "; json_number(o,r.lazy_newton.diagnostic_seconds); o << ",\n";
  o << "  \"lazy_newton_live_iterations_enabled\": " << (r.lazy_newton.live_iterations_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_live_curvature_enabled\": " << (r.lazy_newton.live_curvature_enabled?"true":"false") << ",\n";
  o << "  \"lazy_newton_iteration_history\": [";
  for(std::size_t j=0;j<r.lazy_newton.iteration_history.size();++j) {
    const auto& row=r.lazy_newton.iteration_history[j];
    if(j)o << ",";
    o << "\n    {\"event\": \"" << json_escape(row.event) << "\", \"factor_backend\": \"" << json_escape(row.factor_backend) << "\"";
    o << ", \"iteration\": " << row.iteration;
    o << ", \"attempt\": " << row.attempt;
    o << ", \"active_inequalities\": " << row.active_inequalities;
    o << ", \"total_inequalities\": " << row.total_inequalities;
    o << ", \"active_bounds\": " << row.active_bounds;
    o << ", \"active_quadratic_inequalities\": " << row.active_quadratic_inequalities;
    o << ", \"weighted_quadratic_inequalities\": " << row.weighted_quadratic_inequalities;
    o << ", \"weighted_quadratic_equalities\": " << row.weighted_quadratic_equalities;
    o << ", \"active_dense_constraints\": " << row.active_dense_constraints;
    o << ", \"objective_atoms\": " << row.objective_atoms;
    o << ", \"objective_correction_rank\": " << row.objective_correction_rank;
    o << ", \"inequality_atoms\": " << row.inequality_atoms;
    o << ", \"inequality_correction_rank\": " << row.inequality_correction_rank;
    o << ", \"equality_atoms\": " << row.equality_atoms;
    o << ", \"equality_correction_rank\": " << row.equality_correction_rank;
    o << ", \"completion_atoms\": " << row.completion_atoms;
    o << ", \"atomic_active\": " << row.atomic_active;
    o << ", \"atom_ranks_exact\": " << (row.atom_ranks_exact?"true":"false");
    o << ", \"atomic_trace\": "; json_number(o,row.atomic_trace);
    o << ", \"regularization_rank\": " << row.regularization_rank;
    o << ", \"normal_augmentation_rank\": " << row.normal_augmentation_rank;
    o << ", \"factor_attempts\": " << row.factor_attempts;
    o << ", \"linear_solves\": " << row.linear_solves;
    o << ", \"backtracks\": " << row.backtracks;
    o << ", \"activations\": " << row.activations;
    o << ", \"regularization_increases\": " << row.regularization_increases;
    o << ", \"native_attempts\": " << row.native_attempts;
    o << ", \"recovery_attempts\": " << row.recovery_attempts;
    o << ", \"symbolic_analyses\": " << row.symbolic_analyses;
    o << ", \"symbolic_reuses\": " << row.symbolic_reuses;
    o << ", \"condensation_recoveries\": " << row.condensation_recoveries;
    o << ", \"promoted_inequalities\": " << row.promoted_inequalities;
    o << ", \"promoted_bounds\": " << row.promoted_bounds;
    o << ", \"kkt_dimension\": " << row.kkt_dimension;
    o << ", \"has_direction\": " << (row.has_direction?"true":"false");
    o << ", \"curvature_available\": " << (row.curvature_available?"true":"false");
    o << ", \"inertia_satisfied\": " << (row.inertia_satisfied?"true":"false");
    o << ", \"regularization_reused\": " << (row.regularization_reused?"true":"false");
    o << ", \"objective\": "; json_number(o,row.objective);
    o << ", \"inf_eq\": "; json_number(o,row.inf_eq);
    o << ", \"inf_ineq\": "; json_number(o,row.inf_ineq);
    o << ", \"inf_bnd\": "; json_number(o,row.inf_bnd);
    o << ", \"inf_du\": "; json_number(o,row.inf_du);
    o << ", \"comp\": "; json_number(o,row.comp);
    o << ", \"mu\": "; json_number(o,row.mu);
    o << ", \"direction_inf\": "; json_number(o,row.direction_inf);
    o << ", \"step_inf\": "; json_number(o,row.step_inf);
    o << ", \"alpha_primal\": "; json_number(o,row.alpha_primal);
    o << ", \"alpha_dual\": "; json_number(o,row.alpha_dual);
    o << ", \"regularization\": "; json_number(o,row.regularization);
    o << ", \"normal_augmentation_shift\": "; json_number(o,row.normal_augmentation_shift);
    o << ", \"direction_residual\": "; json_number(o,row.direction_residual);
    o << ", \"objective_curvature\": "; json_number(o,row.objective_curvature);
    o << ", \"inequality_curvature\": "; json_number(o,row.inequality_curvature);
    o << ", \"equality_curvature\": "; json_number(o,row.equality_curvature);
    o << ", \"inequality_barrier_curvature\": "; json_number(o,row.inequality_barrier_curvature);
    o << ", \"bound_barrier_curvature\": "; json_number(o,row.bound_barrier_curvature);
    o << ", \"regularization_curvature\": "; json_number(o,row.regularization_curvature);
    o << ", \"lagrangian_curvature\": "; json_number(o,row.lagrangian_curvature);
    o << ", \"barrier_newton_curvature\": "; json_number(o,row.barrier_newton_curvature);
    o << ", \"factor_seconds\": "; json_number(o,row.factor_seconds);
    o << ", \"solve_seconds\": "; json_number(o,row.solve_seconds);
    o << ", \"elapsed_seconds\": "; json_number(o,row.elapsed_seconds);
    o << "}";
  }
  o << "],\n";
  o << "  \"lazy_newton_display_level\": " << r.lazy_newton.display_level << ",\n";
  o << "  \"lazy_newton_core_solve_seconds\": " << r.lazy_newton.core_solve_seconds << ",\n";
  o << "  \"lazy_newton_solver_wall_seconds\": " << r.lazy_newton.solver_wall_seconds << ",\n";
  o << "  \"lazy_newton_majorant_geometry_parallel_batches\": " << r.lazy_newton.majorant_geometry_parallel_batches << ",\n";
  o << "  \"lazy_newton_majorant_geometry_threads\": " << r.lazy_newton.majorant_geometry_threads << ",\n";
  o << "  \"lazy_newton_cost_seconds\": {";
  for(std::size_t i=0;i<newton_cost_names.size();++i){if(i)o<<", ";o<<"\""<<newton_cost_names[i]<<"\": "<<r.lazy_newton.cost_seconds[i];}
  o << "},\n";
  o << "  \"lazy_newton_lagrangian_hessian_products\": " << r.lazy_newton.lagrangian_hessian_products << ",\n";
  o << "  \"lazy_newton_evaluation_calls\": {";
  const char* phases[]={"preparation","core","diagnostics","certificate"};
  const char* families[]={"objective","equalities","inequalities","bound_rows"};
  for(int phase=0;phase<4;++phase){if(phase)o<<", ";o<<"\""<<phases[phase]<<"\": {";
    for(int family=0;family<4;++family){if(family)o<<", ";const auto& c=r.lazy_newton.calls[phase][family];
      o<<"\""<<families[family]<<"\": {\"values\": "<<c.values<<", \"gradients\": "<<c.gradients<<", \"model_values\": "<<c.model_values<<", \"model_gradients\": "<<c.model_gradients
       <<", \"cached_gradient_accumulations\": "<<c.cached_gradient_accumulations<<", \"jacobian_products\": "<<c.jacobian_products
       <<", \"hessian_assemblies\": "<<c.hessian_assemblies<<", \"hessian_products\": "<<c.hessian_products<<"}";
    }o<<"}";
  }o<<"},\n";
  o << "  \"cli_read_seconds\": " << r.cli_read_seconds << ",\n";
  o << "  \"cli_arguments_seconds\": " << r.cli_arguments_seconds << ",\n";
  o << "  \"cli_solver_call_seconds\": " << r.cli_solver_call_seconds << ",\n";
  o << "  \"lazy_newton_setup_seconds\": "; json_number(o,r.lazy_newton.setup_seconds); o << ",\n";
  o << "  \"lazy_newton_evaluation_seconds\": "; json_number(o,r.lazy_newton.evaluation_seconds); o << ",\n";
  o << "  \"lazy_newton_assembly_seconds\": "; json_number(o,r.lazy_newton.assembly_seconds); o << ",\n";
  o << "  \"lazy_newton_factor_seconds\": "; json_number(o,r.lazy_newton.factor_seconds); o << ",\n";
  o << "  \"lazy_newton_solve_seconds\": "; json_number(o,r.lazy_newton.solve_seconds); o << ",\n";
  o << "  \"lazy_newton_globalization_seconds\": "; json_number(o,r.lazy_newton.globalization_seconds); o << ",\n";
  o << "  \"lazy_newton_certificate_seconds\": "; json_number(o,r.lazy_newton.certificate_seconds); o << ",\n";
  } else {
  o << "  \"lazy_newton_iterations\": " << r.lazy_newton_iterations << ",\n";
  o << "  \"lazy_newton_factorizations\": " << r.lazy_newton_factorizations << ",\n";
  o << "  \"lazy_newton_symbolic_analyses\": " << r.lazy_newton_symbolic_analyses << ",\n";
  o << "  \"lazy_newton_inertia_corrections\": " << r.lazy_newton_inertia_corrections << ",\n";
  o << "  \"lazy_newton_refinement_sweeps\": " << r.lazy_newton_refinement_sweeps << ",\n";
  o << "  \"lazy_newton_quadratic_corrections\": " << r.lazy_newton_quadratic_corrections << ",\n";
  o << "  \"lazy_newton_quadratic_correction_acceptances\": " << r.lazy_newton_quadratic_correction_acceptances << ",\n";
  o << "  \"lazy_newton_lazy_cuts\": " << r.lazy_newton_lazy_cuts << ",\n";
  o << "  \"lazy_newton_proximal_shifts\": " << r.lazy_newton_proximal_shifts << ",\n";
  o << "  \"lazy_newton_exact_curvature_steps\": " << r.lazy_newton_exact_curvature_steps << ",\n";
  o << "  \"lazy_newton_majorant_curvature_steps\": " << r.lazy_newton_majorant_curvature_steps << ",\n";
  o << "  \"lazy_newton_curvature_fallbacks\": " << r.lazy_newton_curvature_fallbacks << ",\n";
  o << "  \"lazy_newton_dual_regularizations\": " << r.lazy_newton_dual_regularizations << ",\n";
  o << "  \"lazy_newton_condensed_rows_max\": " << r.lazy_newton_condensed_rows_max << ",\n";
  o << "  \"lazy_newton_augmented_rows_max\": " << r.lazy_newton_augmented_rows_max << ",\n";
  o << "  \"lazy_newton_working_set_final\": " << r.lazy_newton_working_set_final << ",\n";
  o << "  \"lazy_newton_working_set_max\": " << r.lazy_newton_working_set_max << ",\n";
  o << "  \"lazy_newton_kkt_dimension\": " << r.lazy_newton_kkt_dimension << ",\n";
  o << "  \"lazy_newton_kkt_nnz\": " << r.lazy_newton_kkt_nnz << ",\n";
  o << "  \"lazy_newton_factor_nnz\": " << r.lazy_newton_factor_nnz << ",\n";
  o << "  \"lazy_newton_barrier_updates\": " << r.lazy_newton_barrier_updates << ",\n";
  o << "  \"lazy_newton_line_search_backtracks\": " << r.lazy_newton_line_search_backtracks << ",\n";
  o << "  \"lazy_newton_certificate_evaluations\": " << r.lazy_newton_certificate_evaluations << ",\n";
  o << "  \"lazy_newton_final_mu\": "; json_number(o, r.lazy_newton_final_mu); o << ",\n";
  o << "  \"lazy_newton_objective_scale\": "; json_number(o, r.lazy_newton_objective_scale); o << ",\n";
  o << "  \"lazy_newton_max_delta_x\": "; json_number(o, r.lazy_newton_max_delta_x); o << ",\n";
  o << "  \"lazy_newton_setup_seconds\": "; json_number(o, r.lazy_newton_setup_seconds); o << ",\n";
  o << "  \"lazy_newton_factorization_seconds\": "; json_number(o, r.lazy_newton_factorization_seconds); o << ",\n";
  o << "  \"lazy_newton_assembly_seconds\": "; json_number(o, r.lazy_newton_assembly_seconds); o << ",\n";
  o << "  \"lazy_newton_structure_seconds\": "; json_number(o, r.lazy_newton_structure_seconds); o << ",\n";
  o << "  \"lazy_newton_evaluation_seconds\": "; json_number(o, r.lazy_newton_evaluation_seconds); o << ",\n";
  o << "  \"lazy_newton_line_search_seconds\": "; json_number(o, r.lazy_newton_line_search_seconds); o << ",\n";
  o << "  \"lazy_newton_certificate_seconds\": "; json_number(o, r.lazy_newton_certificate_seconds); o << ",\n";
  }
  o << "  \"dca_filter_spectral_inverse_solves\": " << r.dca_filter.spectral_inverse_solves << ",\n";
  o << "  \"dca_filter_small_block_spectra\": " << r.dca_filter.small_block_spectra << ",\n";
  o << "  \"dca_filter_atoms_active_lagrangian\": " << r.dca_filter.atoms_active_lagrangian << ",\n";
  o << "  \"dca_filter_atoms_active_inequalities\": " << r.dca_filter.atoms_active_inequalities << ",\n";
  o << "  \"dca_filter_used\": " << (r.dca_filter.used?"true":"false") << ",\n";
  o << "  \"dca_filter_inequality_majorants_enabled\": " << (r.dca_filter.inequality_majorants_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_majorant_filter_reanchor_enabled\": " << (r.dca_filter.majorant_filter_reanchor_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_majorant_outer_only_enabled\": " << (r.dca_filter.majorant_outer_only_enabled?"true":"false") << ",\n";
  o << "  \"square_cold_restoration_attempted\": " << (r.dca_filter.square_cold_restoration_attempted ? "true" : "false") << ",\n";
  o << "  \"square_cold_restoration_accepted\": " << (r.dca_filter.square_cold_restoration_accepted ? "true" : "false") << ",\n";
  o << "  \"square_cold_restoration_iterations\": " << r.dca_filter.square_cold_restoration_iterations << ",\n";
  o << "  \"square_continuation_attempted\": " << (r.dca_filter.square_continuation_attempted ? "true" : "false") << ",\n";
  o << "  \"square_continuation_accepted\": " << (r.dca_filter.square_continuation_accepted ? "true" : "false") << ",\n";
  o << "  \"square_continuation_stages\": " << r.dca_filter.square_continuation_stages << ",\n";
  o << "  \"square_continuation_trials\": " << r.dca_filter.square_continuation_trials << ",\n";
  o << "  \"square_continuation_iterations\": " << r.dca_filter.square_continuation_iterations << ",\n";
  o << "  \"square_continuation_factors\": " << r.dca_filter.square_continuation_factors << ",\n";
  o << "  \"square_continuation_solves\": " << r.dca_filter.square_continuation_solves << ",\n";
  o << "  \"square_continuation_evaluations\": " << r.dca_filter.square_continuation_evaluations << ",\n";
  o << "  \"square_continuation_gradient_evaluations\": " << r.dca_filter.square_continuation_gradient_evaluations << ",\n";
  o << "  \"square_continuation_precision_rejections\": " << r.dca_filter.square_continuation_precision_rejections << ",\n";
  o << "  \"square_continuation_krylov_iterations\": " << r.dca_filter.square_continuation_krylov_iterations << ",\n";
  o << "  \"square_continuation_reused_directions\": " << r.dca_filter.square_continuation_reused_directions << ",\n";
  o << "  \"square_continuation_parameter\": " << r.dca_filter.square_continuation_parameter << ",\n";
  o << "  \"square_continuation_seconds\": " << r.dca_filter.square_continuation_seconds << ",\n";
  o << "  \"square_continuation_linear_error\": " << r.dca_filter.square_continuation_linear_error << ",\n";
  o << "  \"square_continuation_attempted_linear_error\": " << r.dca_filter.square_continuation_attempted_linear_error << ",\n";
  o << "  \"dca_filter_majorant_original_cache_evaluations\": " << r.dca_filter.majorant_original_cache_evaluations << ",\n";
  o << "  \"dca_filter_majorant_mu_updates\": " << r.dca_filter.majorant_mu_updates << ",\n";
  o << "  \"dca_filter_majorant_barrier_filter_enabled\": " << (r.dca_filter.majorant_barrier_filter_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_majorant_original_filter_enabled\": " << (r.dca_filter.majorant_original_filter_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_majorant_slack_transport_enabled\": " << (r.dca_filter.majorant_slack_transport_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_majorant_slack_transports\": " << r.dca_filter.majorant_slack_transports << ",\n";
  o << "  \"dca_filter_majorant_outer_predictor_corrector_enabled\": " << (r.dca_filter.majorant_outer_predictor_corrector_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_majorant_predictor_directions\": " << r.dca_filter.majorant_predictor_directions << ",\n";
  o << "  \"dca_filter_majorant_corrector_directions\": " << r.dca_filter.majorant_corrector_directions << ",\n";
  o << "  \"dca_filter_majorant_affine_mu\": "; json_number(o,r.dca_filter.majorant_affine_mu); o << ",\n";
  o << "  \"dca_filter_majorant_centering_sigma\": "; json_number(o,r.dca_filter.majorant_centering_sigma); o << ",\n";
  o << "  \"dca_filter_majorant_startup_predictor_corrector_enabled\": " << (r.dca_filter.majorant_startup_predictor_corrector_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_majorant_terminal_single_newton\": " << (r.dca_filter.majorant_terminal_single_newton?"true":"false") << ",\n";
  o << "  \"dca_filter_majorant_startup_predictor_corrector_steps\": " << r.dca_filter.majorant_startup_predictor_corrector_steps << ",\n";
  o << "  \"dca_filter_majorant_terminal_single_newton_steps\": " << r.dca_filter.majorant_terminal_single_newton_steps << ",\n";
  o << "  \"dca_filter_majorant_direction_precision_retries\": " << r.dca_filter.majorant_direction_precision_retries << ",\n";
  o << "  \"dca_filter_majorant_startup_predictor_corrector_seconds\": "; json_number(o,r.dca_filter.majorant_startup_predictor_corrector_seconds); o << ",\n";
  o << "  \"dca_filter_majorant_barrier_phases\": " << r.dca_filter.majorant_barrier_phases << ",\n";
  o << "  \"dca_filter_majorant_barrier_filter_resets\": " << r.dca_filter.majorant_barrier_filter_resets << ",\n";
  o << "  \"dca_filter_majorant_barrier_line_trials\": " << r.dca_filter.majorant_barrier_line_trials << ",\n";
  o << "  \"dca_filter_majorant_barrier_line_acceptances\": " << r.dca_filter.majorant_barrier_line_acceptances << ",\n";
  o << "  \"dca_filter_majorant_barrier_line_rejections\": " << r.dca_filter.majorant_barrier_line_rejections << ",\n";
  o << "  \"dca_filter_majorant_barrier_mu\": "; json_number(o,r.dca_filter.majorant_barrier_mu); o << ",\n";
  o << "  \"dca_filter_majorant_barrier_residual\": "; json_number(o,r.dca_filter.majorant_barrier_residual); o << ",\n";
  o << "  \"dca_filter_majorant_filter_path_builds\": " << r.dca_filter.majorant_filter_path_builds << ",\n";
  o << "  \"dca_filter_majorant_filter_path_evaluations\": " << r.dca_filter.majorant_filter_path_evaluations << ",\n";
  o << "  \"dca_filter_majorant_early_reanchors\": " << r.dca_filter.majorant_early_reanchors << ",\n";
  o << "  \"dca_filter_majorant_curvature_threads\": " << r.dca_filter.majorant_curvature_threads << ",\n";
  o << "  \"dca_filter_majorant_curvature_certificates\": " << r.dca_filter.majorant_curvature_certificates << ",\n";
  o << "  \"dca_filter_majorant_curvature_reuses\": " << r.dca_filter.majorant_curvature_reuses << ",\n";
  o << "  \"dca_filter_majorant_curvature_small_spectra\": " << r.dca_filter.majorant_curvature_small_spectra << ",\n";
  o << "  \"dca_filter_majorant_curvature_analytic_spectra\": " << r.dca_filter.majorant_curvature_analytic_spectra << ",\n";
  o << "  \"dca_filter_majorant_curvature_gershgorin\": " << r.dca_filter.majorant_curvature_gershgorin << ",\n";
  o << "  \"dca_filter_majorant_curvature_seconds\": "; json_number(o,r.dca_filter.majorant_curvature_seconds); o << ",\n";
  o << "  \"dca_filter_majorant_outer_iterations\": " << r.dca_filter.majorant_outer_iterations << ",\n";
  o << "  \"dca_filter_majorant_inner_convergences\": " << r.dca_filter.majorant_inner_convergences << ",\n";
  o << "  \"dca_filter_majorant_reanchors\": " << r.dca_filter.majorant_reanchors << ",\n";
  o << "  \"dca_filter_majorant_reject_factorization\": " << r.dca_filter.majorant_reject_factorization << ",\n";
  o << "  \"dca_filter_majorant_reject_precision\": " << r.dca_filter.majorant_reject_precision << ",\n";
  o << "  \"dca_filter_majorant_reject_linear_backend\": " << r.dca_filter.majorant_reject_linear_backend << ",\n";
  o << "  \"dca_filter_majorant_reject_globalization\": " << r.dca_filter.majorant_reject_globalization << ",\n";
  o << "  \"dca_filter_majorant_reject_domain\": " << r.dca_filter.majorant_reject_domain << ",\n";
  o << "  \"dca_filter_majorant_inner_kkt\": "; json_number(o,r.dca_filter.majorant_inner_kkt); o << ",\n";
  o << "  \"dca_filter_majorant_inner_primal\": "; json_number(o,r.dca_filter.majorant_inner_primal); o << ",\n";
  o << "  \"dca_filter_majorant_inner_dual\": "; json_number(o,r.dca_filter.majorant_inner_dual); o << ",\n";
  o << "  \"dca_filter_majorant_inner_complementarity\": "; json_number(o,r.dca_filter.majorant_inner_complementarity); o << ",\n";
  o << "  \"dca_filter_majorant_gap_max\": "; json_number(o,r.dca_filter.majorant_gap_max); o << ",\n";
  o << "  \"dca_filter_majorant_stationarity_gap\": "; json_number(o,r.dca_filter.majorant_stationarity_gap); o << ",\n";
  o << "  \"dca_filter_majorant_complementarity_gap\": "; json_number(o,r.dca_filter.majorant_complementarity_gap); o << ",\n";
  o << "  \"dca_filter_majorant_slack_transport_error\": "; json_number(o,r.dca_filter.majorant_slack_transport_error); o << ",\n";
  o << "  \"dca_filter_majorant_lambda_attempts\": " << r.dca_filter.majorant_lambda_attempts << ",\n";
  o << "  \"dca_filter_majorant_lambda_acceptances\": " << r.dca_filter.majorant_lambda_acceptances << ",\n";
  o << "  \"dca_filter_majorant_lambda_iterations\": " << r.dca_filter.majorant_lambda_iterations << ",\n";
  o << "  \"dca_filter_majorant_lambda_last_alpha\": "; json_number(o,r.dca_filter.majorant_lambda_last_alpha); o << ",\n";
  o << "  \"majorant_curvature_attempts\": " << r.dca_filter.majorant_curvature_attempts << ",\n";
  o << "  \"majorant_curvature_solves\": " << r.dca_filter.majorant_curvature_solves << ",\n";
  o << "  \"majorant_curvature_acceptances\": " << r.dca_filter.majorant_curvature_acceptances << ",\n";
  o << "  \"majorant_curvature_correction_seconds\": " << r.dca_filter.majorant_curvature_correction_seconds << ",\n";
  o << "  \"dca_filter_majorant_lambda_interpolations\": " << r.dca_filter.majorant_lambda_interpolations << ",\n";
  o << "  \"dca_filter_majorant_lambda_rho_probes\": " << r.dca_filter.majorant_lambda_rho_probes << ",\n";
  o << "  \"dca_filter_majorant_lambda_seconds\": "; json_number(o,r.dca_filter.majorant_lambda_seconds); o << ",\n";
  o << "  \"dca_filter_majorant_regularization\": "; json_number(o,r.dca_filter.majorant_regularization); o << ",\n";
  o << "  \"dca_filter_majorant_last_rejection\": \"" << json_escape(r.dca_filter.majorant_last_rejection) << "\",\n";
  o << "  \"dca_filter_kkt_growth_max\": " << r.dca_filter.kkt_growth_max << ",\n";
  o << "  \"dca_filter_progress_window\": " << r.dca_filter.progress_window << ",\n";
  o << "  \"dca_filter_progress_events\": " << r.dca_filter.progress_events << ",\n";
  o << "  \"dca_filter_aggregate_progress_stalls\": " << r.dca_filter.aggregate_progress_stalls << ",\n";
  o << "  \"dca_filter_dual_progress_stalls\": " << r.dca_filter.dual_progress_stalls << ",\n";
  o << "  \"dca_filter_terminal_rescue_triggers\": " << r.dca_filter.terminal_rescue_triggers << ",\n";
  o << "  \"dca_filter_terminal_rescue_trials\": " << r.dca_filter.terminal_rescue_trials << ",\n";
  o << "  \"dca_filter_terminal_rescue_acceptances\": " << r.dca_filter.terminal_rescue_acceptances << ",\n";
  o << "  \"dca_filter_aggregate_root_trials\": " << r.dca_filter.aggregate_root_trials << ",\n";
  o << "  \"dca_filter_aggregate_root_acceptances\": " << r.dca_filter.aggregate_root_acceptances << ",\n";
  o << "  \"dca_filter_aggregate_curvature_descents\": " << r.dca_filter.aggregate_curvature_descents << ",\n";
  o << "  \"dca_filter_aggregate_curvature_shortcuts\": " << r.dca_filter.aggregate_curvature_shortcuts << ",\n";
  o << "  \"dca_filter_progress_rescue_trials\": " << r.dca_filter.progress_rescue_trials << ",\n";
  o << "  \"dca_filter_progress_rescue_acceptances\": " << r.dca_filter.progress_rescue_acceptances << ",\n";
  o << "  \"dca_filter_progress_last_event\": \"" << json_escape(r.dca_filter.progress_last_event) << "\",\n";
  o << "  \"dca_filter_initial_restoration_requested\": " << (r.dca_filter.initial_restoration_requested ? "true" : "false") << ",\n";
  o << "  \"dca_filter_normal_atom_weighting\": " << (r.dca_filter.normal_atom_weighting ? "true" : "false") << ",\n";
  o << "  \"dca_filter_active_evaluation\": " << (r.dca_filter.active_evaluation ? "true" : "false") << ",\n";
  o << "  \"dca_filter_packed_atoms\": " << (r.dca_filter.packed_atoms ? "true" : "false") << ",\n";
  o << "  \"dca_filter_lazy_merit\": " << (r.dca_filter.lazy_merit ? "true" : "false") << ",\n";
  o << "  \"dca_filter_block_ritz\": " << (r.dca_filter.block_ritz ? "true" : "false") << ",\n";
  o << "  \"dca_filter_anderson\": " << (r.dca_filter.anderson ? "true" : "false") << ",\n";
  o << "  \"dca_filter_certificate_transport\": " << (r.dca_filter.certificate_transport ? "true" : "false") << ",\n";
  o << "  \"dca_filter_sparse_fallback\": " << (r.dca_filter.sparse_fallback ? "true" : "false") << ",\n";
  o << "  \"dca_filter_equilibration\": " << (r.dca_filter.equilibration ? "true" : "false") << ",\n";
  o << "  \"dca_filter_curvature_backend\": \"" << r.dca_filter.curvature_backend << "\",\n";
  o << "  \"dca_filter_pivoted_backend\": \"" << r.dca_filter.pivoted_backend << "\",\n";
  o << "  \"dca_filter_aggregate_enabled\": " << (r.dca_filter.aggregate_enabled ? "true" : "false") << ",\n";
  o << "  \"dca_filter_rho_complete_search_enabled\": " << (r.dca_filter.rho_complete_search_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_strict_finish_enabled\": " << (r.dca_filter.strict_finish_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_aggregate_kkt_layout\": \"" << json_escape(r.dca_filter.aggregate_kkt_layout) << "\",\n";
  o << "  \"dca_filter_aggregate_last_backend_message\": \"" << json_escape(r.dca_filter.aggregate_last_backend_message) << "\",\n";
  o << "  \"dca_filter_aggregate_rejection_policy\": \"" << json_escape(r.dca_filter.aggregate_rejection_policy) << "\",\n";
  o << "  \"dca_filter_aggregate_last_rejection_kind\": \"" << json_escape(r.dca_filter.aggregate_last_rejection_kind) << "\",\n";
  o << "  \"dca_filter_aggregate_fast_assembly\": " << (r.dca_filter.aggregate_fast_assembly ? "true" : "false") << ",\n";
  o << "  \"dca_filter_aggregate_rho_memory\": " << (r.dca_filter.aggregate_rho_memory ? "true" : "false") << ",\n";
  o << "  \"dca_filter_aggregate_lean\": " << (r.dca_filter.aggregate_lean ? "true" : "false") << ",\n";
  o << "  \"dca_filter_aggregate_border_reserve\": " << r.dca_filter.aggregate_border_reserve << ",\n";
  o << "  \"dca_filter_aggregate_kkt_layout_rebuilds\": " << r.dca_filter.aggregate_kkt_layout_rebuilds << ",\n";
  o << "  \"dca_filter_aggregate_kkt_border_capacity\": " << r.dca_filter.aggregate_kkt_border_capacity << ",\n";
  o << "  \"dca_filter_aggregate_kkt_border_assigned\": " << r.dca_filter.aggregate_kkt_border_assigned << ",\n";
  o << "  \"dca_filter_aggregate_kkt_dimension\": " << r.dca_filter.aggregate_kkt_dimension << ",\n";
  o << "  \"dca_filter_aggregate_reject_inertia\": " << r.dca_filter.aggregate_reject_inertia << ",\n";
  o << "  \"dca_filter_aggregate_reject_backend\": " << r.dca_filter.aggregate_reject_backend << ",\n";
  o << "  \"dca_filter_aggregate_reject_numerical_direction\": " << r.dca_filter.aggregate_reject_numerical_direction << ",\n";
  o << "  \"dca_filter_aggregate_reject_domain\": " << r.dca_filter.aggregate_reject_domain << ",\n";
  o << "  \"dca_filter_aggregate_reject_filter\": " << r.dca_filter.aggregate_reject_filter << ",\n";
  o << "  \"dca_filter_aggregate_reject_kkt\": " << r.dca_filter.aggregate_reject_kkt << ",\n";
  o << "  \"dca_filter_aggregate_reject_globalization\": " << r.dca_filter.aggregate_reject_globalization << ",\n";
  o << "  \"dca_filter_aggregate_rho_memory_uses\": " << r.dca_filter.aggregate_rho_memory_uses << ",\n";
  o << "  \"dca_filter_aggregate_step_regularizations\": " << r.dca_filter.aggregate_step_regularizations << ",\n";
  o << "  \"dca_filter_aggregate_last_positive_rho\": " << r.dca_filter.aggregate_last_positive_rho << ",\n";
  o << "  \"dca_filter_aggregate_backend_assembly_seconds\": " << r.dca_filter.aggregate_backend_assembly_seconds << ",\n";
  o << "  \"dca_filter_aggregate_backend_symbolic_seconds\": " << r.dca_filter.aggregate_backend_symbolic_seconds << ",\n";
  o << "  \"dca_filter_aggregate_backend_numeric_seconds\": " << r.dca_filter.aggregate_backend_numeric_seconds << ",\n";
  o << "  \"dca_filter_aggregate_backend_solve_seconds\": " << r.dca_filter.aggregate_backend_solve_seconds << ",\n";
  o << "  \"dca_filter_aggregate_scatter_builds\": " << r.dca_filter.aggregate_scatter_builds << ",\n";
  o << "  \"dca_filter_aggregate_scatter_reuses\": " << r.dca_filter.aggregate_scatter_reuses << ",\n";
  o << "  \"dca_filter_aggregate_scatter_invalidations\": " << r.dca_filter.aggregate_scatter_invalidations << ",\n";
  o << "  \"dca_filter_model_layout_builds\": " << r.dca_filter.model_layout_builds << ",\n";
  o << "  \"dca_filter_model_layout_seconds\": " << r.dca_filter.model_layout_seconds << ",\n";
  o << "  \"dca_filter_original_point_calls\": " << r.dca_filter.original_point_calls << ",\n";
  o << "  \"dca_filter_original_point_value_reuses\": " << r.dca_filter.original_point_value_reuses << ",\n";
  o << "  \"dca_filter_original_point_seconds\": " << r.dca_filter.original_point_seconds << ",\n";
  o << "  \"dca_filter_aggregate_only_enabled\": " << (r.dca_filter.aggregate_only_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_aggregate_affine_retry_enabled\": " << (r.dca_filter.aggregate_affine_retry_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_aggregate_only_retry_budget\": " << r.dca_filter.aggregate_only_retry_budget << ",\n";
  o << "  \"dca_filter_aggregate_only_attempts\": " << r.dca_filter.aggregate_only_attempts << ",\n";
  o << "  \"dca_filter_aggregate_only_exhaustions\": " << r.dca_filter.aggregate_only_exhaustions << ",\n";
  o << "  \"dca_filter_aggregate_only_inertia_rejections\": " << r.dca_filter.aggregate_only_inertia_rejections << ",\n";
  o << "  \"dca_filter_aggregate_only_step_rejections\": " << r.dca_filter.aggregate_only_step_rejections << ",\n";
  o << "  \"dca_filter_aggregate_only_rho_initial\": " << r.dca_filter.aggregate_only_rho_initial << ",\n";
  o << "  \"dca_filter_aggregate_only_rho_growth\": " << r.dca_filter.aggregate_only_rho_growth << ",\n";
  o << "  \"dca_filter_aggregate_only_rho_decay\": " << r.dca_filter.aggregate_only_rho_decay << ",\n";
  o << "  \"dca_filter_aggregate_only_rho_cap\": " << r.dca_filter.aggregate_only_rho_cap << ",\n";
  o << "  \"dca_filter_rho_grid_completions\": " << r.dca_filter.rho_grid_completions << ",\n";
  o << "  \"dca_filter_strict_finish_budget\": " << r.dca_filter.strict_finish_budget << ",\n";
  o << "  \"dca_filter_strict_finish_refinement_budget\": " << r.dca_filter.strict_finish_refinement_budget << ",\n";
  o << "  \"dca_filter_strict_finish_mu_fraction\": "; json_number(o,r.dca_filter.strict_finish_mu_fraction); o << ",\n";
  o << "  \"dca_filter_strict_finish_mu_target\": "; json_number(o,r.dca_filter.strict_finish_mu_target); o << ",\n";
  o << "  \"dca_filter_strict_finish_eta\": "; json_number(o,r.dca_filter.strict_finish_eta); o << ",\n";
  o << "  \"dca_filter_strict_finish_attempts\": " << r.dca_filter.strict_finish_attempts << ",\n";
  o << "  \"dca_filter_strict_finish_acceptances\": " << r.dca_filter.strict_finish_acceptances << ",\n";
  o << "  \"dca_filter_strict_finish_rejections\": " << r.dca_filter.strict_finish_rejections << ",\n";
  o << "  \"dca_filter_strict_finish_refinements\": " << r.dca_filter.strict_finish_refinements << ",\n";
  o << "  \"dca_filter_strict_finish_certificates\": " << r.dca_filter.strict_finish_certificates << ",\n";
  o << "  \"dca_filter_strict_finish_solved\": " << r.dca_filter.strict_finish_solved << ",\n";
  o << "  \"dca_filter_strict_finish_seconds\": "; json_number(o,r.dca_filter.strict_finish_seconds); o << ",\n";
  o << "  \"dca_filter_strict_finish_linear_residual\": "; json_number(o,r.dca_filter.strict_finish_linear_residual); o << ",\n";
  o << "  \"dca_filter_strict_finish_error_before\": "; json_number(o,r.dca_filter.strict_finish_error_before); o << ",\n";
  o << "  \"dca_filter_strict_finish_error_after\": "; json_number(o,r.dca_filter.strict_finish_error_after); o << ",\n";
  o << "  \"dca_filter_strict_finish_last_rejection\": \"" << json_escape(r.dca_filter.strict_finish_last_rejection) << "\",\n";
  o << "  \"dca_filter_aggregate_preparation_cache_enabled\": " << (r.dca_filter.aggregate_preparation_cache_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_aggregate_strict_probe_stop_enabled\": " << (r.dca_filter.aggregate_strict_probe_stop_enabled?"true":"false") << ",\n";
  o << "  \"dca_filter_aggregate_preparation_requests\": " << r.dca_filter.aggregate_preparation_requests << ",\n";
  o << "  \"dca_filter_aggregate_preparation_builds\": " << r.dca_filter.aggregate_preparation_builds << ",\n";
  o << "  \"dca_filter_aggregate_preparation_hits\": " << r.dca_filter.aggregate_preparation_hits << ",\n";
  o << "  \"dca_filter_aggregate_preparation_invalidations\": " << r.dca_filter.aggregate_preparation_invalidations << ",\n";
  o << "  \"dca_filter_aggregate_preparation_seconds\": " << r.dca_filter.aggregate_preparation_seconds << ",\n";
  o << "  \"dca_filter_aggregate_preparation_key_seconds\": " << r.dca_filter.aggregate_preparation_key_seconds << ",\n";
  o << "  \"dca_filter_aggregate_preparation_geometry_seconds\": " << r.dca_filter.aggregate_preparation_geometry_seconds << ",\n";
  o << "  \"dca_filter_aggregate_preparation_operator_seconds\": " << r.dca_filter.aggregate_preparation_operator_seconds << ",\n";
  o << "  \"dca_filter_aggregate_preparation_peak_bytes\": " << r.dca_filter.aggregate_preparation_peak_bytes << ",\n";
  o << "  \"dca_filter_aggregate_strict_probe_checks\": " << r.dca_filter.aggregate_strict_probe_checks << ",\n";
  o << "  \"dca_filter_aggregate_strict_probe_successes\": " << r.dca_filter.aggregate_strict_probe_successes << ",\n";
  o << "  \"dca_filter_aggregate_strict_probe_failures\": " << r.dca_filter.aggregate_strict_probe_failures << ",\n";
  o << "  \"dca_filter_aggregate_strict_probe_skipped\": " << r.dca_filter.aggregate_strict_probe_skipped << ",\n";
  o << "  \"dca_filter_aggregate_strict_probe_seconds\": " << r.dca_filter.aggregate_strict_probe_seconds << ",\n";
  o << "  \"dca_filter_aggregate_trials\": " << r.dca_filter.aggregate_trials << ",\n";
  o << "  \"dca_filter_aggregate_acceptances\": " << r.dca_filter.aggregate_acceptances << ",\n";
  o << "  \"dca_filter_aggregate_certificates\": " << r.dca_filter.aggregate_certificates << ",\n";
  o << "  \"dca_filter_aggregate_soc\": " << r.dca_filter.aggregate_soc << ",\n";
  o << "  \"dca_filter_aggregate_seconds\": " << r.dca_filter.aggregate_seconds << ",\n";
  o << "  \"dca_filter_aggregate_residual\": " << r.dca_filter.aggregate_residual << ",\n";
  o << "  \"dca_filter_aggregate_factor_seconds\": " << r.dca_filter.aggregate_factor_seconds << ",\n";
  o << "  \"dca_filter_aggregate_max_accepted_residual\": " << r.dca_filter.aggregate_max_accepted_residual << ",\n";
  o << "  \"dca_filter_aggregate_last_rho\": " << r.dca_filter.aggregate_last_rho << ",\n";
  o << "  \"dca_filter_aggregate_max_accepted_rho\": " << r.dca_filter.aggregate_max_accepted_rho << ",\n";
  o << "  \"dca_filter_aggregate_linear_solves\": " << r.dca_filter.aggregate_linear_solves << ",\n";
  o << "  \"dca_filter_aggregate_symbolic_analyses\": " << r.dca_filter.aggregate_symbolic_analyses << ",\n";
  o << "  \"dca_filter_aggregate_symbolic_reuses\": " << r.dca_filter.aggregate_symbolic_reuses << ",\n";
  o << "  \"dca_filter_aggregate_rejections\": " << r.dca_filter.aggregate_rejections << ",\n";
  o << "  \"dca_filter_aggregate_backend\": \"" << r.dca_filter.aggregate_backend << "\",\n";
  o << "  \"dca_filter_aggregate_last_rejection\": \"" << r.dca_filter.aggregate_last_rejection << "\",\n";
  o << "  \"dca_filter_krylov_enabled\": " << (r.dca_filter.krylov_enabled ? "true" : "false") << ",\n";
  o << "  \"dca_filter_krylov_predictor_corrector\": " << (r.dca_filter.krylov_predictor_corrector ? "true" : "false") << ",\n";
  o << "  \"dca_filter_krylov_budget\": " << r.dca_filter.krylov_budget << ",\n";
  o << "  \"dca_filter_krylov_cooldown\": " << r.dca_filter.krylov_cooldown << ",\n";
  o << "  \"dca_filter_krylov_trials\": " << r.dca_filter.krylov_trials << ",\n";
  o << "  \"dca_filter_krylov_acceptances\": " << r.dca_filter.krylov_acceptances << ",\n";
  o << "  \"dca_filter_krylov_iterations\": " << r.dca_filter.krylov_iterations << ",\n";
  o << "  \"dca_filter_krylov_solves\": " << r.dca_filter.krylov_solves << ",\n";
  o << "  \"dca_filter_krylov_linear_rejections\": " << r.dca_filter.krylov_linear_rejections << ",\n";
  o << "  \"dca_filter_krylov_filter_rejections\": " << r.dca_filter.krylov_filter_rejections << ",\n";
  o << "  \"dca_filter_krylov_skips\": " << r.dca_filter.krylov_skips << ",\n";
  o << "  \"dca_filter_krylov_operator_products\": " << r.dca_filter.krylov_operator_products << ",\n";
  o << "  \"dca_filter_krylov_preconditioner_solves\": " << r.dca_filter.krylov_preconditioner_solves << ",\n";
  o << "  \"dca_filter_krylov_seconds\": " << r.dca_filter.krylov_seconds << ",\n";
  o << "  \"dca_filter_krylov_max_accepted_residual\": " << r.dca_filter.krylov_max_accepted_residual << ",\n";
  o << "  \"dca_filter_krylov_last_dimension\": " << r.dca_filter.krylov_last_dimension << ",\n";
  o << "  \"dca_filter_krylov_last_lifted_rows\": " << r.dca_filter.krylov_last_lifted_rows << ",\n";
  o << "  \"dca_filter_krylov_last_iterations\": " << r.dca_filter.krylov_last_iterations << ",\n";
  o << "  \"dca_filter_krylov_last_original_residual\": "; json_number(o, r.dca_filter.krylov_last_original_residual); o << ",\n";
  o << "  \"dca_filter_krylov_last_reduced_residual\": "; json_number(o, r.dca_filter.krylov_last_reduced_residual); o << ",\n";
  o << "  \"dca_filter_krylov_last_dual_residual\": "; json_number(o, r.dca_filter.krylov_last_dual_residual); o << ",\n";
  o << "  \"dca_filter_krylov_last_equality_residual\": "; json_number(o, r.dca_filter.krylov_last_equality_residual); o << ",\n";
  o << "  \"dca_filter_krylov_last_primal_residual\": "; json_number(o, r.dca_filter.krylov_last_primal_residual); o << ",\n";
  o << "  \"dca_filter_krylov_last_complementarity_residual\": "; json_number(o, r.dca_filter.krylov_last_complementarity_residual); o << ",\n";
  o << "  \"dca_filter_krylov_last_rejection\": \"" << json_escape(r.dca_filter.krylov_last_rejection) << "\",\n";
  o << "  \"dca_filter_krylov_last_linear_message\": \"" << json_escape(r.dca_filter.krylov_last_linear_message) << "\",\n";
  o << "  \"dca_filter_krylov_first_accept_seconds\": " << r.dca_filter.krylov_first_accept_seconds << ",\n";
  o << "  \"dca_filter_krylov_first_accept_iteration\": " << r.dca_filter.krylov_first_accept_iteration << ",\n";
  o << "  \"dca_filter_local_trials\": " << r.dca_filter.local_trials << ",\n";
  o << "  \"dca_filter_local_certificates\": " << r.dca_filter.local_certificates << ",\n";
  o << "  \"dca_filter_local_acceptances\": " << r.dca_filter.local_acceptances << ",\n";
  o << "  \"dca_filter_local_pairs\": " << r.dca_filter.local_pairs << ",\n";
  o << "  \"dca_filter_local_fill\": " << r.dca_filter.local_fill << ",\n";
  o << "  \"dca_filter_local_strong\": " << r.dca_filter.local_strong << ",\n";
  o << "  \"dca_filter_local_soc\": " << r.dca_filter.local_soc << ",\n";
  o << "  \"dca_filter_local_seconds\": " << r.dca_filter.local_seconds << ",\n";
  o << "  \"dca_filter_local_residual\": " << r.dca_filter.local_residual << ",\n";
  o << "  \"dca_filter_anderson_trials\": " << r.dca_filter.anderson_trials << ",\n";
  o << "  \"dca_filter_anderson_acceptances\": " << r.dca_filter.anderson_acceptances << ",\n";
  o << "  \"dca_filter_tangent_certificate\": " << (r.dca_filter.tangent_certificate ? "true" : "false") << ",\n";
  o << "  \"dca_filter_certificate_transport_trials\": " << r.dca_filter.certificate_transport_trials << ",\n";
  o << "  \"dca_filter_certificate_transports\": " << r.dca_filter.certificate_transports << ",\n";
  o << "  \"dca_filter_rho_refinement_budget\": " << r.dca_filter.rho_refinement_budget << ",\n";
  o << "  \"dca_filter_rho_refinement_trials\": " << r.dca_filter.rho_refinement_trials << ",\n";
  o << "  \"dca_filter_rho_refinement_acceptances\": " << r.dca_filter.rho_refinement_acceptances << ",\n";
  o << "  \"dca_filter_model_reuses\": " << r.dca_filter.model_reuses << ",\n";
  o << "  \"dca_filter_maximum_model_steps\": " << r.dca_filter.maximum_model_steps << ",\n";
  o << "  \"dca_filter_current_model_steps\": " << r.dca_filter.current_model_steps << ",\n";
  o << "  \"dca_filter_curvature_factor_reuses\": " << r.dca_filter.curvature_factor_reuses << ",\n";
  o << "  \"dca_filter_subspace_spectra\": " << r.dca_filter.subspace_spectra << ",\n";
  o << "  \"dca_filter_dependent_witnesses\": " << r.dca_filter.dependent_witnesses << ",\n";
  o << "  \"dca_filter_curvature_ritz_shift_proposals\": " << r.dca_filter.curvature_ritz_shift_proposals << ",\n";
  o << "  \"dca_filter_curvature_gershgorin_fallbacks\": " << r.dca_filter.curvature_gershgorin_fallbacks << ",\n";
  o << "  \"dca_filter_corrector_steps\": " << r.dca_filter.corrector_steps << ",\n";
  o << "  \"dca_filter_centering_trials\": " << r.dca_filter.centering_trials << ",\n";
  o << "  \"dca_filter_original_line_trials\": " << r.dca_filter.original_line_trials << ",\n";
  o << "  \"dca_filter_original_line_acceptances\": " << r.dca_filter.original_line_acceptances << ",\n";
  o << "  \"dca_filter_original_line_rejections\": " << r.dca_filter.original_line_rejections << ",\n";
  o << "  \"dca_filter_private_line_trials\": " << r.dca_filter.private_line_trials << ",\n";
  o << "  \"dca_filter_private_line_rejections\": " << r.dca_filter.private_line_rejections << ",\n";
  o << "  \"dca_filter_curvature_ensure_seconds\": " << r.dca_filter.curvature_ensure_seconds << ",\n";
  o << "  \"dca_filter_curvature_factor_seconds\": " << r.dca_filter.curvature_factor_seconds << ",\n";
  o << "  \"dca_filter_curvature_inverse_seconds\": " << r.dca_filter.curvature_inverse_seconds << ",\n";
  o << "  \"dca_filter_subspace_seconds\": " << r.dca_filter.subspace_seconds << ",\n";
  o << "  \"dca_filter_last_penalty\": " << r.dca_filter.last_penalty << ",\n";
  o << "  \"dca_filter_last_sigma\": " << r.dca_filter.last_sigma << ",\n";
  o << "  \"dca_filter_extrapolation_trials\": " << r.dca_filter.extrapolation_trials << ",\n";
  o << "  \"dca_filter_extrapolation_acceptances\": " << r.dca_filter.extrapolation_acceptances << ",\n";
  o << "  \"dca_filter_extrapolation_seconds\": " << r.dca_filter.extrapolation_seconds << ",\n";
  o << "  \"dca_filter_basis_compressions\": " << r.dca_filter.basis_compressions << ",\n";
  o << "  \"dca_atom_basis_keep\": " << r.dca_filter.atom_basis_keep << ",\n";
  o << "  \"dca_atom_basis_tail\": " << r.dca_filter.atom_basis_tail << ",\n";
  o << "  \"dca_atom_probe_depth\": " << r.dca_filter.atom_probe_depth << ",\n";
  o << "  \"dca_atom_probe_count_projection\": " << (r.dca_filter.atom_probe_count_projection?"true":"false") << ",\n";
  o << "  \"dca_filter_retired_directions\": " << r.dca_filter.retired_directions << ",\n";
  o << "  \"dca_filter_cumulative_atoms_lagrangian\": " << r.dca_filter.cumulative_atoms_lagrangian << ",\n";
  o << "  \"dca_filter_peak_stored_atoms\": " << r.dca_filter.peak_stored_atoms << ",\n";
  o << "  \"dca_filter_models\": " << r.dca_filter.models << ",\n";
  o << "  \"dca_filter_hessian_updates\": " << r.dca_filter.hessian_updates << ",\n";
  o << "  \"dca_filter_inner_steps\": " << r.dca_filter.inner_steps << ",\n";
  o << "  \"dca_filter_filter_trials\": " << r.dca_filter.filter_trials << ",\n";
  o << "  \"dca_filter_filter_rejections\": " << r.dca_filter.filter_rejections << ",\n";
  o << "  \"dca_filter_private_continuations\": " << r.dca_filter.private_continuations << ",\n";
  o << "  \"dca_filter_model_retries\": " << r.dca_filter.model_retries << ",\n";
  o << "  \"dca_filter_soc_attempts\": " << r.dca_filter.soc_attempts << ",\n";
  o << "  \"dca_filter_soc_acceptances\": " << r.dca_filter.soc_acceptances << ",\n";
  o << "  \"dca_filter_restorations\": " << r.dca_filter.restorations << ",\n";
  o << "  \"dca_filter_atom_events\": " << r.dca_filter.atom_events << ",\n";
  o << "  \"dca_filter_atoms_lagrangian\": " << r.dca_filter.atoms_lagrangian << ",\n";
  o << "  \"dca_filter_atoms_inequalities\": " << r.dca_filter.atoms_inequalities << ",\n";
  o << "  \"dca_filter_curvature_factors\": " << r.dca_filter.curvature_factors << ",\n";
  o << "  \"dca_filter_curvature_reuses\": " << r.dca_filter.curvature_reuses << ",\n";
  o << "  \"dca_filter_first_step_acceptances\": " << r.dca_filter.first_step_acceptances << ",\n";
  o << "  \"dca_filter_eliminated_normal_completion_weight\": "; json_number(o,r.dca_filter.eliminated_normal_completion_weight); o << ",\n";
  o << "  \"dca_filter_normal_completion_weight\": "; json_number(o,r.dca_filter.normal_completion_weight); o << ",\n";
  o << "  \"dca_filter_rho_lagrangian\": "; json_number(o,r.dca_filter.rho_lagrangian); o << ",\n";
  o << "  \"dca_filter_rho_inequalities_max\": "; json_number(o,r.dca_filter.rho_inequalities_max); o << ",\n";
  o << "  \"dca_filter_model_build_seconds\": "; json_number(o,r.dca_filter.model_build_seconds); o << ",\n";
  o << "  \"dca_filter_original_filter_seconds\": "; json_number(o,r.dca_filter.original_filter_seconds); o << ",\n";
  o << "  \"dca_filter_curvature_seconds\": "; json_number(o,r.dca_filter.curvature_seconds); o << ",\n";
  o << "  \"dca_filter_soc_seconds\": "; json_number(o,r.dca_filter.soc_seconds); o << ",\n";
  o << "  \"atomic_engine_used\": " << (r.atomic_engine_used?"true":"false") << ",\n";
  o << "  \"elapsed_seconds\": "; json_number(o, r.elapsed_seconds); o << ",\n";
  const auto emit_vector=[&](const char* name,const Vec& v) {
    o << "  \"" << name << "\": [";
    for(std::size_t i=0;i<v.size();++i) {
      if(i) o << ", ";
      json_number(o,v[i]);
    }
    o << "],\n";
  };
  emit_vector("certificate_equality_multipliers",r.certificate_equality_multipliers);
  emit_vector("certificate_inequality_multipliers",r.certificate_inequality_multipliers);
  emit_vector("certificate_lower_multipliers",r.certificate_lower_multipliers);
  emit_vector("certificate_upper_multipliers",r.certificate_upper_multipliers);
  o << "  \"x\": [";
  for (std::size_t i = 0; i < r.x.size(); ++i) {
    if (i) o << ", ";
    json_number(o, r.x[i]);
  }
  o << "]\n}";
  return o.str();
}

void write_result_json(const Result& r, const std::string& problem_name, const std::string& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write result JSON: " + path);
  out << result_json(r, problem_name) << "\n";
}

} // namespace lazyipdca::qcqp
