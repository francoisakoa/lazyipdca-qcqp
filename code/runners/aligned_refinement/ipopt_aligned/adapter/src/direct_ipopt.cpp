#include "lazyipdca_qcqp/direct_ipopt.hpp"

#include "lazyipdca_qcqp/qcqp_ipopt_termination.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include "lazyipdca_qcqp/version.hpp"
#include "accurate_product_sum.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef LAZYIPDCA_HAVE_IPOPT
#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpIpoptCalculatedQuantities.hpp>
#include <coin-or/IpSolveStatistics.hpp>
#include <coin-or/IpTNLP.hpp>
#include <coin-or/IpIpoptData.hpp>
#include <coin-or/IpIteratesVector.hpp>
#endif

namespace lazyipdca::qcqp {
namespace {

void check_x(const QCQPProblem& problem, const Vec& x, const char* where) {
  if (static_cast<int>(x.size()) != problem.n)
    throw std::invalid_argument(std::string(where) + ": dimension mismatch");
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const unsigned char c : text) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

void json_number(std::ostringstream& out, double value) {
  if (std::isfinite(value)) out << std::setprecision(17) << value;
  else out << "null";
}

void json_vector(std::ostringstream& out, const Vec& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ", ";
    json_number(out, values[i]);
  }
  out << ']';
}

} // namespace

OriginalQCQPDerivatives::OriginalQCQPDerivatives(
    const QCQPProblem& problem) : problem_(&problem) {
  problem.validate();

  constraints_.reserve(problem.equalities.size() +
                       problem.inequalities.size());
  for (const auto& function : problem.equalities)
    constraints_.push_back(&function);
  for (const auto& function : problem.inequalities)
    constraints_.push_back(&function);

  auto make_plan = [&](const QuadraticFunction& function,
                       std::size_t jacobian_offset) {
    if (!function.signed_square.empty())
      throw std::invalid_argument("Ipopt QCQP adapter rejects nonquadratic signed_square");
    QuadraticPlan plan;
    plan.function = &function;
    plan.jacobian_offset = jacobian_offset;

    std::set<int> support;
    function.for_each_linear_nonzero(
        [&](int column, double) { support.insert(column); });
    for (const auto& entry : function.hessian.entries) {
      if (entry.v == 0.0) continue;
      support.insert(entry.i);
      support.insert(entry.j);
    }
    plan.columns.assign(support.begin(), support.end());
    plan.linear_values.assign(plan.columns.size(), 0.0);
    // A sparse row must not allocate/reset an n-vector. PGLIB has >10^6
    // rows; lookup is O(log(row support)), storage O(row support).
    const auto position = [&](int column) -> std::size_t {
      return static_cast<std::size_t>(std::lower_bound(
          plan.columns.begin(), plan.columns.end(), column) - plan.columns.begin());
    };
    function.for_each_linear_nonzero([&](int column, double value) {
      plan.linear_values[position(column)] = value;
    });
    plan.gradient_terms.reserve(function.hessian.entries.size());
    for (const auto& entry : function.hessian.entries) {
      if (entry.v == 0.0) continue;
      plan.gradient_terms.push_back({
          position(entry.i),
          position(entry.j),
          entry.i, entry.j, entry.v});
    }
    return plan;
  };

  objective_plan_ = make_plan(problem.objective, 0);
  constraint_plans_.reserve(constraints_.size());
  std::size_t jacobian_offset = 0;
  for (std::size_t row = 0; row < constraints_.size(); ++row) {
    QuadraticPlan plan = make_plan(*constraints_[row], jacobian_offset);
    for (const int column : plan.columns) {
      jacobian_rows_.push_back(static_cast<int>(row));
      jacobian_columns_.push_back(column);
    }
    jacobian_offset += plan.columns.size();
    constraint_plans_.push_back(std::move(plan));
  }

  // Ipopt requests one fixed lower-triangular structure for the Hessian of
  // the Lagrangian.  The QCQ format stores the symmetric upper triangle, so
  // transpose every structural pair and build a deterministic sorted union.
  std::set<std::pair<int, int>> hessian_structure;
  auto add_hessian_structure = [&](const QuadraticFunction& function) {
    for (const auto& entry : function.hessian.entries)
      if (entry.v != 0.0)
        hessian_structure.emplace(entry.j, entry.i);
  };
  add_hessian_structure(problem.objective);
  for (const auto* function : constraints_)
    add_hessian_structure(*function);

  std::map<std::pair<int, int>, std::size_t> hessian_position;
  for (const auto& coordinate : hessian_structure) {
    const std::size_t position = hessian_rows_.size();
    hessian_position.emplace(coordinate, position);
    hessian_rows_.push_back(coordinate.first);
    hessian_columns_.push_back(coordinate.second);
  }

  hessian_contributions_.resize(1 + constraints_.size());
  auto add_hessian_contributions = [&](const QuadraticFunction& function,
                                       std::size_t function_index) {
    auto& contributions = hessian_contributions_[function_index];
    contributions.reserve(function.hessian.entries.size());
    for (const auto& entry : function.hessian.entries) {
      if (entry.v == 0.0) continue;
      contributions.emplace_back(
          hessian_position.at({entry.j, entry.i}), entry.v);
    }
  };
  add_hessian_contributions(problem.objective, 0);
  for (std::size_t row = 0; row < constraints_.size(); ++row)
    add_hessian_contributions(*constraints_[row], 1 + row);
}

double OriginalQCQPDerivatives::value(const QuadraticPlan& plan,
                                      const Vec& x) const {
  double linear = 0.0;
  bool intermediate_range_loss = false;
  long double magnitude = std::abs(static_cast<long double>(plan.function->constant));
  for (std::size_t k = 0; k < plan.columns.size(); ++k) {
    const double product = plan.linear_values[k] * x[plan.columns[k]];
    linear += product;
    magnitude += std::abs(static_cast<long double>(product));
  }

  double quadratic = 0.0;
  for (const auto& term : plan.gradient_terms) {
    const double first_product = term.value * x[term.first_variable];
    const double product = first_product * x[term.second_variable];
    intermediate_range_loss = intermediate_range_loss ||
        (!std::isnormal(first_product) && term.value != 0.0 &&
         x[term.first_variable] != 0.0);
    quadratic += (term.first_variable == term.second_variable)
                     ? product
                     : 2.0 * product;
    magnitude += std::abs(static_cast<long double>(product)) *
        (term.first_variable == term.second_variable ? 0.5L : 1.L);
  }
  const double fast = plan.function->constant + linear + 0.5 * quadratic;
  // A forward-roundoff envelope detects cancellation across ALL polynomial
  // components, not just within the linear sum. Keep the existing arithmetic
  // when its envelope is within32 binary64 roundoff units of the result.
  // This predicate uses coefficients/coordinates only, never a solver gate,
  // instance name, constraint sense, or optimization tolerance.
  const long double epsilon = std::numeric_limits<double>::epsilon();
  const long double operations = 3.L + 2.L*plan.columns.size() +
      5.L*plan.gradient_terms.size();
  const long double error_factor = operations*epsilon;
  const long double envelope = error_factor < 1.L ?
      error_factor/(1.L-error_factor)*magnitude : INFINITY;
  if (!intermediate_range_loss && std::isfinite(fast) && std::isfinite(envelope) &&
      envelope <= 32.L*epsilon*std::max(1.L,std::abs(static_cast<long double>(fast))))
    return fast;

  detail::AccurateProductSum accurate(plan.function->constant);
  for (std::size_t k = 0; k < plan.columns.size(); ++k)
    accurate.add_product(plan.linear_values[k],x[plan.columns[k]]);
  for (const auto& term : plan.gradient_terms)
    accurate.add_triple_product(term.value,x[term.first_variable],x[term.second_variable],
        term.first_variable == term.second_variable ? 0.5L : 1.L);
  return static_cast<double>(accurate.value());
}

double OriginalQCQPDerivatives::objective_value(const Vec& x) const {
  check_x(*problem_, x, "OriginalQCQPDerivatives::objective_value");
  return value(objective_plan_, x);
}

void OriginalQCQPDerivatives::objective_gradient(const Vec& x,
                                                 Vec& values) const {
  check_x(*problem_, x, "OriginalQCQPDerivatives::objective_gradient");
  values.assign(static_cast<std::size_t>(problem_->n), 0.0);
  for (std::size_t k = 0; k < objective_plan_.columns.size(); ++k)
    values[objective_plan_.columns[k]] += objective_plan_.linear_values[k];
  for (const auto& term : objective_plan_.gradient_terms) {
    values[term.first_variable] += term.value * x[term.second_variable];
    if (term.first_variable != term.second_variable)
      values[term.second_variable] += term.value * x[term.first_variable];
  }
}

void OriginalQCQPDerivatives::constraint_values(const Vec& x,
                                                Vec& values) const {
  check_x(*problem_, x, "OriginalQCQPDerivatives::constraint_values");
  values.resize(constraint_plans_.size());
  for (std::size_t row = 0; row < constraint_plans_.size(); ++row)
    values[row] = value(constraint_plans_[row], x);
}

void OriginalQCQPDerivatives::jacobian_values(const Vec& x,
                                              Vec& values) const {
  check_x(*problem_, x, "OriginalQCQPDerivatives::jacobian_values");
  values.assign(jacobian_rows_.size(), 0.0);
  for (const auto& plan : constraint_plans_) {
    for (std::size_t k = 0; k < plan.columns.size(); ++k)
      values[plan.jacobian_offset + k] = plan.linear_values[k];
    for (const auto& term : plan.gradient_terms) {
      values[plan.jacobian_offset + term.first_position] +=
          term.value * x[term.second_variable];
      if (term.first_variable != term.second_variable)
        values[plan.jacobian_offset + term.second_position] +=
            term.value * x[term.first_variable];
    }
  }
}

void OriginalQCQPDerivatives::lagrangian_hessian_values(
    double objective_factor, const Vec& constraint_multipliers,
    Vec& values) const {
  if (constraint_multipliers.size() != constraints_.size())
    throw std::invalid_argument(
        "OriginalQCQPDerivatives::lagrangian_hessian_values: "
        "multiplier dimension mismatch");
  values.assign(hessian_rows_.size(), 0.0);
  for (const auto& [position, coefficient] : hessian_contributions_[0])
    values[position] += objective_factor * coefficient;
  for (std::size_t row = 0; row < constraints_.size(); ++row) {
    const double multiplier = constraint_multipliers[row];
    if (multiplier == 0.0) continue;
    for (const auto& [position, coefficient] :
         hessian_contributions_[1 + row])
      values[position] += multiplier * coefficient;
  }
}

void classify_direct_ipopt_result(const DirectIpoptSettings& options,
                                  DirectIpoptResult& result) {
  result.acceptable_required = options.acceptable_iterations;
  result.acceptable_tolerance = options.acceptable_tolerance;
  result.acceptable_constraint_violation_tolerance =
      options.acceptable_constraint_violation_tolerance;
  result.acceptable_dual_infeasibility_tolerance =
      options.acceptable_dual_infeasibility_tolerance;
  result.acceptable_complementarity_tolerance =
      options.acceptable_complementarity_tolerance;

  const bool multiplier_signs_valid =
      result.inequality_multiplier_sign_violation <=
          options.dual_infeasibility_tolerance &&
      result.bound_multiplier_sign_violation <=
          options.dual_infeasibility_tolerance;
  const bool strict = result.ipopt_termination_accepted &&
                      result.ipopt_desired_convergence &&
                      multiplier_signs_valid;

  const double stop_dual = std::isfinite(result.ipopt_native_dual_inf) ? result.ipopt_native_dual_inf : result.dual_infeasibility_inf;
  const double stop_primal = std::isfinite(result.ipopt_native_primal_inf) ? result.ipopt_native_primal_inf : result.feasibility.total_inf;
  const double stop_compl = std::isfinite(result.ipopt_native_compl_inf) ? result.ipopt_native_compl_inf : result.complementarity_inf;
  result.acceptable_absolute_limits_pass =
      std::isfinite(stop_primal) &&
      std::isfinite(stop_dual) &&
      std::isfinite(stop_compl) &&
      stop_primal <=
          options.acceptable_constraint_violation_tolerance &&
      stop_dual <=
          (result.ipopt_native_square_problem ? 1e300 : options.acceptable_dual_infeasibility_tolerance) &&
      stop_compl <=
          (result.ipopt_native_square_problem ? 1e300 : options.acceptable_complementarity_tolerance) &&
      multiplier_signs_valid;
  result.acceptable_convergence =
      options.acceptable_iterations > 0 &&
      result.ipopt_termination_accepted &&
      std::isfinite(result.ipopt_internal_scaled_nlp_error) &&
      result.ipopt_internal_scaled_nlp_error <=
          options.acceptable_tolerance &&
      result.acceptable_absolute_limits_pass;

  result.success = strict || result.acceptable_convergence;
  if (strict) {
    result.status = "solved";
    result.message =
        "original QCQP solved directly by Ipopt; native scaled convergence passed; original audit reported separately";
  } else if (result.acceptable_convergence) {
    result.status = "solved_acceptable";
    result.message =
        "original QCQP reached Ipopt acceptable convergence; recomputed "
        "native unscaled ceilings passed (desired Ipopt test did not pass)";
  } else if (!result.ipopt_termination_accepted) {
    result.status = "ipopt_failure";
    result.message = "Ipopt did not report an accepted optimal termination";
  } else {
    result.status = "kkt_failure";
    result.message =
        "Ipopt terminated at an accepted point but neither the strict nor "
        "the guarded acceptable certificate passed";
  }
}

#ifdef LAZYIPDCA_HAVE_IPOPT
namespace {

using Ipopt::Index;
using Ipopt::Number;

std::string application_status_name(Ipopt::ApplicationReturnStatus status) {
  switch(status) {
#define IPOPT_APP_CASE(x) case Ipopt::x: return #x;
    IPOPT_APP_CASE(Solve_Succeeded)
    IPOPT_APP_CASE(Solved_To_Acceptable_Level)
    IPOPT_APP_CASE(Infeasible_Problem_Detected)
    IPOPT_APP_CASE(Search_Direction_Becomes_Too_Small)
    IPOPT_APP_CASE(Diverging_Iterates)
    IPOPT_APP_CASE(User_Requested_Stop)
    IPOPT_APP_CASE(Feasible_Point_Found)
    IPOPT_APP_CASE(Maximum_Iterations_Exceeded)
    IPOPT_APP_CASE(Restoration_Failed)
    IPOPT_APP_CASE(Error_In_Step_Computation)
    IPOPT_APP_CASE(Maximum_CpuTime_Exceeded)
#if IPOPT_VERSION_MAJOR > 3 || (IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR >= 14)
    IPOPT_APP_CASE(Maximum_WallTime_Exceeded)
#endif
    IPOPT_APP_CASE(Not_Enough_Degrees_Of_Freedom)
    IPOPT_APP_CASE(Invalid_Problem_Definition)
    IPOPT_APP_CASE(Invalid_Option)
    IPOPT_APP_CASE(Invalid_Number_Detected)
    IPOPT_APP_CASE(Unrecoverable_Exception)
    IPOPT_APP_CASE(NonIpopt_Exception_Thrown)
    IPOPT_APP_CASE(Insufficient_Memory)
    IPOPT_APP_CASE(Internal_Error)
#undef IPOPT_APP_CASE
    default: break;
  }
  return "ApplicationReturnStatus(" +
         std::to_string(static_cast<int>(status)) + ")";
}

std::string solver_return_name(Ipopt::SolverReturn status) {
  switch(status) {
#define IPOPT_SOLVER_CASE(x) case Ipopt::x: return #x;
    IPOPT_SOLVER_CASE(SUCCESS)
    IPOPT_SOLVER_CASE(MAXITER_EXCEEDED)
    IPOPT_SOLVER_CASE(CPUTIME_EXCEEDED)
#if IPOPT_VERSION_MAJOR > 3 || (IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR >= 14)
    IPOPT_SOLVER_CASE(WALLTIME_EXCEEDED)
#endif
    IPOPT_SOLVER_CASE(STOP_AT_TINY_STEP)
    IPOPT_SOLVER_CASE(STOP_AT_ACCEPTABLE_POINT)
    IPOPT_SOLVER_CASE(LOCAL_INFEASIBILITY)
    IPOPT_SOLVER_CASE(USER_REQUESTED_STOP)
    IPOPT_SOLVER_CASE(FEASIBLE_POINT_FOUND)
    IPOPT_SOLVER_CASE(DIVERGING_ITERATES)
    IPOPT_SOLVER_CASE(RESTORATION_FAILURE)
    IPOPT_SOLVER_CASE(ERROR_IN_STEP_COMPUTATION)
    IPOPT_SOLVER_CASE(INVALID_NUMBER_DETECTED)
    IPOPT_SOLVER_CASE(TOO_FEW_DEGREES_OF_FREEDOM)
    IPOPT_SOLVER_CASE(INVALID_OPTION)
    IPOPT_SOLVER_CASE(OUT_OF_MEMORY)
    IPOPT_SOLVER_CASE(INTERNAL_ERROR)
    IPOPT_SOLVER_CASE(UNASSIGNED)
#undef IPOPT_SOLVER_CASE
    default: break;
  }
  return "SolverReturn(" + std::to_string(static_cast<int>(status)) + ")";
}

class OriginalQCQPTNLP final : public Ipopt::TNLP {
 public:
  OriginalQCQPTNLP(const QCQPProblem& problem,
                   const OriginalQCQPDerivatives& derivatives, double s_max,
                   double wall_seconds, double bound_infinity)
      : s_max_(s_max), problem_(problem), derivatives_(derivatives),
        wall_limit_(wall_seconds), bound_infinity_(bound_infinity),
        optimize_start_(std::chrono::steady_clock::now()) {}

  bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                    Index& nnz_h_lag,
                    IndexStyleEnum& index_style) override {
    const auto index_max =
        static_cast<std::size_t>(std::numeric_limits<Index>::max());
    if (derivatives_.jacobian_rows().size() > index_max ||
        derivatives_.hessian_rows().size() > index_max)
      return false;
    n = static_cast<Index>(problem_.n);
    m = static_cast<Index>(derivatives_.constraints());
    nnz_jac_g = static_cast<Index>(derivatives_.jacobian_rows().size());
    nnz_h_lag = static_cast<Index>(derivatives_.hessian_rows().size());
    index_style = C_STYLE;
    return true;
  }

  bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                       Index m, Number* g_l, Number* g_u) override {
    if (n != problem_.n || m != derivatives_.constraints()) return false;
    const Number ipopt_infinity = bound_infinity_;
    for (Index column = 0; column < n; ++column) {
      x_l[column] = std::isfinite(problem_.lb[column])
                        ? problem_.lb[column]
                        : -ipopt_infinity;
      x_u[column] = std::isfinite(problem_.ub[column])
                        ? problem_.ub[column]
                        : ipopt_infinity;
    }
    const Index equality_count =
        static_cast<Index>(problem_.equalities.size());
    for (Index row = 0; row < equality_count; ++row)
      g_l[row] = g_u[row] = 0.0;
    for (Index row = equality_count; row < m; ++row) {
      g_l[row] = -ipopt_infinity;
      g_u[row] = 0.0;
    }
    return true;
  }

  bool get_starting_point(Index n, bool init_x, Number* x,
                          bool init_z, Number*, Number*, Index,
                          bool init_lambda, Number*) override {
    if (n != problem_.n || !init_x || init_z || init_lambda) return false;
    std::copy(problem_.x0.begin(), problem_.x0.end(), x);
    return true;
  }

  bool eval_f(Index n, const Number* x, bool,
              Number& objective) override {
    try {
      if (n != problem_.n) return false;
      point_.assign(x, x + n);
      objective = derivatives_.objective_value(point_);
      return std::isfinite(objective);
    } catch (...) {
      return false;
    }
  }

  bool eval_grad_f(Index n, const Number* x, bool,
                   Number* gradient) override {
    try {
      if (n != problem_.n) return false;
      point_.assign(x, x + n);
      derivatives_.objective_gradient(point_, work_);
      std::copy(work_.begin(), work_.end(), gradient);
      return true;
    } catch (...) {
      return false;
    }
  }

  bool eval_g(Index n, const Number* x, bool, Index m,
              Number* constraints) override {
    try {
      if (n != problem_.n || m != derivatives_.constraints()) return false;
      point_.assign(x, x + n);
      derivatives_.constraint_values(point_, work_);
      std::copy(work_.begin(), work_.end(), constraints);
      return true;
    } catch (...) {
      return false;
    }
  }

  bool eval_jac_g(Index n, const Number* x, bool, Index m, Index nele_jac,
                  Index* rows, Index* columns, Number* values) override {
    try {
      if (n != problem_.n || m != derivatives_.constraints() ||
          static_cast<std::size_t>(nele_jac) !=
              derivatives_.jacobian_rows().size())
        return false;
      if (!values) {
        for (Index k = 0; k < nele_jac; ++k) {
          rows[k] = derivatives_.jacobian_rows()[k];
          columns[k] = derivatives_.jacobian_columns()[k];
        }
        return true;
      }
      point_.assign(x, x + n);
      derivatives_.jacobian_values(point_, work_);
      std::copy(work_.begin(), work_.end(), values);
      return true;
    } catch (...) {
      return false;
    }
  }

  bool eval_h(Index n, const Number*, bool, Number objective_factor,
              Index m, const Number* lambda, bool, Index nele_hess,
              Index* rows, Index* columns, Number* values) override {
    try {
      if (n != problem_.n || m != derivatives_.constraints() ||
          static_cast<std::size_t>(nele_hess) !=
              derivatives_.hessian_rows().size())
        return false;
      if (!values) {
        for (Index k = 0; k < nele_hess; ++k) {
          rows[k] = derivatives_.hessian_rows()[k];
          columns[k] = derivatives_.hessian_columns()[k];
        }
        return true;
      }
      multipliers_.assign(lambda, lambda + m);
      derivatives_.lagrangian_hessian_values(
          objective_factor, multipliers_, work_);
      std::copy(work_.begin(), work_.end(), values);
      return true;
    } catch (...) {
      return false;
    }
  }

  void finalize_solution(Ipopt::SolverReturn status, Index n,
                         const Number* x, const Number* z_l,
                         const Number* z_u, Index m, const Number*,
                         const Number* lambda, Number,
                         const Ipopt::IpoptData* data,
                         Ipopt::IpoptCalculatedQuantities* quantities) override {
    native_square_=data && data->curr()->x()->Dim()==data->curr()->y_c()->Dim();
    if(data) {
      const auto& v=*data->curr();
      native_y_l1_=v.y_c()->Asum()+v.y_d()->Asum();
      native_z_l1_=v.z_L()->Asum()+v.z_U()->Asum()+v.v_L()->Asum()+v.v_U()->Asum();
      const int zd=v.z_L()->Dim()+v.z_U()->Dim()+v.v_L()->Dim()+v.v_U()->Dim();
      const int yd=zd+v.y_c()->Dim()+v.y_d()->Dim();
      native_sd_=yd ? std::max(1.,(native_y_l1_+native_z_l1_)/yd/s_max_):1.;
      native_sc_=zd ? std::max(1.,native_z_l1_/zd/s_max_):1.;
    }
    solver_return_ = status;
    x_.assign(x, x + n);
    lower_bound_multipliers_.assign(z_l, z_l + n);
    upper_bound_multipliers_.assign(z_u, z_u + n);
    constraint_multipliers_.assign(lambda, lambda + m);
    if (quantities) {
      native_dual_=quantities->unscaled_curr_dual_infeasibility(Ipopt::NORM_MAX);
      native_primal_=quantities->unscaled_curr_nlp_constraint_violation(Ipopt::NORM_MAX);
      native_compl_=quantities->unscaled_curr_complementarity(0.,Ipopt::NORM_MAX);
      ipopt_internal_scaled_nlp_error_ = quantities->curr_nlp_error();
      ipopt_internal_unscaled_nlp_error_ =
          quantities->unscaled_curr_nlp_error();
    }
  }

  double s_max_=100, native_y_l1_=0,native_z_l1_=0,native_sd_=1,native_sc_=1;
  bool native_square_=false;
  double native_dual_=INFINITY,native_primal_=INFINITY,native_compl_=INFINITY;
  Ipopt::SolverReturn solver_return() const { return solver_return_; }
  const Vec& x() const { return x_; }
  const Vec& constraint_multipliers() const {
    return constraint_multipliers_;
  }
  const Vec& lower_bound_multipliers() const {
    return lower_bound_multipliers_;
  }
  const Vec& upper_bound_multipliers() const {
    return upper_bound_multipliers_;
  }
  double ipopt_internal_scaled_nlp_error() const {
    return ipopt_internal_scaled_nlp_error_;
  }
  double ipopt_internal_unscaled_nlp_error() const {
    return ipopt_internal_unscaled_nlp_error_;
  }

 private:
  bool intermediate_callback(Ipopt::AlgorithmMode, Index, Number, Number,
      Number, Number, Number, Number, Number, Number, Index,
      const Ipopt::IpoptData*, Ipopt::IpoptCalculatedQuantities*) override {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-
        optimize_start_).count() < wall_limit_;
  }
  double wall_limit_, bound_infinity_;
  std::chrono::steady_clock::time_point optimize_start_;
  const QCQPProblem& problem_;
  const OriginalQCQPDerivatives& derivatives_;
  Ipopt::SolverReturn solver_return_ = Ipopt::INTERNAL_ERROR;
  Vec point_;
  Vec multipliers_;
  Vec work_;
  Vec x_;
  Vec constraint_multipliers_;
  Vec lower_bound_multipliers_;
  Vec upper_bound_multipliers_;
  double ipopt_internal_scaled_nlp_error_ =
      std::numeric_limits<double>::infinity();
  double ipopt_internal_unscaled_nlp_error_ =
      std::numeric_limits<double>::infinity();
};

} // namespace

DirectIpoptResult solve_direct_ipopt(const QCQPProblem& problem,
                                     const DirectIpoptSettings& requested) {
  validate_aligned_ipopt_settings(requested);
  DirectIpoptSettings options=requested;
  const auto start = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start)
        .count();
  };

  problem.validate();
  if (!(options.tolerance > 0.0) ||
      !(options.constraint_violation_tolerance > 0.0) ||
      !(options.dual_infeasibility_tolerance > 0.0) ||
      !(options.complementarity_tolerance > 0.0) ||
      options.acceptable_iterations < 0 ||
      !(options.acceptable_tolerance > 0.0) ||
      !(options.acceptable_constraint_violation_tolerance > 0.0) ||
      !(options.acceptable_dual_infeasibility_tolerance > 0.0) ||
      !(options.acceptable_complementarity_tolerance > 0.0) ||
      !(options.acceptable_objective_change_tolerance >= 0.0) || !std::isfinite(options.acceptable_objective_change_tolerance) ||
      !(options.nlp_error_scaling_max > 0.0) ||
      !(options.time_limit_seconds > 0.0) ||
      options.max_iterations < 1 || options.print_level < 0 ||
      options.linear_solver.empty())
    throw std::invalid_argument("invalid direct Ipopt settings");

  DirectIpoptResult result;
  result.available = true;
  result.mu_strategy=options.mu_strategy;
  result.linear_solver=options.linear_solver;
  result.requested_mumps_ordering=0; // no ordering override
  result.wall_limit_seconds=options.time_limit_seconds;
  result.max_iterations=options.max_iterations;
  result.n = problem.n;
  result.equalities = static_cast<int>(problem.equalities.size());
  result.inequalities = static_cast<int>(problem.inequalities.size());
  result.acceptable_required = options.acceptable_iterations;

  OriginalQCQPDerivatives derivatives(problem);
  result.jacobian_nnz = derivatives.jacobian_rows().size();
  result.lagrangian_hessian_nnz = derivatives.hessian_rows().size();
  result.derivative_plan_seconds = elapsed();

  auto application = IpoptApplicationFactory();
  auto set_string=[&](const char* name,const std::string& value){
    if(!application->Options()->SetStringValue(name,value))
      throw std::runtime_error(std::string("Ipopt rejected string option ")+name);
  };
  auto set_integer=[&](const char* name,int value){
    if(!application->Options()->SetIntegerValue(name,value))
      throw std::runtime_error(std::string("Ipopt rejected integer option ")+name);
  };
  auto set_numeric=[&](const char* name,double value){
    if(!application->Options()->SetNumericValue(name,value))
      throw std::runtime_error(std::string("Ipopt rejected numeric option ")+name);
  };
  // Make the CLI deterministic: a stray ipopt.opt in the caller's working
  // directory must not silently replace the audited command-line contract.
  set_string("option_file_name", "");
  set_numeric("tol",1e-8);
  set_numeric("constr_viol_tol",1e-6);
  set_numeric("dual_inf_tol",1e-6);
  set_numeric("compl_inf_tol",1e-6);
  set_integer("acceptable_iter",0);
  set_numeric("bound_relax_factor",0.0);
  set_string("mu_strategy",options.mu_strategy);
  set_integer("print_level", options.print_level);
  set_integer("max_iter", options.max_iterations);
  // The campaign budget is wall time in OptimizeTNLP. max_cpu_time is an
  // additional ceiling; a callback supplies a portable wall-time guard.
  set_numeric("max_cpu_time",options.time_limit_seconds);
#if IPOPT_VERSION_MAJOR > 3 || (IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR >= 14)
  set_numeric("max_wall_time",options.time_limit_seconds);
#endif
  double max_finite_bound=0.0;
  for(int j=0;j<problem.n;++j) {
    if(std::isfinite(problem.lb[j])) max_finite_bound=std::max(max_finite_bound,std::abs(problem.lb[j]));
    if(std::isfinite(problem.ub[j])) max_finite_bound=std::max(max_finite_bound,std::abs(problem.ub[j]));
  }
  const double bound_infinity=std::max(2e19,2*max_finite_bound);
  if(!std::isfinite(bound_infinity))
    throw std::runtime_error("finite bound too large for an exact Ipopt infinity sentinel");
  if(max_finite_bound>=1e19) {
    set_numeric("nlp_lower_bound_inf",-bound_infinity);
    set_numeric("nlp_upper_bound_inf",bound_infinity);
    std::ostringstream representation;
    representation<<std::setprecision(17)<<"[{\"reason\":\"preserve finite original bounds beyond the library default infinity cutoff\",\"nlp_lower_bound_inf\":"<<-bound_infinity<<",\"nlp_upper_bound_inf\":"<<bound_infinity<<"}]";
    result.representation_overrides_json=representation.str();
  }
  set_string("linear_solver", options.linear_solver);
  if (!options.hsl_library.empty()) set_string("hsllib",options.hsl_library);
  set_string("print_user_options", "yes");

  const auto initialization_status = application->Initialize();
  if (initialization_status != Ipopt::Solve_Succeeded) {
    result.status = "ipopt_failure";
    result.message = "Ipopt initialization failed";
    result.application_status_code =
        static_cast<int>(initialization_status);
    result.application_status = application_status_name(initialization_status);
    result.elapsed_seconds = elapsed();
    return result;
  }

  // Query effective settings after Initialize; verify the aligned contract.
  double effective_bound_relax=-1,effective_cpu=-1,effective_wall=-1;
  int effective_max_iter=-1;
  std::ostringstream defaults; defaults<<std::setprecision(17)<<"{";
  bool first_default=true;
  const char quote=34;
  auto sep=[&](){if(!first_default)defaults<<',';first_default=false;};
  auto numeric=[&](const char* name,double* destination=nullptr){
    Ipopt::Number value=0;
    // False means the registered library default was returned, not failure.
    application->Options()->GetNumericValue(name,value,"");
    if(destination)*destination=value;
    sep();defaults<<quote<<name<<quote<<':'<<value;
  };
  auto integer=[&](const char* name,int* destination=nullptr){
    Ipopt::Index value=0;
    application->Options()->GetIntegerValue(name,value,"");
    if(destination)*destination=value;
    sep();defaults<<quote<<name<<quote<<':'<<value;
  };
  auto string=[&](const char* name){
    std::string value;
    application->Options()->GetStringValue(name,value,"");
    sep();defaults<<quote<<name<<quote<<':'<<quote<<json_escape(value)<<quote;
    return value;
  };
  numeric("tol",&options.tolerance);
  numeric("constr_viol_tol",&options.constraint_violation_tolerance);
  numeric("dual_inf_tol",&options.dual_infeasibility_tolerance);
  numeric("compl_inf_tol",&options.complementarity_tolerance);
  integer("acceptable_iter",&options.acceptable_iterations);
  numeric("acceptable_tol",&options.acceptable_tolerance);
  numeric("acceptable_constr_viol_tol",&options.acceptable_constraint_violation_tolerance);
  numeric("acceptable_dual_inf_tol",&options.acceptable_dual_infeasibility_tolerance);
  numeric("acceptable_compl_inf_tol",&options.acceptable_complementarity_tolerance);
  numeric("acceptable_obj_change_tol",&options.acceptable_objective_change_tolerance);
  numeric("s_max",&options.nlp_error_scaling_max);
  const auto effective_mu=string("mu_strategy");string("nlp_scaling_method");string("hessian_approximation");
  string("fixed_variable_treatment");const auto effective_backend=string("linear_solver");
  const auto effective_option_file=string("option_file_name");
  const auto effective_print_options=string("print_user_options");
  integer("print_level");
  if(options.linear_solver=="ma27")string("hsllib");
  numeric("bound_relax_factor",&effective_bound_relax);numeric("nlp_lower_bound_inf");numeric("nlp_upper_bound_inf");
  integer("max_iter",&effective_max_iter);numeric("max_cpu_time",&effective_cpu);
#if IPOPT_VERSION_MAJOR > 3 || (IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR >= 14)
  numeric("max_wall_time",&effective_wall);
#endif
  if(options.linear_solver=="mumps")integer("mumps_pivot_order");
  defaults<<"}";result.library_defaults_json=defaults.str();
  validate_aligned_ipopt_settings(options);
  if(effective_mu!=options.mu_strategy||effective_backend!=options.linear_solver||
     effective_bound_relax!=0.0||effective_max_iter!=5000||effective_cpu!=7200.0||
     effective_wall!=7200.0||!effective_option_file.empty()||effective_print_options!="yes")
    throw std::runtime_error("Effective Ipopt options violate aligned contract");
  result.aligned_contract_verified=true;
  result.option_file_disabled=true;
  std::cout<<"IPOPT_ALIGNED_EFFECTIVE {"
           <<quote<<"ipopt_version"<<quote<<':'<<quote<<IPOPT_VERSION<<quote<<','
           <<quote<<"effective_options"<<quote<<':'<<result.library_defaults_json<<','
           <<quote<<"input_representation_overrides"<<quote<<':'<<result.representation_overrides_json<<','
           <<quote<<"explicit_nonrepresentation_overrides"<<quote
           <<":[\"tol\",\"constr_viol_tol\",\"dual_inf_tol\",\"compl_inf_tol\",\"acceptable_iter\",\"bound_relax_factor\",\"mu_strategy\",\"max_iter\",\"max_cpu_time\",\"max_wall_time\",\"linear_solver\",\"hsllib_if_MA27\",\"option_file_name_empty\",\"print_level\",\"print_user_options\"]}"
           <<std::endl;

  Ipopt::SmartPtr<OriginalQCQPTNLP> concrete =
      new OriginalQCQPTNLP(problem, derivatives,options.nlp_error_scaling_max,
                           options.time_limit_seconds,bound_infinity);
  Ipopt::SmartPtr<Ipopt::TNLP> nlp = concrete;
  result.setup_seconds=elapsed();
  const auto optimize_start=std::chrono::steady_clock::now();
  const auto application_status = application->OptimizeTNLP(nlp);
  result.core_s=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-optimize_start).count();
  result.elapsed_seconds = elapsed();
  result.application_status_code = static_cast<int>(application_status);
  result.application_status = application_status_name(application_status);
  result.solver_return_code = static_cast<int>(concrete->solver_return());
  result.solver_return = solver_return_name(concrete->solver_return());
  result.ipopt_termination_accepted =
      application_status == Ipopt::Solve_Succeeded ||
      application_status == Ipopt::Solved_To_Acceptable_Level;

  const auto statistics = application->Statistics();
  if (Ipopt::IsValid(statistics))
    result.iterations = static_cast<int>(statistics->IterationCount());

  result.ipopt_native_square_problem=concrete->native_square_;
  result.ipopt_native_dual_inf=concrete->native_dual_;
  result.ipopt_native_primal_inf=concrete->native_primal_;
  result.ipopt_native_compl_inf=concrete->native_compl_;
  result.x = concrete->x();
  result.lower_bound_multipliers = concrete->lower_bound_multipliers();
  result.upper_bound_multipliers = concrete->upper_bound_multipliers();
  result.ipopt_internal_scaled_nlp_error =
      concrete->ipopt_internal_scaled_nlp_error();
  result.ipopt_internal_unscaled_nlp_error =
      concrete->ipopt_internal_unscaled_nlp_error();
  const auto& lambda = concrete->constraint_multipliers();
  if (lambda.size() == problem.equalities.size() +
                           problem.inequalities.size()) {
    result.equality_multipliers.assign(
        lambda.begin(), lambda.begin() + problem.equalities.size());
    result.inequality_multipliers.assign(
        lambda.begin() + problem.equalities.size(), lambda.end());
  }

  if (result.x.size() == static_cast<std::size_t>(problem.n)) {
    result.objective = problem.objective.value(result.x);
    result.feasibility = feasibility(problem, result.x);

    QCQPTerminationMultipliers multipliers;
    multipliers.equality = result.equality_multipliers;
    multipliers.inequality = result.inequality_multipliers;
    multipliers.lower_bound = result.lower_bound_multipliers;
    multipliers.upper_bound = result.upper_bound_multipliers;
    multipliers.valid = true;

    Settings termination_settings;
    termination_settings.stationarity_tolerance = options.tolerance;
    termination_settings.feasibility_tolerance =
        options.constraint_violation_tolerance;
    termination_settings.dual_infeasibility_tolerance =
        options.dual_infeasibility_tolerance;
    termination_settings.complementarity_tolerance =
        options.complementarity_tolerance;
    termination_settings.nlp_error_scaling_max =
        options.nlp_error_scaling_max;
    SparseQCQPEvaluation sparse_evaluation;
    const auto certificate = qcqp_ipopt_termination_evaluated(
        problem, result.x, multipliers, termination_settings,sparse_evaluation);
    result.dual_infeasibility_inf = certificate.stop.dual_infeasibility;
    result.complementarity_inf = certificate.stop.complementarity;
    result.original_normalized_nlp_error = certificate.stop.overall_nlp_error;
    result.scaled_nlp_error = result.ipopt_internal_scaled_nlp_error;
    result.dual_error_scaling = concrete->native_sd_;
    result.complementarity_error_scaling =
        concrete->native_sc_;
    result.constraint_multiplier_l1 = concrete->native_y_l1_;
    result.bound_slack_multiplier_l1 = concrete->native_z_l1_;
    // Ipopt has already applied its INTERNAL NLP scaling and absolute guards.
    // The original-units normalized diagnostic must not demote native success.
    result.ipopt_desired_convergence = application_status == Ipopt::Solve_Succeeded;

    for (double multiplier : result.inequality_multipliers)
      result.inequality_multiplier_sign_violation = std::max(
          result.inequality_multiplier_sign_violation, -multiplier);
    for (double multiplier : result.lower_bound_multipliers)
      result.bound_multiplier_sign_violation = std::max(
          result.bound_multiplier_sign_violation, -multiplier);
    for (double multiplier : result.upper_bound_multipliers)
      result.bound_multiplier_sign_violation = std::max(
          result.bound_multiplier_sign_violation, -multiplier);
    result.inequality_multiplier_sign_violation =
        std::max(0.0, result.inequality_multiplier_sign_violation);
    result.bound_multiplier_sign_violation =
        std::max(0.0, result.bound_multiplier_sign_violation);
  }

  classify_direct_ipopt_result(options, result);
  result.postsolve_seconds=elapsed()-result.setup_seconds-result.core_s;
  result.elapsed_seconds=elapsed();
  return result;
}

#else

DirectIpoptResult solve_direct_ipopt(const QCQPProblem& problem,
                                     const DirectIpoptSettings& requested) {
  validate_aligned_ipopt_settings(requested);
  problem.validate();
  DirectIpoptResult result;
  result.n = problem.n;
  result.equalities = static_cast<int>(problem.equalities.size());
  result.inequalities = static_cast<int>(problem.inequalities.size());
  OriginalQCQPDerivatives derivatives(problem);
  result.jacobian_nnz = derivatives.jacobian_rows().size();
  result.lagrangian_hessian_nnz = derivatives.hessian_rows().size();
  result.status = "unavailable";
  result.message =
      "direct Ipopt support was not compiled (LAZYIPDCA_WITH_IPOPT=OFF or "
      "Ipopt was not found)";
  return result;
}

#endif

std::string direct_ipopt_result_json(const DirectIpoptResult& result,
                                     const std::string& problem_name) {
  std::ostringstream out;
  out << "{\n";
  out << "  " << char(34) << "adapter_version" << char(34) << ": " << char(34) << "ipopt-aligned-original-qcqp-v1" << char(34) << ",\n";
  out << "  " << char(34) << "reference_policy" << char(34) << ": " << char(34) << "aligned_original_kkt_tolerances_v1" << char(34) << ",\n";
  out << "  " << char(34) << "effective_options" << char(34) << ": " << result.library_defaults_json << ",\n";
  out << "  " << char(34) << "input_representation_overrides" << char(34) << ": " << result.representation_overrides_json << ",\n";
  out << "  \"aligned_contract_verified\": " << (result.aligned_contract_verified?"true":"false") << ",\n";
  out << "  \"option_file_disabled\": " << (result.option_file_disabled?"true":"false") << ",\n";
  out << "  \"mu_strategy\": \"" << json_escape(result.mu_strategy) << "\",\n";
  out << "  \"linear_solver\": \"" << json_escape(result.linear_solver) << "\",\n";
  out << "  \"requested_mumps_ordering\": " << result.requested_mumps_ordering << ",\n";
  out << "  \"optimization_wall_limit_seconds\": " << result.wall_limit_seconds << ",\n";
  out << "  \"max_iterations\": " << result.max_iterations << ",\n";
#ifdef LAZYIPDCA_HAVE_IPOPT
  out << "  \"ipopt_version\": \"" << IPOPT_VERSION << "\",\n";
#endif
  out << "  \"timing_scope\": \"core_s=OptimizeTNLP wall; setup excludes input parse; postsolve separately\",\n";
  out << "  \"parse_seconds\": " << result.parse_seconds << ",\n";
  out << "  \"derivative_plan_seconds\": " << result.derivative_plan_seconds << ",\n";
  out << "  \"setup_seconds\": " << result.setup_seconds << ",\n";
  out << "  \"core_s\": " << result.core_s << ",\n";
  out << "  \"solve_seconds\": " << result.core_s << ",\n";
  out << "  \"postsolve_seconds\": " << result.postsolve_seconds << ",\n";
  out << "  \"start_source\": \"" << json_escape(result.start_source) << "\",\n";
  out << "  \"schema\": \"ipopt-direct-qcqp-result-v1\",\n";
  out << "  \"solver\": \"Ipopt\",\n";
  out << "  \"mode\": \"direct-frozen-qcqp\",\n";
  out << "  \"package_version\": \"" << json_escape(kVersion) << "\",\n";
  out << "  \"package_api_version\": " << kApiVersion << ",\n";
  out << "  \"package_build_id\": \"" << json_escape(kBuildId)
      << "\",\n";
  out << "  \"problem\": \"" << json_escape(problem_name) << "\",\n";
  out << "  \"status\": \"" << json_escape(result.status) << "\",\n";
  out << "  \"message\": \"" << json_escape(result.message) << "\",\n";
  out << "  \"available\": " << (result.available ? "true" : "false")
      << ",\n";
  out << "  \"success\": " << (result.success ? "true" : "false")
      << ",\n";
  out << "  \"ipopt_termination_accepted\": "
      << (result.ipopt_termination_accepted ? "true" : "false") << ",\n";
  out << "  \"ipopt_desired_convergence\": "
      << (result.ipopt_desired_convergence ? "true" : "false") << ",\n";
  out << "  \"acceptable_absolute_limits_pass\": "
      << (result.acceptable_absolute_limits_pass ? "true" : "false")
      << ",\n";
  out << "  \"acceptable_convergence\": "
      << (result.acceptable_convergence ? "true" : "false") << ",\n";
  out << "  \"ipopt_application_status\": \""
      << json_escape(result.application_status) << "\",\n";
  out << "  \"ipopt_application_status_code\": "
      << result.application_status_code << ",\n";
  out << "  \"ipopt_solver_return\": \""
      << json_escape(result.solver_return) << "\",\n";
  out << "  \"ipopt_solver_return_code\": "
      << result.solver_return_code << ",\n";
  out << "  \"n\": " << result.n << ",\n";
  out << "  \"equalities\": " << result.equalities << ",\n";
  out << "  \"inequalities\": " << result.inequalities << ",\n";
  out << "  \"acceptable_required\": " << result.acceptable_required
      << ",\n";
  out << "  \"jacobian_nnz\": " << result.jacobian_nnz << ",\n";
  out << "  \"lagrangian_hessian_nnz\": "
      << result.lagrangian_hessian_nnz << ",\n";
  out << "  \"iterations\": " << result.iterations << ",\n";
  out << "  \"objective\": "; json_number(out, result.objective); out << ",\n";
  out << "  \"feasibility\": ";
  json_number(out, result.feasibility.total_inf); out << ",\n";
  out << "  \"equality_violation\": ";
  json_number(out, result.feasibility.equality_inf); out << ",\n";
  out << "  \"inequality_violation\": ";
  json_number(out, result.feasibility.inequality_inf); out << ",\n";
  out << "  \"bound_violation\": ";
  json_number(out, result.feasibility.bound_inf); out << ",\n";
  out << "  \"dual_infeasibility_inf\": ";
  json_number(out, result.dual_infeasibility_inf); out << ",\n";
  out << "  \"complementarity_inf\": ";
  json_number(out, result.complementarity_inf); out << ",\n";
  out << "  \"ipopt_native_square_problem\": " << (result.ipopt_native_square_problem ? "true" : "false") << ",\n";
  out << "  \"original_normalized_nlp_error\": "; json_number(out,result.original_normalized_nlp_error); out << ",\n";
  out << "  \"ipopt_native_dual_inf\": "; json_number(out,result.ipopt_native_dual_inf); out << ",\n";
  out << "  \"ipopt_native_primal_inf\": "; json_number(out,result.ipopt_native_primal_inf); out << ",\n";
  out << "  \"ipopt_native_compl_inf\": "; json_number(out,result.ipopt_native_compl_inf); out << ",\n";
  out << "  \"scaled_nlp_error\": ";
  json_number(out, result.scaled_nlp_error); out << ",\n";
  out << "  \"ipopt_internal_scaled_nlp_error\": ";
  json_number(out, result.ipopt_internal_scaled_nlp_error); out << ",\n";
  out << "  \"ipopt_internal_unscaled_nlp_error\": ";
  json_number(out, result.ipopt_internal_unscaled_nlp_error); out << ",\n";
  out << "  \"acceptable_tolerance\": ";
  json_number(out, result.acceptable_tolerance); out << ",\n";
  out << "  \"acceptable_constraint_violation_tolerance\": ";
  json_number(out, result.acceptable_constraint_violation_tolerance);
  out << ",\n";
  out << "  \"acceptable_dual_infeasibility_tolerance\": ";
  json_number(out, result.acceptable_dual_infeasibility_tolerance);
  out << ",\n";
  out << "  \"acceptable_complementarity_tolerance\": ";
  json_number(out, result.acceptable_complementarity_tolerance);
  out << ",\n";
  out << "  \"dual_error_scaling\": ";
  json_number(out, result.dual_error_scaling); out << ",\n";
  out << "  \"complementarity_error_scaling\": ";
  json_number(out, result.complementarity_error_scaling); out << ",\n";
  out << "  \"constraint_multiplier_l1\": ";
  json_number(out, result.constraint_multiplier_l1); out << ",\n";
  out << "  \"bound_slack_multiplier_l1\": ";
  json_number(out, result.bound_slack_multiplier_l1); out << ",\n";
  out << "  \"inequality_multiplier_sign_violation\": ";
  json_number(out, result.inequality_multiplier_sign_violation); out << ",\n";
  out << "  \"bound_multiplier_sign_violation\": ";
  json_number(out, result.bound_multiplier_sign_violation); out << ",\n";
  out << "  \"elapsed_seconds\": ";
  json_number(out, result.elapsed_seconds); out << ",\n";
  out << "  \"x\": "; json_vector(out, result.x); out << ",\n";
  out << "  \"equality_multipliers\": ";
  json_vector(out, result.equality_multipliers); out << ",\n";
  out << "  \"inequality_multipliers\": ";
  json_vector(out, result.inequality_multipliers); out << ",\n";
  out << "  \"lower_bound_multipliers\": ";
  json_vector(out, result.lower_bound_multipliers); out << ",\n";
  out << "  \"upper_bound_multipliers\": ";
  json_vector(out, result.upper_bound_multipliers); out << "\n";
  out << '}';
  return out.str();
}

void write_direct_ipopt_result_json(const DirectIpoptResult& result,
                                    const std::string& problem_name,
                                    const std::string& path) {
  std::ofstream output(path);
  if (!output)
    throw std::runtime_error("cannot write direct Ipopt result JSON: " + path);
  output << direct_ipopt_result_json(result, problem_name) << '\n';
}

} // namespace lazyipdca::qcqp
