#include "lazyipdca_qcqp/direct_ipopt.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/version.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable << " --problem FILE.qcq [options]\n"
      << "Solve the original exported QCQP directly with Ipopt.\n\n"
      << "Options:\n"
      << "  --json FILE              write ipopt-direct-qcqp-result-v1 JSON\n"
      << "  --tol X                  scaled overall NLP tolerance (default 1e-8)\n"
      << "  --constr-viol-tol X      original constraint tolerance (default 1e-4)\n"
      << "  --dual-inf-tol X         unscaled dual tolerance (default 1)\n"
      << "  --compl-inf-tol X        unscaled complementarity tolerance (default 1e-4)\n"
      << "  --acceptable-iter N      consecutive acceptable iterations (default 15; 0 disables)\n"
      << "  --acceptable-tol X       Ipopt scaled acceptable NLP tolerance (default 1e-6)\n"
      << "  --acceptable-constr-viol-tol X  original primal acceptable ceiling (default 1e-2)\n"
      << "  --acceptable-dual-inf-tol X     original dual acceptable ceiling (default 1e10)\n"
      << "  --acceptable-compl-inf-tol X    original complementarity acceptable ceiling (default 1e-2)\n"
      << "  --acceptable-obj-change-tol X  relative scaled objective change limit (1e20)\n"
      << "  --s-max X                Ipopt multiplier scaling threshold (default 100)\n"
      << "  --max-iter N             Ipopt iteration limit (default 3000)\n"
      << "  --time-limit SEC         Ipopt CPU-time limit (default 1800)\n"
      << "  --linear-solver NAME     Ipopt linear solver (default mumps)\n"
      << "  --hsl-library FILE       pass FILE to Ipopt's hsllib option (for example with ma27)\n"
      << "  --print-level N          Ipopt print level (default 5)\n"
      << "  --print-timing-statistics enable Ipopt timing table\n"
      << "  --derivative-test        run Ipopt second-order derivative test\n"
      << "  --quiet                  print only the final compact summary\n"
      << "  --version                print version and Ipopt build availability\n";
}

template <class T>
T parse_number(const std::string& text);

template <>
int parse_number<int>(const std::string& text) {
  std::size_t used = 0;
  const int value = std::stoi(text, &used);
  if (used != text.size()) throw std::invalid_argument("invalid integer: " + text);
  return value;
}

template <>
double parse_number<double>(const std::string& text) {
  std::size_t used = 0;
  const double value = std::stod(text, &used);
  if (used != text.size()) throw std::invalid_argument("invalid number: " + text);
  return value;
}

} // namespace

int main(int argc, char** argv) {
  std::string problem_path;
  std::string json_path;
  bool quiet = false;
  DirectIpoptSettings settings;

  try {
    for (int argument = 1; argument < argc; ++argument) {
      const std::string option = argv[argument];
      auto need = [&](const char* name) -> std::string {
        if (argument + 1 >= argc)
          throw std::runtime_error(std::string("missing value after ") + name);
        return argv[++argument];
      };

      if (option == "--problem") problem_path = need("--problem");
      else if (option == "--json") json_path = need("--json");
      else if (option == "--tol")
        settings.tolerance = parse_number<double>(need("--tol"));
      else if (option == "--constr-viol-tol")
        settings.constraint_violation_tolerance =
            parse_number<double>(need("--constr-viol-tol"));
      else if (option == "--dual-inf-tol")
        settings.dual_infeasibility_tolerance =
            parse_number<double>(need("--dual-inf-tol"));
      else if (option == "--compl-inf-tol")
        settings.complementarity_tolerance =
            parse_number<double>(need("--compl-inf-tol"));
      else if (option == "--acceptable-iter")
        settings.acceptable_iterations =
            parse_number<int>(need("--acceptable-iter"));
      else if (option == "--acceptable-tol")
        settings.acceptable_tolerance =
            parse_number<double>(need("--acceptable-tol"));
      else if (option == "--acceptable-constr-viol-tol")
        settings.acceptable_constraint_violation_tolerance =
            parse_number<double>(need("--acceptable-constr-viol-tol"));
      else if (option == "--acceptable-dual-inf-tol")
        settings.acceptable_dual_infeasibility_tolerance =
            parse_number<double>(need("--acceptable-dual-inf-tol"));
      else if (option == "--acceptable-compl-inf-tol")
        settings.acceptable_complementarity_tolerance =
            parse_number<double>(need("--acceptable-compl-inf-tol"));
      else if (option == "--acceptable-obj-change-tol")
        settings.acceptable_objective_change_tolerance = parse_number<double>(need("--acceptable-obj-change-tol"));
      else if (option == "--s-max")
        settings.nlp_error_scaling_max =
            parse_number<double>(need("--s-max"));
      else if (option == "--max-iter")
        settings.max_iterations = parse_number<int>(need("--max-iter"));
      else if (option == "--time-limit")
        settings.time_limit_seconds =
            parse_number<double>(need("--time-limit"));
      else if (option == "--linear-solver")
        settings.linear_solver = need("--linear-solver");
      else if (option == "--hsl-library" || option == "--hsllib")
        settings.hsl_library = need(option.c_str());
      else if (option == "--print-level")
        settings.print_level = parse_number<int>(need("--print-level"));
      else if (option == "--print-timing-statistics")
        settings.print_timing_statistics = true;
      else if (option == "--derivative-test") settings.derivative_test = true;
      else if (option == "--quiet") quiet = true;
      else if (option == "--version") {
        std::cout << "Ipopt-QCQP-Direct " << kVersion << " (API "
                  << kApiVersion << ", " << kBuildId << ")"
#ifdef LAZYIPDCA_HAVE_IPOPT
                  << " [ipopt=compiled]"
#else
                  << " [ipopt=unavailable]"
#endif
                  << '\n';
        return 0;
      } else if (option == "--help" || option == "-h") {
        usage(argv[0]);
        return 0;
      } else {
        throw std::runtime_error("unknown option: " + option);
      }
    }

    if (problem_path.empty()) {
      usage(argv[0]);
      return 2;
    }

    if (quiet) settings.print_level = 0;

    const QCQPProblem problem = read_qcq(problem_path);
    DirectIpoptResult result = solve_direct_ipopt(problem, settings);

    if (!quiet) {
      std::cout << "============================================================\n";
      std::cout << "Ipopt direct original QCQP\n";
      std::cout << "problem              : " << problem.name << '\n';
      std::cout << "n / eq / ineq        : " << result.n << " / "
                << result.equalities << " / " << result.inequalities << '\n';
      std::cout << "Jacobian / Hessian nnz: " << result.jacobian_nnz << " / "
                << result.lagrangian_hessian_nnz << '\n';
      std::cout << "Ipopt status         : " << result.application_status << '\n';
      std::cout << "Ipopt solver return  : " << result.solver_return << '\n';
    }
    std::cout << std::setprecision(17);
    std::cout << "[IPOPT QCQP SUMMARY]\n";
    std::cout << "status     : " << result.status << '\n';
    std::cout << "objective  : " << result.objective << '\n';
    std::cout << std::scientific;
    std::cout << "equality   : " << result.feasibility.equality_inf << '\n';
    std::cout << "inequality : " << result.feasibility.inequality_inf << '\n';
    std::cout << "bounds     : " << result.feasibility.bound_inf << '\n';
    std::cout << "dual       : " << result.dual_infeasibility_inf << '\n';
    std::cout << "complement.: " << result.complementarity_inf << '\n';
    std::cout << "Ipopt NLP err: "
              << result.ipopt_internal_scaled_nlp_error << '\n';
    std::cout << std::defaultfloat;
    std::cout << "iterations : " << result.iterations << '\n';
    std::cout << "solver_s   : " << result.elapsed_seconds << '\n';
    std::cout << "KKT strict : "
              << (result.ipopt_desired_convergence ? "PASS" : "FAIL")
              << '\n';
    std::cout << "ACCEPTABLE : "
              << (result.acceptable_convergence ? "PASS" : "FAIL")
              << '\n';
    if (!json_path.empty()) {
      write_direct_ipopt_result_json(result, problem.name, json_path);
      std::cout << "json       : " << json_path << '\n';
    }

    if (!result.available) return 2;
    return result.success ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
