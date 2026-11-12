#include "lazyipdca_qcqp/direct_ipopt.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/version.hpp"
#include "frozen_start_json.hpp"
#include <chrono>
#ifdef LAZYIPDCA_HAVE_IPOPT
#include <coin-or/IpoptConfig.h>
#endif

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <set>

using namespace lazyipdca::qcqp;

namespace {

void usage(const char* executable) {
  std::cerr
      << "Usage: " << executable << " --problem FILE.qcq [options]\n"
      << "Solve the original exported QCQP with aligned tolerances and explicit barrier strategy.\n\n"
      << "Options:\n"
      << "  --frozen-start FILE      qualified frozen primal x0 JSON override\n"
      << "  --json FILE              write ipopt-direct-qcqp-result-v1 JSON\n"
      << "  --max-iter N             Ipopt iteration limit (default 5000)\n"
      << "  --time-limit SEC         Ipopt optimization wall budget (default 7200)\n"
      << "  --mu-strategy NAME       adaptive or monotone (default monotone)\n"
      << "  --linear-solver NAME     Ipopt linear solver (default mumps)\n"
      << "  --hsl-library FILE       pass FILE to Ipopt's hsllib option (for example with ma27)\n"
      << "  --print-level N          Ipopt print level (default 5)\n"
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
  std::string frozen_start_path;
  bool quiet = false;
  DirectIpoptSettings settings;

  try {
    for (int argument = 1; argument < argc; ++argument) {
      const std::string option = argv[argument];
      const std::set<std::string> allowed={"--problem","--frozen-start","--json","--max-iter","--time-limit","--linear-solver","--mu-strategy","--hsl-library","--print-level","--quiet","--version","--help","-h"};
      if(!allowed.count(option))
        throw std::invalid_argument("Aligned comparator rejects unsupported override: "+option);

      auto need = [&](const char* name) -> std::string {
        if (argument + 1 >= argc)
          throw std::runtime_error(std::string("missing value after ") + name);
        return argv[++argument];
      };

      if (option == "--problem") problem_path = need("--problem");
      else if (option == "--frozen-start") frozen_start_path=need("--frozen-start");
      else if (option == "--json") json_path = need("--json");
      else if (option == "--max-iter")
        settings.max_iterations = parse_number<int>(need("--max-iter"));
      else if (option == "--time-limit")
        settings.time_limit_seconds =
            parse_number<double>(need("--time-limit"));
      else if (option == "--mu-strategy")
        settings.mu_strategy = need("--mu-strategy");
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
                  << " [ipopt=" << IPOPT_VERSION << "; native-sparse-qcqp-20261027-r1]"
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
    validate_aligned_ipopt_settings(settings);

    const auto parse_start=std::chrono::steady_clock::now();
    QCQPProblem problem = read_qcq_compact(problem_path);
    if(!frozen_start_path.empty()) {
      problem.x0=detail::read_frozen_start(frozen_start_path,static_cast<std::size_t>(problem.n));
      problem.validate();
    }
    const double parse_seconds=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-parse_start).count();
    DirectIpoptResult result = solve_direct_ipopt(problem, settings);
    result.parse_seconds=parse_seconds;
    result.start_source=frozen_start_path.empty()?"embedded_x0":"qualified_external_json:"+frozen_start_path;

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
    std::cout << "Native Ipopt desired convergence : "
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
