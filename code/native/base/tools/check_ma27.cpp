// Small real-runtime preflight. This is not a performance benchmark and does
// not establish convergence of any QCQP. No fallback backend is permitted.
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;
namespace {
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

double finite_number(const std::string& value, const std::string& option) {
  std::size_t consumed = 0;
  const double result = std::stod(value, &consumed);
  require(consumed == value.size() && std::isfinite(result),
          option + " requires a finite number");
  return result;
}

void solve_check(SymmetricFactorization& factor, int n,
                 const std::vector<SymmetricTriplet>& entries) {
  Vec expected(n), rhs(n, 0);
  for (int i = 0; i < n; ++i) expected[i] = (i % 2 ? -1. : 1.) * (i + 1);
  for (const auto& entry : entries) {
    rhs[entry.i] += entry.value * expected[entry.j];
    if (entry.i != entry.j) rhs[entry.j] += entry.value * expected[entry.i];
  }
  std::string message;
  require(factor.solve(rhs, message), message);
  for (int i = 0; i < n; ++i)
    require(std::isfinite(rhs[i]) && std::abs(rhs[i] - expected[i]) < 1e-9,
            "MA27 solution failed the original-coordinate check");
  require(std::isfinite(factor.last_backward_error()) &&
              factor.last_backward_error() < 1e-10,
          "MA27 backward error exceeds 1e-10");
}
}

int main(int argc, char** argv) {
  try {
    Settings settings;
    settings.linear_backend = "ma27";
    settings.kkt_adaptive_regularization = true;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help") {
        std::cout << "Usage: check_ma27 --hsl-library FILE [--linear-metis 0|1]"
                     " [--hsl-pivot-tol VALUE] [--hsl-memory-factor VALUE]\n"
                     "Tests the supplied real MA27 runtime on small matrices;"
                     " no QCQP performance claim.\n";
        return 0;
      }
      require(option == "--hsl-library" || option == "--linear-metis" ||
                  option == "--hsl-pivot-tol" || option == "--hsl-memory-factor",
              "unknown option: " + option);
      require(i + 1 < argc, "missing value for " + option);
      const std::string value = argv[++i];
      if (option == "--hsl-library") settings.hsl_library = value;
      else if (option == "--linear-metis") {
        require(value == "0" || value == "1", "--linear-metis requires 0 or 1");
        settings.linear_metis = value == "1";
      } else if (option == "--hsl-pivot-tol") {
        settings.hsl_pivot_tolerance = finite_number(value, option);
        require(settings.hsl_pivot_tolerance > 0 && settings.hsl_pivot_tolerance <= .5,
                "--hsl-pivot-tol must be in (0,0.5]");
      } else {
        settings.hsl_memory_factor = finite_number(value, option);
        require(settings.hsl_memory_factor > 1 && settings.hsl_memory_factor <= 64,
                "--hsl-memory-factor must be in (1,64]");
      }
    }
    require(!settings.hsl_library.empty(), "--hsl-library FILE is required");
    void* handle = dlopen(settings.hsl_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      const char* error = dlerror();
      throw std::runtime_error("cannot load requested HSL library: " +
                               std::string(error ? error : "unknown dynamic-loader error"));
    }
    const auto close_library = [](void* pointer) { if (pointer) dlclose(pointer); };
    std::unique_ptr<void, decltype(close_library)> library(handle, close_library);
    require(dlsym(handle, "mock_ma27_test_double") == nullptr &&
                dlsym(handle, "mock_ma27_analysis_count") == nullptr,
            "refusing the LAPACK test double: this preflight requires real MA27");
    std::string message;
    auto factor = make_symmetric_factorization(settings, message);
    require(bool(factor), message);
    require(std::string(factor->name()).find("ma27_") == 0,
            "factory did not return MA27");
    std::cout << "[MA27-PREFLIGHT] library=" << settings.hsl_library
              << " backend=" << factor->name()
              << " pivot_tol=" << settings.hsl_pivot_tolerance << '\n';
    const int n = 4, primal = 3;
    std::vector<SymmetricTriplet> a{
        {0, 0, 4}, {1, 1, 3}, {2, 2, 2}, {3, 3, -2},
        {0, 1, .25}, {1, 3, -.5}};
    require(factor->factor(n, primal, a, message), message);
    solve_check(*factor, n, a);
    const int analyses = factor->symbolic_analyses();
    require(analyses == 1, "first factorization did not perform one symbolic analysis");
    a[0].value += .5;
    require(factor->factor(n, primal, a, message), message);
    solve_check(*factor, n, a);
    require(factor->symbolic_analyses() == analyses && factor->last_symbolic_reused(),
            "changed values did not reuse symbolic analysis");
    auto subset = a;
    subset.pop_back();
    require(factor->factor(n, primal, subset, message), message);
    solve_check(*factor, n, subset);
    require(factor->symbolic_analyses() == analyses && factor->last_symbolic_reused(),
            "matrix subset did not reuse symbolic analysis");
    a.push_back({0, 2, .125});
    require(factor->factor(n, primal, a, message), message);
    solve_check(*factor, n, a);
    require(factor->symbolic_analyses() == analyses + 1,
            "new edge did not rebuild symbolic analysis");
    const double backward_error = factor->last_backward_error();
    require(!factor->factor(2, 2, {{0, 0, 2}, {1, 1, -1}}, message),
            "wrong inertia authorized Newton");
    Vec rhs{2, -3};
    require(!factor->solve(rhs, message), "rejected inertia authorized ordinary solve");
    require(factor->curvature_defect() == 1 && factor->solve_curvature_probe(rhs, message),
            "complete negative-curvature factor unavailable to probe");
    require(std::abs(rhs[0] - 1) < 1e-9 && std::abs(rhs[1] - 3) < 1e-9,
            "curvature-probe residual");
    require(!factor->factor(2, 1, {{0, 0, 2}, {1, 1, 1}}, message),
            "missing dual-negative pivot accepted");
    rhs = {2, 3};
    require(factor->curvature_defect() == 0 && !factor->solve_curvature_probe(rhs, message),
            "inertia mismatch without negative curvature authorized probe");
    require(!factor->factor(2, 2, {{0, 0, 1}, {1, 1, 0}}, message),
            "singular matrix accepted");
    require(!factor->solve(rhs, message) && !factor->solve_curvature_probe(rhs, message),
            "singular matrix authorized a solve");
    require(!factor->factor(2, 2,
                {{0, 0, std::numeric_limits<double>::quiet_NaN()}}, message),
            "nonfinite matrix accepted");
    require(factor->factor(2, 2, {{0, 0, 2}, {1, 1, 1}}, message), message);
    solve_check(*factor, 2, {{0, 0, 2}, {1, 1, 1}});
    std::cout << "MA27_PREFLIGHT=PASS backend=" << factor->name()
              << " inertia=PASS curvature_probe=PASS residual=" << backward_error
              << " symbolic_reuse=PASS input_checks=PASS runtime=loaded-library\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MA27_PREFLIGHT=FAIL " << error.what() << '\n';
    return 1;
  }
}
