#include "lazyipdca_qcqp/linear_solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#ifdef LAZYIPDCA_HAVE_METIS
extern "C" {
#include <metis.h>
}
#endif

using namespace lazyipdca::qcqp;
namespace {
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

// Independent METIS call: verify the inverse permutation at the Fortran ABI,
// not just that an arbitrary valid permutation was supplied.
std::vector<int> expected_order(int n, const std::vector<SymmetricTriplet>& a,
                                int seed) {
  std::vector<int> result(n);
  std::iota(result.begin(), result.end(), 1);
#ifdef LAZYIPDCA_HAVE_METIS
#ifdef METIS_NOPTIONS
  using Index = idx_t;
#else
  using Index = idxtype;
  (void)seed;
#endif
  std::vector<std::vector<Index>> neighbors(n);
  for (const auto& entry : a) if (entry.i != entry.j) {
    neighbors[entry.i].push_back(entry.j);
    neighbors[entry.j].push_back(entry.i);
  }
  std::vector<Index> ptr(n + 1, 0), adj, perm(n), iperm(n);
  for (int i = 0; i < n; ++i) {
    auto& row = neighbors[i];
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    adj.insert(adj.end(), row.begin(), row.end());
    ptr[i + 1] = static_cast<Index>(adj.size());
  }
  if (adj.empty()) return result;
#ifdef METIS_NOPTIONS
  Index nv = n, options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_SEED] = seed;
  require(METIS_NodeND(&nv, ptr.data(), adj.data(), nullptr, options,
                      perm.data(), iperm.data()) == METIS_OK,
          "independent METIS call failed");
#else
  int nv = n, numflag = 0, options[8]{};
  METIS_NodeND(&nv, ptr.data(), adj.data(), &numflag, options,
               perm.data(), iperm.data());
#endif
  for (int i = 0; i < n; ++i) result[i] = static_cast<int>(iperm[i]) + 1;
#else
  (void)a;
  (void)seed;
#endif
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
    require(std::abs(rhs[i] - expected[i]) < 1e-10,
            "solution is wrong in original variable order");
  require(factor.last_backward_error() < 1e-12, "original-system residual");
}
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "supply the explicit MA27 test-double library");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    require(library != nullptr, "cannot open MA27 test double");
    const auto count = reinterpret_cast<int (*)()>(dlsym(library, "mock_ma27_analysis_count"));
    const auto flag = reinterpret_cast<int (*)()>(dlsym(library, "mock_ma27_last_flag"));
    const auto order = reinterpret_cast<int (*)(int)>(dlsym(library, "mock_ma27_order_at"));
    require(count && flag && order, "test-double inspection symbols missing");
    for (bool metis : {false, true}) {
      Settings settings;
      settings.linear_backend = "ma27";
      settings.linear_metis = metis;
      settings.hsl_library = argv[1];
      settings.kkt_adaptive_regularization = true;
      settings.random_seed = 7;
      std::string message;
      auto factor = make_symmetric_factorization(settings, message);
      require(bool(factor), message);
      const int n = 12;
      std::vector<SymmetricTriplet> a;
      for (int i = 0; i < n; ++i) a.push_back({i, i, i < 9 ? 4. + i : -4. - i});
      for (const auto& edge : std::vector<std::pair<int, int>>{
               {0, 3}, {3, 4}, {1, 7}, {4, 7}, {2, 6}, {6, 8},
               {4, 8}, {4, 9}, {9, 10}})
        a.push_back({edge.first, edge.second, .125});
#ifndef LAZYIPDCA_HAVE_METIS
      if (metis) {
        const int before = count();
        require(!factor->factor(n, 9, a, message), "METIS request silently ignored");
        require(message.find("--linear-metis 0") != std::string::npos,
                "missing METIS diagnostic");
        require(count() == before, "MA27 called despite missing METIS");
        continue;
      }
#endif
      const int before = count();
      auto check_order = [&]() {
        require(flag() == (metis ? 1 : 0), "wrong IFLAG at MA27AD boundary");
        const auto expected = expected_order(n, a, settings.random_seed);
        for (int i = 0; i < n; ++i)
          require(order(i) == (metis ? expected[i] : i + 1),
                  "MA27 IKEEP must contain METIS iperm + 1");
      };
      require(factor->factor(n, 9, a, message), message);
      require(count() == before + 1, "initial analysis count");
      check_order();
      require(std::string(factor->name()).find(metis ? "ma27_metis_" : "ma27_internal_") == 0,
              "reported ordering");
      solve_check(*factor, n, a);
      solve_check(*factor, n, a);
      a[0].value += 1;
      std::reverse(a.begin(), a.end());
      require(factor->factor(n, 9, a, message), message);
      solve_check(*factor, n, a);
      require(count() == before + 1 && factor->last_symbolic_reused(),
              "numeric update rebuilt ordering");
      auto subset = a;
      subset.erase(std::remove_if(subset.begin(), subset.end(), [](const auto& e) {
        return e.i == 9 && e.j == 10;
      }), subset.end());
      require(factor->factor(n, 9, subset, message), message);
      solve_check(*factor, n, subset);
      require(count() == before + 1 && factor->last_symbolic_reused(),
              "subset rebuilt ordering");
      a.push_back({0, 10, .1});
      require(factor->factor(n, 9, a, message), message);
      check_order();
      solve_check(*factor, n, a);
      require(count() == before + 2 && factor->symbolic_invalidations() == 1,
              "new structural edge did not rebuild ordering");
      require(factor->factor(1, 1, {{0, 0, 2}}, message), message);
      solve_check(*factor, 1, {{0, 0, 2}});
      require(factor->factor(3, 2, {{0, 0, 1}, {1, 1, 2}, {2, 2, -3}}, message), message);
      for (int i = 0; i < 3; ++i) require(order(i) == i + 1, "edgeless ordering");
      require(!factor->factor(3, 3, {{0, 0, 1}, {1, 1, 2}, {2, 2, -3}}, message),
              "negative curvature accepted for Newton");
      Vec rhs{1, 2, 3};
      require(!factor->solve(rhs, message), "rejected inertia authorized Newton solve");
      require(factor->curvature_defect() == 1, "wrong negative-curvature defect");
      require(factor->solve_curvature_probe(rhs, message), message);
      require(std::abs(rhs[0] - 1) < 1e-12 && std::abs(rhs[1] - 1) < 1e-12 &&
                  std::abs(rhs[2] + 1) < 1e-12, "curvature-probe residual");
      require(!factor->factor(2, 1, {{0, 0, 1}, {1, 1, 2}}, message),
              "wrong dual inertia accepted");
      rhs = {1, 2};
      require(factor->curvature_defect() == 0 && !factor->solve_curvature_probe(rhs, message),
              "non-curvature inertia mismatch authorized a probe");
      require(!factor->factor(2, 2, {{0, 0, 1}, {1, 1, 0}}, message),
              "singular factor accepted");
      require(!factor->solve_curvature_probe(rhs, message) && !factor->solve(rhs, message),
              "singular factor authorized solve");
      require(!factor->factor(2, 2, {{0, 0, std::numeric_limits<double>::quiet_NaN()}}, message),
              "nonfinite matrix accepted");
      require(!factor->factor(2, 2, {{0, 0, 1e308}, {0, 0, 1e308}}, message),
              "nonfinite duplicate accumulation accepted");
      require(!factor->factor(2, 2, {{2, 0, 1}}, message), "invalid coordinate accepted");
      require(factor->factor(2, 2, {{0, 0, 1}, {0, 0, 2}, {1, 1, 2}, {0, 1, .2}}, message), message);
      solve_check(*factor, 2, {{0, 0, 1}, {0, 0, 2}, {1, 1, 2}, {0, 1, .2}});
      rhs = {1, 2};
      const auto original = rhs;
      setenv("MOCK_MA27_BAD_SOLVE", "1", 1);
      const bool bad_solve = factor->solve(rhs, message);
      unsetenv("MOCK_MA27_BAD_SOLVE");
      require(!bad_solve && rhs == original, "bad residual accepted or failed solve changed RHS");
      require(factor->solve(rhs, message), message);
      rhs = {std::numeric_limits<double>::quiet_NaN(), 2};
      require(!factor->solve(rhs, message), "nonfinite RHS accepted");
      require(std::isnan(rhs[0]) && rhs[1] == 2, "rejected nonfinite RHS changed");
      std::cout << "PASS MA27 " << (metis ? "METIS" : "internal")
                << " ABI/permutation/inertia/probes/residual/symbolic-reuse"
                   " (LAPACK test double, not actual HSL)\n";
    }
    Settings missing;
    missing.linear_backend = "ma27";
    missing.hsl_library = std::string(argv[1]) + ".missing";
    std::string message;
    require(!make_symmetric_factorization(missing, message), "missing library silently fell back");
    dlclose(library);
    return 0;
  } catch (const std::exception& error) {
    unsetenv("MOCK_MA27_BAD_SOLVE");
    std::cerr << "FAIL MA27 adapter test: " << error.what() << '\n';
    return 1;
  }
}
