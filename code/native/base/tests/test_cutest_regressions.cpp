#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef LAZYIPDCA_SOURCE_DIR
#define LAZYIPDCA_SOURCE_DIR "."
#endif

using namespace lazyipdca::qcqp;

static void check_case(const std::string& name, double obj_ref, double obj_tol,
                       double feas_tol, double stat_tol, bool allow_better=false) {
  const std::string path = std::string(LAZYIPDCA_SOURCE_DIR) + "/tests/regression/" + name + ".qcq";
  QCQPProblem p = read_qcq(path);
  Settings st;
  st.outer_algorithm = "legacy-sqp";
  st.verbose = false;
  st.max_outer_iterations = 300;
  st.max_inner_iterations = 500;
  st.time_limit_seconds = 30.0;
  st.feasibility_tolerance = 1e-7;
  st.stationarity_tolerance = 1e-6;
  st.inner_tolerance = 1e-7;
  Solver solver(st);
  const Result r = solver.solve(p);
  if (r.status != Status::Solved) {
    throw std::runtime_error(name + ": expected solved, got " + status_string(r.status) + " / " + r.message);
  }
  if (!(r.feasibility.total_inf <= feas_tol)) {
    throw std::runtime_error(name + ": feasibility regression");
  }
  if (!(r.model_stationarity_inf <= stat_tol)) {
    throw std::runtime_error(name + ": stationarity regression");
  }
  if (std::isfinite(obj_ref)) {
    if (allow_better) {
      if (r.objective > obj_ref + obj_tol)
        throw std::runtime_error(name + ": objective regression");
    } else if (std::abs(r.objective - obj_ref) > obj_tol) {
      throw std::runtime_error(name + ": objective regression");
    }
  }
  std::cout << "[PASS] " << name << " status=solved f=" << r.objective
            << " feas=" << r.feasibility.total_inf
            << " stat=" << r.model_stationarity_inf << "\n";
}

int main() {
  try {
    check_case("BT1", -1.0, 1e-7, 1e-7, 1e-6);
    check_case("HIMMELBD", 0.0, 1e-12, 1e-7, 1e-6);
    check_case("DISCS", 16.6416297775, 5e-7, 1e-7, 1e-6, false);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
