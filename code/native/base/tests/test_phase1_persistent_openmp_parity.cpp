#include "lazyipdca_qcqp/restoration.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#ifdef LAZYIPDCA_HAVE_OPENMP
#include <omp.h>
#endif

using namespace lazyipdca::qcqp;

namespace {
int index(int joint,int coordinate) { return 3*joint+coordinate; }

void add_square_difference(QuadraticFunction& equality,int first,int second) {
  equality.hessian.entries.push_back({std::min(first,second),
                                      std::min(first,second),2.0});
  equality.hessian.entries.push_back({std::max(first,second),
                                      std::max(first,second),2.0});
  equality.hessian.entries.push_back({std::min(first,second),
                                      std::max(first,second),-2.0});
}

QCQPProblem make_problem(int beams) {
  constexpr double fraction=0.6;
  constexpr double total_mass=500.0;
  constexpr double gravity=9.81;
  const double load=(total_mass/beams)*gravity;
  QCQPProblem problem;
  problem.name="CATENARY_phase1_persistent_openmp_parity";
  problem.n=3*(beams+1);
  problem.x0.assign(problem.n,0.0);
  problem.lb.assign(problem.n,-std::numeric_limits<double>::infinity());
  problem.ub.assign(problem.n, std::numeric_limits<double>::infinity());
  problem.objective.name="gravity";
  problem.objective.linear.assign(problem.n,0.0);
  problem.objective.hessian.n=problem.n;
  problem.objective.linear[index(0,1)]=0.5*load;
  for(int beam=1;beam<beams;++beam)
    problem.objective.linear[index(beam,1)]=load;
  problem.objective.linear[index(beams,1)]=0.5*load;
  for(int coordinate=0;coordinate<3;++coordinate)
    problem.lb[index(0,coordinate)]=
        problem.ub[index(0,coordinate)]=0.0;
  problem.lb[index(beams,0)]=problem.ub[index(beams,0)]=fraction*beams;
  for(int beam=1;beam<=beams;++beam) {
    QuadraticFunction equality;
    equality.name="beam_"+std::to_string(beam);
    equality.constant=-1.0;
    equality.linear.assign(problem.n,0.0);
    equality.hessian.n=problem.n;
    add_square_difference(equality,index(beam,0),index(beam-1,0));
    add_square_difference(equality,index(beam,1),index(beam-1,1));
    // Preserve the historical CATENARY-like regression fixture used by r3p3.
    // Its cross-coordinate third term prevents the structured pre-phase from
    // solving the model before the exact matrix-free restoration is reached.
    add_square_difference(equality,index(beam,0),index(beam-1,2));
    problem.equalities.push_back(std::move(equality));
  }
  for(int beam=1;beam<=beams;++beam) problem.x0[index(beam,0)]=fraction;
  problem.x0[index(beams,0)]=fraction*beams;
  return problem;
}

Settings settings() {
  Settings value;
  value.verbose=false;
  value.linear_backend="native";
  value.kkt_adaptive_regularization=true;
  value.restoration_max_iterations=80;
  value.restoration_max_restarts=0;
  value.restoration_minimax_enabled=false;
  value.restoration_tangent_activation=true;
  value.feasibility_tolerance=1e-6;
  value.restoration_exact_parallel_min_rows=64;
  return value;
}

bool binary_equal(const Vec& first,const Vec& second) {
  return first.size()==second.size() &&
      (first.empty() || std::memcmp(first.data(),second.data(),
                                    first.size()*sizeof(double))==0);
}

bool same_contract(const RestorationResult& first,
                   const RestorationResult& second) {
  return first.success==second.success &&
      first.iterations==second.iterations &&
      first.linear_solves==second.linear_solves &&
      first.exact_hessian_solves==second.exact_hessian_solves &&
      first.matrix_free_solves==second.matrix_free_solves &&
      first.cg_iterations==second.cg_iterations &&
      first.line_search_backtracks==second.line_search_backtracks &&
      first.harmonic_attempts==second.harmonic_attempts &&
      first.harmonic_acceptances==second.harmonic_acceptances &&
      first.tangent_activation_attempts==second.tangent_activation_attempts &&
      first.tangent_activation_acceptances==second.tangent_activation_acceptances;
}

bool valid_sparse_runtime_contract(const RestorationResult& result,
                                   bool build_candidate,
                                   const char* route) {
  const int batches=result.exact_sparse_row_dot_batches;
  const int checks=result.exact_sparse_row_dot_verifications;
  const int fallbacks=result.exact_sparse_row_dot_fallbacks;
  bool valid=false;
  if(!build_candidate) {
    valid=batches==0 && checks==0 && fallbacks==0;
  } else if(fallbacks==0) {
    valid=checks==result.matrix_free_solves && batches>0 &&
        batches+checks==result.exact_matvecs;
  } else if(fallbacks==1) {
    // Sparse batches before the first disagreement are legitimate.  The
    // irreversible Phase-I latch makes all later matvecs dense, so only one
    // fallback is permitted and some matvecs may be absent from these two
    // sparse counters.
    valid=checks>0 && checks<result.matrix_free_solves &&
        batches+checks<=result.exact_matvecs;
  }
  if(!valid) {
    std::cerr << "[FAIL] "<<route<<" sparse Phase-I runtime contract"
              << " batches="<<batches
              << " checks="<<checks
              << " fallbacks="<<fallbacks
              << " pcg="<<result.matrix_free_solves
              << " matvecs="<<result.exact_matvecs<<"\n";
  }
  return valid;
}
}  // namespace

int main() {
#ifdef LAZYIPDCA_HAVE_OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(2);
#endif
  const QCQPProblem problem=make_problem(120);

  Settings local=settings();
  local.restoration_exact_workspace_reuse=false;
  local.restoration_exact_parallel_row_dots=false;
  local.restoration_exact_sparse_row_dots=false;
  const auto dense_t0=std::chrono::steady_clock::now();
  const RestorationResult reference=
      FeasibilityRestoration(local).solve(problem,problem.x0);
  const double dense_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-dense_t0).count();

  Settings serial=settings();
  serial.restoration_exact_workspace_reuse=true;
  serial.restoration_exact_parallel_row_dots=false;
  serial.restoration_exact_sparse_row_dots=true;
  const auto sparse_t0=std::chrono::steady_clock::now();
  const RestorationResult reused=
      FeasibilityRestoration(serial).solve(problem,problem.x0);
  const double sparse_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-sparse_t0).count();

  Settings parallel=serial;
  parallel.restoration_exact_parallel_row_dots=true;
  const auto parallel_t0=std::chrono::steady_clock::now();
  const RestorationResult persistent=
      FeasibilityRestoration(parallel).solve(problem,problem.x0);
  const double parallel_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-parallel_t0).count();

  if(!reference.success || reference.matrix_free_solves<=0 ||
     reference.cg_iterations<=0) {
    std::cerr << "[FAIL] fixture did not exercise matrix-free Phase I"
              << " success="<<reference.success
              << " mf="<<reference.matrix_free_solves
              << " cg="<<reference.cg_iterations
              << " theta="<<reference.final_feasibility<<"\n";
    return 1;
  }
  if(!same_contract(reference,reused) ||
     !binary_equal(reference.x,reused.x) ||
     reference.final_feasibility!=reused.final_feasibility) {
    std::cerr << "[FAIL] workspace reuse changed the Phase-I trajectory\n";
    return 2;
  }
  if(reused.exact_workspace_growths<=0 || reused.exact_matvecs<=0) {
    std::cerr << "[FAIL] persistent exact workspace was not exercised\n";
    return 3;
  }
  if(!same_contract(reference,persistent) ||
     !binary_equal(reference.x,persistent.x) ||
     reference.final_feasibility!=persistent.final_feasibility) {
    std::cerr << "[FAIL] persistent OpenMP team changed the Phase-I trajectory\n";
    return 4;
  }
  // Absence of __FAST_MATH__ only makes sparse Jv a candidate.  The runtime
  // bytewise probe remains authoritative because contraction/vectorization
  // may still differ across compilers and CPUs.  A single disagreement must
  // latch the complete Phase I onto its historical dense kernel.
  const bool sparse_jv_build_candidate =
      restoration_exact_sparse_row_dots_build_compatible();
  if(!valid_sparse_runtime_contract(
         reused,sparse_jv_build_candidate,"serial") ||
     !valid_sparse_runtime_contract(
         persistent,sparse_jv_build_candidate,"OpenMP"))
    return 5;
  if(reused.exact_sparse_row_dot_batches!=
         persistent.exact_sparse_row_dot_batches ||
     reused.exact_sparse_row_dot_verifications!=
         persistent.exact_sparse_row_dot_verifications ||
     reused.exact_sparse_row_dot_fallbacks!=
         persistent.exact_sparse_row_dot_fallbacks) {
    std::cerr << "[FAIL] serial/OpenMP sparse Phase-I latch decisions differ\n";
    return 5;
  }
#ifdef LAZYIPDCA_HAVE_OPENMP
  if(persistent.exact_parallel_teams<=0 ||
     persistent.exact_parallel_row_dot_batches<=
         persistent.exact_parallel_teams ||
     persistent.exact_parallel_row_dot_batches!=persistent.exact_matvecs) {
    std::cerr << "[FAIL] OpenMP team/batch counters do not prove persistence"
              << " batches="<<persistent.exact_parallel_row_dot_batches
              << " teams="<<persistent.exact_parallel_teams
              << " matvecs="<<persistent.exact_matvecs<<"\n";
    return 6;
  }
#endif

  std::cout << "[PASS] Phase-I workspace and persistent-OpenMP parity"
            << " iterations="<<reference.iterations
            << " pcg_solves="<<reference.matrix_free_solves
            << " cg="<<reference.cg_iterations
            << " matvecs="<<persistent.exact_matvecs
            << " batches="<<persistent.exact_parallel_row_dot_batches
            << " teams="<<persistent.exact_parallel_teams
            << " sparse_batches="<<persistent.exact_sparse_row_dot_batches
            << " sparse_checks="<<persistent.exact_sparse_row_dot_verifications
            << " sparse_fallbacks="<<persistent.exact_sparse_row_dot_fallbacks
            << " sparse_mode="
            << (!sparse_jv_build_candidate
                    ? "disabled-fast-fp"
                    : (persistent.exact_sparse_row_dot_fallbacks==0
                           ? "exact" : "dense-latched"))
            << " seconds_dense/sparse/omp="<<dense_seconds<<"/"
            << sparse_seconds<<"/"<<parallel_seconds
            << " theta="<<reference.final_feasibility<<"\n";
  return 0;
}
