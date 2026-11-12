#include "lazyipdca_qcqp/linear_solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lazyipdca::qcqp;

namespace {
void require(bool condition,const std::string& message) {
  if(!condition)throw std::runtime_error(message);
}

// The first diagonal deliberately distinguishes stable duplicate summation
// from reassociation: ((1e16 + 1) - 1e16) + 4 is exactly 4, not 5.
std::vector<SymmetricTriplet> matrix(double coupling) {
  return {{0,0,1e16},{1,2,-0.5},{1,1,3.0},{0,1,2.0},{0,0,1.0},
          {2,2,-2.0},{0,0,-1e16},{0,1,coupling-2.0},{0,0,4.0},{2,0,1.0}};
}

Vec rhs(double coupling,double primal_shift=0,double dual_shift=0) {
  // Exact right-hand side for x = (1, -2, 3).
  return {7.0-2.0*coupling+primal_shift,
          coupling-7.5-2.0*primal_shift,-4.0-3.0*dual_shift};
}

void check_solution(const Vec& x) {
  require(x.size()==3 && std::abs(x[0]-1.0)<1e-12 &&
          std::abs(x[1]+2.0)<1e-12 && std::abs(x[2]-3.0)<1e-12,
          "stable duplicate sum or original KKT solution changed");
}

void compare_cached_and_reference(const std::string& layout,bool adaptive) {
  Settings fast;
  fast.linear_backend="mumps";fast.linear_metis=true;fast.live_display_level=0;
  fast.dca_aggregate=true;fast.dca_aggregate_fast_assembly=true;
  fast.dca_aggregate_kkt_layout=layout;fast.kkt_preserve_structural_zeros=true;
  fast.kkt_adaptive_regularization=adaptive;
  fast.kkt_primal_regularization=adaptive?0.0:0.125;
  fast.kkt_dual_regularization=adaptive?0.0:0.25;
  Settings reference=fast;reference.dca_aggregate_fast_assembly=false;
  std::string message;
  auto cached=make_symmetric_factorization(fast,message);
  auto legacy=make_symmetric_factorization(reference,message);
  require(cached && legacy,"MUMPS factory: "+message);
  const auto started=std::chrono::steady_clock::now();
  for(double coupling:{0.0,0.25,0.0,-0.5}) {
    const auto entries=matrix(coupling);
    require(cached->factor(3,2,entries,message),"cached factor: "+message);
    require(legacy->factor(3,2,entries,message),"reference factor: "+message);
    Vec a=rhs(coupling,fast.kkt_primal_regularization,fast.kkt_dual_regularization);
    Vec b=a;
    require(cached->solve(a,message),"cached solve: "+message);
    require(legacy->solve(b,message),"reference solve: "+message);
    require(a==b,"cache on/off changed the computed direction");
    require(cached->last_backward_error()==legacy->last_backward_error(),
            "cache on/off changed backward error");
    check_solution(a);
  }
  require(cached->numeric_scatter_builds()==1 && cached->numeric_scatter_reuses()==3 &&
          cached->numeric_scatter_invalidations()==0,"stable layout did not reuse scatter");
  require(legacy->numeric_scatter_builds()==0 && legacy->numeric_scatter_reuses()==0,
          "disabled cache was used");
  const int expected_analyses=layout=="persistent"?1:2;
  require(cached->symbolic_analyses()==expected_analyses &&
          legacy->symbolic_analyses()==expected_analyses,
          "structural-zero policy or symbolic reuse changed");

  // Moving an unrelated coordinate changes raw layout, but not the duplicate
  // sequence for any coordinate. Recompile scatter without reanalyzing graph.
  auto reordered=matrix(-0.5);
  std::swap(reordered[1],reordered[2]);
  require(cached->factor(3,2,reordered,message),"reordered cached factor: "+message);
  require(legacy->factor(3,2,reordered,message),"reordered reference factor: "+message);
  require(cached->numeric_scatter_builds()==2 && cached->numeric_scatter_invalidations()==1,
          "changed raw layout did not invalidate scatter");
  require(cached->symbolic_analyses()==expected_analyses,
          "raw layout change unnecessarily reanalyzed graph");
  Vec a=rhs(-0.5,fast.kkt_primal_regularization,fast.kkt_dual_regularization),b=a;
  require(cached->solve(a,message) && legacy->solve(b,message),"reordered solve: "+message);
  require(a==b,"recompiled scatter changed direction");
  check_solution(a);

  // The optimization must not turn numerical success into inertia success.
  require(!cached->factor(3,3,reordered,message) &&
          message.find("inertia mismatch")!=std::string::npos,
          "cached factor bypassed inertia guard");
  require(!legacy->factor(3,3,reordered,message) &&
          message.find("inertia mismatch")!=std::string::npos,
          "reference factor bypassed inertia guard");
  require(cached->factor(3,2,reordered,message),"recovery factor: "+message);
  const int before_quality=cached->symbolic_analyses();
  require(cached->increase_quality(message),"quality recovery failed: "+message);
  require(cached->symbolic_analyses()==before_quality,"quality recovery reanalyzed graph");
  a=rhs(-0.5,fast.kkt_primal_regularization,fast.kkt_dual_regularization);
  require(cached->solve(a,message),"quality solve: "+message);
  check_solution(a);
  a={1.0};
  require(!cached->solve(a,message),"invalid RHS dimension accepted");

  const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  const double phases=cached->assembly_seconds()+cached->symbolic_seconds()+
      cached->numeric_seconds()+cached->linear_solve_seconds();
  require(cached->assembly_seconds()>0 && cached->symbolic_seconds()>0 &&
          cached->numeric_seconds()>0 && cached->linear_solve_seconds()>0,
          "backend phase timer missing");
  require(phases<=wall+1e-6,"exclusive backend phase timers overlap");
}

void preserve_unrelated_callers() {
  Settings settings;
  settings.linear_backend="mumps";settings.dca_aggregate=false;
  settings.dca_aggregate_fast_assembly=true;
  settings.dca_aggregate_kkt_layout="persistent";
  settings.kkt_preserve_structural_zeros=true;
  settings.kkt_adaptive_regularization=true;
  std::string message;
  auto factor=make_symmetric_factorization(settings,message);
  require(bool(factor),"unrelated factory: "+message);
  require(factor->factor(3,2,matrix(0),message),"unrelated initial factor: "+message);
  require(factor->factor(3,2,matrix(0.25),message),"unrelated second factor: "+message);
  require(factor->numeric_scatter_builds()==0 && factor->symbolic_analyses()==2,
          "aggregate-only optimization changed unrelated MUMPS caller");
}
}

int main() {
  if(!linear_backend_compiled("mumps")) {
    std::cout<<"[SKIP] MUMPS unavailable\n";return 0;
  }
  try {
    for(const auto* layout:{"dynamic","persistent"})
      for(bool adaptive:{true,false})compare_cached_and_reference(layout,adaptive);
    preserve_unrelated_callers();
    std::cout<<"[PASS] MUMPS aggregate scatter parity, ordered sums, zeros, inertia, quality and phase timers\n";
  }catch(const std::exception& error) {
    std::cerr<<"[FAIL] "<<error.what()<<'\n';return 1;
  }
}
