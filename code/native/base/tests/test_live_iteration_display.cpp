#include "lazyipdca_qcqp/solver.hpp"

#include <iostream>
#include <sstream>
#include <string>

using namespace lazyipdca::qcqp;

int main() {
  QCQPProblem p;
  p.name="live_display_contract";
  p.n=2;
  p.x0={1.0,0.0};
  p.lb={-100.0,-100.0};
  p.ub={100.0,100.0};
  p.objective.name="linear_descent";
  p.objective.linear={0.0,-1.0};
  p.objective.hessian.n=2;

  QuadraticFunction h;
  h.name="unit_circle";
  h.constant=-1.0;
  h.linear={0.0,0.0};
  h.hessian.n=2;
  h.hessian.entries={{0,0,2.0},{1,1,2.0}};
  p.equalities.push_back(h);

  // Include a genuine original inequality so the dedicated inf_ineq and Lazy
  // working-set/rank columns are covered by the public display contract.
  QuadraticFunction g;
  g.name="upper_halfspace";
  g.constant=-2.0;
  g.linear={0.0,1.0};
  g.hessian.n=2;
  p.inequalities.push_back(g);

  Settings st;
  st.outer_algorithm="lazy-dca";
  st.verbose=false;
  st.live_iterations=true;
  st.live_iteration_every=1;
  st.live_iteration_header_every=1;
  st.restoration_enabled=false;
  st.max_outer_iterations=1;
  st.max_inner_iterations=200;
  st.feasibility_tolerance=1e-6;
  st.inner_tolerance=1e-9;
  st.dca_local_corrector=false;

  std::ostringstream capture;
  std::streambuf* previous=std::cout.rdbuf(capture.rdbuf());
  (void)Solver(st).solve(p);
  std::cout.rdbuf(previous);

  const std::string text=capture.str();
  const char* required[]={"[ITER]", "objective", "inf_eq", "inf_ineq",
                          "inf_bnd", "inf_du", "lg(mu)", "r0", "rI",
                          "ws", "bF", "tag", "eqM"};
  for(const char* token:required) {
    if(text.find(token)==std::string::npos) {
      std::cerr<<"[FAIL] live table missing token '"<<token<<"'\n"<<text;
      return 1;
    }
  }
  if(text.find(" 0/1")==std::string::npos &&
     text.find(" 1/1")==std::string::npos) {
    std::cerr<<"[FAIL] live table missing inequality working-set cardinality\n"
             <<text;
    return 2;
  }
  std::cout<<"[PASS] live iteration display exposes original NLP and Lazy columns\n";
  return 0;
}
