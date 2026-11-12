#include "lazyipdca_qcqp/qcqp_ipopt_termination.hpp"
#include <cmath>
#include <iostream>
#include <limits>
using namespace lazyipdca::qcqp;

static QuadraticFunction linear_q(const char* name, double c, double a) {
  QuadraticFunction q; q.name=name; q.constant=c; q.linear={a}; q.hessian.n=1; return q;
}
int main(){
  Settings st;
  st.feasibility_tolerance=1e-7; st.stationarity_tolerance=1e-6;
  st.dual_infeasibility_tolerance=1.0; st.complementarity_tolerance=1e-4;
  st.nlp_error_scaling_max=100.0;
  const double INF=std::numeric_limits<double>::infinity();

  QCQPProblem p; p.name="bad_comp"; p.n=1; p.x0={-1.0}; p.lb={-INF}; p.ub={INF};
  p.objective=linear_q("f",0.0,-1.0); p.inequalities.push_back(linear_q("g",0.0,1.0));
  QCQPTerminationMultipliers m; m.inequality={1.0}; m.lower_bound={0.0}; m.upper_bound={0.0};
  auto t=qcqp_ipopt_termination(p,{-1.0},m,st);
  if(std::abs(t.stop.dual_infeasibility)>1e-12 || std::abs(t.stop.complementarity-1.0)>1e-12 || t.stop.converged){
    std::cerr<<"[FAIL] complementarity guard dual="<<t.stop.dual_infeasibility<<" comp="<<t.stop.complementarity<<"\n"; return 1;
  }
  t=qcqp_ipopt_termination(p,{0.0},m,st);
  if(!t.stop.converged || t.stop.dual_infeasibility>1e-12 || t.stop.complementarity>1e-12){
    std::cerr<<"[FAIL] valid KKT pair rejected\n"; return 2;
  }

  QCQPProblem pf; pf.name="fixed"; pf.n=1; pf.x0={2.0}; pf.lb={2.0}; pf.ub={2.0};
  pf.objective=linear_q("f",0.0,123.0);
  QCQPTerminationMultipliers mf;
  auto tf=qcqp_ipopt_termination(pf,{2.0},mf,st);
  if(!tf.stop.converged || tf.stop.dual_infeasibility>1e-12 || tf.z_L_dimension!=0 || tf.z_U_dimension!=0){
    std::cerr<<"[FAIL] fixed make_parameter contract\n"; return 3;
  }
  std::cout<<"[PASS] KKT-stop certificate enforces complementarity and make_parameter fixed-variable semantics\n";
  return 0;
}
