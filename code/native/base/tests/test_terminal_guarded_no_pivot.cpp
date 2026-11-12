#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;

int main(){
  QCQPProblem p; p.name="terminal_guarded_no_pivot"; p.n=3;
  p.x0={0.18758514685756175,0.10945702434350951,0.23110018888939368};
  p.lb={-2.0,-2.0,-2.0}; p.ub={2.0,2.0,2.0};
  p.objective.name="f";
  p.objective.linear={-0.07486217763775338,0.83772205062691785,-0.88284989350862519};
  p.objective.hessian.n=3;
  p.objective.hessian.entries={{0,0,-0.25311434927018184},{1,1,0.35072557287732042},{2,2,0.3064942118041134},{0,1,0.079680032488642194}};

  QuadraticFunction h; h.name="h"; h.hessian.n=3; h.constant=-0.5281423600904649; h.linear={1.0,1.0,1.0}; p.equalities.push_back(h);

  auto addg=[&](double c, std::initializer_list<double> lin,
                double h0,double h1,double h2){
    QuadraticFunction g; g.name="g"; g.constant=c; g.linear=Vec(lin); g.hessian.n=3;
    g.hessian.entries={{0,0,h0},{1,1,h1},{2,2,h2}}; p.inequalities.push_back(g);
  };
  addg(0.016048870765421976,{-0.86466008586279131,0.64652600439729047,0.31309633787473024},0.023268245848060132,0.096408126193968727,0.076288231298064502);
  addg(-0.2236511086206544,{-0.099812345828981797,0.50188253994128473,0.7971866508203862},0.051459763548131893,0.0060351605864389368,0.084869531396645728);
  addg(-0.15136506919959464,{0.79426252247097673,0.96998945518941948,-0.44741263807021636},-0.097337220859014528,-0.090768676867093778,0.069335308753982847);
  addg(0.29001448794388801,{-0.95255938240917237,0.10945962330396797,-0.537297763669488},-0.055505533449200756,0.048491890766181968,0.057747781766032617);

  Settings st; st.outer_algorithm="legacy-sqp"; st.verbose=false; st.max_outer_iterations=80; st.max_inner_iterations=200;
  st.feasibility_tolerance=1e-7; st.stationarity_tolerance=1e-6; st.inner_tolerance=1e-7;
  st.max_rank_per_quadratic=8; st.max_lagrangian_rank=8; st.exact_eigen_threshold=50;
  st.terminal_kkt_start_outer=0; st.terminal_kkt_trigger_stationarity=1e3;
  st.terminal_kkt_max_iterations=12; st.terminal_kkt_multiplier_tolerance=1e-12;

  Result r=Solver(st).solve(p);
  if(r.status!=Status::Solved){
    std::cerr<<"[FAIL] pivot regression not solved: "<<status_string(r.status)
             <<" stat="<<r.model_stationarity_inf<<"\n"; return 1;
  }
  if(r.terminal_kkt_active_set_drops!=0){
    std::cerr<<"[FAIL] guarded terminal KKT must never pivot/drop constraints from Newton dy: drops="
             <<r.terminal_kkt_active_set_drops<<"\n"; return 2;
  }
  if(r.terminal_kkt_attempts<=0 || r.terminal_kkt_accepted_steps<=0){
    std::cerr<<"[FAIL] guarded KKT path was not exercised: attempts="
             <<r.terminal_kkt_attempts<<" accept="<<r.terminal_kkt_accepted_steps<<"\n"; return 3;
  }
  if(r.feasibility.total_inf>st.feasibility_tolerance || r.model_stationarity_inf>st.stationarity_tolerance){
    std::cerr<<"[FAIL] residuals feas="<<r.feasibility.total_inf<<" stat="<<r.model_stationarity_inf<<"\n"; return 4;
  }
  std::cout<<"[PASS] guarded KKT attempts="<<r.terminal_kkt_attempts
           <<" accept="<<r.terminal_kkt_accepted_steps
           <<" drops="<<r.terminal_kkt_active_set_drops
           <<" stat="<<r.model_stationarity_inf<<"\n";
  return 0;
}
