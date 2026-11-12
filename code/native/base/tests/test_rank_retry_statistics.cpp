#include "../src/rank_retry_statistics.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool good,const char* message){if(!good)throw std::runtime_error(message);}
int main(){try{
  Result primary,retry;
  primary.status=Status::InnerFailure;primary.outer_iterations=7;primary.elapsed_seconds=2.5;
  primary.restoration_iterations=13;primary.restoration_total_seconds=.2;
  primary.lazy_newton.factor_attempts=11;primary.lazy_newton.linear_solves=12;
  primary.lazy_newton.core_solve_seconds=2;primary.lazy_newton.cost_seconds[3]=.6;
  primary.lazy_newton.calls[1][2].values=31;primary.lazy_newton.calls[1][2].hessian_products=17;
  primary.lazy_newton.kkt_dimension=111;primary.lazy_newton.objective_correction_rank=5;
  primary.lazy_newton.iteration_history.resize(1);primary.lazy_newton.iteration_history[0].iteration=7;
  retry.status=Status::Solved;retry.outer_iterations=3;retry.elapsed_seconds=.75;
  retry.restoration_iterations=4;retry.restoration_total_seconds=.1;
  retry.lazy_newton.factor_attempts=8;retry.lazy_newton.linear_solves=9;
  retry.lazy_newton.core_solve_seconds=.5;retry.lazy_newton.cost_seconds[3]=.2;
  retry.lazy_newton.calls[1][2].values=4;retry.lazy_newton.calls[1][2].hessian_products=6;
  retry.lazy_newton.kkt_dimension=77;retry.lazy_newton.objective_correction_rank=2;
  retry.lazy_newton.iteration_history.resize(1);retry.lazy_newton.iteration_history[0].iteration=3;
  retry.lazy_newton.iteration_history[0].elapsed_seconds=.5;
  retry.x={1,2};retry.objective=-4;retry.raw_kkt_error=1e-8;retry.certificate_inequality_multipliers={3};
  detail::merge_rank_retry_work(retry,primary);
  check(retry.outer_iterations==10&&retry.restoration_iterations==17,"iteration budgets were not cumulative");
  check(retry.elapsed_seconds==3.25&&retry.lazy_newton.core_solve_seconds==2.5,"time was not cumulative");
  check(retry.lazy_newton.factor_attempts==19&&retry.lazy_newton.linear_solves==21,"factor/solve work was hidden");
  check(retry.lazy_newton.calls[1][2].values==35&&retry.lazy_newton.calls[1][2].hessian_products==23,"function work was hidden");
  check(retry.lazy_newton.kkt_dimension==77&&retry.lazy_newton.objective_correction_rank==2,"end-state dimensions or ranks were added");
  check(retry.status==Status::Solved&&retry.x==Vec({1,2})&&retry.objective==-4&&retry.raw_kkt_error==1e-8&&retry.certificate_inequality_multipliers==Vec({3}),"final point or certificate was changed");
  check(retry.lazy_newton.iteration_history.size()==2&&retry.lazy_newton.iteration_history[1].iteration==10&&retry.lazy_newton.iteration_history[1].elapsed_seconds==3,"phase history offsets wrong");
  const auto& r=retry.rank_presolve;
  check(r.exact_fixed_retry_attempted&&r.exact_fixed_primary_status=="inner_failure"&&r.exact_fixed_retry_status=="solved"&&r.exact_fixed_primary_seconds==2.5&&r.exact_fixed_retry_seconds==.75,"phase accounting metadata missing");
  check(r.exact_fixed_primary_outer_iterations==7&&r.exact_fixed_retry_outer_iterations==3&&r.exact_fixed_primary_history_records==1&&r.exact_fixed_primary_cost_seconds[3]==.6,"phase work metadata incorrect");
  std::cout<<"rank retry statistics: actual work, phase provenance, final certificate and history offsets PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
