#include "../src/rank_retry_statistics.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool good,const char* message){if(!good)throw std::runtime_error(message);}
int main(int argc,char** argv){try{
  Result primary,retry;
  primary.status=Status::InnerFailure;primary.message="protected original failure";
  primary.x={2};primary.objective=-7;primary.feasibility.total_inf=1e-9;
  primary.dual_infeasibility_inf=3;primary.certificate_equality_multipliers={4};
  primary.outer_iterations=7;primary.elapsed_seconds=2.5;primary.restoration_iterations=5;
  primary.lazy_newton.factor_attempts=11;primary.lazy_newton.linear_solves=12;
  primary.lazy_newton.cost_seconds[3]=.6;primary.lazy_newton.calls[1][2].values=31;
  primary.lazy_newton.first_sparse_failure_iteration=6;
  primary.lazy_newton.iteration_history.resize(1);
  primary.lazy_newton.iteration_history[0].iteration=7;
  primary.lazy_newton.iteration_history[0].elapsed_seconds=2;
  primary.dca_filter.majorant_outer_iterations=7;primary.dca_filter.majorant_barrier_line_trials=13;
  primary.dca_filter.majorant_curvature_seconds=.2;primary.dca_filter.majorant_barrier_mu=.3;
  primary.dca_filter.models=19;primary.multiplier_row_space_solves=3;
  retry.status=Status::Solved;retry.x={1};retry.objective=-9;
  retry.feasibility.total_inf=1e-12;retry.dual_infeasibility_inf=1e-8;
  retry.certificate_equality_multipliers={6};retry.termination_audit_valid=true;
  retry.outer_iterations=3;retry.elapsed_seconds=.75;retry.restoration_iterations=2;
  retry.lazy_newton.factor_attempts=8;retry.lazy_newton.linear_solves=9;
  retry.lazy_newton.cost_seconds[3]=.2;retry.lazy_newton.calls[1][2].values=4;
  retry.lazy_newton.first_sparse_failure_iteration=2;
  retry.lazy_newton.iteration_history.resize(1);
  retry.lazy_newton.iteration_history[0].iteration=3;
  retry.lazy_newton.iteration_history[0].elapsed_seconds=.5;
  retry.dca_filter.majorant_outer_iterations=3;retry.dca_filter.majorant_barrier_line_trials=5;
  retry.dca_filter.majorant_curvature_seconds=.1;retry.dca_filter.majorant_barrier_mu=.02;
  retry.dca_filter.models=4;retry.multiplier_row_space_solves=2;
  for(bool accepted:{true,false}) {
    Result chosen=accepted?retry:primary;
    detail::set_native_barrier_retry_statistics(chosen,primary,retry,accepted);
    detail::merge_rank_retry_work(chosen,accepted?primary:retry,false,accepted);
    check(chosen.outer_iterations==10&&chosen.restoration_iterations==7,"phase iteration work hidden");
    check(chosen.elapsed_seconds==3.25&&chosen.lazy_newton.factor_attempts==19&&chosen.lazy_newton.linear_solves==21,"phase costs/factors/solves hidden");
    check(chosen.lazy_newton.calls[1][2].values==35&&chosen.multiplier_row_space_solves==5,"function/fit work hidden");
    check(chosen.dca_filter.majorant_outer_iterations==10&&chosen.dca_filter.majorant_barrier_line_trials==18&&chosen.dca_filter.models==23,"DCA work hidden");
    check(std::abs(chosen.dca_filter.majorant_curvature_seconds-.3)<1e-14,"DCA time hidden");
    check(chosen.dca_filter.majorant_barrier_mu==(accepted?.02:.3),"retained barrier end state altered");
    const Result& endpoint=accepted?retry:primary;
    check(chosen.status==endpoint.status&&chosen.message==endpoint.message&&chosen.x==endpoint.x&&
          chosen.objective==endpoint.objective&&chosen.feasibility.total_inf==endpoint.feasibility.total_inf&&
          chosen.dual_infeasibility_inf==endpoint.dual_infeasibility_inf&&
          chosen.certificate_equality_multipliers==endpoint.certificate_equality_multipliers&&
          chosen.termination_audit_valid==endpoint.termination_audit_valid,"selected endpoint or original certificate changed");
    check(!chosen.rank_presolve.exact_fixed_retry_attempted,"native retry mislabeled exact-fixed retry");
    const auto& history=chosen.lazy_newton.iteration_history;
    check(history.size()==2&&history[0].iteration==7&&history[1].iteration==10&&
          history[0].elapsed_seconds==2&&history[1].elapsed_seconds==3,"rejected or accepted phase history order wrong");
    check(chosen.lazy_newton.first_sparse_failure_iteration==6,"first chronological failure lost");
    const auto& a=chosen.native_barrier_retry;
    check(a.attempted&&a.accepted==accepted&&a.primary_status=="inner_failure"&&a.retry_status=="solved"&&
          a.primary_outer_iterations==7&&a.retry_outer_iterations==3&&
          a.primary_cost_seconds[3]==.6&&a.retry_cost_seconds[3]==.2,"native phase provenance wrong");
    const auto output=result_json(chosen,"independent test");
    check(output.find("\"native_barrier_retry\"")!=std::string::npos&&
          output.find("\"attempted\":true")!=std::string::npos,"JSON native provenance missing");
    if(argc>1){std::ofstream f(std::string(argv[1])+(accepted?"_accepted.json":"_rejected.json"));f<<output;}
  }
  // If the primary phase had no failure, the retry's index must be shifted
  // even when its endpoint is rejected and the primary is retained.
  primary.lazy_newton.first_sparse_failure_iteration=-1;
  Result rejected=primary;
  detail::merge_rank_retry_work(rejected,retry,false,false);
  check(rejected.lazy_newton.first_sparse_failure_iteration==9,"retry failure offset wrong");
  std::cout<<"native_barrier_retry_statistics PASS accepted_and_rejected_endpoints=1 "
           <<"chronological_history=1 full_costs_and_DCA_counts=1 exact_fixed_labels_unchanged=1 JSON=1\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
