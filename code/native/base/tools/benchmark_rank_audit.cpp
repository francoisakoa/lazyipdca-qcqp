#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(int argc,char** argv){try{
  if(argc<2||argc>4){std::cerr<<"Usage: benchmark_rank_audit input.qcq [audits-per-sample:10] [samples:3]\n";return 2;}
  const int loops=argc>2?std::stoi(argv[2]):10,samples=argc>3?std::stoi(argv[3]):3;
  if(loops<1||loops>1000||samples<1||samples>30)throw std::invalid_argument("invalid sample budget");
  auto original=read_qcq_compact(argv[1]);ConstraintPresolve reduction(original);const auto& p=reduction.reduced();
  Result seed;seed.x=p.x0;seed.certificate_equality_multipliers.assign(p.equalities.size(),.5);
  seed.certificate_inequality_multipliers.assign(p.inequalities.size(),.1);
  seed.certificate_lower_multipliers.resize(p.n);seed.certificate_upper_multipliers.resize(p.n);
  for(int j=0;j<p.n;++j){seed.certificate_lower_multipliers[j]=std::isfinite(p.lb[j])?.2:0;seed.certificate_upper_multipliers[j]=std::isfinite(p.ub[j])?.3:0;}
  seed.termination_scaling_available=true;seed.termination_objective_scale=.01;
  seed.termination_equality_scales.assign(p.equalities.size(),.5);seed.termination_inequality_scales.assign(p.inequalities.size(),.25);
  seed.termination_inequality_slacks.assign(p.inequalities.size(),.2);seed.outer_iterations=2;
  Settings st;st.rank_presolve_context=&reduction;
  std::cout<<std::setprecision(17)<<"{\"schema\":\"rank-audit-microbenchmark-v1\",\"optimization_run\":false,\"point\":\"qcq-x0-with-synthetic-multipliers-and-slacks\",\"audits_per_sample\":"<<loops
    <<",\"samples\":"<<samples<<",\"shared_value_rows\":"<<reduction.statistics().audit_shared_values
    <<",\"distinct_recipe_exceptions\":"<<reduction.statistics().audit_recipe_exceptions<<",\"measurements\":[";
  bool comma=false;
  for(const char* profile:{"project","ipopt"}){
    st.termination_test=profile;auto check=seed;st.dca_rank_audit_mode="check";detail::OriginalSignedResidual residual;
    detail::audit_original_termination(p,st,check,nullptr,true,&residual);
    for(int sample=0;sample<samples;++sample){
      // Alternate order to reduce a systematic warm-cache ordering effect.
      for(int arm=0;arm<2;++arm){const char* mode=((sample+arm)%2==0)?"full":"mapped";st.dca_rank_audit_mode=mode;
        auto r=seed;const auto before=reduction.statistics();
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<loops;++i)detail::audit_original_termination(p,st,r);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(!r.termination_audit_valid)throw std::runtime_error("invalid benchmark state");
        const auto after=reduction.statistics();
        if(comma)std::cout<<',';
        comma=true;
        std::cout<<"{\"profile\":\""<<profile<<"\",\"mode\":\""<<mode<<"\",\"sample\":"<<sample<<",\"seconds\":"<<seconds
          <<",\"value_evaluations\":"<<(after.audit_value_evaluations-before.audit_value_evaluations)
          <<",\"reused_values\":"<<(after.audit_value_reuses-before.audit_value_reuses)
          <<",\"result_copies\":"<<(after.audit_result_copies-before.audit_result_copies)
          <<",\"lift_calls\":"<<(after.audit_lift_calls-before.audit_lift_calls)<<'}';
      }
    }
  }
  std::cout<<"],\"check_mode_exact_parity\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
