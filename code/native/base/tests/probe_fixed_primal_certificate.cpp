#include "lazyipdca_qcqp/terminal_multiplier_fit.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>

using namespace lazyipdca::qcqp;
Vec vector_from(std::istream& input) {
  std::uint64_t n=0;input.read(reinterpret_cast<char*>(&n),sizeof(n));
  if(n>100000000)throw std::runtime_error("invalid diagnostic vector size");
  Vec values(n);input.read(reinterpret_cast<char*>(values.data()),n*sizeof(double));
  if(!input)throw std::runtime_error("truncated diagnostic vector");
  return values;
}
int main(int argc,char** argv) {try {
  if(argc!=5)throw std::runtime_error("probe_fixed_primal_certificate input.qcq vectors.bin output.json seconds");
  const auto problem=read_qcq(argv[1]);std::ifstream input(argv[2],std::ios::binary);
  Result result;result.status=Status::InnerFailure;result.x=vector_from(input);
  result.certificate_equality_multipliers=vector_from(input);
  result.certificate_inequality_multipliers=vector_from(input);
  result.certificate_lower_multipliers=vector_from(input);
  result.certificate_upper_multipliers=vector_from(input);
  Settings settings;settings.termination_test="project";
  settings.feasibility_tolerance=settings.stationarity_tolerance=
      settings.dual_infeasibility_tolerance=settings.complementarity_tolerance=1e-6;
  settings.live_display_level=1;
  detail::audit_original_termination(problem,settings,result);
  const Vec original=result.x;
  const bool strict=detail::terminal_original_multiplier_fit(problem,settings,result,std::stod(argv[4]));
  if(result.x!=original)throw std::runtime_error("fixed-primal certificate moved its point");
  write_result_json(result,problem.name,argv[3]);
  std::cout<<"fixed_primal_strict="<<strict<<" primal="<<result.feasibility.total_inf
      <<" dual="<<result.dual_infeasibility_inf<<" complementarity="<<result.complementarity_inf<<'\n';
  return strict?0:1;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
