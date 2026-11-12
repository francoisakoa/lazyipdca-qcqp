#pragma once
#include "lazyipdca_qcqp/result.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
inline void print_newton_report(const lazyipdca::qcqp::Result& r) {
  using namespace lazyipdca::qcqp;
  const auto& s=r.lazy_newton;if(s.display_level<2)return;
  std::ostringstream out;out<<std::fixed<<std::setprecision(6);
  out<<"\nTiming  |  wall seconds, exclusive sectors\n";
  auto row=[&](const char* name,double value){out<<"  "<<std::left<<std::setw(26)<<name<<std::right<<std::setw(12)<<value<<" s\n";};
  row("Core solve",s.core_solve_seconds);
  for(int i=(int)NewtonCost::Evaluation;i<=(int)NewtonCost::CoreOther;++i)row(newton_cost_names[i],s.cost_seconds[i]);
  out<<"\nPeripheral costs\n";
  for(auto sector:{NewtonCost::Preparation,NewtonCost::Certificate,NewtonCost::Diagnostics,NewtonCost::Cleanup})row(newton_cost_names[(int)sector],s.cost_seconds[(int)sector]);
  row("Solver wall (sum)",s.solver_wall_seconds);row("QCQP file read",r.cli_read_seconds);
  row("Solver API call",r.cli_solver_call_seconds);
  row("Solver wrapper",r.cli_solver_call_seconds-s.solver_wall_seconds);
  out<<"  JSON writing is measured in <result>.json.timing.json.\n";
  out<<"\nEvaluations  |  core only, individual scalar functions\n"
     <<"  family                values    gradients   cached J'v         Jv   H assembly\n";
  const char* families[]={"objective","equalities","inequalities","bound rows"};
  for(int f=0;f<4;++f){const auto& c=s.calls[1][f];out<<"  "<<std::left<<std::setw(17)<<families[f]<<std::right
    <<std::setw(11)<<c.values<<std::setw(13)<<c.gradients<<std::setw(13)<<c.cached_gradient_accumulations
    <<std::setw(11)<<c.jacobian_products<<std::setw(13)<<c.hessian_assemblies<<"\n";}
  out<<"  Lagrangian Hessian products: "<<s.lagrangian_hessian_products
     <<"; Hessians are constant QCQP coefficients (no fresh Hessian callbacks).\n";
  if(s.display_level>=3) {
    out<<"\nEvaluations outside core  |  values / gradients\n";
    const char* phases[]={"preparation","core","diagnostics","certificate"};
    for(int phase:{0,2,3})for(int f=0;f<4;++f){const auto& c=s.calls[phase][f];
      out<<"  "<<std::left<<std::setw(14)<<phases[phase]<<std::setw(17)<<families[f]<<std::right<<std::setw(11)<<c.values<<" / "<<c.gradients<<"\n";}
  }
  std::cout<<out.str()<<std::flush;
}
inline void write_newton_cli_timing(const std::string& path,const lazyipdca::qcqp::Result& r,
    double summary,double write,double wall) {
  std::ofstream out(path);if(!out)throw std::runtime_error("cannot write CLI timing report: "+path);
  out<<std::setprecision(17)<<"{\n  \"schema\": \"lazy-newton-cli-timing-v1\",\n"
    <<"  \"arguments_seconds\": "<<r.cli_arguments_seconds<<",\n"
    <<"  \"qcqp_read_seconds\": "<<r.cli_read_seconds<<",\n"
    <<"  \"solver_call_seconds\": "<<r.cli_solver_call_seconds<<",\n"
    <<"  \"summary_output_seconds\": "<<summary<<",\n"
    <<"  \"result_json_write_seconds\": "<<write<<",\n"
    <<"  \"process_until_result_written_seconds\": "<<wall<<",\n"
    <<"  \"excludes\": [\"writing this timing sidecar\", \"process teardown\"]\n}\n";
  out.close();if(!out)throw std::runtime_error("failed writing CLI timing report: "+path);
}
