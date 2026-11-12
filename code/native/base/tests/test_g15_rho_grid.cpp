#include "lazyipdca_qcqp/linear_solver.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include "../src/dca_progress_guard.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
int main(){try{
  const auto v=detail::dca_complete_inertia_schedule(0,100,100,false);
  require(v==std::vector<double>({0,20,100}),"missed current cap / previous certificate");
  const auto cap=detail::dca_complete_inertia_schedule(0,40,100,false);
  require(cap==std::vector<double>({0,8,80,40,100}),"legacy proposals changed or anchors omitted");
  const auto tiny=detail::dca_complete_inertia_schedule(0,100,1e-9,false);
  require(tiny==std::vector<double>({0,1e-9}),"sub-1e-6 cap not exercised");
  const auto zero=detail::dca_complete_inertia_schedule(0,100,0,false);
  require(zero==std::vector<double>({0}),"zero cap exceeded");
  const auto remembered=detail::dca_complete_inertia_schedule(0,1000,100,false);
  require(remembered==std::vector<double>({0,100}),"stale certificate above current cap was used");
  Settings st;st.linear_backend="native";st.kkt_primal_regularization=st.kkt_dual_regularization=0;
  std::string message;auto factor=make_symmetric_factorization(st,message);require(bool(factor),"native factor unavailable");
  int count=0;double certified=-1;
  for(double rho:v){++count;std::vector<SymmetricTriplet> a={{0,0,-50+rho},{1,1,1+rho}};
    if(factor->factor(2,2,a,message)){certified=rho;break;}}
  require(certified==100 && count==3,"real inertia factorization did not reach authorized certificate");
  Result diagnostic;diagnostic.dca_filter.strict_finish_linear_residual=INFINITY;
  diagnostic.dca_filter.strict_finish_error_before=INFINITY;
  const auto json=result_json(diagnostic,"nonfinite-finish-diagnostic");
  require(json.find("\"dca_filter_strict_finish_linear_residual\": null")!=std::string::npos &&
          json.find("\"dca_filter_strict_finish_error_before\": null")!=std::string::npos,
          "nonfinite failed-finish diagnostic corrupted JSON");
  std::cout<<"PASS rho=0,20 rejected; rho=100 certified by actual native factor; bounds, duplicates, legacy order, tiny cap\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
