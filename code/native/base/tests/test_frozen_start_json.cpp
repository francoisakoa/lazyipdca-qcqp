#include "../src/frozen_start_json.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(){int checks=0;try{
  auto require=[&](bool value,const char* why){++checks;if(!value)throw std::runtime_error(why);};
  const std::string valid=R"({"input_sha256":"public-source","x0":[1.25,-0.0],"lambda_eq":[999],"nested":{"x0":[999],"ignored":[true,false,null,{"a":"escaped\\string"}]}})";
  auto x=detail::FrozenStartJson(valid,2).read();require(x.size()==2&&x[0]==1.25&&x[1]==0&&std::signbit(x[1]),"frozen vector or signed zero changed");
  x=detail::FrozenStartJson(R"({"x\u0030":[-1.2e+2,4.9406564584124654e-324]})",2).read();
  require(x[0]==-120&&x[1]>0&&std::isfinite(x[1]),"finite exponent/subnormal rejected or changed");
  for(const std::string bad:{
      R"({})",R"({"nested":{"x0":[1,2]}})",R"({"x0":[1]})",R"({"x0":[1,2,3]})",
      R"({"x0":["1",2]})",R"({"x0":[true,2]})",R"({"x0":[null,2]})",R"({"x0":{"0":1,"1":2}})",
      R"({"x0":[NaN,2]})",R"({"x0":[Infinity,2]})",R"({"x0":[1e999,2]})",
      R"({"x0":[01,2]})",R"({"x0":[1.,2]})",R"({"x0":[+1,2]})",R"({"x0":[1e,2]})",
      R"({"x0":[1,2],"x0":[3,4]})",R"({"x0":[1,2],"x\u0030":[3,4]})",
      R"({"x0":[1,2]} trailing)",R"({"x0":[1,2],})",R"([1,2])",R"({"x0":[1,2})"}) {
    bool rejected=false;try{detail::FrozenStartJson(bad,2).read();}catch(const std::exception&){rejected=true;}
    require(rejected,"malformed, nonfinite or wrong-sized start was accepted");
  }
  bool rejected=false;try{detail::read_frozen_start("/nonexistent/frozen/start.json",2);}catch(const std::exception&){rejected=true;}
  require(rejected,"missing source file was accepted");
  std::cout<<"FROZEN_START_JSON_PASS checks="<<checks<<" finite_primal_only=1 strict_dimension=1 duplicate_x0_rejected=1\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
