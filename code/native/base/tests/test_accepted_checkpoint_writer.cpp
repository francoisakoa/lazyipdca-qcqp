#include "../src/accepted_checkpoint_writer.hpp"
#include <fstream>
#include <iterator>
#include <iostream>
using namespace lazyipdca::qcqp;
int main(int argc,char**argv){
 const std::string path=argc==2?argv[1]:("/tmp/lazyipdca_checkpoint_writer_"+std::to_string(::getpid())+".json");
 Result r;r.x={0.1,-1e-200,1e200};r.certificate_equality_multipliers={-2};r.certificate_inequality_multipliers={3};r.certificate_lower_multipliers={0,0,0};r.certificate_upper_multipliers={0,0,0};
 detail::write_accepted_checkpoint(r,path,1,.25);
 std::ifstream f(path);std::string before((std::istreambuf_iterator<char>(f)),{});
 r.x[1]=std::numeric_limits<double>::quiet_NaN();bool caught=false;
 try{detail::write_accepted_checkpoint(r,path,2,.5);}catch(const std::runtime_error&){caught=true;}
 std::ifstream g(path);std::string after((std::istreambuf_iterator<char>(g)),{});
 if(!caught||before!=after)throw std::runtime_error("failed write replaced prior checkpoint");
 ::unlink(path.c_str());
 std::cout<<"writer PASS finite/roundtrip/preserve-previous-on-nonfinite\n";
}
