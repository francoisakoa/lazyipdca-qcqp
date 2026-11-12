#include "lazyipdca_qcqp/native_inexact_contract.hpp"
#include "lazyipdca_qcqp/hybrid_inexact.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(){try {
 // min m(d)=d*d/2-d. At d=.9999 the kernel's absolute 1e-10
 // stationarity target fails, while the stated relative model KKT and
 // model-decrease contract are already certified (no constraints).
 double d=.9999,residual=std::abs(d-1),value=.5*d*d-d;
 auto g=hybrid_inner::check_model(residual,d,1,.25,1e-10,0,1e-6,0,1e-6,value,0);
 require(residual>1e-10&&g.passed,"analytic inexact certificate not exercised");
 require(native_model_certificate_authorized(true,g.passed,80,2,1,1200),"valid complete iteration-limit endpoint rejected");
 auto bad=hybrid_inner::check_model(.9999,.0001,1,.25,1e-10,0,1e-6,0,1e-6,-.000099995,0);
 require(!bad.passed&&!native_model_certificate_authorized(true,bad.passed,80,2,1,1200),"bad model certificate accepted");
 require(!native_model_certificate_authorized(false,true,80,2,1,1200),"incomplete/nonfinite/domain-invalid state accepted");
 require(!native_model_certificate_authorized(true,true,80,2,1200,1200),"expired certificate accepted");
 require(!native_model_certificate_authorized(true,true,1,2,1,1200),"minimum budget bypassed");
 require(!native_model_certificate_authorized(true,true,80,2,NAN,1200),"nonfinite time accepted");
 // A historical weight from a rejected model can reverse an otherwise
 // valid exact-L1 decrease near affine feasibility. A fresh multiplier-based
 // seed is common to anchor and trial; coefficients/tolerances do not change.
 double seed=hybrid_inner::model_penalty_seed(100);
 double anchor=10+seed*1e-14,trial=(10-1e-9)+seed*2e-14;
 require(trial<anchor,"fresh common model merit seed lost decrease");
 require((10-1e-9)+1e10*2e-14>10+1e10*1e-14,"stale penalty counterexample not exercised");
 std::cout<<"PASS native inexact contract: analytic model-relative certificate, strict-kernel disagreement, invalid model, incomplete domain, deadline, minimum steps\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
