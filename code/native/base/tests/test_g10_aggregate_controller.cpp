#include "../src/dca_aggregate_controller.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp::detail;
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){try{
  DcaPositiveRhoMemory memory;
  memory.certificate(200,true);
  memory.certificate(0,true);
  require(memory.value==200,"zero erased positive curvature history");
  memory.certificate(2e7,false);
  require(memory.value==200,"step regularization contaminated curvature history");
  require(memory.seed(1e-6,0.2,1e9)==40,"historical seed incorrect");
  require(memory.seed(1e-6,0.2,20)==20,"seed exceeded cap");
  memory.certificate(4,true);
  require(memory.value==4,"new positive certificate did not replace history");
  require(dca_aggregate_factor_rejection("MUMPS inertia mismatch: negative pivots=3, expected=2")==DcaAggregateRejection::inertia,"inertia classification");
  require(dca_aggregate_factor_rejection("MUMPS factorization failed INFOG(1)=-10")==DcaAggregateRejection::backend,"singularity guessed as inertia");
  require(!dca_aggregate_retry_permitted(DcaAggregateRejection::backend,true),"backend error retried by rho inflation");
  require(!dca_aggregate_retry_permitted(DcaAggregateRejection::numerical_direction,true),"precision error retried by rho inflation");
  require(dca_aggregate_retry_permitted(DcaAggregateRejection::filter,true),"filtered step retry removed");
  require(dca_aggregate_retry_permitted(DcaAggregateRejection::numerical_direction,false),"legacy policy not preserved");
  require(dca_aggregate_point_rejection("nonpositive-local-state")==DcaAggregateRejection::domain,"domain classification");
  require(dca_aggregate_point_rejection("insufficient-kkt-contraction")==DcaAggregateRejection::kkt,"KKT classification");
  std::cout<<"Aggregate rejection policy and independent positive-rho memory passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
