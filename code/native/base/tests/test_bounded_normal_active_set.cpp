#include "lazyipdca_qcqp/bounded_normal_active_set.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
using namespace lazyipdca::qcqp::detail;
static void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
int main() {
  const double inf=std::numeric_limits<double>::infinity();
  {
    std::vector<BoundedNormalColumn> c={{{{0,1}},-inf,inf},{{{0,1},{1,1e-8}},-inf,inf}};
    auto r=bounded_normal_active_set({0,1e-5},c,1);
    require(r.valid&&inf_norm(r.residual)<1e-10,"near-dependent free SVD fit failed");
  }
  {
    std::vector<BoundedNormalColumn> c={{{{0,1}},-inf,inf},{{{0,1},{1,1e-4}},0,2}};
    auto r=bounded_normal_active_set({-1,-1e-4},c,1);
    require(r.valid&&inf_norm(r.residual)<1e-10,"nonnegative bound release failed");
    require(r.multipliers[1]>=0&&r.multipliers[1]<=2,"fit violated bound");
  }
  {
    std::vector<BoundedNormalColumn> c={{{{0,1}},0,.25},{{{1,1}},0,inf}};
    auto r=bounded_normal_active_set({-1,2},c,1);
    require(r.valid&&std::abs(r.multipliers[0]-.25)<1e-14&&r.multipliers[1]==0,"bound clipping optimum failed");
    require(std::abs(r.residual[0]+.75)<1e-14&&r.residual[1]==2,"original residual wrong");
  }
  {
    std::vector<BoundedNormalColumn> c={{{{0,1}},-inf,inf},{{{0,1}},0,inf}};
    const Vec warm={-1e12,1e12};
    auto r=bounded_normal_active_set({1},c,1,2000,&warm);
    require(r.valid&&inf_norm(r.residual)<1e-10&&inf_norm(r.multipliers)<2,"warm dependent gauge retained");
  }
  {
    std::vector<BoundedNormalColumn> c={{{{0,.25},{0,.75}},.5,1},{{},3,4}};
    auto r=bounded_normal_active_set({0},c,1);
    require(r.valid&&r.multipliers[0]==.5&&r.multipliers[1]==3,"nonzero lower/fixed zero normal failed");
    require(r.residual[0]==.5,"duplicate normal accumulation wrong");
    require(!bounded_normal_active_set({0},c,0).valid,"zero budget ignored");
    c[0].lower=2;require(!bounded_normal_active_set({0},c,1).valid,"invalid interval accepted");
  }
  {
    std::vector<BoundedNormalColumn> c={{{{0,1},{1,1e-5}},-inf,inf},{{{0,-1}},0,2}};
    auto r=bounded_normal_active_set({0,-1e-5},c,1);
    require(r.valid&&inf_norm(r.residual)<1e-10,"dependent active face fit failed");
  }
  std::cout<<"bounded_normal_active_set: SVD / bounds / gauge / duplicate / invalid / zero budget PASS\n";
}
