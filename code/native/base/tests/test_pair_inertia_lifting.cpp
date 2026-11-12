#include "../src/sparse_pair_elimination.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
// Deterministic diagnostic mock: exact numerical elimination/inverse tests
// remain in g7_pair_elimination. Here isolate wrapper coordinate accounting.
class Counts final:public SymmetricFactorization {
 public:
  bool unknown=false,fail_reduced=false;int n=0,p=0;
  bool factor(int N,int P,const std::vector<SymmetricTriplet>&,std::string&)override{
    n=N;p=P;return !(fail_reduced&&N<5);
  }
  bool solve(Vec&,std::string&)const override{return false;}
  int last_positive_pivots()const override{return unknown?-1:p;}
  int last_negative_pivots()const override{return unknown?-1:n-p;}
  int last_zero_pivots()const override{return unknown?-1:0;}
  const char* name()const override{return "diagnostic-test-mock";}
};
int main(){try{
  const std::vector<SymmetricTriplet> a={{0,0,4},{1,1,5},{2,2,6},{3,3,-1e-6},
    {4,4,-.02},{0,3,1},{1,3,.5},{1,4,1},{2,4,.5},{0,1,.2}};
  std::string message;
  auto kernel=std::make_unique<Counts>();auto* observed=kernel.get();
  detail::PairReducedFactorization f(std::move(kernel),[](std::string&){return std::make_unique<Counts>();});
  check(f.factor(5,3,a,message),"reduced factor failed");
  check(f.elimination().pairs==2,"fixture has no two eliminated pairs");
  check(observed->n==1,"mock not operating on reduced coordinates");
  check(f.last_positive_pivots()==3&&f.last_negative_pivots()==2&&f.last_zero_pivots()==0,"original inertia not lifted");
  observed->unknown=true;
  check(f.last_positive_pivots()==-1&&f.last_negative_pivots()==-1&&f.last_zero_pivots()==-1,"unknown inertia fabricated");
  observed->unknown=false;observed->fail_reduced=true;
  check(f.factor(5,3,a,message),"full fallback failed");
  check(f.full_factor_calls==1,"full fallback not exercised");
  check(f.last_positive_pivots()==3&&f.last_negative_pivots()==2,"pairs double counted on full fallback");
  observed->fail_reduced=false;
  check(f.factor(5,3,a,message),"return to reduced failed");
  check(f.last_positive_pivots()==3&&f.last_negative_pivots()==2,"stale full/reduced accounting");
  std::cout<<"PASS original inertia, unknown sentinel, full fallback, return to reduced\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
