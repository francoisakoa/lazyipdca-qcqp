#include "../src/newton_atom_projection.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using lazyipdca::qcqp::detail::NewtonAtomProjection;
int main(){try{
  // Four original coordinates, three retained coordinates; overlapping
  // rows, structural zeros, an empty row and exact cancellations in T' u.
  const std::vector<int> begin={0,2,4,4,6},column={0,2,0,1,1,2};
  const std::vector<int> source={0,1,2,3};
  const std::vector<double> u={1,-1,3,2};
  NewtonAtomProjection cache;std::vector<int> support;std::vector<double> values;
  const NewtonAtomProjection::Entry* plan=nullptr;
  for(int k=0;k<100;++k) {
    std::vector<double> t={k%3==0?0.0:1.0,2.0,1.0,double(k%4),.5*double(k%4),-1.0};
    cache.update(source,u,begin,column,t,support,values);
    if(support!=std::vector<int>({0,1,2}))throw std::runtime_error("structural zeros dropped");
    if(k==0)plan=cache.entries.data();
    if(plan!=cache.entries.data()||cache.entries.size()!=6)throw std::runtime_error("projection recompiled");
    // Independent dense algebra T' u; same operation order for exact parity.
    std::vector<double> dense(12,0.0),expected(3,0.0);
    for(int i=0;i<4;++i)for(int p=begin[i];p<begin[i+1];++p)dense[3*i+column[p]]=t[p];
    for(int j=0;j<3;++j)for(int i=0;i<4;++i)expected[j]+=u[i]*dense[3*i+j];
    if(values!=expected)throw std::runtime_error("moving tangent or cancellation mismatch");
    // Atom action must agree with the dense outer product after projection.
    const double w=.25;const std::vector<double> d={2,-3,.5};double overlap=0;
    for(int i=0;i<3;++i)overlap+=values[i]*d[i];
    for(int i=0;i<3;++i){double action=0;for(int j=0;j<3;++j)action+=w*expected[i]*expected[j]*d[j];
      if(std::abs(action-w*values[i]*overlap)>1e-12)throw std::runtime_error("rank-one action mismatch");}
  }
  if(cache.updates!=100)throw std::runtime_error("projection update count");
  NewtonAtomProjection empty;std::vector<int> ids;std::vector<double> v;
  empty.update({2},{1},begin,column,{0,0,0,0,0,0},ids,v);
  empty.update({2},{1},begin,column,{1,2,3,4,5,6},ids,v);
  if(!empty.ready||!ids.empty()||!v.empty()||!empty.entries.empty())throw std::runtime_error("empty projection mismatch");
  std::cout<<"PASS fixed structural projection, changing tangents, zeros and dense atom action\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
