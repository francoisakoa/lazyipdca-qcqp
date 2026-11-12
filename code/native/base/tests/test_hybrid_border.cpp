#include "lazyipdca_qcqp/hybrid_border.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>
using namespace lazyipdca::qcqp;
using Matrix=std::vector<long double>;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Matrix dense(int n,const std::vector<SymmetricTriplet>& triplets,const Vec& values){
  require(triplets.size()==values.size(),"matrix/value dimension mismatch");
  Matrix a(n*n,0);
  for(std::size_t k=0;k<triplets.size();++k){auto e=triplets[k];a[e.i*n+e.j]+=values[k];if(e.i!=e.j)a[e.j*n+e.i]+=values[k];}
  return a;
}
std::vector<long double> solve(Matrix a,std::vector<long double> b,int n){
 for(int k=0;k<n;++k){int p=k;for(int i=k+1;i<n;++i)if(std::abs(a[i*n+k])>std::abs(a[p*n+k]))p=i;
  require(std::abs(a[p*n+k])>1e-24L,"test singular matrix");
  for(int j=k;j<n;++j)std::swap(a[k*n+j],a[p*n+j]);std::swap(b[k],b[p]);
  for(int i=k+1;i<n;++i){long double m=a[i*n+k]/a[k*n+k];for(int j=k+1;j<n;++j)a[i*n+j]-=m*a[k*n+j];b[i]-=m*b[k];}
 }
 for(int i=n-1;i>=0;--i){for(int j=i+1;j<n;++j)b[i]-=a[i*n+j]*b[j];b[i]/=a[i*n+i];}return b;
}
int main(){
 try {
  constexpr int primal=3,base=5;
  std::vector<SymmetricTriplet> sparse;Vec values;
  auto add=[&](int i,int j,double v){sparse.push_back({i,j,0});values.push_back(v);};
  // Base KKT includes independent primal and dual numerical shifts.
  add(0,0,2.2);add(1,1,1.7);add(2,2,3.2);add(0,1,-.2);add(0,2,.1);
  add(0,3,1);add(1,3,-1);add(1,4,.5);add(2,4,1.2);add(3,3,-.03);add(4,4,-.43);
  const auto original=dense(base,sparse,values);auto explicit_matrix=original;int dimension=base;
  const std::vector<std::pair<Vec,double>> atoms={{{1,0,2},.75},{{0,1,-1},2.5*4.2},{{0,1,0},.125*.3}};
  std::size_t entries=0;
  for(const auto& item:atoms){entries+=detail::append_hybrid_border(primal,item.first,item.second,dimension,sparse,values);
   for(int i=0;i<primal;++i)for(int j=0;j<primal;++j)explicit_matrix[i*base+j]+=(long double)item.second*item.first[i]*item.first[j];
  }
  require(dimension==8&&entries==8,"border did not preserve sparse rank-sized footprint");
  const auto border=dense(dimension,sparse,values);std::vector<long double> rhs={1,-2,.5,.3,-.7},augmented=rhs;augmented.resize(dimension,0);
  auto direct=solve(explicit_matrix,rhs,base),lifted=solve(border,augmented,dimension);
  long double difference=0,residual=0;
  for(int i=0;i<base;++i)difference=std::max(difference,std::abs(direct[i]-lifted[i]));
  for(int i=0;i<dimension;++i){long double v=-augmented[i];for(int j=0;j<dimension;++j)v+=border[i*dimension+j]*lifted[j];residual=std::max(residual,std::abs(v));}
  require(difference<1e-12L&&residual<1e-12L,"explicit and bordered solve mismatch");
  auto count=sparse.size();auto zero=detail::append_hybrid_border(primal,Vec(primal,0),1,dimension,sparse,values);require(zero==0&&count==sparse.size(),"zero atom appended");
  bool rejected=false;try{detail::append_hybrid_border(primal,{1,0,0},-1,dimension,sparse,values);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected&&count==sparse.size(),"negative atom accepted or mutated matrix");
  rejected=false;try{detail::append_hybrid_border(primal,{1,0,INFINITY},1,dimension,sparse,values);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected&&count==sparse.size(),"nonfinite atom accepted");
  Vec huge(10000,0);huge[2]=1;huge[9999]=-1;int dim=10000;std::vector<SymmetricTriplet> m;Vec v;
  require(detail::append_hybrid_border(10000,huge,2,dim,m,v)==3,"sparse border densified large atom");
  std::cout<<"{\"hybrid_border_equivalence\":true,\"objective_atoms\":1,\"inequality_atoms\":2,\"nonunit_inequality_multipliers\":true,\"primal_regularization\":0.2,\"dual_regularization\":0.03,\"max_solution_difference\":"<<double(difference)<<",\"border_residual\":"<<double(residual)<<",\"sparse_large_atom_entries\":3,\"unsafe_weights_rejected\":true}"<<std::endl;
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
