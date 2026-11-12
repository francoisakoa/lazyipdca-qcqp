#include "lazyipdca_qcqp/hybrid_border.hpp"
#include "../src/bounded_pivot_recovery.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
using Matrix=std::vector<long double>;
static void require(bool ok,const char* message) {
  if(!ok)throw std::runtime_error(message);
}
static std::vector<long double> dense_solve(Matrix a,std::vector<long double> b,int n) {
  for(int k=0;k<n;++k) {
    int p=k;
    for(int i=k+1;i<n;++i)if(std::abs(a[i*n+k])>std::abs(a[p*n+k]))p=i;
    require(std::abs(a[p*n+k])>1e-25L,"singular independent explicit test matrix");
    for(int j=k;j<n;++j)std::swap(a[k*n+j],a[p*n+j]);
    std::swap(b[k],b[p]);
    for(int i=k+1;i<n;++i) {
      const long double m=a[i*n+k]/a[k*n+k];
      for(int j=k+1;j<n;++j)a[i*n+j]-=m*a[k*n+j];
      b[i]-=m*b[k];
    }
  }
  for(int i=n-1;i>=0;--i) {
    for(int j=i+1;j<n;++j)b[i]-=a[i*n+j]*b[j];
    b[i]/=a[i*n+i];
  }
  return b;
}
static Matrix dense(int n,const std::vector<SymmetricTriplet>& entries) {
  Matrix a(n*n,0);
  for(const auto& e:entries) {
    a[e.i*n+e.j]+=e.value;
    if(e.i!=e.j)a[e.j*n+e.i]+=e.value;
  }
  return a;
}
struct Evidence {
  double difference=0,residual=0,prior_bug_difference=0,recovery_difference=0;
  int rank=0,dimension=0;
};
static Evidence check(bool objective,bool inequalities) {
  constexpr int primal=3,base=5;
  const std::vector<SymmetricTriplet> entries={
    {0,0,2},{1,1,1.5},{2,2,3},{0,1,-.2},{0,2,.1},
    {0,3,1},{1,3,-1},{1,4,.5},{2,4,1.2},{4,4,-.4}};
  // T eliminates the fourth full-space coordinate: x3=.5*x0-2*x1+x2.
  // Project every full atom with T', rather than dropping eliminated entries.
  const auto project=[](const Vec& a) {
    return Vec{a[0]+.5*a[3],a[1]-2*a[3],a[2]+a[3]};
  };
  std::vector<std::pair<Vec,double>> atoms;
  if(objective)atoms.push_back({project({1,0,2,.75}),.4*1.875});
  if(inequalities) {
    atoms.push_back({project({0,1,-1,.5}),2.5*.7*6.0});
    atoms.push_back({project({0,1,0,-.25}),.125*1.2*.25});
  }
  auto sparse=entries;
  Vec values;
  for(const auto& e:sparse)values.push_back(e.value);
  int dimension=base;
  for(const auto& a:atoms)
    detail::append_hybrid_border(primal,a.first,a.second,dimension,sparse,values);
  for(std::size_t k=0;k<sparse.size();++k)sparse[k].value=values[k];
  require(dimension==base+int(atoms.size()),"border rank/dimension mismatch");
  const Vec rhs={1,-2,.5,.3,-.7};
  const auto explicit_matrix=[&](double primal_shift,double dual_shift) {
    auto a=dense(base,entries);
    for(int i=0;i<base;++i)a[i*base+i]+=i<primal?primal_shift:-dual_shift;
    for(const auto& atom:atoms)
      for(int i=0;i<primal;++i)for(int j=0;j<primal;++j)
        a[i*base+j]+=(long double)atom.second*atom.first[i]*atom.first[j];
    return a;
  };
  const auto expected=dense_solve(explicit_matrix(.2,.03),
      std::vector<long double>(rhs.begin(),rhs.end()),base);
  Settings st;
  st.linear_backend="mumps";st.linear_metis=true;
  st.kkt_adaptive_regularization=false;
  st.kkt_primal_regularization=.2;st.kkt_dual_regularization=.03;
  st.kkt_dual_regularization_end=base;
  std::string message;
  auto factor=make_symmetric_factorization(st,message);
  require(bool(factor),"MUMPS+METIS not available: this test requires actual backend");
  require(std::string(factor->name()).find("mumps_sym_indef_metis")!=std::string::npos,"unexpected fallback backend");
  require(factor->factor(dimension,primal,sparse,message),message.c_str());
  Vec lifted=rhs;lifted.resize(dimension,0);
  require(factor->solve(lifted,message),message.c_str());
  Evidence out;out.rank=atoms.size();out.dimension=dimension;
  for(int i=0;i<base;++i)out.difference=std::max(out.difference,double(std::abs(expected[i]-lifted[i])));
  auto shifted=dense(dimension,sparse);
  for(int i=0;i<base;++i)shifted[i*dimension+i]+=i<primal?.2:-.03;
  for(int i=0;i<dimension;++i) {
    long double r=i<base?-rhs[i]:0;
    for(int j=0;j<dimension;++j)r+=shifted[i*dimension+j]*lifted[j];
    out.residual=std::max(out.residual,double(std::abs(r)));
  }
  require(out.difference<1e-11&&out.residual<1e-11,"actual MUMPS bordered metric differs from explicit correction");
  // This control reproduces the former backend bug and must measurably differ.
  st.kkt_dual_regularization_end=-1;
  auto old=make_symmetric_factorization(st,message);
  require(old&&old->factor(dimension,primal,sparse,message),message.c_str());
  Vec attenuated=rhs;attenuated.resize(dimension,0);
  require(old->solve(attenuated,message),message.c_str());
  for(int i=0;i<base;++i)out.prior_bug_difference=std::max(out.prior_bug_difference,
      double(std::abs(expected[i]-attenuated[i])));
  require(out.prior_bug_difference>1e-6,"prior-bug control did not detect atom pivot attenuation");
  // The recovery preconditioner must protect the same mathematical -I pivots.
  st.kkt_dual_regularization_end=base;
  detail::BoundedPivotRecovery recovery(st,.03,message);
  require(recovery.factor(dimension,primal,sparse,message),message.c_str());
  Vec recovered=rhs;recovered.resize(dimension,0);
  require(recovery.solve(recovered,message),message.c_str());
  const auto recovery_expected=dense_solve(explicit_matrix(.03,.03),
      std::vector<long double>(rhs.begin(),rhs.end()),base);
  for(int i=0;i<base;++i)out.recovery_difference=std::max(out.recovery_difference,
      double(std::abs(recovery_expected[i]-recovered[i])));
  require(out.recovery_difference<1e-11,"recovery regularized mathematical atom pivots");
  return out;
}
static double check_large_sparse() {
  constexpr int primal=10000,base=primal+1;
  std::vector<SymmetricTriplet> entries;
  Vec values;
  const auto add=[&](int i,int j,double value) {
    entries.push_back({i,j,value});values.push_back(value);
  };
  for(int i=0;i<primal;++i)add(i,i,2);
  add(0,primal,1);
  Vec a(primal,0),b(primal,0);
  a[1]=1;a[9999]=-.5;b[2]=.75;b[9000]=1.25;
  const std::vector<std::pair<Vec,double>> atoms={{a,.4*1.875},{b,2.5*.7*6}};
  int dimension=base;std::size_t atom_entries=0;
  for(const auto& atom:atoms)
    atom_entries+=detail::append_hybrid_border(primal,atom.first,atom.second,dimension,entries,values);
  require(atom_entries==6&&dimension==base+2,"large sparse border filled the base Hessian");
  for(std::size_t k=0;k<entries.size();++k)entries[k].value=values[k];
  Vec expected(base,0),rhs(dimension,0);
  expected[0]=.75;expected[1]=-.5;expected[2]=.25;
  expected[9000]=-.75;expected[9999]=.125;expected[primal]=-.125;
  for(int i=0;i<primal;++i)rhs[i]=2.2*expected[i];
  rhs[0]+=expected[primal];rhs[primal]=expected[0]-.03*expected[primal];
  for(const auto& atom:atoms) {
    long double product=0;
    for(int i=0;i<primal;++i)product+=(long double)atom.first[i]*expected[i];
    for(int i=0;i<primal;++i)rhs[i]+=double(atom.second*atom.first[i]*product);
  }
  Settings st;st.linear_backend="mumps";st.linear_metis=true;
  st.kkt_adaptive_regularization=false;
  st.kkt_primal_regularization=.2;st.kkt_dual_regularization=.03;
  st.kkt_dual_regularization_end=base;
  std::string message;auto factor=make_symmetric_factorization(st,message);
  require(factor&&factor->factor(dimension,primal,entries,message),message.c_str());
  require(factor->solve(rhs,message),message.c_str());
  double difference=0;
  for(int i=0;i<base;++i)difference=std::max(difference,std::abs(rhs[i]-expected[i]));
  require(difference<1e-10,"actual large sparse MUMPS border solution mismatch");
  return difference;
}
int main() {
  try {
    std::cout<<std::setprecision(17)<<"{\"actual_backend\":\"MUMPS+METIS\",\"primal_regularization\":0.2,\"dual_regularization\":0.03,\"nonunit_inequality_multipliers\":true,\"function_scaling\":true,\"eliminated_projection\":true,\"atom_auxiliary_diagonal\":-1,\"modes\":{";
    int position=0;
    for(const auto& mode:{std::string("HH"),std::string("HS"),std::string("SH")}) {
      auto e=check(mode!="SH",mode!="HS");
      if(position++)std::cout<<',';
      std::cout<<'"'<<mode<<"\":{\"rank\":"<<e.rank<<",\"dimension\":"<<e.dimension
        <<",\"explicit_solution_difference\":"<<e.difference
        <<",\"shifted_border_residual\":"<<e.residual
        <<",\"prior_bug_control_difference\":"<<e.prior_bug_difference
        <<",\"recovery_solution_difference\":"<<e.recovery_difference<<'}';
    }
    const double sparse_difference=check_large_sparse();
    std::cout<<"},\"large_sparse\":{\"primal_dimension\":10000,\"rank\":2,\"objective_atoms\":1,\"inequality_atoms\":1,\"border_nnz\":6,\"explicit_atom_outer_products\":0,\"solution_difference\":"<<sparse_difference<<"}}"<<std::endl;
    return 0;
  }catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
