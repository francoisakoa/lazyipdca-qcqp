#include "../src/dca_stable_border.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using lazyipdca::qcqp::SymmetricTriplet;
using lazyipdca::qcqp::detail::DcaStableBorderSlots;
namespace {
using Vector=std::vector<double>;
using Matrix=std::vector<Vector>;
struct Gradient {std::vector<int> support;Vector values;};
void require(bool ok,const std::string& message) {
  if(!ok)throw std::runtime_error(message);
}
void close(double a,double b,const std::string& message) {
  require(std::abs(a-b)<=2e-11*std::max({1.0,std::abs(a),std::abs(b)}),message);
}
Matrix dense(int n,const std::vector<SymmetricTriplet>& entries) {
  Matrix result(n,Vector(n,0));
  for(const auto& e:entries) {
    require(e.i>=0 && e.j>=0 && e.i<n && e.j<n,"triplet outside layout");
    result[e.i][e.j]+=e.value;
    if(e.i!=e.j)result[e.j][e.i]+=e.value;
  }
  return result;
}
Vector solve(Matrix a,Vector b) {
  const int n=static_cast<int>(a.size());
  for(int k=0;k<n;++k) {
    int p=k;for(int i=k+1;i<n;++i)if(std::abs(a[i][k])>std::abs(a[p][k]))p=i;
    require(std::abs(a[p][k])>1e-14,"singular test matrix");
    std::swap(a[p],a[k]);std::swap(b[p],b[k]);
    for(int i=k+1;i<n;++i) {
      const double t=a[i][k]/a[k][k];
      for(int j=k+1;j<n;++j)a[i][j]-=t*a[k][j];
      b[i]-=t*b[k];a[i][k]=0;
    }
  }
  for(int i=n-1;i>=0;--i) {
    for(int j=i+1;j<n;++j)b[i]-=a[i][j]*b[j];
    b[i]/=a[i][i];
  }
  return b;
}
std::pair<int,int> inertia(Matrix a) {
  // These fixtures have a positive definite primal block and independent
  // constraint gradients; their unpivoted LDL has no zero leading pivots.
  int positive=0,negative=0;
  for(int k=0;k<static_cast<int>(a.size());++k) {
    require(std::abs(a[k][k])>1e-12,"zero test LDL pivot");
    if(a[k][k]>0)++positive;else ++negative;
    for(int i=k+1;i<static_cast<int>(a.size());++i)
      for(int j=i;j<static_cast<int>(a.size());++j) {
        a[j][i]-=a[i][k]*a[j][k]/a[k][k];a[i][j]=a[j][i];
      }
  }
  return {positive,negative};
}
std::vector<std::pair<int,int>> graph(const std::vector<SymmetricTriplet>& matrix) {
  std::vector<std::pair<int,int>> result;
  for(const auto& e:matrix)result.emplace_back(std::min(e.i,e.j),std::max(e.i,e.j));
  std::sort(result.begin(),result.end());
  result.erase(std::unique(result.begin(),result.end()),result.end());return result;
}

void solve_equivalence_and_dummy_inertia() {
  const std::vector<Gradient> gradients={{{0,1},{1,.3}},{{1,2},{.8,-.2}},{{0,2},{.4,1.1}}};
  const Vector s={2,1.2,.8},z={.3,1.5,.7},border_rhs={.2,-.4,.9};
  const std::vector<SymmetricTriplet> baseline={{0,0,4},{1,1,3},{2,2,2},
      {0,1,.2},{1,2,-.1},{0,3,1},{1,3,.2},{2,3,-.1},{3,3,0}};
  DcaStableBorderSlots layout;layout.configure(3,1,3);
  std::vector<int> first_ids;
  std::vector<std::pair<int,int>> previous_graph;
  std::uint64_t previous_generation=0;
  // Includes permutations, de-promotion, re-promotion, an inactive inequality,
  // and all inequalities condensed. The requested classification stays free.
  const std::vector<std::vector<int>> requested={{2,0},{0,2},{1},{2,1,0},{},{0}};
  for(std::size_t turn=0;turn<requested.size();++turn) {
    std::vector<unsigned char> active={1,1,1};if(turn==4)active[1]=0;
    layout.update(requested[turn]);
    auto entries=baseline;Vector rhs(layout.dimension(),0);
    rhs[0]=.4;rhs[1]=-.7;rhs[2]=1.2;rhs[3]=.1;
    auto reference=baseline;Vector reference_rhs={.4,-.7,1.2,.1};
    for(int i=0;i<3;++i) {
      const auto& g=gradients[i];const double ratio=z[i]/s[i];
      for(std::size_t a=0;a<g.support.size();++a) {
        if(active[i])reference_rhs[g.support[a]]+=ratio*g.values[a]*border_rhs[i];
        if(active[i] && layout.active_row_for(i)<0)
          rhs[g.support[a]]+=ratio*g.values[a]*border_rhs[i];
        for(std::size_t b=a;b<g.support.size();++b) {
          const double v=active[i]?ratio*g.values[a]*g.values[b]:0.0;
          reference.push_back({g.support[a],g.support[b],v});
          // Preserve condensed structural entries even while the row is
          // numerically represented at its persistent border position.
          entries.push_back({g.support[a],g.support[b],layout.active_row_for(i)<0?v:0.0});
        }
      }
      if(layout.active_row_for(i)>=0)rhs[layout.active_row_for(i)]=border_rhs[i];
    }
    layout.append_borders(gradients,s,z,entries);
    const Matrix matrix=dense(layout.dimension(),entries);
    const Vector actual=solve(matrix,rhs),expected=solve(dense(4,reference),reference_rhs);
    for(int i=0;i<4;++i)close(actual[i],expected[i],"persistent Schur solve differs from full condensation");
    for(int i=0;i<3;++i)if(layout.active_row_for(i)>=0) {
      double gx=0;for(std::size_t k=0;k<gradients[i].support.size();++k)
        gx+=gradients[i].values[k]*actual[gradients[i].support[k]];
      close(actual[layout.active_row_for(i)],(gx-border_rhs[i])*z[i]/s[i],"border recovery differs");
    }
    int dummy_count=0;
    for(int row=4;row<layout.dimension();++row) {
      bool live=false;for(int i=0;i<3;++i)live|=layout.active_row_for(i)==row;
      if(!live) {
        ++dummy_count;close(matrix[row][row],-1,"dummy diagonal changed");
        for(int j=0;j<layout.dimension();++j)if(j!=row)close(matrix[row][j],0,"dummy coupled to operator");
        close(actual[row],0,"zero dummy RHS produces nonzero solution");
      }
    }
    require(dummy_count==layout.capacity()-static_cast<int>(requested[turn].size()),"dummy count mismatch");
    const auto signs=inertia(matrix);
    require(signs.first==3 && signs.second==1+layout.capacity(),"dummy inertia accounting incorrect");
    if(turn==0)first_ids={layout.row_for(0),layout.row_for(2)};
    require(layout.row_for(0)==first_ids[0] && layout.row_for(2)==first_ids[1],"row identities were reassigned");
    const auto current_graph=graph(entries);
    if(turn==1 || turn>=3) {
      require(current_graph==previous_graph,"classification changed persistent graph");
      require(layout.generation()==previous_generation,"classification invalidated generation");
    }
    previous_graph=current_graph;previous_generation=layout.generation();
  }
}
void bounded_capacity_and_support_union() {
  DcaStableBorderSlots layout;layout.configure(4,0,1000);
  layout.update({17,3,17});
  require(layout.assigned_count()==2 && layout.capacity()==67,"headroom must stay proportional to assigned borders");
  require(layout.row_for(3)==4 && layout.row_for(17)==5,"initial identity assignment is not canonical");
  std::vector<Gradient> rows(1000);rows[3]={{2,0},{.3,1}};rows[17]={{1},{.2}};
  Vector s(1000,1),z(1000,1);std::vector<SymmetricTriplet> entries;
  layout.append_borders(rows,s,z,entries);const auto generation=layout.generation();
  const auto original_graph=graph(entries);
  rows[3]={{0,2},{2,.5}};entries.clear();layout.append_borders(rows,s,z,entries);
  require(layout.generation()==generation && graph(entries)==original_graph,"support permutation changed envelope");
  auto matrix=dense(layout.dimension(),entries);
  close(matrix[0][layout.row_for(3)],2,"support permutation lost value");
  close(matrix[2][layout.row_for(3)],.5,"support permutation lost second value");
  rows[3]={{0,1},{2,.4}};entries.clear();layout.append_borders(rows,s,z,entries);
  require(layout.generation()>generation,"new support did not invalidate generation");
  matrix=dense(layout.dimension(),entries);
  close(matrix[1][layout.row_for(3)],.4,"support addition lost value");
  close(matrix[2][layout.row_for(3)],0,"removed support did not become structural zero");
  const auto expanded_graph=graph(entries);const auto expanded_generation=layout.generation();
  layout.update({});entries.clear();layout.append_borders(rows,s,z,entries);
  require(layout.generation()==expanded_generation && graph(entries)==expanded_graph,"de-promotion erased support");
  std::vector<int> requested(68);for(int i=0;i<68;++i)requested[i]=i;
  layout.update(requested);
  require(layout.row_for(3)==4 && layout.row_for(17)==5,"capacity growth reassigned row");
  require(layout.capacity()>=68 && layout.capacity()<1000,"capacity growth allocated every inequality");
  layout.configure(5,0,1000);require(layout.capacity()==0 && layout.row_for(3)==-1,"dimension change did not reset layout");
  bool rejected=false;try{layout.update({1000});}catch(const std::out_of_range&){rejected=true;}
  require(rejected,"invalid row was silently accepted");
}
void large_dense_support() {
  constexpr int n=10000;
  DcaStableBorderSlots layout;layout.configure(n,0,1);layout.update({0});
  std::vector<Gradient> rows(1);
  rows[0].support.reserve(n);rows[0].values.reserve(n);
  for(int i=n-1;i>=0;--i) {
    rows[0].support.push_back(i);rows[0].values.push_back(1.0+i);
  }
  const Vector s={2},z={3};std::vector<SymmetricTriplet> entries;
  layout.append_borders(rows,s,z,entries);
  const auto generation=layout.generation();
  require(entries.size()==n+1,"large support lost entries");
  for(int i=0;i<n;++i) {
    require(entries[i+1].i==n-1-i && entries[i+1].j==n,
        "large support first-seen emission order changed");
    close(entries[i+1].value,n-i,"large support initial value");
  }
  std::reverse(rows[0].support.begin(),rows[0].support.end());
  std::reverse(rows[0].values.begin(),rows[0].values.end());
  entries.clear();layout.append_borders(rows,s,z,entries);
  require(layout.generation()==generation,"large support permutation changed graph");
  for(int i=0;i<n;++i) {
    require(entries[i+1].i==n-1-i,"large support permutation changed emission order");
    close(entries[i+1].value,n-i,"large support permutation lost value");
  }
  rows[0].support.push_back(rows[0].support.front());rows[0].values.push_back(1);
  bool rejected=false;
  try{layout.append_borders(rows,s,z,entries);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"large support duplicate silently accepted");
}
} // namespace
int main() {
  try {
    solve_equivalence_and_dummy_inertia();bounded_capacity_and_support_union();large_dense_support();
    std::cout<<"stable border: exact Schur solves, row identities, dummy inertia, amortized envelope PASS\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
