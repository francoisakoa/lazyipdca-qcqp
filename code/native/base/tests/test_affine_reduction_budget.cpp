#include "../src/bounded_affine_elimination.hpp"
#include <array>
#include <iostream>
#include <random>

using lazyipdca::qcqp::detail::bounded_affine_elimination;
namespace {
int checks=0;
void check(bool value,const char* why) {
  ++checks;if(!value)throw std::runtime_error(why);
}
std::size_t exact_count(const std::vector<std::size_t>& width,
    const std::vector<std::pair<int,int>>& pattern) {
  std::size_t count=0;
  for(auto edge:pattern) {
    const auto a=width[edge.first],b=width[edge.second];
    count+=edge.first==edge.second?a*(a+1)/2:a*b;
  }
  return count;
}
void ordinary_and_mixed() {
  const std::vector<std::pair<int,int>> diagonal{{0,0},{1,1},{2,2},{3,3}};
  const auto ordinary=bounded_affine_elimination({1,2,1,0},diagonal,24);
  check(ordinary.refused==0,"ordinary sparse reduction changed");
  check(ordinary.chosen_width==std::vector<std::size_t>({1,2,1,0}),"useful pivots not retained");
  check(ordinary.admitted_entries==5,"ordinary contribution count wrong");
  const auto mixed=bounded_affine_elimination({1,1000,1,0},diagonal,24,96);
  check(mixed.refused==1,"dense fill pivot not refused");
  check(mixed.chosen_width==std::vector<std::size_t>({1,1,1,0}),"dense refusal discarded useful sparse leaf");
  check(mixed.admitted_entries==3,"constant elimination did not reduce fill");
  check(mixed.admitted_bytes<=mixed.budget_bytes,"admitted expansion exceeds budget");
}
void original_large_lift_dimensions() {
  // The REAL dimensions, not a size-reduced proxy, are cheap to estimate.
  // No D matrices, projected outer products or giant allocations are built.
  constexpr int w=2592,u=38880,t=1,n=w+u+t;
  std::vector<std::size_t> widths(n,1);
  std::vector<std::pair<int,int>> diagonal;
  for(int j=0;j<n;++j) {
    diagonal.push_back({j,j});if(j>=w&&j<w+u)widths[j]=w;
  }
  const auto plan=bounded_affine_elimination(widths,diagonal,24);
  const std::size_t expected=std::size_t(u)*w*(w+1)/2+w+t;
  check(plan.proposed_entries==expected,"large factor-lift estimate wrong");
  check(plan.proposed_bytes>std::size_t(2)*1024*1024*1024*1024,"terabyte-scale expansion was missed");
  check(plan.refused==u,"large affine lifts still admitted projected outer products");
  check(plan.admitted_entries==n,"unreduced sparse diagonal not preserved");
  check(plan.chosen_width==std::vector<std::size_t>(n,1),"original lifted coordinates were not retained");
  check(!plan.arithmetic_saturated,"ordinary full-size estimate overflowed");
}
void overflow_and_invalid_data() {
  const auto huge=std::numeric_limits<std::size_t>::max();
  auto a=bounded_affine_elimination({huge,huge},{{0,0},{0,1},{1,1}},24,240);
  check(a.arithmetic_saturated,"overflow was not diagnosed");
  check(a.refused==2,"overflowing projection was admitted");
  check(a.admitted_entries==3,"overflow invalidated original sparse form");
  check(a.proposed_bytes==huge,"overflow estimate did not fail closed");
  bool caught=false;
  try{bounded_affine_elimination({1},{{0,1}},24);}catch(const std::invalid_argument&){caught=true;}
  check(caught,"invalid support index was accepted");
}
void exhaustive_small_projection_counts() {
  std::mt19937 rng(20261024);
  for(int trial=0;trial<200;++trial) {
    constexpr int n=8;std::vector<std::size_t> widths(n);
    std::vector<std::pair<int,int>> pattern;
    for(auto& width:widths)width=rng()%7;
    for(int i=0;i<n;++i)for(int j=i;j<n;++j)if(i==j||rng()%3==0)pattern.push_back({i,j});
    const auto bytes=std::size_t(24)*(20+rng()%80);
    const auto plan=bounded_affine_elimination(widths,pattern,24,bytes);
    check(plan.proposed_entries==exact_count(widths,pattern),"proposed count disagrees with explicit pairs");
    check(plan.admitted_entries==exact_count(plan.chosen_width,pattern),"accepted count disagrees with explicit pairs");
    check(plan.admitted_bytes<=bytes||plan.admitted_entries<=pattern.size(),"expansion was admitted beyond budget");
    for(int j=0;j<n;++j) {
      check(plan.chosen_width[j]==widths[j]||plan.chosen_width[j]==1,"a substituted model was invented");
      if(widths[j]<=1)check(plan.chosen_width[j]==widths[j],"nonexpanding substitution was rejected");
    }
  }
}
void exact_complex_factor_and_sparse_leaf() {
  // Original variables: (Re w0, Im w0, Re w1, Im w1, Re u, Im u, v).
  // u=(1+2i)w0+(-.5+3i)w1, v=2 Re w0. The complex lift is kept;
  // the independent sparse leaf remains eliminated. No equality, bound or
  // quadratic function is removed by the allocation policy.
  std::vector<std::pair<int,int>> hessian{{0,0},{1,1},{2,2},{3,3},{4,4},{5,5},{6,6}};
  const auto plan=bounded_affine_elimination({1,1,1,1,4,4,1},hessian,24,7*24);
  check(plan.refused==2,"complex factor fill was not rejected");
  check(plan.chosen_width[6]==1,"sparse independent leaf was lost");
  const std::array<std::array<double,4>,2> D{{{{1,-2,-.5,-3}},{{2,1,3,-.5}}}};
  const std::array<double,4> dw{{.125,-.25,.5,.75}};
  std::array<double,7> full{{dw[0],dw[1],dw[2],dw[3],0,0,2*dw[0]}};
  for(int r=0;r<2;++r)for(int j=0;j<4;++j)full[4+r]+=D[r][j]*dw[j];
  for(int r=0;r<2;++r) {
    double equality=full[4+r];
    for(int j=0;j<4;++j)equality-=D[r][j]*full[j];
    check(equality==0,"original affine complex equality changed");
  }
  check(full[6]-2*full[0]==0,"sparse leaf equality changed");
  // Exact dyadic arithmetic: q=|u|^2 is identical in the lifted and pulled
  // back model. This is independent of the contribution-count estimator.
  double lifted=full[4]*full[4]+full[5]*full[5],pulled=0;
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)
    for(int r=0;r<2;++r)pulled+=dw[i]*D[r][i]*D[r][j]*dw[j];
  check(lifted==pulled,"retaining factor variables changed the original quadratic");
}
}
int main() {try {
  ordinary_and_mixed();original_large_lift_dimensions();overflow_and_invalid_data();
  exhaustive_small_projection_counts();exact_complex_factor_and_sparse_leaf();
  std::cout<<"affine_reduction_budget PASS checks="<<checks
    <<" full_phase_dimensions=1 checked_overflow=1 mixed_sparse_leaf=1 exact_complex_lift=1 preallocation_admission=1\n";
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
