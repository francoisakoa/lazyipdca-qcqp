#pragma once
// Candidate-only homogeneous affine facial reduction. The floating point
// search cannot authorize a reduction: every returned face is checked using
// exact rational arithmetic on the stored binary64 coefficients.
#include "lazyipdca_qcqp/model.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <vector>

namespace lazyipdca::qcqp::detail {
using AffineFaceExact=boost::multiprecision::cpp_rational;
struct AffineFaceCertificate {
  std::vector<int> rows;
  std::vector<AffineFaceExact> weights;
  std::map<int,AffineFaceExact> coefficients;
  int candidate_iterations=0;
  std::int64_t rational_grid=0;
};
struct AffineFaceDiscovery {
  std::vector<AffineFaceCertificate> certificates;
  std::vector<int> forced_zero;
  int homogeneous_rows=0,support_components=0,candidate_updates=0;
  int exact_checks=0,rejected_exact_candidates=0;
};
inline AffineFaceDiscovery discover_homogeneous_affine_faces(
    const QCQPProblem& p,int iteration_budget=16384,
    const std::function<bool()>& expired={}) {
  using Exact=AffineFaceExact;
  struct Row {int original;std::map<int,Exact> exact;};
  AffineFaceDiscovery out;std::vector<Row> rows;
  std::vector<int> first(p.n,-1),parent;
  auto root=[&](int k){while(parent[k]!=k){parent[k]=parent[parent[k]];k=parent[k];}return k;};
  for(int i=0;i<int(p.inequalities.size());++i) {
    if(expired&&expired())return out;
    const auto& q=p.inequalities[i];
    if(q.constant!=0||!q.signed_square.empty()||
       std::any_of(q.hessian.entries.begin(),q.hessian.entries.end(),[](auto e){return e.v!=0;}))continue;
    Row row;row.original=i;q.for_each_linear_nonzero([&](int j,double a){if(a)row.exact[j]+=Exact(a);});
    for(auto k=row.exact.begin();k!=row.exact.end();)if(k->second==0)k=row.exact.erase(k);else ++k;
    // The proof relies on these exact, supplied lower bound sides.
    if(row.exact.empty()||std::any_of(row.exact.begin(),row.exact.end(),[&](const auto& e){return p.lb[e.first]!=0;}))continue;
    const int r=rows.size();parent.push_back(r);rows.push_back(std::move(row));
    for(const auto& e:rows.back().exact) {
      const int j=e.first;
      if(first[j]<0)first[j]=r;else {int a=root(r),b=root(first[j]);if(a!=b)parent[a]=b;}
    }
  }
  out.homogeneous_rows=rows.size();std::map<int,std::vector<int>> groups;
  for(int r=0;r<int(rows.size());++r)groups[root(r)].push_back(r);
  out.support_components=groups.size();std::set<int> forced;
  for(const auto& group:groups) {
    if(expired&&expired())break;
    const auto& ids=group.second;const int m=ids.size();
    std::set<int> sup;for(int r:ids)for(const auto& e:rows[r].exact)sup.insert(e.first);
    const std::vector<int> support(sup.begin(),sup.end());const int n=support.size();
    if(m==0||n==0||m>64||n>4096||static_cast<std::uint64_t>(m)*n>262144)continue;
    std::vector<std::vector<double>> a(m,std::vector<double>(n,0));
    std::vector<double> scale(m,0),logits(m,0),weights(m),score(n),original_weights(m);
    bool finite=true;
    for(int i=0;i<m;++i)for(const auto& e:rows[ids[i]].exact) {
      const double value=e.second.convert_to<double>();finite=finite&&std::isfinite(value);
      scale[i]=std::max(scale[i],std::abs(value));
      a[i][std::lower_bound(support.begin(),support.end(),e.first)-support.begin()]=value;
    }
    for(int i=0;i<m;++i) {
      finite=finite&&scale[i]>0&&std::isfinite(scale[i]);
      if(scale[i]>0)for(double& v:a[i])v/=scale[i];
    }
    if(!finite)continue;
    bool found=false;
    auto certify=[&](const Vec& candidate,int iteration) {
        double maximum=0;for(int i=0;i<m;++i){original_weights[i]=candidate[i]/scale[i];maximum=std::max(maximum,original_weights[i]);}
        if(!(maximum>0)||!std::isfinite(maximum))return false;
        for(auto grid:{10LL,100LL,1000LL,10000LL,1000000LL,100000000LL}) {
          ++out.exact_checks;AffineFaceCertificate proof;proof.candidate_iterations=iteration;proof.rational_grid=grid;
          for(int i=0;i<m;++i) {
            const auto numerator=static_cast<std::int64_t>(std::llround(grid*original_weights[i]/maximum));
            if(numerator<=0)continue;
            const Exact weight=Exact(numerator)/Exact(grid);
            proof.rows.push_back(rows[ids[i]].original);proof.weights.push_back(weight);
            for(const auto& e:rows[ids[i]].exact)proof.coefficients[e.first]+=weight*e.second;
          }
          bool positive=false,valid=!proof.rows.empty();
          for(const auto& e:proof.coefficients){valid=valid&&e.second>=0;positive=positive||e.second>0;}
          if(valid&&positive) {
            // The full sum is now exact: constants and every off-face
            // coefficient are zero, all remaining coefficients nonnegative.
            for(const auto& e:proof.coefficients)if(e.second>0)forced.insert(e.first);
            out.certificates.push_back(std::move(proof));return true;
          }
          ++out.rejected_exact_candidates;
        }
      return false;
    };
    for(int it=0;it<=(m==1?0:iteration_budget);++it) {
      if(expired&&expired())break;
      const double largest=*std::max_element(logits.begin(),logits.end());double sum=0;
      for(int i=0;i<m;++i){weights[i]=std::exp(logits[i]-largest);sum+=weights[i];}
      if(!(sum>0)||!std::isfinite(sum))break;
      std::fill(score.begin(),score.end(),0);
      for(int i=0;i<m;++i){weights[i]/=sum;for(int j=0;j<n;++j)score[j]+=weights[i]*a[i][j];}
      const int worst=std::min_element(score.begin(),score.end())-score.begin();
      if(score[worst]>1e-8)found=certify(weights,it);
      if(found)break;
      for(int i=0;i<m;++i)logits[i]+=.5*a[i][worst];
      ++out.candidate_updates;
    }
    // A narrow exposed face may need much more accurate separating weights
    // than the inexpensive entropic proposal provides. A bounded accelerated
    // projection of the coefficient deficits supplies a second candidate.
    // This is finite-vector arithmetic only; exact verification above remains
    // the sole authority. Repeated identical columns carry no new sign test.
    if(!found&&m>1&&m<=64) {
      std::set<Vec> unique;
      for(int j=0;j<n;++j){Vec column(m);for(int i=0;i<m;++i)column[i]=a[i][j];unique.insert(std::move(column));}
      if(unique.size()<=256) {
        const std::vector<Vec> columns(unique.begin(),unique.end());
        double lipschitz=0;for(const auto& column:columns)for(double v:column)lipschitz+=v*v;
        if(lipschitz>0&&std::isfinite(lipschitz)) {
          Vec current(m,1),extrapolated=current,gradient(m),next(m);
          double acceleration=1;
          const int fallback_budget=std::min(16384,std::max(0,iteration_budget))*8;
          for(int it=0;it<=fallback_budget;++it) {
            if(expired&&expired())break;
            double minimum=std::numeric_limits<double>::infinity();
            for(const auto& column:columns){double v=0;for(int i=0;i<m;++i)v+=current[i]*column[i];minimum=std::min(minimum,v);}
            if(minimum>1e-9&&certify(current,iteration_budget+it+1)){found=true;break;}
            std::fill(gradient.begin(),gradient.end(),0);
            for(const auto& column:columns) {
              double v=-1;for(int i=0;i<m;++i)v+=extrapolated[i]*column[i];
              if(v<0)for(int i=0;i<m;++i)gradient[i]+=column[i]*v;
            }
            const double next_acceleration=.5*(1+std::sqrt(1+4*acceleration*acceleration));
            const double momentum=(acceleration-1)/next_acceleration;
            bool finite_step=true;
            for(int i=0;i<m;++i){next[i]=std::max(0.0,extrapolated[i]-gradient[i]/lipschitz);finite_step=finite_step&&std::isfinite(next[i]);}
            if(!finite_step)break;
            for(int i=0;i<m;++i)extrapolated[i]=next[i]+momentum*(next[i]-current[i]);
            current.swap(next);acceleration=next_acceleration;++out.candidate_updates;
          }
        }
      }
    }
  }
  out.forced_zero.assign(forced.begin(),forced.end());return out;
}
} // namespace lazyipdca::qcqp::detail
