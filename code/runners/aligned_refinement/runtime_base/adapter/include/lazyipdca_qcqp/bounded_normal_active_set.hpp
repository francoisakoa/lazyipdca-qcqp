#pragma once
#include "lazyipdca_qcqp/bounded_normal_restart.hpp"
#include <numeric>

extern "C" void dgelss_(const int*,const int*,const int*,double*,const int*,
                         double*,const int*,double*,const double*,int*,
                         double*,const int*,int*);

namespace lazyipdca::qcqp::detail {

// Small dense bounded least squares fallback for dependent original normals.
// Function convexification and the primal/filter state are not touched. A
// caller must separately audit every original KKT component before admitting
// these multipliers. The bounded coordinate method remains the large-system
// fast path; this SVD fallback is deliberately bounded in space and work.
inline BoundedNormalRestart bounded_normal_active_set(
    const Vec& gradient,const std::vector<BoundedNormalColumn>& columns,
    double budget_seconds,int max_iterations=2000,const Vec* initial=nullptr) {
  using L=long double;using Clock=std::chrono::steady_clock;
  const auto began=Clock::now();
  auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-began).count();};
  BoundedNormalRestart result;
  const std::size_t m=gradient.size(),n=columns.size();
  if(!(budget_seconds>0)||max_iterations<1||m==0||m>512||n>1536||
      m*n>1000000||m*n*std::min(m,n)>250000000||
      !std::all_of(gradient.begin(),gradient.end(),[](double x){return std::isfinite(x);})||
      (initial&&initial->size()!=n))return result;
  std::vector<double> matrix(m*n),scale(n,1),lower(n),upper(n),u(n);
  std::vector<int> state(n,0); // -1 lower, +1 upper, 0 free, 2 fixed.
  for(std::size_t j=0;j<n;++j) {
    const auto& c=columns[j];
    if(std::isnan(c.lower)||std::isnan(c.upper)||c.lower>c.upper||
       c.lower==std::numeric_limits<double>::infinity()||
       c.upper==-std::numeric_limits<double>::infinity())return result;
    std::vector<L> dense(m,0);
    for(auto [i,a]:c.entries) {
      if(i<0||std::size_t(i)>=m||!std::isfinite(a))return result;
      dense[i]+=a;
    }
    L max_entry=0;for(L a:dense)max_entry=std::max(max_entry,std::abs(a));
    if(max_entry>std::numeric_limits<double>::max())return result;
    scale[j]=max_entry>0?double(max_entry):1.;
    for(std::size_t i=0;i<m;++i)matrix[j*m+i]=double(dense[i]/scale[j]);
    auto scaled_bound=[&](double b) {
      const L v=L(b)*scale[j];
      return std::abs(v)>std::numeric_limits<double>::max()?
          std::copysign(std::numeric_limits<double>::infinity(),double(v)):double(v);
    };
    lower[j]=scaled_bound(c.lower);upper[j]=scaled_bound(c.upper);
    const double start=initial?(*initial)[j]:0.;
    if(!std::isfinite(start))return result;
    const L normalized=L(std::min(c.upper,std::max(c.lower,start)))*scale[j];
    if(!std::isfinite(normalized)||std::abs(normalized)>std::numeric_limits<double>::max())return result;
    u[j]=double(normalized);
    if(lower[j]==upper[j]||max_entry==0)state[j]=2;
    else if(u[j]<=lower[j]){u[j]=lower[j];state[j]=-1;}
    else if(u[j]>=upper[j]){u[j]=upper[j];state[j]=1;}
    if(elapsed()>=budget_seconds)return result;
  }
  std::vector<L> residual(m);
  auto reconstruct=[&] {
    residual.assign(gradient.begin(),gradient.end());
    for(std::size_t j=0;j<n;++j)if(u[j]!=0)
      for(std::size_t i=0;i<m;++i)residual[i]+=L(matrix[j*m+i])*u[j];
  };
  reconstruct();
  std::vector<double> best=u;
  auto squared=[&]{L s=0;for(L r:residual)s+=r*r;return s;};
  L best_cost=squared();
  // A rank-revealing solve is repeated only after the active set changes.
  // The cutoff preserves directions down to near binary64 precision rather
  // than declaring small flow/concentration derivatives identically zero.
  const double rcond=8*std::numeric_limits<double>::epsilon();
  for(;result.sweeps<max_iterations&&elapsed()<budget_seconds;++result.sweeps) {
    std::vector<int> free;
    for(std::size_t j=0;j<n;++j)if(state[j]==0)free.push_back(int(j));
    if(!free.empty()) {
      const int rows=int(m),cols=int(free.size()),nrhs=1,lda=rows;
      const int ldb=std::max(rows,cols);
      std::vector<double> a(m*free.size()),b(ldb,0),singular(std::min(rows,cols));
      for(std::size_t i=0;i<m;++i) {
        L rhs=-L(gradient[i]);
        for(std::size_t j=0;j<n;++j)if(state[j]!=0)rhs-=L(matrix[j*m+i])*u[j];
        b[i]=double(rhs);
      }
      for(std::size_t k=0;k<free.size();++k)
        std::copy_n(matrix.data()+std::size_t(free[k])*m,m,a.data()+k*m);
      int rank=0,info=0,lwork=-1;double query=0;
      dgelss_(&rows,&cols,&nrhs,a.data(),&lda,b.data(),&ldb,singular.data(),
              &rcond,&rank,&query,&lwork,&info);
      if(info||!std::isfinite(query)||query<1||query>16000000)break;
      lwork=int(std::ceil(query));std::vector<double> work(lwork);
      dgelss_(&rows,&cols,&nrhs,a.data(),&lda,b.data(),&ldb,singular.data(),
              &rcond,&rank,work.data(),&lwork,&info);
      if(info||!std::all_of(b.begin(),b.begin()+cols,[](double x){return std::isfinite(x);}))break;
      L alpha=1;int hit=-1,hit_side=0;
      for(std::size_t k=0;k<free.size();++k) {
        const int j=free[k];const L step=L(b[k])-u[j];
        L candidate=1;int side=0;
        if(step<0&&b[k]<lower[j]){candidate=(L(lower[j])-u[j])/step;side=-1;}
        if(step>0&&b[k]>upper[j]){candidate=(L(upper[j])-u[j])/step;side=1;}
        if(side&&candidate<=alpha){alpha=std::max(L(0),candidate);hit=j;hit_side=side;}
      }
      for(std::size_t k=0;k<free.size();++k) {
        const int j=free[k];
        const L next=L(u[j])+alpha*(L(b[k])-u[j]);
        u[j]=std::min(upper[j],std::max(lower[j],double(next)));
      }
      if(hit>=0) {
        state[hit]=hit_side;u[hit]=hit_side<0?lower[hit]:upper[hit];
        reconstruct();const L cost=squared();if(cost<best_cost){best_cost=cost;best=u;}
        continue;
      }
    }
    reconstruct();
    const L cost=squared();if(cost<best_cost){best_cost=cost;best=u;}
    L rnorm=0;for(L r:residual)rnorm=std::max(rnorm,std::abs(r));
    // A vanishing residual already solves the fit; continued release of
    // numerically indifferent normals only cycles between equivalent gauges.
    // This is an absolute fitting threshold, not a KKT acceptance tolerance.
    if(rnorm<=1e-12L){++result.sweeps;break;}
    const L tolerance=64*std::numeric_limits<double>::epsilon()*std::max(L(1),rnorm);
    int release=-1;L violation=tolerance;
    for(std::size_t j=0;j<n;++j)if(state[j]==-1||state[j]==1) {
      L product=0,norm=0;
      for(std::size_t i=0;i<m;++i){const L a=matrix[j*m+i];product+=a*residual[i];norm+=a*a;}
      const L v=(state[j]==-1?-product:product)/std::max(L(1),std::sqrt(norm));
      if(v>violation){violation=v;release=int(j);}
    }
    if(release<0){++result.sweeps;break;}
    state[release]=0;
  }
  result.multipliers.resize(n);
  for(std::size_t j=0;j<n;++j) {
    const double value=double(L(best[j])/scale[j]);
    if(!std::isfinite(value))return BoundedNormalRestart{};
    result.multipliers[j]=std::min(columns[j].upper,std::max(columns[j].lower,value));
  }
  residual.assign(gradient.begin(),gradient.end());
  for(std::size_t j=0;j<n;++j)for(auto [i,a]:columns[j].entries)
    residual[i]+=L(result.multipliers[j])*a;
  for(L r:residual) {
    if(!std::isfinite(r)||std::abs(r)>std::numeric_limits<double>::max())return BoundedNormalRestart{};
    result.residual.push_back(double(r));
  }
  result.seconds=elapsed();result.valid=true;return result;
}

} // namespace lazyipdca::qcqp::detail
