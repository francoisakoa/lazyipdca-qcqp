#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <map>
#include <chrono>
namespace lazyipdca::qcqp::detail {
// On the retained affine face h=c+a(t+u)=0, a pair d(t²-u²)
// equals -d*c/a*(t-u). The exact identity is q=q_reduced+f*h,
// f=d/a*(t-u). No coordinate or x0 is changed. The dual lift is
// y_original=y_reduced-z*f; the complete original audit remains required.
struct AffineFaceQuadraticRewrite {
  struct Link{int inequality,equality,t,u;double scale;};
  QCQPProblem model;std::vector<Link> links;
  AffineFaceQuadraticRewrite(const QCQPProblem& p,double deadline) {
    using E=boost::multiprecision::cpp_rational;
    auto expired=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=deadline;};
    if(p.has_signed_square()||p.equalities.empty()||p.inequalities.empty())return;
    // Admission precedes the first private-model copy. Proof-work limits by
    // themselves do not bound the cost of copying as-yet-unscanned rows.
    // Count all represented source storage conservatively, with saturated
    // bounds and deadline checks; uncertain/large models keep the old path.
    constexpr std::size_t max_elements=2000000,max_bytes=128u*1024u*1024u;
    std::size_t admitted_elements=0,admitted_bytes=0;
    auto admit=[&](std::size_t count,std::size_t width) {
      if(expired()||count>max_elements-admitted_elements||
          (width!=0&&count>(max_bytes-admitted_bytes)/width))return false;
      admitted_elements+=count;admitted_bytes+=count*width;return true;
    };
    if(!admit(p.x0.size(),sizeof(double))||!admit(p.lb.size(),sizeof(double))||
        !admit(p.ub.size(),sizeof(double))||!admit(p.name.size(),sizeof(char)))return;
    auto admit_row=[&](const QuadraticFunction& q) {
      return admit(1,sizeof(QuadraticFunction))&&admit(q.name.size(),sizeof(char))&&
          admit(q.linear.size(),sizeof(double))&&
          admit(q.compact_linear.size(),sizeof(std::pair<int,double>))&&
          admit(q.hessian.entries.size(),sizeof(SymTriplet))&&
          admit(q.signed_square.size(),sizeof(std::pair<int,double>));
    };
    if(!admit_row(p.objective))return;
    for(const auto& q:p.equalities)if(!admit_row(q))return;
    for(const auto& q:p.inequalities)if(!admit_row(q))return;
    struct Face{int e,t,u;double a,c;};std::vector<Face> faces;std::size_t proof_work=0;
    auto work_available=[&](std::size_t next) {
      if(next>2000000-proof_work)return false;proof_work+=next;
      return !expired();
    };
    for(int e=0;e<int(p.equalities.size());++e){
      const auto& h=p.equalities[e];
      if(!work_available(1+h.hessian.entries.size()+h.linear.size()+h.compact_linear.size()))return;
      if(!h.signed_square.empty()||std::any_of(h.hessian.entries.begin(),h.hessian.entries.end(),[](const auto& v){return v.v!=0;}))continue;
      std::vector<std::pair<int,double>> entries;h.for_each_linear_nonzero([&](int j,double a){if(a!=0)entries.push_back({j,a});});
      if(entries.size()!=2||entries[0].first==entries[1].first||entries[0].second!=entries[1].second)continue;
      const int t=entries[0].first,u=entries[1].first;const double a=entries[0].second;
      if(E(h.constant)+E(a)*(E(p.x0[t])+E(p.x0[u]))!=0)continue;
      faces.push_back({e,t,u,a,h.constant});
    }
    if(faces.empty())return;
    for(int i=0;i<int(p.inequalities.size());++i){
      if(expired())return;
      const auto& q=p.inequalities[i];if(!q.signed_square.empty())continue;
      if(!work_available(1+q.hessian.entries.size()+q.linear.size()+q.compact_linear.size()))return;
      std::map<std::pair<int,int>,E> H;for(auto h:q.hessian.entries)H[{h.i,h.j}]+=E(h.v);
      for(const auto& face:faces){
        if(!work_available(1))return;
        const E ht=H[{face.t,face.t}],hu=H[{face.u,face.u}];if(ht==0||ht!=-hu)continue;
        bool cross=false;for(const auto& h:H)if(h.second!=0&&h.first.first!=h.first.second&&
            (h.first.first==face.t||h.first.second==face.t||h.first.first==face.u||h.first.second==face.u))cross=true;
        if(cross)continue;
        const E d=ht/2,scale=d/E(face.a),b=-scale*E(face.c);
        const double ss=scale.convert_to<double>();if(!std::isfinite(ss)||E(ss)!=scale)continue;
        std::map<int,E> linear;q.for_each_linear_nonzero([&](int j,double a){linear[j]+=E(a);});
        linear[face.t]+=b;linear[face.u]-=b;
        bool exact=true;std::vector<std::pair<int,double>> represented;
        for(const auto& v:linear){const double a=v.second.convert_to<double>();if(!std::isfinite(a)||E(a)!=v.second){exact=false;break;}if(a!=0)represented.push_back({v.first,a});}
        if(!exact)continue;
        if(links.empty())model=p;
        auto& reduced=model.inequalities[i];reduced.linear.clear();reduced.compact_linear_dimension=p.n;reduced.compact_linear=std::move(represented);
        reduced.hessian.entries.erase(std::remove_if(reduced.hessian.entries.begin(),reduced.hessian.entries.end(),[&](auto h){return h.i==h.j&&(h.i==face.t||h.i==face.u);}),reduced.hessian.entries.end());
        links.push_back({i,face.e,face.t,face.u,ss});break;
      }
    }
    if(!links.empty())model.validate();
  }
  bool applied() const{return !links.empty();}
  bool lift(Result& r) const {
    if(r.x.size()!=std::size_t(model.n)||r.certificate_equality_multipliers.size()!=model.equalities.size()||r.certificate_inequality_multipliers.size()!=model.inequalities.size())return false;
    Vec lifted=r.certificate_equality_multipliers;
    for(const auto& l:links){
      const long double f=static_cast<long double>(l.scale)*(static_cast<long double>(r.x[l.t])-r.x[l.u]);
      const long double y=static_cast<long double>(lifted[l.equality])-static_cast<long double>(r.certificate_inequality_multipliers[l.inequality])*f;
      if(!std::isfinite(y)||!std::isfinite(double(y)))return false;lifted[l.equality]=double(y);
    }
    r.certificate_equality_multipliers=std::move(lifted);
    r.termination_scaling_available=false;r.termination_audit_valid=false;return true;
  }
};
} // namespace
