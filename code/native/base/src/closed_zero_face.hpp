#pragma once
// Exact lower-zero face proof; projection is a declared native feasibility phase.
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <map>
#include <set>
#include <limits>
namespace lazyipdca::qcqp::detail {
class ClosedZeroFace {
 public:
  using Exact=boost::multiprecision::cpp_rational;
  struct Row {int original;std::vector<std::pair<int,Exact>> terms;};
  const QCQPProblem& original;QCQPProblem model;
  std::vector<Row> proofs;
  std::vector<int> introduced,zero_coordinates,kept_equalities,kept_inequalities;
  std::vector<unsigned char> zero,introduced_mask;
  std::string reason="no-positive-affine-zero-face";
  double projection_inf=0,projection_l1=0;
  std::size_t inspected_terms=0;
  static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
  static double upward(const Exact& x){
    double d=x.convert_to<double>();
    if(!std::isfinite(d))throw std::runtime_error("closed zero face: nonfinite lifted multiplier");
    if(Exact(d)<x)d=std::nextafter(d,INFINITY);
    if(!std::isfinite(d)||Exact(d)<x)throw std::runtime_error("closed zero face: upward conversion failed");
    return d;
  }
  bool projected_constant(const QuadraticFunction& q)const{
    // Exact zero substitution removes terms touching the proven face. Exact
    // combination of repeated coefficients prevents cancellation guesses.
    if(!q.signed_square.empty())return false;
    std::map<int,Exact> l;
    q.for_each_linear_nonzero([&](int j,double v){if(!zero[j])l[j]+=Exact(v);});
    for(const auto& x:l)if(x.second!=0)return false;
    std::map<std::pair<int,int>,Exact> h;
    for(const auto& e:q.hessian.entries)if(!zero[e.i]&&!zero[e.j])h[std::minmax(e.i,e.j)]+=Exact(e.v);
    for(const auto& x:h)if(x.second!=0)return false;
    return true;
  }
  bool box_implied(const QuadraticFunction& q)const{
    if(!q.signed_square.empty()||!std::isfinite(q.constant))return false;
    std::map<std::pair<int,int>,Exact> h;
    for(const auto& e:q.hessian.entries)if(!zero[e.i]&&!zero[e.j])h[std::minmax(e.i,e.j)]+=Exact(e.v);
    for(const auto& e:h)if(e.second!=0)return false;
    std::map<int,Exact> l;
    q.for_each_linear_nonzero([&](int j,double a){if(!zero[j])l[j]+=Exact(a);});
    Exact maximum(q.constant);
    for(const auto& t:l)if(t.second!=0){
      const double b=t.second>0?original.ub[t.first]:original.lb[t.first];
      if(!std::isfinite(b))return false;
      maximum+=t.second*Exact(b);
    }
    return maximum<=0;
  }
  explicit ClosedZeroFace(const QCQPProblem& p,double deadline=INFINITY):original(p){
    p.validate();
    if(p.has_signed_square()){reason="signed-square-model-refused";return;}
    if(!(deadline>now())){reason="deadline";return;}
    std::size_t terms=std::size_t(p.n)*5,bytes=std::size_t(p.n)*5*sizeof(double);
    auto count=[&](const QuadraticFunction& q){
      const std::size_t t=q.linear.size()+q.compact_linear.size()+q.hessian.entries.size()+q.signed_square.size();
      if(t>2000000||terms>2000000-t)return false;terms+=t;
      const std::size_t b=sizeof(q)+q.linear.size()*sizeof(double)+q.compact_linear.size()*sizeof(std::pair<int,double>)+q.hessian.entries.size()*sizeof(SymTriplet)+q.signed_square.size()*sizeof(std::pair<int,double>);
      if(b>134217728||bytes>134217728-b)return false;bytes+=b;
      return now()<deadline;
    };
    if(!count(p.objective)){reason="source-budget";return;}
    for(const auto* family:{&p.equalities,&p.inequalities})for(const auto& q:*family)if(!count(q)){reason="source-budget";return;}
    zero.assign(p.n,0);introduced_mask.assign(p.n,0);
    for(int j=0;j<p.n;++j)if(p.lb[j]==0&&p.ub[j]==0)zero[j]=1;
    for(int i=0;i<int(p.inequalities.size());++i){
      if(now()>=deadline){reason="deadline";proofs.clear();return;}
      const auto& q=p.inequalities[i];if(q.constant!=0||!q.signed_square.empty())continue;
      bool affine=true;for(const auto& h:q.hessian.entries)if(h.v!=0)affine=false;
      if(!affine)continue;
      std::map<int,Exact> combined;
      q.for_each_linear_nonzero([&](int j,double a){++inspected_terms;combined[j]+=Exact(a);});
      Row row{i,{}};bool supported=true,has_new=false;
      for(const auto& term:combined)if(term.second!=0){
        if(!(term.second>0)||p.lb[term.first]!=0||!(p.ub[term.first]>=0)){supported=false;break;}
        row.terms.push_back(term);has_new=has_new||p.ub[term.first]!=0;
      }
      if(!supported||!has_new||row.terms.empty())continue;
      proofs.push_back(std::move(row));
    }
    for(const auto& row:proofs)for(const auto& term:row.terms){
      const int j=term.first;zero[j]=1;
      if(p.ub[j]!=0)introduced_mask[j]=1;
    }
    for(int j=0;j<p.n;++j){if(zero[j])zero_coordinates.push_back(j);if(introduced_mask[j])introduced.push_back(j);}
    if(introduced.empty())return;
    if(now()>=deadline){reason="deadline";proofs.clear();introduced.clear();return;}
    model=p;
    for(int j:zero_coordinates){
      projection_inf=std::max(projection_inf,std::abs(p.x0[j]));projection_l1+=std::abs(p.x0[j]);model.x0[j]=0;
    }
    if(!std::isfinite(projection_inf)||!std::isfinite(projection_l1)){reason="nonfinite-projection-diagnostic";model=QCQPProblem{};introduced.clear();proofs.clear();return;}
    for(int j:introduced)model.ub[j]=0;
    model.equalities.clear();model.inequalities.clear();
    for(int i=0;i<int(p.equalities.size());++i){
      const auto& q=p.equalities[i];
      if(q.constant==0&&projected_constant(q))continue;
      kept_equalities.push_back(i);model.equalities.push_back(q);
    }
    for(int i=0;i<int(p.inequalities.size());++i){
      const auto& q=p.inequalities[i];
      if(box_implied(q))continue;
      kept_inequalities.push_back(i);model.inequalities.push_back(q);
    }
    if(now()>=deadline){reason="deadline";model=QCQPProblem{};introduced.clear();proofs.clear();return;}
    model.validate();reason="exact-positive-affine-IQ-and-original-lower-zero-bounds";
  }
  bool applied()const{return !introduced.empty()&&!model.x0.empty();}
  bool lift(Result& r)const{
    if(!applied()||r.x.size()!=std::size_t(original.n)||r.certificate_equality_multipliers.size()!=kept_equalities.size()||r.certificate_inequality_multipliers.size()!=kept_inequalities.size()||r.certificate_lower_multipliers.size()!=std::size_t(original.n)||r.certificate_upper_multipliers.size()!=std::size_t(original.n))return false;
    try{
      for(double x:r.x)if(!std::isfinite(x))return false;
      for(int j:zero_coordinates)if(r.x[j]!=0)return false;
      auto lower=r.certificate_lower_multipliers,upper=r.certificate_upper_multipliers;
      for(double x:lower)if(!std::isfinite(x)||x<0)return false;
      for(double x:upper)if(!std::isfinite(x)||x<0)return false;
      Vec eq(original.equalities.size(),0),iq(original.inequalities.size(),0);
      for(std::size_t k=0;k<kept_equalities.size();++k)eq[kept_equalities[k]]=r.certificate_equality_multipliers[k];
      for(std::size_t k=0;k<kept_inequalities.size();++k)iq[kept_inequalities[k]]=r.certificate_inequality_multipliers[k];
      for(double x:eq)if(!std::isfinite(x))return false;
      for(double x:iq)if(!std::isfinite(x)||x<0)return false;
      std::vector<Exact> needed(proofs.size(),0),increments(proofs.size(),0),added(original.n,0);
      for(int j:introduced){
        if(!std::isfinite(lower[j])||lower[j]<0||!std::isfinite(upper[j])||upper[j]<0)return false;
        bool assigned=false;
        for(std::size_t i=0;i<proofs.size()&&!assigned;++i)for(const auto& t:proofs[i].terms)if(t.first==j){
          const Exact q=Exact(upper[j])/t.second;needed[i]=std::max(needed[i],q);assigned=true;break;
        }
        if(!assigned)return false;
      }
      for(std::size_t i=0;i<proofs.size();++i){
        const int row=proofs[i].original;const double old=iq[row];
        iq[row]=upward(Exact(old)+needed[i]);increments[i]=Exact(iq[row])-Exact(old);
        for(const auto& t:proofs[i].terms)added[t.first]+=increments[i]*t.second;
      }
      for(int j:zero_coordinates){
        if(!std::isfinite(lower[j])||lower[j]<0||!std::isfinite(upper[j])||upper[j]<0)return false;
        Exact v=Exact(lower[j])+added[j];if(introduced_mask[j]){v-=Exact(upper[j]);upper[j]=0;}
        if(v<0)return false;lower[j]=v.convert_to<double>();if(!std::isfinite(lower[j]))return false;
      }
      r.certificate_equality_multipliers=std::move(eq);r.certificate_inequality_multipliers=std::move(iq);
      r.certificate_lower_multipliers=std::move(lower);r.certificate_upper_multipliers=std::move(upper);
      r.termination_scaling_available=false;r.termination_equality_scales.clear();r.termination_inequality_scales.clear();r.termination_inequality_slacks.clear();
      return true;
    }catch(const std::exception&){return false;}
  }
};
}
