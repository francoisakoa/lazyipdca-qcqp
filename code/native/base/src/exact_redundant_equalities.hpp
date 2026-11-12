#pragma once
// Standalone experimental, coefficient-exact removal of redundant equalities.
// No variable substitution, projection, approximate rank decision, or optimizer.
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <map>
#include <set>
#include <ostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <limits>

namespace lazyipdca::qcqp::detail {
class ExactRedundantEqualities {
 public:
  using Exact=boost::multiprecision::cpp_rational;
  using Monomial=std::vector<int>;
  using Polynomial=std::map<Monomial,Exact>;
  struct Definition {int row,variable,a,b;double pivot;};
  struct Simplex {int row;std::vector<int> variables;double pivot;};
  struct Witness {
    int row=-1,simplex_row=-1,y=-1;
    std::vector<int> definition_rows,product_variables,factor_variables;
    std::vector<Exact> definition_weights;
    Exact simplex_weight=0,absolute_definition_weight_sum=0;
    Exact y_bound_with_original_tolerance=0,residual_amplification_bound=0;
    bool exact_polynomial_zero=false;
  };
  const QCQPProblem& original;
  QCQPProblem reduced;
  std::vector<int> kept_equalities;
  std::vector<Witness> witnesses;
  std::vector<Definition> definitions;
  std::vector<Simplex> simplexes;
  double original_tolerance=1e-6,inner_feasibility_tolerance=1e-6;
  Exact maximum_residual_amplification=1;
  std::string reason="no-proven-bounded-redundant-equality";
  struct DeadlineExceeded {};
  double deadline=std::numeric_limits<double>::infinity();
  void check_deadline() const {
    if(std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=deadline)throw DeadlineExceeded{};
  }

  static Exact absolute(const Exact& x){return x<0?-x:x;}
  static Polynomial coefficients(const QuadraticFunction& q) {
    if(!q.signed_square.empty())throw std::runtime_error("exact EQ proof: signed-square unsupported");
    Polynomial p;
    if(!std::isfinite(q.constant))throw std::runtime_error("exact EQ proof: nonfinite constant");
    p[{}]+=Exact(q.constant);
    q.for_each_linear_nonzero([&](int j,double a){
      if(!std::isfinite(a))throw std::runtime_error("exact EQ proof: nonfinite linear coefficient");
      p[{j}]+=Exact(a);
    });
    for(auto h:q.hessian.entries){
      if(!std::isfinite(h.v))throw std::runtime_error("exact EQ proof: nonfinite Hessian");
      const auto ij=std::minmax(h.i,h.j);
      p[{ij.first,ij.second}]+=Exact(h.v)/(h.i==h.j?2:1);
    }
    return p;
  }
  static void add(Polynomial& dst,const Polynomial& src,const Exact& weight,int multiply_variable=-1){
    for(const auto& term:src){
      Monomial key=term.first;
      if(multiply_variable>=0){key.push_back(multiply_variable);std::sort(key.begin(),key.end());}
      dst[key]+=weight*term.second;
    }
  }
  static bool zero(const Polynomial& p){for(const auto& t:p)if(t.second!=0)return false;return true;}
  bool verify(const Witness& w)const {
    if(w.row<0||w.simplex_row<0||w.y<0||w.y>=original.n||
       w.row>=int(original.equalities.size())||w.simplex_row>=int(original.equalities.size())||
       w.definition_rows.size()!=w.definition_weights.size())return false;
    check_deadline();
    Polynomial residual=coefficients(original.equalities[w.row]);
    for(std::size_t k=0;k<w.definition_rows.size();++k){
      check_deadline();
      const int row=w.definition_rows[k];
      if(row<0||row>=int(original.equalities.size()))return false;
      add(residual,coefficients(original.equalities[row]),-w.definition_weights[k]);
    }
    add(residual,coefficients(original.equalities[w.simplex_row]),-w.simplex_weight,w.y);
    return zero(residual);
  }
  explicit ExactRedundantEqualities(const QCQPProblem& p,double tolerance=1e-6,
      double proof_deadline=std::numeric_limits<double>::infinity())
    :original(p),reduced(p),original_tolerance(tolerance),inner_feasibility_tolerance(tolerance),deadline(proof_deadline) {
    try {
    check_deadline();
    p.validate();
    if(!(std::isfinite(tolerance)&&tolerance>0))throw std::runtime_error("exact EQ proof: invalid original tolerance");
    for(double v:p.x0)if(!std::isfinite(v))throw std::runtime_error("exact EQ proof: nonfinite start");
    if(p.has_signed_square()){reason="signed-square-unsupported";for(int i=0;i<int(p.equalities.size());++i)kept_equalities.push_back(i);return;}
    std::map<int,int> by_product;
    std::set<int> ambiguous;
    std::map<std::vector<int>,std::vector<int>> by_simplex;
    for(int row=0;row<int(p.equalities.size());++row){
      check_deadline();
      const auto& q=p.equalities[row];
      std::vector<std::pair<int,double>> linear;
      q.for_each_linear_nonzero([&](int j,double a){if(a)linear.push_back({j,a});});
      std::vector<SymTriplet> quadratic;
      for(auto h:q.hessian.entries)if(h.v)quadratic.push_back(h);
      if(q.constant==0&&linear.size()==1&&quadratic.size()==1){
        const auto l=linear[0];const auto h=quadratic[0];
        if(std::isfinite(l.second)&&l.second!=0&&h.i!=h.j&&l.first!=h.i&&l.first!=h.j&&Exact(h.v)==-Exact(l.second)){
          const int index=int(definitions.size());definitions.push_back({row,l.first,h.i,h.j,l.second});
          if(by_product.count(l.first))ambiguous.insert(l.first);else by_product[l.first]=index;
        }
      }
      if(quadratic.empty()&&q.constant!=0&&!linear.empty()&&std::isfinite(q.constant)){
        const double pivot=-q.constant;
        bool uniform=true;std::vector<int> variables;
        for(auto l:linear){uniform=uniform&&std::isfinite(l.second)&&Exact(l.second)==Exact(pivot);variables.push_back(l.first);}
        std::sort(variables.begin(),variables.end());
        if(uniform&&std::adjacent_find(variables.begin(),variables.end())==variables.end()){
          by_simplex[variables].push_back(int(simplexes.size()));simplexes.push_back({row,variables,pivot});
        }
      }
    }
    for(int j:ambiguous)by_product.erase(j);
    std::vector<bool> remove(p.equalities.size(),false);
    for(int row=0;row<int(p.equalities.size());++row){
      check_deadline();
      const auto& q=p.equalities[row];if(q.constant!=0)continue;
      bool affine=true;for(auto h:q.hessian.entries)if(h.v)affine=false;if(!affine)continue;
      std::vector<std::pair<int,double>> linear;
      q.for_each_linear_nonzero([&](int j,double a){if(a)linear.push_back({j,a});});
      if(linear.size()<2)continue;
      int y=-1,other=0;double cy=0,scale=0;
      bool compatible=true;
      for(auto l:linear){
        if(!std::isfinite(l.second)){compatible=false;break;}
        if(!by_product.count(l.first)){y=l.first;cy=l.second;++other;}
        else if(scale==0)scale=l.second;else if(Exact(scale)!=Exact(l.second))compatible=false;
      }
      if(!compatible||other!=1||scale==0||Exact(cy)!=-Exact(scale))continue;
      if(!std::isfinite(p.lb[y])||!std::isfinite(p.ub[y]))continue;
      Witness w;w.row=row;w.y=y;
      for(auto l:linear)if(l.first!=y){
        const auto& d=definitions[by_product.at(l.first)];
        const int factor=d.a==y?d.b:(d.b==y?d.a:-1);
        if(factor<0){compatible=false;break;}
        w.definition_rows.push_back(d.row);w.product_variables.push_back(d.variable);w.factor_variables.push_back(factor);
        w.definition_weights.push_back(Exact(scale)/Exact(d.pivot));
        w.absolute_definition_weight_sum+=absolute(w.definition_weights.back());
      }
      if(!compatible)continue;
      auto support=w.factor_variables;std::sort(support.begin(),support.end());
      if(std::adjacent_find(support.begin(),support.end())!=support.end())continue;
      const auto found=by_simplex.find(support);
      if(found==by_simplex.end()||found->second.size()!=1)continue;
      const auto& simplex=simplexes[found->second[0]];
      w.simplex_row=simplex.row;w.simplex_weight=Exact(scale)/Exact(simplex.pivot);
      w.y_bound_with_original_tolerance=std::max(absolute(Exact(p.lb[y])),absolute(Exact(p.ub[y])))+Exact(tolerance);
      w.residual_amplification_bound=w.absolute_definition_weight_sum+
          absolute(w.simplex_weight)*w.y_bound_with_original_tolerance;
      w.exact_polynomial_zero=verify(w);if(!w.exact_polynomial_zero)continue;
      maximum_residual_amplification=std::max(maximum_residual_amplification,w.residual_amplification_bound);
      remove[row]=true;witnesses.push_back(std::move(w));
    }
    reduced.equalities.clear();
    for(int row=0;row<int(p.equalities.size());++row)if(!remove[row]){
      kept_equalities.push_back(row);reduced.equalities.push_back(p.equalities[row]);
    }
    for(const auto& w:witnesses){
      if(remove[w.simplex_row])throw std::runtime_error("exact EQ proof: removed proof premise");
      for(int d:w.definition_rows)if(remove[d])throw std::runtime_error("exact EQ proof: removed product premise");
    }
    if(changed()){
      // Half the exact algebraic bound leaves a safety margin; final physical
      // audit is mandatory because polynomial evaluation itself is rounded.
      const Exact target=Exact(tolerance)/(2*maximum_residual_amplification);
      inner_feasibility_tolerance=std::nextafter(target.convert_to<double>(),0.0);
      if(!(inner_feasibility_tolerance>0&&std::isfinite(inner_feasibility_tolerance)&&Exact(inner_feasibility_tolerance)<=target))
        throw std::runtime_error("exact EQ proof: tighter tolerance is not representable");
      reason="exact-product-simplex-identity-with-finite-factor-bounds";
    }
    check_deadline();
    reduced.validate();
    } catch(const DeadlineExceeded&) {
      // Transactional refusal: never expose a partial exact deletion. The
      // caller subtracts this proof time and must not start a spent child.
      reduced=p;kept_equalities.clear();witnesses.clear();definitions.clear();simplexes.clear();
      for(int row=0;row<int(p.equalities.size());++row)kept_equalities.push_back(row);
      maximum_residual_amplification=1;inner_feasibility_tolerance=tolerance;
      reason="exact-proof-deadline-refused-no-partial-deletion";
    }
  }
  bool changed()const{return !witnesses.empty();}
  void lift(Result& r)const {
    if(r.x.size()!=std::size_t(original.n)||r.certificate_equality_multipliers.size()!=kept_equalities.size()||
       r.certificate_inequality_multipliers.size()!=original.inequalities.size()||
       r.certificate_lower_multipliers.size()!=std::size_t(original.n)||
       r.certificate_upper_multipliers.size()!=std::size_t(original.n))
      throw std::runtime_error("exact EQ lift: incomplete certificate");
    Vec eq(original.equalities.size(),0.0);
    for(std::size_t k=0;k<kept_equalities.size();++k)eq[kept_equalities[k]]=r.certificate_equality_multipliers[k];
    r.certificate_equality_multipliers=std::move(eq);
    r.termination_scaling_available=false;
    r.termination_equality_scales.clear();r.termination_inequality_scales.clear();r.termination_inequality_slacks.clear();
    r.termination_multiplier_source="exact-redundant-equality-zero-lift:"+r.termination_multiplier_source;
  }
  void write_proof(std::ostream& out)const {
    auto integers=[&](const std::vector<int>& a){out<<'[';for(std::size_t i=0;i<a.size();++i){if(i)out<<',';out<<a[i];}out<<']';};
    out<<std::setprecision(17)<<std::boolalpha
       <<"{\"schema\":\"exact-product-simplex-equality-proof-v1\",\"indices\":\"zero-based\",\"n\":"<<original.n
       <<",\"equalities_original\":"<<original.equalities.size()<<",\"equalities_retained\":"<<reduced.equalities.size()
       <<",\"removed\":"<<witnesses.size()<<",\"definition_candidates\":"<<definitions.size()<<",\"simplex_candidates\":"<<simplexes.size()
       <<",\"original_tolerance\":"<<original_tolerance<<",\"inner_feasibility_tolerance\":"<<inner_feasibility_tolerance
       <<",\"maximum_amplification_exact\":\""<<maximum_residual_amplification<<"\",\"all_variables_retained\":true,\"start_unchanged\":true"
       <<",\"objective_iq_bounds_unchanged\":true,\"qualified\":false,\"kept_equalities\":";
    integers(kept_equalities);out<<",\"witnesses\":[";
    for(std::size_t i=0;i<witnesses.size();++i){const auto& w=witnesses[i];if(i)out<<',';
      out<<"{\"row\":"<<w.row<<",\"simplex_row\":"<<w.simplex_row<<",\"y\":"<<w.y
         <<",\"exact_polynomial_zero\":"<<w.exact_polynomial_zero<<",\"definition_rows\":";integers(w.definition_rows);
      out<<",\"product_variables\":";integers(w.product_variables);out<<",\"factor_variables\":";integers(w.factor_variables);
      out<<",\"definition_weights_exact\":[";for(std::size_t j=0;j<w.definition_weights.size();++j){if(j)out<<',';out<<'"'<<w.definition_weights[j]<<'"';}
      out<<"],\"simplex_weight_exact\":\""<<w.simplex_weight<<"\",\"absolute_definition_weight_sum_exact\":\""<<w.absolute_definition_weight_sum
         <<"\",\"y_bound_with_original_tolerance_exact\":\""<<w.y_bound_with_original_tolerance
         <<"\",\"residual_amplification_bound_exact\":\""<<w.residual_amplification_bound<<"\"}";
    }
    out<<"]}\n";
  }
};
} // namespace lazyipdca::qcqp::detail
