#pragma once
// Experimental exact product substitution. No coefficient tolerance, degree
// truncation or auxiliary optimizer enters detection or postsolve.
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>

namespace lazyipdca::qcqp::detail {
class ProductDefinitionPresolve {
 public:
  struct Definition {int variable,row,a,b;double sign,pivot;int lower=-1,upper=-1;};
  const QCQPProblem& original;
  QCQPProblem reduced;
  std::vector<Definition> definitions;
  std::vector<int> kept,map,kept_equalities;
  std::string reason="no-exact-product-definitions";
  bool initial_projection=false;
  double projection_inf=0;
  explicit ProductDefinitionPresolve(const QCQPProblem& p):original(p) {
    p.validate();if(p.has_signed_square()){reason="signed-square-unsupported";return;}
    std::vector<bool> nonlinear(p.n,false);
    auto visit=[&](const QuadraticFunction& q){for(auto e:q.hessian.entries)if(e.v){nonlinear[e.i]=true;nonlinear[e.j]=true;}};
    visit(p.objective);for(const auto& q:p.equalities)visit(q);for(const auto& q:p.inequalities)visit(q);
    std::vector<int> counts(p.n,0);
    std::vector<Definition> candidates;
    for(int i=0;i<int(p.equalities.size());++i){
      const auto& q=p.equalities[i];if(q.constant!=0)continue;
      int count=0,j=-1;double coefficient=0;
      q.for_each_linear_nonzero([&](int k,double a){if(a){++count;j=k;coefficient=a;}});
      if(count!=1||nonlinear[j]||(coefficient!=1&&coefficient!=-1))continue;
      SymTriplet term;int nt=0;
      for(auto e:q.hessian.entries)if(e.v){term=e;++nt;}
      if(nt!=1||term.i==term.j||(term.v!=1&&term.v!=-1))continue;
      candidates.push_back({j,i,term.i,term.j,-term.v/coefficient,coefficient});++counts[j];
    }
    std::vector<bool> remove(p.n,false),remove_equality(p.equalities.size(),false);
    for(auto d:candidates)if(counts[d.variable]==1){definitions.push_back(d);remove[d.variable]=true;remove_equality[d.row]=true;}
    if(definitions.empty())return;
    map.assign(p.n,-1);std::vector<int> definition(p.n,-1);
    for(int j=0;j<p.n;++j)if(!remove[j]){map[j]=kept.size();kept.push_back(j);}
    if(kept.empty()){definitions.clear();reason="no-retained-variable";return;}
    for(int i=0;i<int(definitions.size());++i)definition[definitions[i].variable]=i;
    reduced.name=p.name;reduced.n=kept.size();
    for(int j:kept){reduced.x0.push_back(p.x0[j]);reduced.lb.push_back(p.lb[j]);reduced.ub.push_back(p.ub[j]);}
    using Exact=boost::multiprecision::cpp_rational;
    auto exact_double=[](const Exact& a){double v=a.convert_to<double>();if(!std::isfinite(v)||Exact(v)!=a)throw std::runtime_error("product-definition: exact coefficient not representable");return v;};
    auto rewrite=[&](const QuadraticFunction& q){
      QuadraticFunction out;out.name=q.name;out.constant=q.constant;out.hessian.n=reduced.n;out.compact_linear_dimension=reduced.n;
      std::map<int,Exact> linear;std::map<std::pair<int,int>,Exact> hessian;
      q.for_each_linear_nonzero([&](int j,double a){
        if(!a)return;
        if(map[j]>=0)linear[map[j]]+=Exact(a);
        else {const auto& d=definitions[definition[j]];const auto pair=std::minmax(map[d.a],map[d.b]);
          if(pair.first<0)throw std::runtime_error("product-definition: recursive definition");
          hessian[{pair.first,pair.second}]+=Exact(d.sign)*Exact(a);
        }
      });
      for(auto e:q.hessian.entries)if(e.v){
        if(map[e.i]<0||map[e.j]<0)throw std::runtime_error("product-definition: nonlinear pivot incidence");
        const auto pair=std::minmax(map[e.i],map[e.j]);hessian[{pair.first,pair.second}]+=Exact(e.v);
      }
      for(const auto& e:linear)if(e.second!=0)out.compact_linear.push_back({e.first,exact_double(e.second)});
      for(const auto& e:hessian)if(e.second!=0)out.hessian.entries.push_back({e.first.first,e.first.second,exact_double(e.second)});
      return out;
    };
    try {
      reduced.objective=rewrite(p.objective);
      for(int i=0;i<int(p.equalities.size());++i)if(!remove_equality[i]){kept_equalities.push_back(i);reduced.equalities.push_back(rewrite(p.equalities[i]));}
      for(const auto& q:p.inequalities)reduced.inequalities.push_back(rewrite(q));
      for(auto& d:definitions){
        // Exact rational interval proof only; an omitted auxiliary bound
        // receives multiplier zero. Every retained-factor bound remains.
        bool finite_interval=true;for(int j:{d.a,d.b})finite_interval=finite_interval&&std::isfinite(p.lb[j])&&std::isfinite(p.ub[j]);
        Exact lo=0,hi=0;
        if(finite_interval){bool first=true;for(double a:{p.lb[d.a],p.ub[d.a]})for(double b:{p.lb[d.b],p.ub[d.b]}){Exact v=Exact(d.sign)*Exact(a)*Exact(b);if(first){lo=hi=v;first=false;}else{lo=std::min(lo,v);hi=std::max(hi,v);}}}
        auto bound=[&](bool lower,double value){
          QuadraticFunction q;q.name=std::string("product_")+(lower?"lower_":"upper_")+std::to_string(d.variable);
          q.constant=lower?value:-value;q.compact_linear_dimension=reduced.n;q.hessian.n=reduced.n;
          const auto pair=std::minmax(map[d.a],map[d.b]);q.hessian.entries.push_back({pair.first,pair.second,(lower?-1:1)*d.sign});
          const int index=reduced.inequalities.size();reduced.inequalities.push_back(std::move(q));return index;
        };
        if(std::isfinite(p.lb[d.variable])&&!(finite_interval&&lo>=Exact(p.lb[d.variable])))d.lower=bound(true,p.lb[d.variable]);
        if(std::isfinite(p.ub[d.variable])&&!(finite_interval&&hi<=Exact(p.ub[d.variable])))d.upper=bound(false,p.ub[d.variable]);
      }
      reduced.validate();
      for(const auto& d:definitions){const double v=d.sign*p.x0[d.a]*p.x0[d.b];if(!std::isfinite(v))throw std::runtime_error("product-definition: nonfinite initial projection");
        initial_projection=initial_projection||std::memcmp(&v,&p.x0[d.variable],sizeof(double))!=0;projection_inf=std::max(projection_inf,std::abs(v-p.x0[d.variable]));}
      reason="exact-unit-product-definitions";
    } catch(const std::exception& e){reason=e.what();definitions.clear();reduced={};kept.clear();kept_equalities.clear();map.clear();}
  }
  bool changed()const{return !definitions.empty();}
  void lift(Result& r)const {
    if(!changed())return;
    if(r.x.size()!=std::size_t(reduced.n)||r.certificate_equality_multipliers.size()!=reduced.equalities.size()||
       r.certificate_inequality_multipliers.size()!=reduced.inequalities.size()||r.certificate_lower_multipliers.size()!=r.x.size()||
       r.certificate_upper_multipliers.size()!=r.x.size())throw std::runtime_error("product-definition: incomplete reduced certificate");
    Vec x=original.x0,lower(original.n,0),upper(original.n,0),eq(original.equalities.size(),0),ineq(original.inequalities.size(),0);
    for(int k=0;k<int(kept.size());++k){int j=kept[k];x[j]=r.x[k];lower[j]=r.certificate_lower_multipliers[k];upper[j]=r.certificate_upper_multipliers[k];}
    for(int k=0;k<int(kept_equalities.size());++k)eq[kept_equalities[k]]=r.certificate_equality_multipliers[k];
    for(int k=0;k<int(ineq.size());++k)ineq[k]=r.certificate_inequality_multipliers[k];
    for(const auto& d:definitions){x[d.variable]=d.sign*x[d.a]*x[d.b];if(d.lower>=0)lower[d.variable]=r.certificate_inequality_multipliers[d.lower];if(d.upper>=0)upper[d.variable]=r.certificate_inequality_multipliers[d.upper];}
    // Product pivots occur linearly in every original row. Their own
    // stationarity equation fixes the eliminated equality multiplier.
    std::vector<long double> stationarity(original.n,0);
    auto add=[&](const QuadraticFunction& q,double w){if(w)q.for_each_linear_nonzero([&](int j,double a){stationarity[j]+=static_cast<long double>(w)*a;});};
    add(original.objective,1);
    for(int i:kept_equalities)add(original.equalities[i],eq[i]);
    for(int i=0;i<int(ineq.size());++i)add(original.inequalities[i],ineq[i]);
    for(const auto& d:definitions)eq[d.row]=static_cast<double>(-(stationarity[d.variable]-lower[d.variable]+upper[d.variable])/d.pivot);
    r.x=std::move(x);r.certificate_equality_multipliers=std::move(eq);r.certificate_inequality_multipliers=std::move(ineq);
    r.certificate_lower_multipliers=std::move(lower);r.certificate_upper_multipliers=std::move(upper);
    // Experimental route supports project termination only: do not export
    // private-dimensional scaling arrays as original-dimensional metadata.
    r.termination_scaling_available=false;r.termination_equality_scales.clear();r.termination_inequality_scales.clear();r.termination_inequality_slacks.clear();
  }
};

inline Result solve_with_product_definition_presolve(const QCQPProblem& p,const Settings& st){
  using Clock=std::chrono::steady_clock;const auto began=Clock::now();
  auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-began).count();};
  ProductDefinitionPresolve reduction(p);const double preparation=elapsed();Settings inner=st;inner.product_definition_presolve=false;
  const auto& model=reduction.changed()?reduction.reduced:p;
  if(st.live_display_level)std::cout<<"[PRODUCT-DEFINITION-PRESOLVE] applied="<<reduction.changed()<<" definitions="<<reduction.definitions.size()
      <<" variables="<<p.n<<"->"<<model.n<<" equalities="<<p.equalities.size()<<"->"<<model.equalities.size()
      <<" inequalities="<<p.inequalities.size()<<"->"<<model.inequalities.size()<<" input_start_retained=1 internal_lift_projection="<<reduction.initial_projection
      <<" projection_inf="<<reduction.projection_inf<<" exact_coefficients=1 reason="<<reduction.reason<<std::endl;
  if(reduction.changed()&&st.accepted_checkpoint_callback){const auto observer=st.accepted_checkpoint_callback;
    inner.accepted_checkpoint_callback=[&,observer](const QCQPProblem& current,Result& snapshot,const ConstraintPresolve* context,int iteration,double){
      if(&current!=&reduction.reduced&&!(context&&&context->original()==&reduction.reduced&&&context->reduced()==&current))throw std::runtime_error("product-definition checkpoint owner unavailable");
      Result full=snapshot;if(context)context->lift_checkpoint(full);reduction.lift(full);observer(p,full,nullptr,iteration,elapsed());
    };
  }
  if(std::isfinite(st.time_limit_seconds))inner.time_limit_seconds=std::max(0.,st.time_limit_seconds-elapsed());
  Result r;
  if(inner.time_limit_seconds>0)r=Solver(inner).solve(model);
  else {r.status=Status::TimeLimit;r.message="product-definition: preparation deadline";r.x=model.x0;r.certificate_equality_multipliers.assign(model.equalities.size(),0);r.certificate_inequality_multipliers.assign(model.inequalities.size(),0);r.certificate_lower_multipliers.assign(model.n,0);r.certificate_upper_multipliers.assign(model.n,0);}
  const auto post=Clock::now();
  if(reduction.changed()){
    try {reduction.lift(r);Settings original=st;original.rank_presolve_context=nullptr;original.product_definition_presolve=false;
      const bool audit_ok=audit_original_termination(p,original,r);
      const bool strict=audit_ok&&r.feasibility.total_inf<=std::min(1e-6,st.feasibility_tolerance)&&
          r.dual_infeasibility_inf<=std::min(1e-6,st.dual_infeasibility_tolerance)&&
          r.complementarity_inf<=std::min(1e-6,st.complementarity_tolerance);
      if((r.status==Status::Solved||r.status==Status::SolvedAcceptable||r.status==Status::SolvedAcceptableOnFailure)&&!strict){r.status=Status::NumericalError;r.message="product-definition: full ORIGINAL KKT audit failed";r.acceptable_point=false;r.acceptable_convergence=false;}
      else if(r.status==Status::Solved)r.message+="; exact product definitions lifted and ORIGINAL KKT audited";
      r.termination_multiplier_source="product-definition-original-stationarity-lift:"+r.termination_multiplier_source;
    } catch(const std::exception& e){
      r.status=Status::NumericalError;r.message=e.what();r.x=p.x0;r.certificate_equality_multipliers.assign(p.equalities.size(),0);r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);r.termination_scaling_available=false;r.termination_audit_valid=false;r.acceptable_point=false;r.acceptable_convergence=false;r.objective=p.objective.value(r.x);r.feasibility=feasibility(p,r.x);r.dual_infeasibility_inf=r.complementarity_inf=r.raw_kkt_error=INFINITY;
    }
  }
  const double post_seconds=std::chrono::duration<double>(Clock::now()-post).count();r.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=preparation;r.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=post_seconds;
  r.elapsed_seconds=elapsed();r.lazy_newton.solver_wall_seconds=r.elapsed_seconds;
  if(std::isfinite(st.time_limit_seconds)&&r.elapsed_seconds>st.time_limit_seconds){r.status=Status::TimeLimit;r.message="product-definition: original solve/audit deadline";}
  if(st.live_display_level&&reduction.changed())std::cout<<"[PRODUCT-DEFINITION-POSTSOLVE] original_n="<<r.x.size()<<" strict_original="<<(r.termination_audit_valid&&r.raw_kkt_error<=st.feasibility_tolerance)<<" status="<<status_string(r.status)<<" primal="<<r.feasibility.total_inf<<" dual="<<r.dual_infeasibility_inf<<" compl="<<r.complementarity_inf<<" seconds="<<r.elapsed_seconds<<std::endl;
  return r;
}
} // namespace lazyipdca::qcqp::detail
