#pragma once
// Research exact facial presolve. Source functions/starts are never modified.
// A caller supplies candidate positive combinations; every implication is
// checked with exact rational arithmetic against original coefficients.
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <map>
#include <set>
#include <sstream>
#include <istream>
#include <stdexcept>

namespace lazyipdca::qcqp::detail {
class ExactAffineFacePresolve {
 public:
  using Exact=boost::multiprecision::cpp_rational;
  using Certificate=std::vector<std::pair<int,Exact>>;
  struct EqualityOrigin {bool original_inequality=false;int row=-1;};
  struct ZeroPropagation {int row=-1,variable=-1;double coefficient=0;int simplex_row=-1;std::vector<std::pair<int,double>> product_rows;std::vector<int> fraction_variables;};
  const QCQPProblem& original;
  QCQPProblem reduced;
  std::vector<int> kept,map,initial_forced,forced,kept_inequalities;
  std::vector<EqualityOrigin> kept_equalities;
  std::vector<ZeroPropagation> zero_propagations;
  std::vector<Exact> certificate_coefficient;
  std::map<int,Exact> certificate_weights;
  std::string reason="no-exact-affine-face-certificate";
  bool applied=false,initial_projection=false,proof_rows_vanish=false;
  double projection_inf=0,preparation_seconds=0,lift_seconds=0,last_shift=0;
  int removed_equalities=0,removed_inequalities=0,converted_inequalities=0;

  // Format: "face", then one "original_row numerator/denominator" per line,
  // then "end_face". Multiple blocks are allowed; comments begin with '#'.
  static std::vector<Certificate> read_certificate_text(std::istream& stream) {
    std::vector<Certificate> out;Certificate current;bool open=false;std::string line;
    while(std::getline(stream,line)) {
      const auto hash=line.find('#');if(hash!=std::string::npos)line.resize(hash);
      std::istringstream row(line);std::string token;if(!(row>>token))continue;
      if(token=="face") {if(open)throw std::runtime_error("face: nested certificate");open=true;current.clear();continue;}
      if(token=="end_face") {if(!open||current.empty())throw std::runtime_error("face: empty certificate");out.push_back(std::move(current));current={};open=false;continue;}
      if(!open)throw std::runtime_error("face: coefficient outside certificate");
      int index=std::stoi(token);Exact weight;if(!(row>>weight))throw std::runtime_error("face: invalid exact weight");
      std::string extra;if(row>>extra)throw std::runtime_error("face: trailing certificate content");current.push_back({index,weight});
    }
    if(open)throw std::runtime_error("face: unclosed certificate");return out;
  }
  static double upward(const Exact& a) {
    double d=a.convert_to<double>();if(!std::isfinite(d))throw std::runtime_error("face: multiplier overflow");
    while(Exact(d)<a)d=std::nextafter(d,INFINITY);
    if(!std::isfinite(d))throw std::runtime_error("face: multiplier overflow");return d;
  }
  explicit ExactAffineFacePresolve(const QCQPProblem& p,const std::vector<Certificate>& proofs):original(p),reduced(p) {
    const auto began=std::chrono::steady_clock::now();p.validate();
    if(p.has_signed_square())throw std::runtime_error("face: true QCQP required");
    certificate_coefficient.assign(p.n,Exact(0));std::vector<bool> fixed(p.n,false);
    for(const auto& proof:proofs) {
      std::vector<Exact> coefficient(p.n,Exact(0));Exact constant=0;bool positive=false;
      std::map<int,Exact> local;
      for(const auto& term:proof) {
        if(term.first<0||term.first>=int(p.inequalities.size())||term.second<0)
          throw std::runtime_error("face: invalid original row or negative certificate weight");
        if(term.second==0)continue;positive=true;local[term.first]+=term.second;
        const auto& q=p.inequalities[term.first];
        for(const auto& e:q.hessian.entries)if(e.v!=0)throw std::runtime_error("face: certificate row is nonlinear");
        constant+=term.second*Exact(q.constant);
        q.for_each_linear_nonzero([&](int j,double a){coefficient[j]+=term.second*Exact(a);});
      }
      if(!positive||constant!=0)throw std::runtime_error("face: certificate must be nonempty homogeneous combination");
      for(int j=0;j<p.n;++j) {
        if(coefficient[j]<0)throw std::runtime_error("face: combined coefficient is negative");
        if(coefficient[j]>0) {
          if(p.lb[j]!=0||p.ub[j]<0)throw std::runtime_error("face: forced-zero coordinate lacks original zero lower bound");
          fixed[j]=true;
        }
        certificate_coefficient[j]+=coefficient[j];
      }
      for(const auto& e:local)certificate_weights[e.first]+=e.second;
    }
    for(int j=0;j<p.n;++j)if(fixed[j])initial_forced.push_back(j);
    if(certificate_weights.empty()) {finish_time(began);return;}
    struct Simplex {int row;std::vector<int> q;};
    struct Product {int row,z,a,b;double pivot;};
    std::vector<Simplex> simplexes;std::vector<Product> products;
    for(int e=0;e<int(p.equalities.size());++e) {
      const auto& q=p.equalities[e];std::vector<std::pair<int,double>> linear;
      q.for_each_linear_nonzero([&](int j,double a){if(a!=0)linear.push_back({j,a});});
      std::vector<SymTriplet> h;for(auto a:q.hessian.entries)if(a.v!=0)h.push_back(a);
      if(q.constant==-1&&h.empty()&&!linear.empty()&&
         std::all_of(linear.begin(),linear.end(),[](const auto& a){return a.second==1;})) {
        Simplex a;a.row=e;for(auto term:linear)a.q.push_back(term.first);simplexes.push_back(std::move(a));
      }
      if(q.constant==0&&linear.size()==1&&h.size()==1&&h[0].i!=h[0].j&&
         h[0].v==-linear[0].second&&linear[0].first!=h[0].i&&linear[0].first!=h[0].j)
        products.push_back({e,linear[0].first,h[0].i,h[0].j,linear[0].second});
    }
    // Exact zero singleton closure. No product disjunction or assumption that
    // a nonnegative factor is strictly positive is used.
    for(bool changed=true;changed;) {
      changed=false;
      for(int e=0;e<int(p.equalities.size());++e) {
        const auto& q=p.equalities[e];if(q.constant!=0)continue;
        bool nonlinear=false;for(const auto& h:q.hessian.entries)
          if(h.v!=0&&!fixed[h.i]&&!fixed[h.j]){nonlinear=true;break;}
        if(nonlinear)continue;
        std::map<int,Exact> linear;q.for_each_linear_nonzero([&](int j,double a){if(!fixed[j])linear[j]+=Exact(a);});
        for(auto it=linear.begin();it!=linear.end();)if(it->second==0)it=linear.erase(it);else ++it;
        if(linear.size()!=1)continue;
        const int j=linear.begin()->first;const Exact a=linear.begin()->second;
        if(p.lb[j]>0||p.ub[j]<0)throw std::runtime_error("face: singleton zero conflicts with original bounds");
        const double coefficient=a.convert_to<double>();
        if(!std::isfinite(coefficient)||Exact(coefficient)!=a)throw std::runtime_error("face: singleton coefficient not representable");
        fixed[j]=true;ZeroPropagation event;event.row=e;event.variable=j;event.coefficient=coefficient;
        zero_propagations.push_back(std::move(event));changed=true;
      }
      // Global simplex/product identity: sum(q)=1 and every z_q=q*y=0
      // implies y=sum(z_q)=0. This never chooses a zero-product branch.
      for(const auto& simplex:simplexes) {
        std::set<int> members(simplex.q.begin(),simplex.q.end());
        std::map<int,std::map<int,const Product*>> by_flow;
        for(const auto& d:products) {
          if(members.count(d.a))by_flow[d.b][d.a]=&d;
          if(members.count(d.b))by_flow[d.a][d.b]=&d;
        }
        for(const auto& group:by_flow) {
          const int flow=group.first;if(fixed[flow]||group.second.size()!=simplex.q.size())continue;
          bool all=true;for(int q:simplex.q){auto it=group.second.find(q);if(it==group.second.end()||!fixed[it->second->z]){all=false;break;}}
          if(!all)continue;
          if(p.lb[flow]>0||p.ub[flow]<0)throw std::runtime_error("face: simplex zero flow conflicts with original bounds");
          ZeroPropagation event;event.variable=flow;event.simplex_row=simplex.row;event.fraction_variables=simplex.q;
          for(int q:simplex.q){const auto* d=group.second.at(q);event.product_rows.push_back({d->row,d->pivot});}
          fixed[flow]=true;zero_propagations.push_back(std::move(event));changed=true;
        }
      }
    }
    map.assign(p.n,-1);for(int j=0;j<p.n;++j)if(fixed[j])forced.push_back(j);else{map[j]=kept.size();kept.push_back(j);}
    if(kept.empty())throw std::runtime_error("face: zero-dimensional private model not admitted by this prototype");
    reduced={};reduced.name=p.name;reduced.n=kept.size();
    for(int j:kept){reduced.x0.push_back(p.x0[j]);reduced.lb.push_back(p.lb[j]);reduced.ub.push_back(p.ub[j]);}
    auto rewrite=[&](const QuadraticFunction& q) {
      QuadraticFunction r;r.name=q.name;r.constant=q.constant;r.compact_linear_dimension=reduced.n;r.hessian.n=reduced.n;
      q.for_each_linear_nonzero([&](int j,double a){if(map[j]>=0&&a!=0)r.compact_linear.push_back({map[j],a});});
      for(const auto& h:q.hessian.entries)if(h.v!=0&&map[h.i]>=0&&map[h.j]>=0)
        r.hessian.entries.push_back({map[h.i],map[h.j],h.v});
      return r;
    };
    auto constant=[](const QuadraticFunction& q){return q.compact_linear.empty()&&q.hessian.entries.empty();};
    reduced.objective=rewrite(p.objective);
    for(int e=0;e<int(p.equalities.size());++e) {
      auto q=rewrite(p.equalities[e]);if(constant(q)) {
        if(q.constant!=0)throw std::runtime_error("face: inconsistent constant equality");++removed_equalities;
      }else {kept_equalities.push_back({false,e});reduced.equalities.push_back(std::move(q));}
    }
    proof_rows_vanish=true;
    for(int i=0;i<int(p.inequalities.size());++i) {
      auto q=rewrite(p.inequalities[i]);
      if(constant(q)) {if(q.constant>0)throw std::runtime_error("face: inconsistent constant inequality");++removed_inequalities;continue;}
      if(certificate_weights.count(i)&&certificate_weights.at(i)>0) {
        // Since every original g<=0 and sum(alpha*g)=0 on this face,
        // every positively weighted source row equals zero on all feasibles.
        proof_rows_vanish=false;++converted_inequalities;
        kept_equalities.push_back({true,i});reduced.equalities.push_back(std::move(q));
      }else {kept_inequalities.push_back(i);reduced.inequalities.push_back(std::move(q));}
    }
    for(int j:forced){initial_projection=initial_projection||p.x0[j]!=0;projection_inf=std::max(projection_inf,std::abs(p.x0[j]));}
    reduced.validate();applied=!forced.empty()||removed_equalities||removed_inequalities||converted_inequalities;
    reason="exact-positive-affine-combination-zero-face-and-singleton-closure";finish_time(began);
  }
  bool changed() const{return applied;}

  // Lifts a private endpoint, not a solver success status. The caller must run
  // the ordinary full-original KKT audit and enforce the unchanged thresholds.
  void lift(Result& result) {
    const auto began=std::chrono::steady_clock::now();
    if(!applied)return;
    if(result.x.size()!=kept.size()||result.certificate_equality_multipliers.size()!=kept_equalities.size()||
       result.certificate_inequality_multipliers.size()!=kept_inequalities.size()||
       result.certificate_lower_multipliers.size()!=kept.size()||result.certificate_upper_multipliers.size()!=kept.size())
      throw std::runtime_error("face: incomplete reduced endpoint certificate");
    Vec x(original.n,0),y(original.equalities.size(),0),z(original.inequalities.size(),0),lower(original.n,0),upper(original.n,0);
    for(std::size_t k=0;k<kept.size();++k){int j=kept[k];x[j]=result.x[k];lower[j]=result.certificate_lower_multipliers[k];upper[j]=result.certificate_upper_multipliers[k];}
    for(std::size_t k=0;k<kept_equalities.size();++k){auto e=kept_equalities[k];if(e.original_inequality)z[e.row]=result.certificate_equality_multipliers[k];else y[e.row]=result.certificate_equality_multipliers[k];}
    for(std::size_t k=0;k<kept_inequalities.size();++k)z[kept_inequalities[k]]=result.certificate_inequality_multipliers[k];
    std::vector<bool> is_forced(original.n,false);for(int j:forced)is_forced[j]=true;
    // Exact original gradients on the fixed coordinates, using the ACTUAL
    // rounded endpoint multipliers. This makes sign checks independent of a
    // reduced residual or a model scaling convention.
    auto residual=[&]() {
      std::vector<Exact> r(original.n,Exact(0));
      auto add=[&](const QuadraticFunction& q,double mult) {
        if(mult==0)return;if(!std::isfinite(mult))throw std::runtime_error("face: nonfinite endpoint multiplier");
        const Exact w(mult);q.for_each_linear_nonzero([&](int j,double a){if(is_forced[j])r[j]+=w*Exact(a);});
        for(const auto& h:q.hessian.entries)if(h.v!=0) {
          if(is_forced[h.i])r[h.i]+=w*Exact(h.v)*Exact(x[h.j]);
          if(h.i!=h.j&&is_forced[h.j])r[h.j]+=w*Exact(h.v)*Exact(x[h.i]);
        }
      };
      add(original.objective,1);for(std::size_t e=0;e<y.size();++e)add(original.equalities[e],y[e]);
      for(std::size_t i=0;i<z.size();++i)add(original.inequalities[i],z[i]);return r;
    };
    // Reverse causal closure. A propagation equality has nonzero derivative
    // only in its new coordinate and coordinates fixed earlier; hence this
    // adjoint back-substitution never changes a retained-coordinate residual.
    for(auto it=zero_propagations.rbegin();it!=zero_propagations.rend();++it) {
      const auto r=residual();
      if(it->product_rows.empty()) {
        const Exact correction=-r[it->variable]/Exact(it->coefficient);
        const double value=correction.convert_to<double>();if(!std::isfinite(value))throw std::runtime_error("face: equality lift overflow");
        y[it->row]+=value;
      } else {
        Exact sum=0;for(int q:it->fraction_variables)sum+=Exact(x[q]);
        if(sum==0)throw std::runtime_error("face: simplex lift has zero actual sum; original equality infeasible");
        const Exact correction=r[it->variable]/sum;
        for(const auto& product:it->product_rows) {
          const Exact scaled_correction=correction/Exact(product.second);
          const double value=scaled_correction.convert_to<double>();
          if(!std::isfinite(value))throw std::runtime_error("face: simplex adjoint lift overflow");
          y[product.first]+=value;
        }
      }
    }
    const auto before=residual();Exact shift=0;
    for(int j:initial_forced)if(before[j]<0){const Exact need=-before[j]/certificate_coefficient[j];if(need>shift)shift=need;}
    for(const auto& e:certificate_weights)if(z[e.first]<0){const Exact need=-Exact(z[e.first])/e.second;if(need>shift)shift=need;}
    const Vec base_z=z;
    // Pay a tiny exact positive margin, then check the actual rounded shift.
    const Exact margin=(shift>1?shift:Exact(1))/Exact(1ULL<<40);shift+=margin;
    bool signs=false;std::vector<Exact> after;
    for(int retry=0;retry<12;++retry) {
      z=base_z;for(const auto& e:certificate_weights)z[e.first]=upward(Exact(base_z[e.first])+shift*e.second);
      after=residual();signs=true;for(int j:initial_forced)if(after[j]<0)signs=false;
      for(const auto& e:certificate_weights)if(z[e.first]<0)signs=false;
      if(signs)break;shift=2*shift+margin;
    }
    if(!signs)throw std::runtime_error("face: rounded certificate shift failed sign audit");
    for(int j:forced) {
      // Singleton-propagated coordinates should have zero residual after the
      // causal lift; a harmless positive remainder uses its original lower
      // normal only when zero is an original active lower bound.
      if(original.lb[j]==0&&after[j]>=0)lower[j]=after[j].convert_to<double>();
      else if(original.ub[j]==0&&after[j]<=0)upper[j]=(-after[j]).convert_to<double>();
      // For an interior zero coordinate the equality lift must carry the
      // stationarity. Any rounding remainder is exposed to the original audit.
    }
    for(double v:x)if(!std::isfinite(v))throw std::runtime_error("face: nonfinite lifted point");
    for(const auto* values:{&y,&z,&lower,&upper})for(double v:*values)if(!std::isfinite(v))throw std::runtime_error("face: nonfinite lifted multiplier");
    result.x=std::move(x);result.certificate_equality_multipliers=std::move(y);result.certificate_inequality_multipliers=std::move(z);
    result.certificate_lower_multipliers=std::move(lower);result.certificate_upper_multipliers=std::move(upper);
    result.termination_scaling_available=false;result.termination_equality_scales.clear();result.termination_inequality_scales.clear();result.termination_inequality_slacks.clear();
    result.termination_multiplier_source="exact-affine-face-original-adjoint-lift:"+result.termination_multiplier_source;
    last_shift=shift.convert_to<double>();lift_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
  }
 private:
  void finish_time(std::chrono::steady_clock::time_point began){preparation_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();}
};
} // namespace lazyipdca::qcqp::detail
