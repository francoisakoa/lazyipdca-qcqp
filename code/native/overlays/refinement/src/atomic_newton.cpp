#include "lazyipdca_qcqp/native_original_anchor_finish.hpp"
#include "factorial_trace.hpp"
#include "lazyipdca_qcqp/gr_source_newton_curve.hpp"
#include "lazyipdca_qcqp/source_centered_rhs.hpp"
#include "lazyipdca_qcqp/native_restoration_trace.hpp"
#include "lazyipdca_qcqp/gr_source_curve.hpp"
#include "native_affine_progress.hpp"
#include "rank_retry_statistics.hpp"
#include "gr_bound_stagnation_watch.hpp"
#include "lazyipdca_qcqp/convexification_study.hpp"
#include "lazyipdca_qcqp/zero_barrier_forcing.hpp"
#include "lazyipdca_qcqp/constant_square_policy.hpp"
#include "lazyipdca_qcqp/hybrid_selection.hpp"
#include "lazyipdca_qcqp/hybrid_recovery.hpp"
#include "lazyipdca_qcqp/hybrid_border.hpp"
#include "lazyipdca_qcqp/hybrid_inexact.hpp"
#include "lazyipdca_qcqp/hybrid_terminal_certificate.hpp"
#include "lazyipdca_qcqp/hybrid_slack_path.hpp"
#include "lazyipdca_qcqp/h4_domain.hpp"
#include "lazyipdca_qcqp/majorant_filter_recenter.hpp"
#include "lazyipdca_qcqp/majorant_auto_startup.hpp"
#include "lazyipdca_qcqp/majorant_startup_retry.hpp"
#include "lazyipdca_qcqp/majorant_terminal_retry.hpp"
#include "lazyipdca_qcqp/majorant_equality_stabilization.hpp"
#include "historical_equality_watch.hpp"
#include "shared_restoration_budget.hpp"
#include "positive_square_epigraph_restoration.hpp"
#include "lazyipdca_qcqp/primal_dual_merit.hpp"
#include "failure_recovery_policy.hpp"
#include "gr_proximal_policy.hpp"
#include "gr_fixed_centering.hpp"
#include "small_step_soc_policy.hpp"
#include "nonlinear_trial_soc_policy.hpp"
#include "trial_soc_arc_policy.hpp"
#include "soc_defect_control_policy.hpp"
#include "rho_decay_feasibility_guard.hpp"
#include "cold_dual_balance.hpp"
#include "lazyipdca_qcqp/lazy_newton.hpp"
#include "lazyipdca_qcqp/restoration.hpp"
#include "lazyipdca_qcqp/inner_ipm.hpp"
#include "lazyipdca_qcqp/convex_affine_pullback.hpp"
#include "lazyipdca_qcqp/native_inexact_contract.hpp"
#include "lazyipdca_qcqp/hybrid_split_step.hpp"
#include "lazyipdca_qcqp/hybrid_merit_recovery.hpp"
#include "lazyipdca_qcqp/hybrid_source_soc.hpp"
#include "lazyipdca_qcqp/dca_filter.hpp"
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/terminal_multiplier_fit.hpp"
#include "lazyipdca_qcqp/original_kkt_gmres.hpp"
#include "lazyipdca_qcqp/dual_step.hpp"
#include "dca_curvature.hpp"
#include "sparse_pair_elimination.hpp"
#include "bounded_pivot_recovery.hpp"
#include "bounded_affine_elimination.hpp"
#include "accurate_product_sum.hpp"
#include "coefficient_quadratic_increment.hpp"
#include "dca_filter_contract.hpp"
#include "dca_progress_guard.hpp"
#include "dca_stable_border.hpp"
#include "dca_aggregate_controller.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <chrono>
#include "newton_atom_projection.hpp"
#include "newton_cost_ledger.hpp"
#include <map>
#include <set>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <unordered_map>
#if defined(LAZYIPDCA_HAVE_OPENMP)
#include <omp.h>
#endif
extern "C" void dsyev_(const char*,const char*,const int*,double*,const int*,double*,double*,const int*,int*);

namespace lazyipdca::qcqp {
namespace {
using Clock=std::chrono::steady_clock;
using CostScope=detail::NewtonCostLedger::Scope;
double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
bool finite(const Vec& a) {return std::all_of(a.begin(),a.end(),[](double x){return std::isfinite(x);});}
using Pair=std::pair<int,int>;
Pair upper(int i,int j) {return {std::min(i,j),std::max(i,j)};}

// Sparse values, gradients and Hessians share permanent index maps. Fixed
// coordinates are substituted once, including cross terms and constants.
struct Quadratic {
  detail::NewtonCostLedger* ledger=nullptr;
  int family=0;
  // Majorant line polynomials and reanchored values use compatible accurate
  // sums. This flag changes values only; legacy routes keep their arithmetic.
  bool majorant_values_extended=false;
  bool majorant_gradients_extended=false;
  void note(std::uint64_t NewtonCalls::*field) const {++(ledger->active_calls[family].*field);}

  bool model_active=false;
  // Ambient scalar correction; at an anchor its derivative is exactly zero,
  // so Jacobian sparsity is preserved without restricting rho*I to support.
  double study_model_scalar=0;
  const Vec* model_center=nullptr;
  double model_constant=0;
  Vec model_gradient;
  std::vector<long double> model_gradient_extended;
  std::vector<SymTriplet> model_hessian;
  std::vector<LowRankTerm> model_atoms;
  // Optional exact common support of lifted DCA atoms. Empty preserves the
  // generic full-coordinate representation. DcaModels supplies retained
  // coordinates only for corrections E' U, where E*T=I.
  std::vector<int> model_atom_support;
  std::vector<int> model_hi,model_hj,model_hglobal;
  double model_atom_displacement(const LowRankTerm& a,const Vec& point) const {
    double d=0;
    if(model_atom_support.empty()) {
      for(std::size_t j=0;j<a.direction.size();++j)d+=a.direction[j]*(point[j]-(*model_center)[j]);
    } else for(int j:model_atom_support)d+=a.direction[j]*(point[j]-(*model_center)[j]);
    return d;
  }
  double model_atom_product(const LowRankTerm& a,const Vec& direction) const {
    if(model_atom_support.empty())return dot(a.direction,direction);
    double d=0;for(int j:model_atom_support)d+=a.direction[j]*direction[j];return d;
  }
  void add_model_atom(const LowRankTerm& a,double weight,Vec& destination) const {
    if(model_atom_support.empty())axpy(weight,a.direction,destination);
    else for(int j:model_atom_support)destination[j]+=weight*a.direction[j];
  }
  double model_at(const Vec& point) const {
    if(majorant_values_extended) {
      long double v=model_constant;
      if(study_model_scalar)for(std::size_t j=0;j<point.size();++j){const long double d=(long double)point[j]-(*model_center)[j];v+=0.5L*study_model_scalar*d*d;}
      for(std::size_t k=0;k<support.size();++k) {
        const int j=support[k];
        const long double coefficient=majorant_gradients_extended&&
            model_gradient_extended.size()==support.size()?model_gradient_extended[k]:
            static_cast<long double>(model_gradient[k]);
        v+=coefficient*(static_cast<long double>(point[j])-(*model_center)[j]);
      }
      for(auto e:model_hessian)
        v+=(e.i==e.j?0.5L:1.0L)*e.v*
            (static_cast<long double>(point[e.i])-(*model_center)[e.i])*
            (static_cast<long double>(point[e.j])-(*model_center)[e.j]);
      for(const auto& a:model_atoms) {
        long double displacement=0;
        if(model_atom_support.empty()) {
          for(std::size_t j=0;j<a.direction.size();++j)
            displacement+=static_cast<long double>(a.direction[j])*
                (static_cast<long double>(point[j])-(*model_center)[j]);
        } else for(int j:model_atom_support)
          displacement+=static_cast<long double>(a.direction[j])*
              (static_cast<long double>(point[j])-(*model_center)[j]);
        v+=0.5L*a.beta*displacement*displacement;
      }
      // Model coefficients already contain their constraint scale.
      return static_cast<double>(v);
    }
    double v=model_constant;
    if(study_model_scalar)for(std::size_t j=0;j<point.size();++j){const double d=point[j]-(*model_center)[j];v+=0.5*study_model_scalar*d*d;}
    for(std::size_t k=0;k<support.size();++k)v+=model_gradient[k]*(point[support[k]]-(*model_center)[support[k]]);
    for(auto e:model_hessian)v+=(e.i==e.j?0.5:1.0)*e.v*(point[e.i]-(*model_center)[e.i])*(point[e.j]-(*model_center)[e.j]);
    for(const auto& a:model_atoms){const double d=model_atom_displacement(a,point);v+=0.5*a.beta*d*d;}
    return v;
  }
  void set_model(const Vec& center,std::vector<SymTriplet> h,const std::map<Pair,int>& slots) {
    model_center=&center;model_constant=value;model_gradient=gradient;
    if(majorant_gradients_extended)model_gradient_extended=gradient_extended;
    bool same = model_hessian.size()==h.size();
    if(same) for(std::size_t k=0;k<h.size();++k)
      if(model_hessian[k].i!=h[k].i||model_hessian[k].j!=h[k].j){same=false;break;}
    model_hessian=std::move(h);
    if(same){model_active=true;return;}
    model_hi.clear();model_hj.clear();model_hglobal.clear();
    for(auto e:model_hessian) {
      model_hi.push_back((int)(std::lower_bound(support.begin(),support.end(),e.i)-support.begin()));
      model_hj.push_back((int)(std::lower_bound(support.begin(),support.end(),e.j)-support.begin()));
      model_hglobal.push_back(slots.at(upper(e.i,e.j)));
    }
    model_active=true;
  }
  double constant=0.0,scale=1.0;
  std::vector<std::pair<int,double>> linear;
  std::vector<SymTriplet> hessian;
  std::vector<SymTriplet> base_hessian;
  std::vector<std::pair<int,double>> signed_square;
  std::vector<int> signed_square_slots;
  void refresh_native_hessian(const Vec& x) {
    if(signed_square.empty())return;
    hessian=base_hessian;
    for(std::size_t k=0;k<signed_square.size();++k){const auto e=signed_square[k];hessian[signed_square_slots[k]].v+=2*e.second*((x[e.first]>0)-(x[e.first]<0));}
  }
  std::vector<int> support,linear_pos,hi,hj,hglobal;
  Vec gradient;
  // Persistent majorant-only scratch/cache: no allocation after first sizing.
  std::vector<long double> gradient_extended;
  const int* pooled_support=nullptr;
  double* pooled_gradient=nullptr;
  double value=0.0;
  Quadratic()=default;
  Quadratic(const QuadraticFunction& q,const std::vector<int>& index,const Vec& fixed,std::vector<int>* original_linear=nullptr) {
    constant=q.constant;
    std::map<int,double> l;
    std::map<Pair,double> h;
    q.for_each_linear_nonzero([&](int j,double coefficient) {
      if(!std::isfinite(coefficient))throw std::runtime_error("nonfinite quadratic coefficient");
      if(original_linear)original_linear->push_back(j);
      if(index[j]<0)constant+=coefficient*fixed[j];
      else l[index[j]]+=coefficient;
    });
    for(const auto& e:q.hessian.entries) {
      int a=index[e.i],b=index[e.j];
      if(a>=0&&b>=0)h[upper(a,b)]+=e.v;
      else if(a<0&&b<0)constant+=(e.i==e.j?0.5:1.0)*e.v*fixed[e.i]*fixed[e.j];
      else if(a>=0)l[a]+=e.v*fixed[e.j];
      else l[b]+=e.v*fixed[e.i];
    }
    for(auto e:q.signed_square){if(index[e.first]<0)constant+=e.second*fixed[e.first]*std::abs(fixed[e.first]);else{signed_square.push_back({index[e.first],e.second});h[upper(index[e.first],index[e.first])]+=0.;}}
    for(auto e:l)if(e.second!=0.0)linear.push_back(e);
    for(auto e:h)if(e.second!=0.0||std::any_of(signed_square.begin(),signed_square.end(),[&](auto a){return a.first==e.first.first&&a.first==e.first.second;}))hessian.push_back({e.first.first,e.first.second,e.second});
    if(!signed_square.empty())base_hessian=hessian;
    for(auto e:signed_square){auto it=std::find_if(hessian.begin(),hessian.end(),[&](const auto& a){return a.i==e.first&&a.j==e.first;});signed_square_slots.push_back(int(it-hessian.begin()));}
    compile();
  }
  void compile() {
    std::set<int> s;
    for(auto e:linear)s.insert(e.first);
    for(auto e:hessian){s.insert(e.i);s.insert(e.j);}
    support.assign(s.begin(),s.end());gradient.resize(support.size());
    linear_pos.clear();hi.clear();hj.clear();
    auto pos=[&](int j){return (int)(std::lower_bound(support.begin(),support.end(),j)-support.begin());};
    for(auto e:linear)linear_pos.push_back(pos(e.first));
    for(auto e:hessian){hi.push_back(pos(e.i));hj.push_back(pos(e.j));}
  }
  double at(const Vec& x) const {
    if(model_active){note(&NewtonCalls::values);note(&NewtonCalls::model_values);return model_at(x);}
    return original_at(x);
  }
  double original_at(const Vec& x) const {
    note(&NewtonCalls::values);
    if(majorant_values_extended) {
      long double v=constant;
      for(auto e:linear)v+=static_cast<long double>(e.second)*x[e.first];
      for(auto e:(signed_square.empty()?hessian:base_hessian))v+=(e.i==e.j?0.5L:1.0L)*e.v*x[e.i]*x[e.j];
      for(auto e:signed_square)v+=static_cast<long double>(e.second)*x[e.first]*std::abs(x[e.first]);
      return static_cast<double>(static_cast<long double>(scale)*v);
    }
    double v=constant;
    for(auto e:linear)v+=e.second*x[e.first];
    for(auto e:(signed_square.empty()?hessian:base_hessian))v+=(e.i==e.j?0.5:1.0)*e.v*x[e.i]*x[e.j];
    for(auto e:signed_square)v+=e.second*x[e.first]*std::abs(x[e.first]);
    return scale*v;
  }
  long double original_difference(const Vec& x,const Vec& trial) const {
    return detail::coefficient_quadratic_increment(x,trial,linear,
        signed_square.empty()?hessian:base_hessian,signed_square,scale);
  }
  void evaluate(const Vec& x) {
    // This outer-only study only evaluates auxiliary gradients at their
    // anchor. General off-anchor gradients use add_gradient_at (full Vec).
    if(model_active&&study_model_scalar&&x!=*model_center)
      throw std::runtime_error("ambient scalar auxiliary sparse gradient requested away from anchor");
    if(!model_active)refresh_native_hessian(x);
    note(&NewtonCalls::gradients);
    value=at(x);
    if(majorant_gradients_extended) {
      gradient_extended.resize(support.size());
      if(model_active) {
        note(&NewtonCalls::model_gradients);
        if(model_gradient_extended.size()==support.size())
          std::copy(model_gradient_extended.begin(),model_gradient_extended.end(),gradient_extended.begin());
        else std::copy(model_gradient.begin(),model_gradient.end(),gradient_extended.begin());
        for(std::size_t k=0;k<model_hessian.size();++k) {
          const auto e=model_hessian[k];
          gradient_extended[model_hi[k]]+=static_cast<long double>(e.v)*
              (static_cast<long double>(x[e.j])-(*model_center)[e.j]);
          if(e.i!=e.j)gradient_extended[model_hj[k]]+=static_cast<long double>(e.v)*
              (static_cast<long double>(x[e.i])-(*model_center)[e.i]);
        }
        for(const auto& atom:model_atoms) {
          long double displacement=0;
          if(model_atom_support.empty()) {
            for(std::size_t j=0;j<atom.direction.size();++j)
              displacement+=static_cast<long double>(atom.direction[j])*
                  (static_cast<long double>(x[j])-(*model_center)[j]);
          } else for(int j:model_atom_support)
            displacement+=static_cast<long double>(atom.direction[j])*
                (static_cast<long double>(x[j])-(*model_center)[j]);
          const long double weight=static_cast<long double>(atom.beta)*displacement;
          if(model_atom_support.empty()) {
            for(std::size_t k=0;k<support.size();++k)
              gradient_extended[k]+=weight*atom.direction[support[k]];
          } else for(int j:model_atom_support) {
            const auto k=std::size_t(j)<support.size()&&support[j]==j?std::size_t(j):
                std::size_t(std::lower_bound(support.begin(),support.end(),j)-support.begin());
            if(k==support.size()||support[k]!=j)throw std::runtime_error("DCA atom outside model support");
            gradient_extended[k]+=weight*atom.direction[j];
          }
        }
      } else {
        std::fill(gradient_extended.begin(),gradient_extended.end(),0.0L);
        for(std::size_t k=0;k<linear.size();++k)
          gradient_extended[linear_pos[k]]+=linear[k].second;
        for(std::size_t k=0;k<hessian.size();++k) {
          const auto e=hessian[k];
          gradient_extended[hi[k]]+=static_cast<long double>(e.v)*x[e.j];
          if(e.i!=e.j)gradient_extended[hj[k]]+=static_cast<long double>(e.v)*x[e.i];
        }
        for(auto& value:gradient_extended)value*=static_cast<long double>(scale);
      }
      for(std::size_t k=0;k<gradient.size();++k)
        gradient[k]=static_cast<double>(gradient_extended[k]);
      if(pooled_gradient)std::copy(gradient.begin(),gradient.end(),pooled_gradient);
      return;
    }
    if(model_active) {
      note(&NewtonCalls::model_gradients);gradient=model_gradient;
      for(std::size_t k=0;k<model_hessian.size();++k){auto e=model_hessian[k];gradient[model_hi[k]]+=e.v*(x[e.j]-(*model_center)[e.j]);if(e.i!=e.j)gradient[model_hj[k]]+=e.v*(x[e.i]-(*model_center)[e.i]);}
      for(const auto& a:model_atoms){
        const double d=model_atom_displacement(a,x);
        if(model_atom_support.empty()) {
          for(std::size_t k=0;k<support.size();++k)gradient[k]+=a.beta*d*a.direction[support[k]];
        } else for(int j:model_atom_support) {
          // The DCA objective reserves every coordinate, so its common case
          // uses identity positions. Retain the generic sparse support map.
          const auto k=std::size_t(j)<support.size()&&support[j]==j?std::size_t(j):
              std::size_t(std::lower_bound(support.begin(),support.end(),j)-support.begin());
          if(k==support.size()||support[k]!=j)throw std::runtime_error("DCA atom outside model support");
          gradient[k]+=a.beta*d*a.direction[j];
        }
      }
      if(pooled_gradient)std::copy(gradient.begin(),gradient.end(),pooled_gradient);
      return;
    }
    std::fill(gradient.begin(),gradient.end(),0.0);
    for(std::size_t i=0;i<linear.size();++i)gradient[linear_pos[i]]+=linear[i].second;
    for(std::size_t i=0;i<hessian.size();++i) {
      const auto e=hessian[i];gradient[hi[i]]+=e.v*x[e.j];
      if(e.i!=e.j)gradient[hj[i]]+=e.v*x[e.i];
    }
    for(double& v:gradient)v*=scale;
    if(pooled_gradient)std::copy(gradient.begin(),gradient.end(),pooled_gradient);
  }
  void add_gradient_at(const Vec& point,double weight,Vec& g) const {
    if(model_active) {
      note(&NewtonCalls::gradients);note(&NewtonCalls::model_gradients);
      for(std::size_t k=0;k<support.size();++k)g[support[k]]+=weight*model_gradient[k];
      if(study_model_scalar)for(std::size_t j=0;j<point.size();++j)g[j]+=weight*study_model_scalar*(point[j]-(*model_center)[j]);
      for(auto e:model_hessian){g[e.i]+=weight*e.v*(point[e.j]-(*model_center)[e.j]);if(e.i!=e.j)g[e.j]+=weight*e.v*(point[e.i]-(*model_center)[e.i]);}
      for(const auto& a:model_atoms){const double d=model_atom_displacement(a,point);add_model_atom(a,weight*a.beta*d,g);}
      return;
    }
    original_gradient_at(point,weight,g);
  }
  void original_gradient_at(const Vec& point,double weight,Vec& g) const {
    note(&NewtonCalls::gradients);
    const double a=weight*scale;
    for(auto e:linear)g[e.first]+=a*e.second;
    for(auto e:(signed_square.empty()?hessian:base_hessian)){g[e.i]+=a*e.v*point[e.j];if(e.i!=e.j)g[e.j]+=a*e.v*point[e.i];}
    for(auto e:signed_square)g[e.first]+=a*2*e.second*std::abs(point[e.first]);
  }
  void add_gradient(double a,Vec& g) const {note(&NewtonCalls::cached_gradient_accumulations);const int* columns=pooled_support?pooled_support:support.data();const double* values=pooled_gradient?pooled_gradient:gradient.data();for(std::size_t k=0;k<support.size();++k)g[columns[k]]+=a*values[k];}
  void add_gradient_extended(double weight,std::vector<long double>& destination) const {
    note(&NewtonCalls::cached_gradient_accumulations);
    const int* columns=pooled_support?pooled_support:support.data();
    for(std::size_t k=0;k<support.size();++k)
      destination[columns[k]]+=static_cast<long double>(weight)*gradient_extended[k];
  }
  double directional(const Vec& d) const {note(&NewtonCalls::jacobian_products);const int* columns=pooled_support?pooled_support:support.data();const double* values=pooled_gradient?pooled_gradient:gradient.data();double v=0;for(std::size_t k=0;k<support.size();++k)v+=values[k]*d[columns[k]];return v;}
};

bool certify_dense_original(const QCQPProblem& p,const Settings& st,Result& r) {
  ++r.lazy_newton.original_certificates;
  r.ipopt_desired_convergence=false;
  const auto& eq=r.certificate_equality_multipliers;const auto& in=r.certificate_inequality_multipliers;
  const auto& lo=r.certificate_lower_multipliers;const auto& up=r.certificate_upper_multipliers;
  r.dual_infeasibility_inf=r.complementarity_inf=r.scaled_nlp_error=std::numeric_limits<double>::infinity();
  if(r.x.size()!=(std::size_t)p.n||eq.size()!=p.equalities.size()||in.size()!=p.inequalities.size()||
     lo.size()!=r.x.size()||up.size()!=r.x.size()||!finite(r.x)||!finite(eq)||!finite(in)||!finite(lo)||!finite(up))return false;
  for(const Vec* v:{&in,&lo,&up})for(double a:*v)if(a<0.0)return false;
  ++r.lazy_newton.calls[3][0].gradients;
  Vec g=p.objective.gradient(r.x);double comp=0.0;
  for(std::size_t e=0;e<eq.size();++e){++r.lazy_newton.calls[3][1].gradients;axpy(eq[e],p.equalities[e].gradient(r.x),g);}
  for(std::size_t i=0;i<in.size();++i) {
    ++r.lazy_newton.calls[3][2].gradients;++r.lazy_newton.calls[3][2].values;
    axpy(in[i],p.inequalities[i].gradient(r.x),g);
    const double v=p.inequalities[i].value(r.x);if(!std::isfinite(v))return false;
    comp=std::max(comp,std::abs(in[i]*v));
  }
  for(int j=0;j<p.n;++j) {
    if(std::isfinite(p.lb[j])&&p.lb[j]==p.ub[j]){g[j]=0.0;continue;}
    g[j]+=up[j]-lo[j];
    if(std::isfinite(p.lb[j]))comp=std::max(comp,std::abs(lo[j]*(r.x[j]-p.lb[j])));else if(lo[j]!=0.0)return false;
    if(std::isfinite(p.ub[j]))comp=std::max(comp,std::abs(up[j]*(p.ub[j]-r.x[j])));else if(up[j]!=0.0)return false;
  }
  for(const auto& q:p.equalities){++r.lazy_newton.calls[3][1].values;if(!std::isfinite(q.value(r.x)))return false;}
  if(!finite(g))return false;
  ++r.lazy_newton.calls[3][0].values;r.lazy_newton.calls[3][1].values+=p.equalities.size();r.lazy_newton.calls[3][2].values+=p.inequalities.size();
  r.objective=p.objective.value(r.x);r.feasibility=feasibility(p,r.x);
  if(!std::isfinite(r.objective)||!std::isfinite(r.feasibility.total_inf))return false;
  r.dual_infeasibility_inf=inf_norm(g);r.complementarity_inf=comp;
  r.scaled_nlp_error=std::max({r.feasibility.total_inf,inf_norm(g),comp});
  r.model_stationarity_inf=inf_norm(g);r.termination_multiplier_source="lazy-newton-original-primal-dual-state";
  r.ipopt_desired_convergence=r.feasibility.total_inf<=st.feasibility_tolerance&&
      r.dual_infeasibility_inf<=st.dual_infeasibility_tolerance&&comp<=st.complementarity_tolerance&&
      r.scaled_nlp_error<=st.stationarity_tolerance;
  return r.ipopt_desired_convergence;
}

using OriginalLinearSupport=std::vector<std::vector<int>>;
bool certify_indexed_original(const QCQPProblem& p,const Settings& st,Result& r,
                              const OriginalLinearSupport* support) {
  ++r.lazy_newton.original_certificates;
  r.ipopt_desired_convergence=false;
  const auto& eq=r.certificate_equality_multipliers;const auto& in=r.certificate_inequality_multipliers;
  const auto& lo=r.certificate_lower_multipliers;const auto& up=r.certificate_upper_multipliers;
  r.dual_infeasibility_inf=r.complementarity_inf=r.scaled_nlp_error=std::numeric_limits<double>::infinity();
  if(r.x.size()!=(std::size_t)p.n||eq.size()!=p.equalities.size()||in.size()!=p.inequalities.size()||
     lo.size()!=r.x.size()||up.size()!=r.x.size()||!finite(r.x)||!finite(eq)||!finite(in)||!finite(lo)||!finite(up))return false;
  for(const Vec* v:{&in,&lo,&up})for(double a:*v)if(a<0.0)return false;
  Vec g(p.n,0.0),gradient(p.n,0.0);std::vector<unsigned char> seen(p.n,0);std::vector<int> touched;
  // The cache contains only indices of ORIGINAL linear coefficients, collected
  // in the already-required input scan. Values/Hessians are read from p here.
  // Each source gradient is accumulated before multiplication, like the dense
  // certificate; no transformed/convexified model or fitted dual is consulted.
  auto evaluate=[&](const QuadraticFunction& q,double weight,std::size_t id) {
    auto& calls=r.lazy_newton.calls[3][id==0?0:id<=eq.size()?1:2];++calls.values;++calls.gradients;
    auto add=[&](int j,double value){if(!seen[j]){seen[j]=1;touched.push_back(j);}gradient[j]+=value;};
    double quadratic=0.0,linear=0.0;
    for(auto e:q.hessian.entries) {
      add(e.i,e.v*r.x[e.j]);if(e.i!=e.j)add(e.j,e.v*r.x[e.i]);
      quadratic+=(e.i==e.j?e.v*r.x[e.i]*r.x[e.i]:2.0*e.v*r.x[e.i]*r.x[e.j]);
    }
    auto coefficient=[&](int j){linear+=q.linear[j]*r.x[j];add(j,q.linear[j]);};
    if(q.has_compact_linear())for(auto e:q.compact_linear){linear+=e.second*r.x[e.first];add(e.first,e.second);}
    else if(support)for(int j:(*support)[id])coefficient(j);
    else for(int j=0;j<p.n;++j)if(q.linear[j]!=0.0)coefficient(j);
    for(int j:touched){g[j]+=weight*gradient[j];gradient[j]=0.0;seen[j]=0;}touched.clear();
    return q.constant+linear+0.5*quadratic;
  };
  double objective=evaluate(p.objective,1.0,0),comp=0.0;Feasibility f;
  if(!std::isfinite(objective))return false;
  for(std::size_t e=0;e<eq.size();++e) {
    const double v=evaluate(p.equalities[e],eq[e],1+e);if(!std::isfinite(v))return false;
    f.equality_inf=std::max(f.equality_inf,std::abs(v));
  }
  for(std::size_t i=0;i<in.size();++i) {
    const double v=evaluate(p.inequalities[i],in[i],1+eq.size()+i);if(!std::isfinite(v))return false;
    f.inequality_inf=std::max(f.inequality_inf,std::max(0.0,v));comp=std::max(comp,std::abs(in[i]*v));
  }
  for(int j=0;j<p.n;++j) {
    if(std::isfinite(p.lb[j]))f.bound_inf=std::max(f.bound_inf,std::max(0.0,p.lb[j]-r.x[j]));
    if(std::isfinite(p.ub[j]))f.bound_inf=std::max(f.bound_inf,std::max(0.0,r.x[j]-p.ub[j]));
    if(std::isfinite(p.lb[j])&&p.lb[j]==p.ub[j]){g[j]=0.0;continue;}
    g[j]+=up[j]-lo[j];
    if(std::isfinite(p.lb[j]))comp=std::max(comp,std::abs(lo[j]*(r.x[j]-p.lb[j])));else if(lo[j]!=0.0)return false;
    if(std::isfinite(p.ub[j]))comp=std::max(comp,std::abs(up[j]*(p.ub[j]-r.x[j])));else if(up[j]!=0.0)return false;
  }
  if(!finite(g))return false;
  f.total_inf=std::max({f.equality_inf,f.inequality_inf,f.bound_inf});
  r.objective=objective;r.feasibility=f;
  if(!std::isfinite(f.total_inf))return false;
  r.dual_infeasibility_inf=inf_norm(g);r.complementarity_inf=comp;
  r.scaled_nlp_error=std::max({f.total_inf,inf_norm(g),comp});
  r.model_stationarity_inf=inf_norm(g);r.termination_multiplier_source="lazy-newton-original-primal-dual-state";
  r.ipopt_desired_convergence=f.total_inf<=st.feasibility_tolerance&&
      r.dual_infeasibility_inf<=st.dual_infeasibility_tolerance&&comp<=st.complementarity_tolerance&&
      r.scaled_nlp_error<=st.stationarity_tolerance;
  return r.ipopt_desired_convergence;
}

struct JacobianMap {int gradient,t,output;};
struct ReducedGradient {
  std::vector<int> support;
  Vec values;
  std::vector<JacobianMap> map;
  struct Gram {int a,b,k;};
  std::vector<Gram> gram;
  std::vector<int> border;
  int dense=-1;
};
struct HessianMap {int h,a,b,k;double multiplier;};
struct Direction {Vec x,lambda,s,z;double residual=0.0;};
// Complete remains the default contract, including validate=false and raw
// depth>0 calls. Reduced results require an explicit internal request.
enum class DirectionPurpose { Complete, ResidualCorrection };

#include "lazy_newton_atoms.hpp"

class Workspace {
public:
  const QCQPProblem& original;
  const Settings& st;
  LazyNewtonStatistics& stats;
  DcaFilterStatistics* dca_stats=nullptr;
  detail::NewtonCostLedger& costs;
  std::vector<int> index,free,retained,to_retained,pivot_variable,pivot_equation,eq_retained;
  Vec fixed,pivot_coefficient;
  int n=0,nr=0,me=0,mi=0,mr=0,ndense=0,dim=0;
  OriginalLinearSupport original_linear_support;
  std::vector<int> pooled_columns;
  Vec pooled_values;
  Quadratic objective;
  // Refreshed from the actual installed base and atom borders. A positive
  // margin is either exact Gershgorin or a proof of A-gamma*I PSD.
  double hybrid_objective_base_lower=0;
  std::shared_ptr<const study::ObjectiveMarginCertificate> hybrid_objective_margin_certificate;
  int hybrid_objective_margin_serial=0;
  std::vector<Quadratic> equalities,inequalities;
  // kind=0: original inequality; -1/+1: lower/upper bound.
  std::vector<int> inequality_kind,inequality_origin;
  std::vector<unsigned char> active;
  std::vector<int> tbegin,tcolumn;
  Vec tvalue;
  std::vector<std::pair<int,int>> tgradient;
  std::vector<ReducedGradient> jr,ar;
  std::vector<Pair> hpattern;
  Vec hvalue;
  std::vector<HessianMap> hmap;
  std::vector<SymmetricTriplet> kkt;
  std::vector<int> diagonal,dense_diagonal,equality_diagonal;
  Vec base_values;
  std::size_t atom_border_begin=0;
  bool has_model_borders=false;
  std::size_t hybrid_border_assemblies=0;
  int hybrid_border_objective_atoms=0,hybrid_border_inequality_atoms=0;
  int hybrid_last_reported_objective=-1,hybrid_last_reported_inequalities=-1;
  bool hybrid_initial_model_startup_consumed=false;

  bool dca_active_evaluation=false;
  void strip_model_borders(){if(has_model_borders){kkt.resize(atom_border_begin);base_values.resize(atom_border_begin);dim=nr+mr+ndense;has_model_borders=false;}}

  detail::AutoStartupDecision auto_startup;
  Vec x,lambda,z,s,rd,rp,h;
  detail::MajorantEqualityStabilization equality_stabilization;
  bool source_primal_barrier_mode=false;
  double source_barrier_mu=0;
  bool source_primal_barrier(int i) const {
    return source_primal_barrier_mode&&equality_stabilization.enabled&&inequality_kind[i]==0;
  }
  double centered_complementarity_rhs(int i,double mu) const {
    return detail::source_centered_complementarity_rhs(source_primal_barrier(i),s[i],z[i],mu);
  }
  bool requires_positive_dual(int i) const {return !source_primal_barrier(i);}
  const char* source_stabilization_variant() const {
    return source_primal_barrier_mode?detail::MajorantEqualityStabilization::primal_barrier_variant:
        detail::MajorantEqualityStabilization::variant;
  }
  void set_barrier_mu(double mu) {
    // An equality-only workspace has no slack or bound barrier terms.
    // Its exact central-path parameter is zero; retain strict positivity
    // whenever any effective source inequality or free-variable bound exists.
    if(source_primal_barrier_mode&&(!std::isfinite(mu)||mu<0||(mi>0&&mu==0)))
      throw std::invalid_argument("source primal barrier requires finite positive mu");
    source_barrier_mu=mu;
  }
  double source_stationarity_inf(double mu) const {
    double value=0;
    for(int i=0;i<mi;++i)if(active[i]&&source_primal_barrier(i)) {
      if(!(s[i]>0)||!std::isfinite(s[i])||!std::isfinite(z[i]))return INFINITY;
      value=std::max(value,static_cast<double>(std::abs(static_cast<long double>(z[i])-
          static_cast<long double>(mu)/s[i])));
    }
    return value;
  }
  long double source_barrier_denominator(int i) const {
    return static_cast<long double>(s[i])*s[i]+
        static_cast<long double>(inequality_algorithmic_rho(i))*source_barrier_mu;
  }
  long double slack_newton_coefficient(int i) const {
    return source_primal_barrier(i)?static_cast<long double>(source_barrier_mu)/s[i]:
        static_cast<long double>(z[i]);
  }
  double balance_source_slack_direction(long double a,long double b,int i,double old) const {
    const long double linear=b/s[i];
    const long double weight=static_cast<long double>(source_barrier_mu)/s[i]/s[i];
    auto error=[&](double value) {
      return std::max(std::abs(a+value),std::abs(linear+weight*value));
    };
    double best=old;long double current=error(old);
    const double balanced=static_cast<double>(-(a+linear)/(1.L+weight));
    for(double value:{balanced,std::nextafter(balanced,-INFINITY),std::nextafter(balanced,INFINITY)})
      if(std::isfinite(value)&&error(value)<current){best=value;current=error(value);}
    return best;
  }
  double source_stationarity_direction_backward_error(const Direction& d,const Vec& rc) const {
    long double worst=0;
    for(int i=0;i<mi;++i)if(active[i]&&source_primal_barrier(i)) {
      detail::AccurateProductSum equation(rc[i]);
      equation.add_product(s[i],d.z[i]);
      equation.add(slack_newton_coefficient(i)*d.s[i]);
      const long double error=std::abs(equation.value()/s[i]);
      const long double hessian_step=slack_newton_coefficient(i)*d.s[i]/s[i];
      const long double scale=1+std::abs(static_cast<long double>(rc[i])/s[i])+
          std::abs(hessian_step)+std::abs(static_cast<long double>(d.z[i]));
      if(!std::isfinite(error)||!std::isfinite(scale)||!(scale>0))return INFINITY;
      worst=std::max(worst,error/scale);
    }
    return static_cast<double>(worst);
  }
  Vec equality_newton_residual() const {
    return equality_stabilization.residual(h,lambda);
  }
  double inequality_algorithmic_rho(int i) const {
    return equality_stabilization.enabled&&inequality_kind[i]==0?equality_stabilization.rho:0.0;
  }
  Vec primal_newton_residual() const {
    return equality_stabilization.primal_residual(rp,z,inequality_kind);
  }
  long double stabilized_slack_denominator(int i) const {
    return static_cast<long double>(s[i])+static_cast<long double>(inequality_algorithmic_rho(i))*z[i];
  }
  double inequality_gram_weight(int i) const {
    if(source_primal_barrier(i))return static_cast<double>(
        static_cast<long double>(source_barrier_mu)/source_barrier_denominator(i));
    return inequality_algorithmic_rho(i)>0?
        static_cast<double>(static_cast<long double>(z[i])/stabilized_slack_denominator(i)):z[i]/s[i];
  }
  long double inequality_gram_weight_extended(int i) const {
    if(source_primal_barrier(i))return static_cast<long double>(source_barrier_mu)/source_barrier_denominator(i);
    return inequality_algorithmic_rho(i)>0?
        static_cast<long double>(z[i])/stabilized_slack_denominator(i):static_cast<long double>(z[i])/s[i];
  }
  double true_source_constraint_residual() const {
    double value=0;
    for(int e=0;e<me;++e)value=std::max(value,std::abs(h[e])/equalities[e].scale);
    for(int i=0;i<mi;++i)if(active[i]&&inequality_kind[i]==0)
      value=std::max(value,std::abs(rp[i])/inequalities[i].scale);
    return value;
  }
  double stabilized_barrier_residual(double mu) const {
    double value=std::max(inf_norm(rd),inf_norm(equality_newton_residual()));
    const Vec fp=primal_newton_residual();
    for(int i=0;i<mi;++i)if(active[i])
      value=std::max({value,std::abs(fp[i]),std::abs(s[i]*z[i]-mu)});
    return std::max(value,source_stationarity_inf(mu));
  }
  std::vector<long double> majorant_rd_extended;
  std::vector<detail::AccurateProductSum> majorant_direction_wide;
  // All factor objects borrow their settings; these objects must outlive them.
  Settings factor_settings,pivoted_settings,recovery_settings;
  std::unique_ptr<SymmetricFactorization> factor,pivoted,recovery,pivot_rescue;
  SymmetricFactorization* current_factor=nullptr;
  double regularization=0.0,penalty=1.0;
  int recovery_equation=-1;
  double recovery_scale=1.0,recovery_shift=0.0;
  Vec recovery_normal;
  std::string message;
  bool condensation_scanned=false,precision_condensation_scanned=false,precision_quality_used=false;
  std::unique_ptr<NewtonAtoms> atoms;
  bool atom_refresh_pending=false;
  std::vector<AtomWitness> atom_guides;int atom_guide_iteration=-1;
  int atom_defect_iteration=-1,atom_last_defect=-1,atom_stagnant_batches=0;
  int atom_defect_factor=-1;
  bool atom_adaptation_active=false;
  std::vector<std::vector<std::pair<int,int>>> atom_rows,atom_dense_rows;
  std::size_t atom_mapped_slots=0;
  Vec atom_probe_values,atom_dense_dot;
  std::vector<unsigned char> atom_dense_seen;
  std::vector<int> atom_dense_used;
  std::pair<double,double> atom_rayleigh(const NewtonAtom& a) {
    if(atom_rows.empty()){atom_rows.resize(nr);atom_dense_rows.resize(nr);atom_probe_values.assign(nr,0.0);}
    for(;atom_mapped_slots<kkt.size();++atom_mapped_slots) {
      const auto& e=kkt[atom_mapped_slots];
      if(e.i<nr&&e.j<nr) {
        atom_rows[e.i].push_back({e.j,(int)atom_mapped_slots});
        if(e.i!=e.j)atom_rows[e.j].push_back({e.i,(int)atom_mapped_slots});
      } else if(e.i<nr&&e.j>=nr+mr)atom_dense_rows[e.i].push_back({e.j-nr-mr,(int)atom_mapped_slots});
    }
    atom_dense_dot.resize(ndense,0.0);atom_dense_seen.resize(ndense,0);atom_dense_used.clear();
    for(std::size_t i=0;i<a.support.size();++i)atom_probe_values[a.support[i]]=a.values[i];
    double quadratic=0.0,magnitude=0.0;
    for(std::size_t i=0;i<a.support.size();++i) {
      const int j=a.support[i];const double u=a.values[i];
      for(auto e:atom_rows[j]) {const double v=u*kkt[e.second].value*atom_probe_values[e.first];quadratic+=v;magnitude+=std::abs(v);}
      for(auto e:atom_dense_rows[j]) {
        if(!atom_dense_seen[e.first]){atom_dense_seen[e.first]=1;atom_dense_used.push_back(e.first);}
        atom_dense_dot[e.first]+=u*kkt[e.second].value;
      }
    }
    // Retained inequality-dual borders represent PSD barrier terms. They
    // must be included in a primal Rayleigh probe, although not condensed.
    for(int i:atom_dense_used) {
      const double v=-atom_dense_dot[i]*atom_dense_dot[i]/kkt[dense_diagonal[i]].value;
      quadratic+=v;magnitude+=std::abs(v);atom_dense_dot[i]=0.0;atom_dense_seen[i]=0;
    }
    for(int j:a.support)atom_probe_values[j]=0.0;
    const double norm=dot(a.values,a.values);
    return {quadratic/norm,64*std::numeric_limits<double>::epsilon()*std::max(1.0,magnitude/norm)};
  }
  double hybrid_objective_margin(double tau) const {
    const bool hybrid=st.convexification_study=="HH"||st.convexification_study=="HS"||
                      st.convexification_study=="SH"||st.convexification_study=="SS"||st.convexification_study=="H0";
    if(!hybrid||!objective.model_active||tau<0||!std::isfinite(tau))return 0;
    // Convexification-model certificates precede this call. KKT inertia,
    // barrier curvature, inequality multipliers and backend shifts are not
    // used as substitutes for the certified objective-model PSD property.
    for(int j=0;j<nr;++j) {
      const int k=retained[j];
      if(tbegin[k+1]-tbegin[k]!=1||tcolumn[tbegin[k]]!=j||tvalue[tbegin[k]]!=1.0)
        return 0;
    }
    // Identity rows at retained coordinates give ||T d|| >= ||d||.
    // Hence a certified ambient base lower bound b and the retained-coordinate
    // proximal diagonal jointly certify (b+tau)*I in reduced coordinates.
    return study::certified_objective_margin_sum(hybrid_objective_base_lower,tau);
  }
  double atom_strength() const {return atoms?atoms->trace():0.0;}
  bool corrected() const {
    return equality_stabilization.enabled||regularization>0.0||(atoms&&atoms->active_count()>0)||
      (current_factor&&(current_factor->last_primal_regularization()>0.0||
                       current_factor->last_dual_regularization()>0.0));
  }
  bool pivoted_sparse_fallback_enabled() const {
    return (st.dca_engine=="filter"||st.dca_engine=="majorants")&&st.dca_filter_sparse_fallback&&
      (st.linear_backend=="ldl"||st.linear_backend=="cholesk");
  }

  Workspace(const QCQPProblem& p,const Settings& settings,LazyNewtonStatistics& diagnostics,detail::NewtonCostLedger& cost_ledger)
      :original(p),st(settings),stats(diagnostics),costs(cost_ledger),index(p.n,-1),fixed(p.x0),factor_settings(settings) {
    equality_stabilization.enabled=st.majorant_equality_stabilization;
    source_primal_barrier_mode=detail::original_source_primal_barrier_requested(st);
    if(equality_stabilization.enabled&&(st.dca_engine!="majorants"||st.dca_pair_elimination))
      throw std::invalid_argument("algorithmic equality stabilization requires native majorants and no equality pair elimination");
    if(st.dca_pair_elimination&&(st.convexification_study=="HH"||
       st.convexification_study=="HS"||st.convexification_study=="SH"||st.convexification_study=="SS"||
       st.convexification_study=="H0"))
      throw std::invalid_argument("hybrid bordered models require dca_pair_elimination=false; primary affine elimination remains enabled");
    dca_active_evaluation=(st.dca_engine=="filter"||st.dca_engine=="majorants")&&
        (st.outer_algorithm=="lazy-dca"||st.outer_algorithm=="auto")&&
        st.dca_filter_active_evaluation;
    // An unbounded coordinate absent from every original function has zero
    // primal residual and zero stationarity for every value. Keep its supplied
    // value when lifting, but do not put its exact null row into a Newton KKT.
    // This is coefficient-exact elimination, not a regularization or a bound.
    std::vector<unsigned char> original_coordinate_used(p.n,0);
    auto mark_original_support=[&](const QuadraticFunction& q) {
      q.for_each_linear_nonzero([&](int j,double value) {
        if(value!=0.0)original_coordinate_used[j]=1;
      });
      for(const auto& entry:q.hessian.entries)if(entry.v!=0.0) {
        original_coordinate_used[entry.i]=1;original_coordinate_used[entry.j]=1;
      }
      for(const auto& entry:q.signed_square)if(entry.second!=0.0)
        original_coordinate_used[entry.first]=1;
    };
    mark_original_support(p.objective);
    for(const auto& q:p.equalities)mark_original_support(q);
    for(const auto& q:p.inequalities)mark_original_support(q);
    int unused_free_coordinates=0;
    for(int j=0;j<p.n;++j) {
      if(std::isfinite(p.lb[j])&&p.lb[j]==p.ub[j])fixed[j]=p.lb[j];
      else if(!original_coordinate_used[j]&&!std::isfinite(p.lb[j])&&
              !std::isfinite(p.ub[j]))++unused_free_coordinates;
      else {index[j]=(int)free.size();free.push_back(j);}
    }
    // Preserve the pre-existing handling of wholly constant models.
    if(free.empty()&&unused_free_coordinates>0)for(int j=0;j<p.n;++j)
      if(!original_coordinate_used[j]&&!std::isfinite(p.lb[j])&&!std::isfinite(p.ub[j])) {
        index[j]=0;free.push_back(j);--unused_free_coordinates;break;
      }
    if(unused_free_coordinates>0&&(st.verbose||st.live_display_level>0))
      std::cout<<"[EXACT-FREE-COORDINATES] removed="<<unused_free_coordinates
        <<" original_dimension="<<p.n<<" retained_free="<<free.size()
        <<" supplied_values_preserved=1 original_KKT_unchanged=1"<<std::endl;
    n=(int)free.size();if(n==0)throw std::runtime_error("lazy-newton: no free coordinate");
    x.resize(n);for(int j=0;j<n;++j)x[j]=p.x0[free[j]];
    if(!finite(x))throw std::runtime_error("lazy-newton: nonfinite initial point");
    if(st.lazy_newton_sparse_certificate)original_linear_support.resize(1+p.equalities.size()+p.inequalities.size());
    auto original_support=[&](std::size_t i)->std::vector<int>* {return st.lazy_newton_sparse_certificate?&original_linear_support[i]:nullptr;};
    objective=Quadratic(p.objective,index,fixed,original_support(0));objective.ledger=&costs;
    objective.majorant_values_extended=st.dca_engine=="majorants";
    objective.majorant_gradients_extended=st.dca_engine=="majorants";
    if((st.dca_engine=="filter"||st.dca_engine=="majorants")){
      // Reserve diagonal curvature and the signed equality Hessian support.
      // Structural zeros change no original coefficient or certificate.
      for(int j=0;j<n;++j){objective.linear.push_back({j,0});objective.hessian.push_back({j,j,0});}
      if(!objective.signed_square.empty())objective.base_hessian=objective.hessian;
      objective.compile();
    }
    objective.evaluate(x);
    objective.scale=std::min(1.0,st.lazy_newton_objective_gradient_target/std::max(1.0,inf_norm(objective.gradient)));
    for(std::size_t e=0;e<p.equalities.size();++e)equalities.emplace_back(p.equalities[e],index,fixed,original_support(1+e));
    for(int i=0;i<(int)p.inequalities.size();++i) {
      inequalities.emplace_back(p.inequalities[i],index,fixed,original_support(1+p.equalities.size()+i));inequality_kind.push_back(0);inequality_origin.push_back(i);
    }
    for(int j=0;j<n;++j)for(int side:{-1,1}) {
      const double b=side<0?p.lb[free[j]]:p.ub[free[j]];
      if(!std::isfinite(b))continue;
      Quadratic q;q.constant=-side*b;q.linear.push_back({j,(double)side});q.compile();
      inequalities.push_back(std::move(q));inequality_kind.push_back(side);inequality_origin.push_back(free[j]);
    }
    for(std::size_t i=0;i<equalities.size();++i){equalities[i].ledger=&costs;equalities[i].family=1;}
    for(std::size_t i=0;i<inequalities.size();++i){inequalities[i].ledger=&costs;inequalities[i].family=inequality_kind[i]==0?2:3;}
    std::size_t homogeneous_fill_slots=0,homogeneous_extra_slots=0;
    int homogeneous_filled_rows=0;
    for(auto* family:{&equalities,&inequalities})for(auto& q:*family) {
      q.majorant_values_extended=st.dca_engine=="majorants";
      q.majorant_gradients_extended=st.dca_engine=="majorants";
      if((st.dca_engine=="filter"||st.dca_engine=="majorants")&&!q.hessian.empty()){
        std::set<int> diagonal;for(auto e:q.hessian){diagonal.insert(e.i);diagonal.insert(e.j);}
        for(int j:diagonal)q.hessian.push_back({j,j,0});
        // A small homogeneous quadratic may have a singular initial normal
        // and sparse indefinite curvature. Reserve its complete small block
        // so the existing certified positive-part majorant is representable.
        // These zeros affect assembly support only, never original data.
        const bool homogeneous=q.constant==0.&&q.signed_square.empty()&&
            std::all_of(q.linear.begin(),q.linear.end(),[](const auto& a){return a.second==0.;});
        const std::size_t block=diagonal.size()*(diagonal.size()+1)/2;
        if(family==&inequalities&&st.dca_engine=="majorants"&&
            st.convexification_study=="off"&&homogeneous&&diagonal.size()<=16&&
            homogeneous_fill_slots+block<=200000) {
          const auto previous_size=q.hessian.size();
          std::set<Pair> present;for(const auto& e:q.hessian)present.insert(upper(e.i,e.j));
          for(int j:diagonal)for(int k:diagonal)if(j<=k&&!present.count({j,k}))
            q.hessian.push_back({j,k,0.});
          homogeneous_fill_slots+=block;
          homogeneous_extra_slots+=q.hessian.size()-previous_size;
          homogeneous_filled_rows+=q.hessian.size()>previous_size;
        }
        if(!q.signed_square.empty())q.base_hessian=q.hessian;
        q.compile();
      }
      q.evaluate(x);q.scale=std::min(1.0,st.lazy_newton_constraint_gradient_target/std::max(1.0,inf_norm(q.gradient)));
    }
    if(homogeneous_filled_rows&&(st.verbose||st.live_display_level>0))
      std::cout<<"[HOMOGENEOUS-SPECTRAL-ENVELOPE] rows="<<homogeneous_filled_rows
          <<" extra_slots="<<homogeneous_extra_slots
          <<" original_coefficients_unchanged=1 maximum_support=16 slot_budget=200000"<<std::endl;
    if(st.lazy_newton_packed_gradients) {
    std::size_t pooled_size=objective.support.size();
    for(const auto* family:{&equalities,&inequalities})for(const auto& q:*family)pooled_size+=q.support.size();
    pooled_columns.resize(pooled_size);pooled_values.resize(pooled_size);
    std::size_t pooled_offset=0;
    auto bind=[&](Quadratic& q){
      if(q.support.empty())return;
      std::copy(q.support.begin(),q.support.end(),pooled_columns.begin()+pooled_offset);
      std::copy(q.gradient.begin(),q.gradient.end(),pooled_values.begin()+pooled_offset);
      q.pooled_support=pooled_columns.data()+pooled_offset;
      q.pooled_gradient=pooled_values.data()+pooled_offset;
      pooled_offset+=q.support.size();
    };
    bind(objective);for(auto* family:{&equalities,&inequalities})for(auto& q:*family)bind(q);
    }
    me=(int)equalities.size();mi=(int)inequalities.size();
    // Generic independent eliminations: a pivot occurs linearly in exactly
    // one equality, and in no equality Hessian. No names or case sizes enter.
    std::vector<int> count(n,0);std::vector<unsigned char> nonlinear(n,0);
    for(const auto& q:equalities) {
      for(auto e:q.linear)++count[e.first];
      for(auto e:q.hessian){nonlinear[e.i]=1;nonlinear[e.j]=1;}
    }
    pivot_variable.assign(me,-1);pivot_equation.assign(n,-1);pivot_coefficient.assign(me,0.0);
    if(st.lazy_newton_reduction&&!equality_stabilization.enabled)for(int e=0;e<me;++e) {
      int best=-1;double coefficient=0.0;
      for(auto a:equalities[e].linear)if(count[a.first]==1&&!nonlinear[a.first]&&std::abs(a.second)>std::abs(coefficient)) {
        best=a.first;coefficient=a.second;
      }
      // An extremely weak pivot is left in the unreduced KKT.
      if(best>=0&&std::abs(coefficient)>=1e-10) {
        pivot_variable[e]=best;pivot_equation[best]=e;pivot_coefficient[e]=coefficient*equalities[e].scale;
      }
    }
    if(st.lazy_newton_reduction&&!equality_stabilization.enabled) {
      // Cost the CONTRIBUTION map T' H T before allocating a single reduced
      // row. Affine factor lifts may have dense substitution rows even though
      // their original Hessians and unreduced KKT remain sparse.
      std::vector<std::size_t> widths(n,1);
      for(int e=0;e<me;++e)if(pivot_variable[e]>=0)
        widths[pivot_variable[e]]=equalities[e].support.size()-1;
      std::set<Pair> structural;
      auto include_structure=[&](const Quadratic& q) {
        for(auto h:q.hessian)structural.insert(upper(h.i,h.j));
      };
      include_structure(objective);
      for(const auto* family:{&equalities,&inequalities})for(const auto& q:*family)include_structure(q);
      if(st.convexification_study!="off") {
        for(int j=0;j<n;++j)structural.insert({j,j});
        auto include_study=[&](const Quadratic& q) {
          std::set<int> support;
          for(auto h:q.hessian)if(h.v!=0){support.insert(h.i);support.insert(h.j);}
          if(support.size()<=16)for(int i:support)for(int j:support)if(i<=j)
            structural.insert({i,j});
        };
        include_study(objective);
        for(int i=0;i<mi;++i)if(inequality_kind[i]==0)include_study(inequalities[i]);
      }
      const std::vector<Pair> pattern(structural.begin(),structural.end());
      const auto budget=detail::bounded_affine_elimination(widths,pattern,sizeof(HessianMap));
      for(int e=0;e<me;++e) {
        const int j=pivot_variable[e];
        if(j>=0&&budget.chosen_width[j]!=widths[j]) {
          pivot_variable[e]=-1;pivot_equation[j]=-1;pivot_coefficient[e]=0;
        }
      }
      if(budget.refused&&st.live_display_level>=1)
        std::cout<<"[AFFINE-REDUCTION-BUDGET] refused="<<budget.refused
          <<" proposed_map_entries="<<budget.proposed_entries
          <<" admitted_map_entries="<<budget.admitted_entries
          <<" proposed_map_bytes="<<budget.proposed_bytes
          <<" admitted_map_bytes="<<budget.admitted_bytes
          <<" map_budget_bytes="<<budget.budget_bytes
          <<" arithmetic_saturated="<<budget.arithmetic_saturated
          <<" rejected_pivots_retained=1 original_equalities_preserved=1"<<std::endl;
    }
    to_retained.assign(n,-1);
    for(int j=0;j<n;++j)if(pivot_equation[j]<0){to_retained[j]=(int)retained.size();retained.push_back(j);}
    if(retained.empty()) {
      // Leave one scalar equality in the bordered solve when all free
      // coordinates are independently determined by linear equalities.
      int j=n-1,e=pivot_equation[j];pivot_variable[e]=-1;pivot_equation[j]=-1;
      pivot_coefficient[e]=0.0;to_retained[j]=0;retained.push_back(j);
    }
    nr=(int)retained.size();
    for(int e=0;e<me;++e)if(pivot_variable[e]<0) {
      if(equalities[e].support.empty()) {
        if(equalities[e].constant!=0.0)throw std::runtime_error("lazy-newton: inconsistent constant equality");
        // Stabilization retains one multiplier per original equation. A row
        // that becomes identically zero after fixed-coordinate substitution
        // still has its -rho dual diagonal and must keep that multiplier slot.
        if(equality_stabilization.enabled)eq_retained.push_back(e);
      } else eq_retained.push_back(e);
    }
    mr=(int)eq_retained.size();
    // Counting rows is not a rank test. Retain ALL original equations in
    // every route, including redundant and zero-Jacobian quadratic rows.
    // The existing audited dual-regularized KKT or original Phase I must
    // establish a usable step; no equation is silently deleted here.
    for(int j=0;j<n;++j) {
      tbegin.push_back((int)tcolumn.size());
      if(to_retained[j]>=0){tcolumn.push_back(to_retained[j]);tvalue.push_back(1.0);tgradient.push_back({-1,-1});}
      else {
        int e=pivot_equation[j];const auto& q=equalities[e];
        for(int k=0;k<(int)q.support.size();++k)if(q.support[k]!=j) {
          int r=to_retained[q.support[k]];
          if(r<0)throw std::runtime_error("lazy-newton: dependent elimination pivots");
          tcolumn.push_back(r);tvalue.push_back(0.0);tgradient.push_back({e,k});
        }
      }
    }
    tbegin.push_back((int)tcolumn.size());
    auto compile_gradient=[&](const Quadratic& q) {
      ReducedGradient g;std::set<int> cols;
      for(int j:q.support)for(int t=tbegin[j];t<tbegin[j+1];++t)cols.insert(tcolumn[t]);
      g.support.assign(cols.begin(),cols.end());g.values.resize(cols.size());
      for(int a=0;a<(int)q.support.size();++a)for(int t=tbegin[q.support[a]];t<tbegin[q.support[a]+1];++t) {
        int b=(int)(std::lower_bound(g.support.begin(),g.support.end(),tcolumn[t])-g.support.begin());
        g.map.push_back({a,t,b});
      }
      return g;
    };
    for(auto& q:inequalities)jr.push_back(compile_gradient(q));
    for(int e:eq_retained)ar.push_back(compile_gradient(equalities[e]));
    for(auto& g:jr)if((int)g.support.size()>=st.lazy_newton_dense_threshold)g.dense=ndense++;
    dim=nr+mr+ndense;
    std::map<Pair,int> hp;
    auto insert_h=[&](Quadratic& q) {
      for(auto e:q.hessian) {
        auto key=upper(e.i,e.j);auto it=hp.find(key);
        if(it==hp.end()){int v=(int)hpattern.size();hp[key]=v;hpattern.push_back(key);q.hglobal.push_back(v);}
        else q.hglobal.push_back(it->second);
      }
    };
    insert_h(objective);for(auto& q:equalities)insert_h(q);for(auto& q:inequalities)insert_h(q);
    if(st.convexification_study!="off") {
      auto reserve_study=[&](const Quadratic& q) {
        std::set<int> support;for(auto e:q.hessian)if(e.v!=0){support.insert(e.i);support.insert(e.j);}
        if(support.size()<=16)for(int i:support)for(int j:support)if(i<=j) {
          const auto key=upper(i,j);if(!hp.count(key)){hp[key]=hpattern.size();hpattern.push_back(key);}
        }
      };
      reserve_study(objective);for(int i=0;i<mi;++i)if(inequality_kind[i]==0)reserve_study(inequalities[i]);
      for(int j=0;j<n;++j)if(!hp.count({j,j})){hp[{j,j}]=hpattern.size();hpattern.push_back({j,j});}
    }
    // Native bounded dense inequality spectra may contain zero coefficients
    // in an otherwise dense block. Reserve those model slots once, before
    // compiling the symbolic Newton envelope; original functions stay exact.
    if(st.convexification_study=="off" && (st.dca_engine=="majorants"||st.dca_engine=="filter")) {
      constexpr std::size_t cubic_cap=536870912, entries_cap=2000000, fill_cap=131072;
      std::size_t cubic_work=0,model_entries=0,added=0;
      for(const auto& q:inequalities) {
        if(!q.signed_square.empty()||q.hessian.empty())continue;
        std::set<int> support;for(const auto& e:q.hessian){support.insert(e.i);support.insert(e.j);}
        const std::size_t count=support.size();
        if(count<=64||count>256||q.hessian.size()<count*(count+1)/4)continue;
        const std::size_t cubic=count*count*count,entries=count*(count+1)/2;
        if(cubic>cubic_cap-cubic_work||entries>entries_cap-model_entries)continue;
        std::vector<Pair> missing;
        for(int i:support)for(int j:support)if(i<=j&&!hp.count(upper(i,j)))missing.push_back(upper(i,j));
        if(missing.size()>fill_cap-added)continue;
        cubic_work+=cubic;model_entries+=entries;added+=missing.size();
        for(const auto& key:missing){hp[key]=hpattern.size();hpattern.push_back(key);}
      }
      if(added&&(st.verbose||st.live_display_level>0))
        std::cout<<"[BOUNDED-DENSE-ENVELOPE] added_slots="<<added<<" added_slots_cap="<<fill_cap
            <<" model_entries="<<model_entries<<" model_entries_cap="<<entries_cap
            <<" original_coefficients_unchanged=1"<<std::endl;
    }
    hvalue.resize(hpattern.size());
    std::map<Pair,int> kp;
    auto slot=[&](int i,int j) {
      auto key=upper(i,j);auto it=kp.find(key);if(it!=kp.end())return it->second;
      int v=(int)kkt.size();kp[key]=v;kkt.push_back({key.first,key.second,0.0});return v;
    };
    for(int i=0;i<nr;++i)diagonal.push_back(slot(i,i));
    // The explicit algorithmic -rho is distinct from backend rescue shifts.
    // Legacy/default equality diagonals remain exactly zero.
    for(int a=0;a<mr;++a)equality_diagonal.push_back(slot(nr+a,nr+a));
    for(int k=0;k<(int)hpattern.size();++k) {
      auto [i,j]=hpattern[k];
      for(int a=tbegin[i];a<tbegin[i+1];++a)for(int b=tbegin[j];b<tbegin[j+1];++b) {
        if(i==j&&a>b)continue;
        double c=i!=j&&tcolumn[a]==tcolumn[b]?2.0:1.0;
        hmap.push_back({k,a,b,slot(tcolumn[a],tcolumn[b]),c});
      }
    }
    for(auto& g:jr) {
      if(g.dense>=0) {
        for(int j:g.support)g.border.push_back(slot(j,nr+mr+g.dense));
        dense_diagonal.push_back(slot(nr+mr+g.dense,nr+mr+g.dense));
      } else for(int a=0;a<(int)g.support.size();++a)for(int b=a;b<(int)g.support.size();++b)
        g.gram.push_back({a,b,slot(g.support[a],g.support[b])});
    }
    for(int e=0;e<mr;++e)for(int j:ar[e].support)ar[e].border.push_back(slot(j,nr+e));
    base_values.resize(kkt.size());lambda.assign(me,0.0);z.assign(mi,0.0);s.assign(mi,1.0);active.assign(mi,0);
    rd.resize(n);rp.resize(mi);h.resize(me);
    // The majorant route keeps the complete inequality working set. Cache
    // each row's values and gradients once before initializing s and z.
    const bool reuse_initial_cache=st.dca_engine=="majorants"&&!st.lazy_newton_lazy_constraints;
    evaluate(reuse_initial_cache);
    // Do not initialize a strongly infeasible start with a near-terminal
    // barrier. Use scaled ORIGINAL constraint violations, not slack residuals.
    double initial_violation=inf_norm(h);
    for(const auto& q:inequalities)initial_violation=std::max(initial_violation,std::max(0.0,q.value));
    const double legacy_initial_mu=st.lazy_newton_adaptive_barrier&&initial_violation>std::sqrt(st.lazy_newton_initial_mu)
        ?std::max(st.lazy_newton_initial_mu,std::min(1.0,initial_violation)):st.lazy_newton_initial_mu;
    if(reuse_initial_cache && st.majorant_startup_policy=="auto") {
      detail::AutoStartupFeatures f;
      f.estimated_dense_restoration_bytes=sizeof(double)*static_cast<long double>(original.n)*
          (static_cast<long double>(original.equalities.size())+original.inequalities.size());
      auto curved=[](const Quadratic& q) {return std::any_of(q.hessian.begin(),q.hessian.end(),
          [](const auto& e){return e.v!=0.0;});};
      f.mixed=me>0 && mi>0;
      f.scaled_objective_gradient=inf_norm(objective.gradient);
      f.finite_data=finite(objective.gradient)&&std::isfinite(objective.value);
      f.affine_nonconstant_objective=!curved(objective)&&f.scaled_objective_gradient>0;
      f.native_signed_squares=original.has_signed_square();
      for(const auto& q:equalities) {
        f.curved_equalities=f.curved_equalities||curved(q);
        f.infeasible_curved_equality=f.infeasible_curved_equality||
          (curved(q)&&std::abs(q.value)/q.scale>st.feasibility_tolerance);
        const double gn=inf_norm(q.gradient);
        f.finite_data=f.finite_data&&std::isfinite(q.value)&&finite(q.gradient)&&q.scale>0;
        f.blind_violated_row=f.blind_violated_row||
          (std::abs(q.value)/q.scale>st.feasibility_tolerance &&
           gn<=64*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(q.value)));
      }
      for(const auto& q:inequalities) {
        f.curved_inequalities=f.curved_inequalities||curved(q);
        const double gn=inf_norm(q.gradient);
        f.finite_data=f.finite_data&&std::isfinite(q.value)&&finite(q.gradient)&&q.scale>0;
        f.blind_violated_row=f.blind_violated_row||
          (q.value/q.scale>st.feasibility_tolerance &&
           gn<=64*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(q.value)));
      }
      auto_startup=detail::choose_auto_startup(f,legacy_initial_mu);
    }
    const double initial_mu=auto_startup.central_first?auto_startup.initial_mu:legacy_initial_mu;
    stats.initial_scaled_violation=initial_violation;stats.initial_barrier=initial_mu;
    // Start the outer-only route with central slack products without changing
    // its bounded initial multipliers: mu/max(mu,s_old)=min(1,mu/s_old).
    // This happens once. Reanchoring later transports, never resets, s/z.
    const double initial_slack_floor=reuse_initial_cache?std::max(0.01,initial_mu):0.01;
    for(int i=0;i<mi;++i) {
      active[i]=!st.lazy_newton_lazy_constraints||inequality_kind[i]!=0||inequalities[i].value>=-st.lazy_newton_activation_margin;
      s[i]=std::max(initial_slack_floor,-inequalities[i].value);
      if(active[i])z[i]=std::min(1.0,initial_mu/s[i]);
    }
    if(reuse_initial_cache)refresh_majorant_residuals_from_cache();
    else evaluate();
    if(st.majorant_initial_dual_balance&&reuse_initial_cache)balance_cold_duals();
    if(equality_stabilization.enabled) {
      if(nr!=n||mr!=me)throw std::logic_error("stabilized equality rows must remain uncondensed");
      equality_stabilization.reset(lambda,true_source_constraint_residual(),z);
    }
    set_barrier_mu(stats.initial_barrier);
    // Equality multipliers start at zero and are advanced by the coupled
    // Newton direction; no separate full-Jacobian fit is performed.
    factor_settings.kkt_adaptive_regularization=st.dca_engine!="majorants";
    factor_settings.kkt_primal_first_ordering=!st.lazy_newton_interleaved_ordering;
    factor_settings.kkt_interleaved_equality_ordering=st.lazy_newton_interleaved_ordering;
    factor_settings.kkt_preserve_structural_zeros=true;
    factor_settings.kkt_primal_regularization=0.0;factor_settings.kkt_dual_regularization=0.0;
    // Raw QDLDL has no sign-aware recovery for the changing atomic KKT.  The
    // cholesk adapter is deliberately allowed: this workspace assembles its
    // primal shift explicitly and cholesk accepts a factor only after the full
    // expected inertia and pivot-quality checks have passed.
    if(factor_settings.linear_backend=="qdldl")
      throw std::runtime_error("lazy-newton requires a general symmetric-indefinite backend");
    if(factor_settings.linear_backend=="cholesk") {
      // Atomic source-atoms own primal curvature repair.  Cholesk owns only
      // the complementary dual safeguard: it floors zero/weak equality
      // diagonals and certifies the complete KKT inertia before acceptance.
      factor_settings.kkt_adaptive_regularization=false;
      factor_settings.kkt_rho_x_floor=0.0;
      factor_settings.kkt_rho_x_max=0.0;
      if(st.dca_engine=="majorants") {
        // Numerical rescue is independent of inequality majorant curvature.
        // Exact first, then bounded block-specific shifts; no rhoI search.
        factor_settings.kkt_rho_x_floor=st.kkt_rho_x_floor;
        factor_settings.kkt_rho_x_max=std::min(st.kkt_rho_x_max,1e-8);
        factor_settings.kkt_delta_dual_max=std::min(st.kkt_delta_dual_max,1e-8);
        factor_settings.kkt_regularization_retry_max=std::min(st.kkt_regularization_retry_max,5);
      }
      // The exact KKT is always attempted first.  The Cholesk retry ladder
      // owns the smallest authorized dual floor and activates it only after a
      // failed factor/solve, so a successful unshifted Newton direction is not
      // biased merely because the backend supports regularization.
      factor_settings.kkt_dual_regularization=0.0;
    }
    factor=make_symmetric_factorization(factor_settings,message);
    // A backend-created dual floor must remain visible in original KKT
    // coordinates for the residual audit below. Pair elimination would apply
    // it only after removing some equality coordinates, so keep this atomic
    // Cholesk system unreduced. Other backends retain the optional reduction.
    if(st.dca_pair_elimination&&factor_settings.linear_backend!="cholesk")
      factor=detail::with_sparse_pair_elimination(std::move(factor),
        [this](std::string& m){return make_symmetric_factorization(factor_settings,m);},st.live_display_level>=4);
    if(!factor)throw std::runtime_error(message);
    stats.structure_builds=1;stats.fixed_variables=original.n-n;stats.reduced_variables=nr;
    stats.reduced_equalities=mr;stats.eliminated_variables=n-nr;stats.objective_scale=objective.scale;
    stats.initial_active_constraints=std::count(active.begin(),active.end(),1);
    stats.objective_hessian_nnz=(int)objective.hessian.size();
    for(const auto& q:equalities)if(!q.hessian.empty()) {
      ++stats.quadratic_equalities;stats.equality_hessian_nnz+=(int)q.hessian.size();
    }
    for(const auto& q:inequalities)if(!q.hessian.empty()) {
      ++stats.quadratic_inequalities;stats.inequality_hessian_nnz+=(int)q.hessian.size();
    }
    stats.kkt_dimension=dim;stats.kkt_structural_nnz=(int)kkt.size();stats.dense_constraint_borders=ndense;
  }

  void balance_cold_duals() {
    const char* reason="cold-context-ineligible";bool accepted=false,original_verified=false;
    int original_j=-1,j=-1,contributors=0;double raw=0,gamma=1,before=inf_norm(rd),after=before;
    double mean=0,new_mu=stats.initial_barrier;
    const double floor=objective.scale*std::min(st.complementarity_tolerance,st.stationarity_tolerance)/11.0;
    auto attempt=[&]() {
      if(!st.dca_endpoint_start_file.empty()||st.restoration_failed_linearization_context||
          st.restoration_private_precision_context||stats.factor_attempts!=0)return;
      const auto* reduction=st.rank_presolve_context;
      const QCQPProblem* full=&original;
      if(reduction) {
        if(&original!=&reduction->reduced()){reason="rank-context-mismatch";return;}
        full=&reduction->original();
      }
      const auto column=detail::affine_objective_column(*full);reason=column.reason;
      if(column.coordinate<0)return;
      original_verified=true;original_j=column.coordinate;int reduced_j=original_j;
      if(reduction) {
        const auto& map=reduction->variable_map();reduced_j=map.at(original_j);
        if(std::count(map.begin(),map.end(),reduced_j)!=1){reason="objective-coordinate-not-unique-after-rank-map";return;}
      }
      if(reduced_j<0||reduced_j>=int(index.size())||(j=index[reduced_j])<0){reason="objective-coordinate-fixed";return;}
      // Verify the actual active, scaled Workspace columns as well. Native
      // bounds are included here; a positive opposing side forbids balancing.
      long double cj=0,sum=0,products=0;int count=0;
      for(std::size_t k=0;k<objective.support.size();++k) {
        const long double a=objective.gradient_extended[k];
        if(objective.support[k]==j)cj=a;else if(a!=0){reason="workspace-objective-not-one-coordinate";return;}
      }
      if(!std::isfinite(cj)||cj==0){reason="invalid-scaled-objective-column";return;}
      const long double orientation=cj>0?1:-1;
      for(int i=0;i<mi;++i)if(active[i]) {
        if(!(s[i]>0&&z[i]>0)||!std::isfinite(s[i])||!std::isfinite(z[i]))return;
        const auto& q=inequalities[i];
        for(const auto& e:q.hessian)if(e.v!=0&&(e.i==j||e.j==j)){reason="workspace-inequality-column-curved";return;}
        for(const auto& e:q.signed_square)if(e.second!=0&&e.first==j){reason="workspace-inequality-column-curved";return;}
        long double a=0;
        for(std::size_t k=0;k<q.support.size();++k)if(q.support[k]==j)a=q.gradient_extended[k];
        if(!std::isfinite(a)||orientation*a>0){reason="workspace-active-column-opposes-objective";return;}
        if(a!=0){++contributors;sum+=a*z[i];}
        products+=static_cast<long double>(s[i])*z[i];++count;
      }
      mean=count?double(products/count):0;
      const auto plan=detail::cold_dual_balance_plan(cj,sum,contributors,mean,floor);
      reason=plan.reason;raw=plan.raw;gamma=plan.scale;if(!plan.allowed)return;
      Vec proposed_z=z,proposed_lambda=lambda;
      for(int i=0;i<mi;++i)if(active[i]) {
        proposed_z[i]*=gamma;
        if(!(proposed_z[i]>0)||!std::isfinite(proposed_z[i])){reason="nonpositive-proposed-active-dual";return;}
      }
      for(auto& a:proposed_lambda)a*=gamma;
      if(!finite(proposed_lambda)){reason="nonfinite-proposed-equality-dual";return;}
      const Vec old_rd=rd;const auto old_extended=majorant_rd_extended;
      z.swap(proposed_z);lambda.swap(proposed_lambda);
      try {refresh_majorant_residuals_from_cache();after=inf_norm(rd);}
      catch(const std::runtime_error&){after=INFINITY;}
      if(!std::isfinite(after)||!(after<before)) {
        z.swap(proposed_z);lambda.swap(proposed_lambda);rd=old_rd;majorant_rd_extended=old_extended;
        reason="whole-dual-residual-not-improved";return;
      }
      long double new_products=0;
      for(int i=0;i<mi;++i)if(active[i])new_products+=static_cast<long double>(s[i])*z[i];
      new_mu=std::max(floor,double(new_products/count));
      stats.initial_barrier=new_mu;auto_startup.initial_mu=new_mu;
      accepted=true;reason="accepted-whole-dual-decrease";
    };
    attempt();
    if(st.live_display_level>=1)std::cout<<"[COLD-DUAL-BALANCE] accepted="<<accepted<<" reason="<<reason
        <<" original_coordinate="<<original_j<<" free_coordinate="<<j<<" contributors="<<contributors
        <<" raw_scale="<<raw<<" effective_scale="<<gamma<<" old_product_mean="<<mean
        <<" effective_initial_mu="<<new_mu<<" barrier_floor="<<floor
        <<" dual_before_internal="<<before<<" dual_after_internal="<<after
        <<" objective_scale="<<objective.scale<<" x_bitwise_unchanged=1 s_bitwise_unchanged=1"
        <<" full_original_coefficients_verified="<<original_verified<<" before_first_factor="<<(stats.factor_attempts==0)<<std::endl;
  }

  // Only s/z/lambda changed; x and every cached value/gradient are unchanged.
  // This is not a function evaluation and does not revisit QCQP coefficients.
  void refresh_majorant_residuals_from_cache() {
    CostScope cost(costs,NewtonCost::Evaluation);
    majorant_rd_extended.assign(n,0.0L);
    objective.add_gradient_extended(1.0,majorant_rd_extended);
    for(int e=0;e<me;++e) {
      h[e]=equalities[e].value;
      if(lambda[e]!=0.0)equalities[e].add_gradient_extended(lambda[e],majorant_rd_extended);
    }
    for(int i=0;i<mi;++i) {
      rp[i]=inequalities[i].value+s[i];
      if(active[i])inequalities[i].add_gradient_extended(z[i],majorant_rd_extended);
    }
    for(int j=0;j<n;++j)rd[j]=static_cast<double>(majorant_rd_extended[j]);
    if(!finite(rd)||!finite(h)||!finite(rp)||!std::isfinite(objective.value))
      throw std::runtime_error("lazy-newton: nonfinite cached initial residual");
  }

  OmittedBoundResidual omitted_original_bounds(const Vec& point) const {
    const auto* reduction=st.rank_presolve_context;
    if(!reduction||!reduction->has_omitted_equality_bounds())return {};
    if(point.size()!=std::size_t(n))throw std::invalid_argument("original bound contract: private point dimension");
    Vec complete=fixed;
    for(int j=0;j<n;++j)complete[free[j]]=point[j];
    if(!finite(complete)) {
      OmittedBoundResidual invalid;
      invalid.maximum=INFINITY;invalid.sum=INFINITY;return invalid;
    }
    return reduction->omitted_bound_residual(complete);
  }
  double omitted_original_bound_scale() const {
    // A box row has a unit gradient, so this is exactly the scaling policy
    // used by a retained finite bound in the Workspace constructor.
    return std::min(1.0,st.lazy_newton_constraint_gradient_target);
  }
  void evaluate(bool all_rows=false) {
    CostScope cost(costs,NewtonCost::Evaluation);
    const bool extended=st.dca_engine=="majorants";
    if(extended)majorant_rd_extended.assign(n,0.0L);
    else std::fill(rd.begin(),rd.end(),0.0);
    auto accumulate=[&](const Quadratic& q,double weight) {
      if(extended)q.add_gradient_extended(weight,majorant_rd_extended);
      else q.add_gradient(weight,rd);
    };
    objective.evaluate(x);accumulate(objective,1.0);
    for(int e=0;e<me;++e){equalities[e].evaluate(x);h[e]=equalities[e].value;accumulate(equalities[e],lambda[e]);}
    for(int i=0;i<mi;++i){
      // Inactive rows contribute neither stationarity nor KKT coefficients.
      // Keep their values current for initial activation and diagnostics;
      // a newly active row is fully reevaluated before the next model/matrix.
      if(dca_active_evaluation&&!active[i]&&!all_rows)inequalities[i].value=inequalities[i].at(x);
      else inequalities[i].evaluate(x);
      rp[i]=inequalities[i].value+s[i];
      if(active[i])accumulate(inequalities[i],z[i]);
    }
    if(extended)for(int j=0;j<n;++j)rd[j]=static_cast<double>(majorant_rd_extended[j]);
    if(!finite(rd)||!finite(h)||!finite(rp)||!std::isfinite(objective.value))throw std::runtime_error("lazy-newton: nonfinite evaluation");
  }
  void update_geometry() {
    CostScope cost(costs,NewtonCost::Assembly);
    condensation_scanned=false;precision_condensation_scanned=false;precision_quality_used=false;strip_model_borders();
    auto update=[&](ReducedGradient& g,const Quadratic& q) {
      std::fill(g.values.begin(),g.values.end(),0.0);
      for(auto m:g.map)g.values[m.output]+=q.gradient[m.gradient]*tvalue[m.t];
    };
#if defined(LAZYIPDCA_HAVE_OPENMP)
    // Parallelize independent rows only. Each worker preserves the original
    // sum order within its row, and writes a disjoint tvalue segment/buffer.
    // No cost-ledger calls, atomics or floating-point reductions occur here.
    // One team handles both stages; the first for's barrier publishes T before
    // any reduced gradient reads it. Small and legacy problems stay serial.
    if(st.dca_engine=="majorants" && mi+mr>=1024 &&
       omp_get_max_threads()>1 && !omp_in_parallel()) {
      int actual_threads=1;
#pragma omp parallel shared(actual_threads)
      {
#pragma omp single nowait
        actual_threads=omp_get_num_threads();
#pragma omp for schedule(static)
        for(int j=0;j<n;++j)if(pivot_equation[j]>=0)for(int t=tbegin[j];t<tbegin[j+1];++t) {
          auto [e,k]=tgradient[t];tvalue[t]=-equalities[e].gradient[k]/pivot_coefficient[e];
        }
#pragma omp for schedule(static)
        for(int row=0;row<mi+mr;++row) {
          if(row<mi) {
            if(!dca_active_evaluation||active[row])update(jr[row],inequalities[row]);
          } else {
            const int a=row-mi;update(ar[a],equalities[eq_retained[a]]);
          }
        }
      }
      if(actual_threads>1)++stats.majorant_geometry_parallel_batches;
      stats.majorant_geometry_threads=std::max(stats.majorant_geometry_threads,actual_threads);
      return;
    }
#endif
    for(int j=0;j<n;++j)if(pivot_equation[j]>=0)for(int t=tbegin[j];t<tbegin[j+1];++t) {
      auto [e,k]=tgradient[t];tvalue[t]=-equalities[e].gradient[k]/pivot_coefficient[e];
    }
    for(int i=0;i<mi;++i)if(!dca_active_evaluation||active[i])update(jr[i],inequalities[i]);
    for(int a=0;a<mr;++a)update(ar[a],equalities[eq_retained[a]]);
  }
  void update_matrix() {
    CostScope cost(costs,NewtonCost::Assembly);
    update_geometry();
    std::fill(hvalue.begin(),hvalue.end(),0.0);
    auto add=[&](const Quadratic& q,double weight) {
      if(q.model_active){if(!q.model_hessian.empty())q.note(&NewtonCalls::hessian_assemblies);for(std::size_t j=0;j<q.model_hessian.size();++j)hvalue[q.model_hglobal[j]]+=weight*q.model_hessian[j].v;}
      else {if(!q.hessian.empty())q.note(&NewtonCalls::hessian_assemblies);for(std::size_t j=0;j<q.hessian.size();++j)hvalue[q.hglobal[j]]+=weight*q.scale*q.hessian[j].v;}
    };
    add(objective,1.0);for(int e=0;e<me;++e)add(equalities[e],lambda[e]);
    for(int i=0;i<mi;++i)if(active[i])add(inequalities[i],z[i]);
    if(st.convexification_study!="off") {
      double scalar=objective.model_active?objective.study_model_scalar:0;
      for(int i=0;i<mi;++i)if(active[i]&&inequalities[i].model_active)scalar+=z[i]*inequalities[i].study_model_scalar;
      for(std::size_t k=0;k<hpattern.size();++k)if(hpattern[k].first==hpattern[k].second)hvalue[k]+=scalar;
    }
    std::fill(base_values.begin(),base_values.end(),0.0);
    for(auto m:hmap)base_values[m.k]+=m.multiplier*hvalue[m.h]*tvalue[m.a]*tvalue[m.b];
    for(int i=0;i<mi;++i) {
      const auto& g=jr[i];
      if(g.dense>=0) {
        for(int a=0;a<(int)g.values.size();++a)base_values[g.border[a]]=active[i]?g.values[a]:0.0;
        base_values[dense_diagonal[g.dense]]=active[i]?
            (source_primal_barrier(i)?
              -static_cast<double>(static_cast<long double>(s[i])*s[i]/source_barrier_mu+
                  inequality_algorithmic_rho(i)):(inequality_algorithmic_rho(i)>0?
              -static_cast<double>(static_cast<long double>(s[i])/z[i]+inequality_algorithmic_rho(i)):
              -s[i]/z[i])):-1.0;
      } else if(active[i]) {
        const double ratio=inequality_gram_weight(i);for(auto a:g.gram)base_values[a.k]+=ratio*g.values[a.a]*g.values[a.b];
      }
    }
    for(int a=0;a<mr;++a) {
      for(int j=0;j<(int)ar[a].values.size();++j)base_values[ar[a].border[j]]=ar[a].values[j];
      if(equality_stabilization.enabled)base_values[equality_diagonal[a]]=-equality_stabilization.rho;
    }
    if(!finite(base_values))throw std::runtime_error("lazy-newton: nonfinite KKT coefficients");
    if((st.convexification_study=="HH"||st.convexification_study=="HS"||st.convexification_study=="SH"||st.convexification_study=="SS"||st.convexification_study=="H0")) {
      atom_border_begin=kkt.size();
      // The manuscript's atom pivots are not dual stabilization variables.
      factor_settings.kkt_dual_regularization_end=nr+mr+ndense;
      pivoted_settings.kkt_dual_regularization_end=nr+mr+ndense;
      recovery_settings.kkt_dual_regularization_end=nr+mr+ndense;
      hybrid_border_objective_atoms=hybrid_border_inequality_atoms=0;
      auto append=[&](const Quadratic& q,double multiplier,bool objective_role) {
        if(!q.model_active||q.model_atoms.empty())return;
        if(!std::isfinite(multiplier)||multiplier<0)
          throw std::runtime_error("hybrid inequality atom requires a nonnegative finite multiplier");
        for(const auto& a:q.model_atoms) {
          if(a.direction.size()!=std::size_t(n)||!std::isfinite(a.beta)||a.beta<0)
            throw std::runtime_error("invalid hybrid model atom");
          const long double scaled=static_cast<long double>(multiplier)*a.beta;
          if(!std::isfinite(scaled)||scaled>std::numeric_limits<double>::max())
            throw std::runtime_error("hybrid weighted atom overflows");
          Vec reduced(nr,0);
          if(!q.model_atom_support.empty()) {
            for(int j=0;j<nr;++j)reduced[j]=a.direction[retained[j]];
          } else for(int j=0;j<n;++j)if(a.direction[j]!=0)
            for(int t=tbegin[j];t<tbegin[j+1];++t)
              reduced[tcolumn[t]]+=a.direction[j]*tvalue[t];
          const int before=dim;
          detail::append_hybrid_border(nr,reduced,double(scaled),dim,kkt,base_values);
          if(dim>before) {
            if(objective_role)++hybrid_border_objective_atoms;
            else ++hybrid_border_inequality_atoms;
          }
        }
      };
      append(objective,1.0,true);
      for(int i=0;i<mi;++i)if(active[i])append(inequalities[i],z[i],false);
      has_model_borders=dim>nr+mr+ndense;
      stats.kkt_dimension=dim;stats.kkt_structural_nnz=kkt.size();++hybrid_border_assemblies;
      if(hybrid_last_reported_objective!=hybrid_border_objective_atoms||
         hybrid_last_reported_inequalities!=hybrid_border_inequality_atoms) {
        std::cout<<"[HYBRID-BORDER] objective_atoms="<<hybrid_border_objective_atoms
          <<" inequality_atoms="<<hybrid_border_inequality_atoms
          <<" base_dimension="<<(nr+mr+ndense)<<" dimension="<<dim
          <<" border_nnz="<<(kkt.size()-atom_border_begin)
          <<" explicit_atom_outer_products=0 positive_dual_weighting=1"<<std::endl;
        hybrid_last_reported_objective=hybrid_border_objective_atoms;
        hybrid_last_reported_inequalities=hybrid_border_inequality_atoms;
      }
    } else if(objective.model_active&&!objective.model_atoms.empty()) {
      atom_border_begin=kkt.size();has_model_borders=true;
      for(const auto& a:objective.model_atoms){
        Vec reduced(nr,0);
        if(!objective.model_atom_support.empty()) {
          // The support is E's retained coordinates and the atom is E' u.
          // Thus T' E' u=u exactly, with no scan of eliminated coordinates.
          for(int j=0;j<nr;++j)reduced[j]=a.direction[retained[j]];
        } else for(int j=0;j<n;++j)for(int t=tbegin[j];t<tbegin[j+1];++t)reduced[tcolumn[t]]+=a.direction[j]*tvalue[t];
        const int row=dim++;const double weight=std::sqrt(a.beta);
        for(int j=0;j<nr;++j){kkt.push_back({j,row,0});base_values.push_back(weight*reduced[j]);}
        kkt.push_back({row,row,0});base_values.push_back(-1);
      }
      stats.kkt_dimension=dim;stats.kkt_structural_nnz=kkt.size();
    }
    ++stats.numeric_updates;
    if(atoms){CostScope atom_cost(costs,NewtonCost::Atoms);atoms->project(tbegin,tcolumn,tvalue,retained,kkt,base_values,stats);}
  }
  bool recover_condensation(const char* trigger) {
    CostScope cost(costs,NewtonCost::Assembly);
    if(!st.lazy_newton_condensation_guard||condensation_scanned)return false;
    condensation_scanned=true;
    // Reconstruct just the reduced Lagrangian block, without barrier Gram
    // terms. This scan is performed only after a factor/accuracy failure.
    Vec projected(kkt.size(),0.0);
    for(auto m:hmap)projected[m.k]+=m.multiplier*hvalue[m.h]*tvalue[m.a]*tvalue[m.b];
    const double scale=std::max({1.0,regularization,inf_norm(projected)});
    const bool majorants=st.dca_engine=="majorants";
    // A single-coordinate bound can lose the same stationarity digits as a
    // multi-coordinate row when z/s is large: reconstructing dz subtracts two
    // nearly equal quantities after its Gram contribution has dominated H.
    // Keep those rows eligible in the majorant route, and recover before the
    // historical 1/sqrt(epsilon) ratio has consumed most working precision.
    // The trigger remains an actual factor/expanded-residual failure; this
    // is an equivalent bordered representation, not extra regularization.
    const double ratio_limit=majorants?std::min(st.lazy_newton_condensation_ratio,1e4):
        st.lazy_newton_condensation_ratio;
    std::vector<int> promote;
    for(int i=0;i<mi;++i)if(active[i]&&jr[i].dense<0&&
        (majorants?!jr[i].support.empty():jr[i].support.size()>1)) {
      const double gradient=inf_norm(jr[i].values);
      const double ratio=inequality_gram_weight(i)*gradient*gradient/scale;
      stats.maximum_condensation_ratio=std::max(stats.maximum_condensation_ratio,ratio);
      if(ratio>ratio_limit)promote.push_back(i);
    }
    return promote_condensed_rows(promote,trigger,false);
  }
  // Called only after refinement and the historical condensation rescue have
  // both failed. A global Hessian norm can hide locally dominant Gram terms.
  // Bound their row-wise rounding amplification against the actual expanded
  // residual budget, independently of large coefficients in unrelated rows.
  // This is a conservative selection rule, NOT a new acceptance criterion:
  // the returned direction must still pass the unchanged expanded audit.
  bool recover_condensation_precision(const Direction& rejected,double rhs_scale) {
    if(st.dca_engine!="majorants"||!st.lazy_newton_condensation_guard||
       precision_condensation_scanned||!finite(rejected.x)||
       !std::isfinite(rhs_scale)||rhs_scale<=0)return false;
    CostScope cost(costs,NewtonCost::Assembly);
    precision_condensation_scanned=true;
    std::vector<int> incidence(nr,0);
    for(int i=0;i<mi;++i)if(active[i]&&jr[i].dense<0)
      for(std::size_t a=0;a<jr[i].support.size();++a)
        if(jr[i].values[a]!=0)++incidence[jr[i].support[a]];
    const long double direction_scale=std::max(1.0,inf_norm(rejected.x));
    const long double budget=0.01L*st.kkt_backward_error_reject*rhs_scale/
        (std::numeric_limits<double>::epsilon()*direction_scale);
    std::vector<int> promote;
    for(int i=0;i<mi;++i)if(active[i]&&jr[i].dense<0) {
      const auto& g=jr[i];long double norm1=0;
      for(double value:g.values)norm1+=std::abs(static_cast<long double>(value));
      const long double weight=inequality_gram_weight_extended(i);
      for(std::size_t a=0;a<g.support.size();++a) {
        const long double amplification=weight*std::abs(static_cast<long double>(g.values[a]))*
            norm1*incidence[g.support[a]];
        if(amplification>budget){promote.push_back(i);break;}
      }
    }
    return promote_condensed_rows(promote,"expanded-precision",true);
  }
  bool promote_condensed_rows(const std::vector<int>& promote,const char* trigger,bool precision) {
    if(promote.empty())return false;
    const bool precision_scanned=precision_condensation_scanned;
    const bool quality_used=precision_quality_used;
    strip_model_borders();
    // Append only new borders; all gradient/Hessian maps remain persistent.
    // Old Gram slots stay structural zeros where no other term uses them.
    for(int i:promote) {
      auto& g=jr[i];g.dense=ndense++;
      const int border_index=nr+mr+g.dense;
      for(int j:g.support){g.border.push_back((int)kkt.size());kkt.push_back({j,border_index,0.0});}
      dense_diagonal.push_back((int)kkt.size());kkt.push_back({border_index,border_index,0.0});
      if(inequality_kind[i]==0)++stats.promoted_inequalities;else ++stats.promoted_bounds;
    }
    dim=nr+mr+ndense;base_values.resize(kkt.size());
    ++stats.condensation_recoveries;
    stats.kkt_dimension=dim;stats.kkt_structural_nnz=(int)kkt.size();stats.dense_constraint_borders=ndense;
    update_matrix();condensation_scanned=true;precision_condensation_scanned=precision_scanned;
    precision_quality_used=quality_used;
    if(precision) {
      ++stats.precision_condensation_recoveries;
      stats.precision_condensation_rows+=static_cast<int>(promote.size());
    }
    if(st.live_curvature||st.verbose||(precision&&st.live_display_level>0)) {
      CostScope display_cost(costs,NewtonCost::Diagnostics);const auto display_timer=Clock::now();
      std::cout<<"[KKT-CONDENSATION-RECOVERY] k="<<stats.accepted_steps<<" trigger="<<trigger
               <<" promoted="<<promote.size()<<" dim="<<dim<<" rho="<<regularization
               <<" max_ratio="<<stats.maximum_condensation_ratio<<std::endl;
      stats.diagnostic_seconds+=seconds(display_timer);
    }
    return true;
  }

  // A successful normwise backend solve can still fail the expanded audit.
  // After representation recovery, let a pivoting backend improve the SAME
  // matrix, once per outer assembly. Its factor routine rechecks inertia;
  // the caller still requires the original expanded direction tolerance.
  bool recover_precision_pivoting() {
    if(st.dca_engine!="majorants"||current_factor!=factor.get()||precision_quality_used||
       (st.linear_backend!="mumps"&&st.linear_backend!="ma27"))return false;
    precision_quality_used=true;
    const double before=current_factor->pivot_tolerance();
    bool ok;
    {CostScope factor_cost(costs,NewtonCost::Factorization);
      ok=current_factor->increase_quality(message);}
    const bool attempted=current_factor->pivot_tolerance()>before;
    if(attempted)++stats.factor_attempts;
    if(!ok) {
      if(attempted) {
        ++stats.backend_factor_failures;
        // Better numerical pivoting can reveal a genuine extra negative
        // eigenvalue hidden in the first factor. Route that new inertia
        // diagnosis through the ordinary explicit proximal ladder instead
        // of converting it into a terminal linear-solver error. The new
        // direction is still audited against the full auxiliary equations.
        const bool excess=message.find("inertia mismatch")!=std::string::npos &&
            current_factor->last_negative_pivots()>dim-nr;
        const double next=regularization>0?10*regularization:st.lazy_newton_regularization_initial;
        if(excess && !st.lazy_newton_atomic_regularization && std::isfinite(next) && next<=1e12) {
          ++stats.inertia_rejections;++stats.regularization_increases;
          if(st.live_display_level>0)std::cout<<"[KKT-LATE-INERTIA] k="<<stats.accepted_steps
              <<" rho="<<regularization<<"->"<<next<<" original_audit_required=1"<<std::endl;
          if(factorize(next))return true;
        }
        throw std::runtime_error("lazy-newton accuracy recovery factorization failed: "+message);
      }
      return false;
    }
    if(st.live_display_level>0||st.verbose) {
      CostScope display_cost(costs,NewtonCost::Diagnostics);
      std::cout<<"[KKT-PIVOT-RECOVERY] k="<<stats.accepted_steps
               <<" before="<<before<<" after="<<current_factor->pivot_tolerance()
               <<" backend="<<current_factor->name()<<std::endl;
    }
    return true;
  }

  bool native_factor(const char* trigger,const std::string& sparse_reason) {
    CostScope cost(costs,NewtonCost::Factorization);
    const auto timer=Clock::now();
    if(st.live_curvature) {
      CostScope display_cost(costs,NewtonCost::Diagnostics);const auto display_timer=Clock::now();
      std::cout<<"[KKT-PIVOTED-BEGIN] backend="
        <<((st.dca_engine=="filter"||st.dca_engine=="majorants") && st.dca_filter_sparse_fallback && linear_backend_compiled("mumps") ? "mumps" : "native")
        <<" k="<<stats.accepted_steps<<" trigger="<<trigger
               <<" dim="<<dim<<" rho="<<regularization<<" reason="<<sparse_reason<<std::endl;
      stats.diagnostic_seconds+=seconds(display_timer);
    }
    if(!pivoted) {
      pivoted_settings=factor_settings;pivoted_settings.linear_backend="native";
      if ((st.dca_engine=="filter"||st.dca_engine=="majorants") && st.dca_filter_sparse_fallback && linear_backend_compiled("mumps"))
        pivoted_settings.linear_backend="mumps";
      pivoted=make_symmetric_factorization(pivoted_settings,message);
    }
    bool ok=false;std::string native_message=message;
    if(pivoted) {
      native_message.clear();++stats.factor_attempts;++stats.pivoted_fallbacks;
      stats.last_native_iteration=stats.accepted_steps;
      ok=pivoted->factor(dim,nr,kkt,native_message);
      if(ok){current_factor=pivoted.get();++stats.native_factor_successes;}
    }
    const double elapsed=seconds(timer);stats.native_factor_seconds+=elapsed;
    if(st.verbose)std::cout<<"[LAZY-NEWTON-NATIVE] k="<<stats.accepted_steps<<" trigger="<<trigger
        <<" rho="<<regularization<<" dim="<<dim<<" seconds="<<elapsed<<" factor_ok="<<ok
        <<" sparse_reason="<<sparse_reason<<" native_reason="<<native_message<<"\n";
    message=native_message;return ok;
  }

  bool normal_recovery(int bad) {
    // K=[B A' C'; A 0 0; C 0 -D]. Adding tau*u times equality e
    // to the primal equations changes B to B+tau*u*u', where a_e=c*u.
    // The RHS must simultaneously change by tau*u*b_e/c. This invertible
    // row operation leaves BOTH primal and dual solutions unchanged. The
    // symmetric new matrix is also congruent to K, so its inertia is equal.
    // No normal regularization of the original Newton equations is introduced.
    // The matched normal row operation below assumes a zero equality-dual
    // block. It is not algebraically equivalent when algorithmic rho>0.
    if(equality_stabilization.enabled||!st.lazy_newton_sparse_recovery||bad<0||bad>=nr+mr)return false;
    const auto timer=Clock::now();
    recovery_equation=-1;double best=0.0;
    for(int e=0;e<mr;++e) {
      const auto& a=ar[e];const double norm=inf_norm(a.values);
      if(!(norm>0.0)||!std::isfinite(norm))continue;
      if(bad>=nr) {
        if(e==bad-nr){recovery_equation=e;recovery_scale=norm;best=1.0;break;}
        continue;
      }
      for(std::size_t i=0;i<a.support.size();++i)if(a.support[i]==bad) {
        const double ratio=std::abs(a.values[i])/norm;
        if(ratio>best){best=ratio;recovery_equation=e;recovery_scale=norm;}
      }
    }
    const double root_eps=std::sqrt(std::numeric_limits<double>::epsilon());
    if(recovery_equation<0||best<root_eps) {
      stats.normal_recovery_factor_seconds+=seconds(timer);return false;
    }
    ++stats.normal_recovery_attempts;stats.last_normal_recovery_equation=recovery_equation;
    const auto& a=ar[recovery_equation];recovery_normal=a.values;
    for(double& v:recovery_normal)v/=recovery_scale;
    double row_scale=1.0;
    for(const auto& entry:kkt)if(entry.i==bad||entry.j==bad)row_scale=std::max(row_scale,std::abs(entry.value));
    recovery_shift=std::max(1.0,root_eps*row_scale)/(best*best);
    std::vector<SymmetricTriplet> augmented=kkt;
    bool finite_values=std::isfinite(recovery_shift);
    for(std::size_t i=0;i<a.support.size();++i)for(std::size_t j=i;j<a.support.size();++j) {
      const double v=recovery_shift*recovery_normal[i]*recovery_normal[j];
      finite_values=finite_values&&std::isfinite(v);
      augmented.push_back({a.support[i],a.support[j],v});
    }
    bool ok=false;std::string recovery_message;
    if(finite_values) {
      if(!recovery){recovery_settings=factor_settings;recovery=make_symmetric_factorization(recovery_settings,recovery_message);}
      if(recovery){++stats.factor_attempts;ok=recovery->factor(dim,nr,augmented,recovery_message);}
    } else recovery_message="nonfinite equality augmentation";
    if(ok) {
      current_factor=recovery.get();++stats.normal_recovery_successes;
      stats.normal_recovery_maximum_shift=std::max(stats.normal_recovery_maximum_shift,recovery_shift);
    }
    const double elapsed=seconds(timer);stats.normal_recovery_factor_seconds+=elapsed;
    if(st.verbose)std::cout<<"[LAZY-NEWTON-SPARSE-RECOVERY] k="<<stats.accepted_steps<<" pivot="<<bad
        <<" equality="<<recovery_equation<<" tau="<<recovery_shift<<" seconds="<<elapsed
        <<" factor_ok="<<ok<<" reason="<<recovery_message<<"\n";
    return ok;
  }

  #include "lazy_newton_marginal.hpp"

  bool add_atoms(int retry) {
    CostScope cost(costs,NewtonCost::Atoms);
    const auto timer=Clock::now();
    if(!atoms)atoms=std::make_unique<NewtonAtoms>();
    if(atom_defect_iteration!=stats.accepted_steps) {
      atom_defect_iteration=stats.accepted_steps;atom_last_defect=-1;atom_stagnant_batches=0;
    }
    const int defect=current_factor?current_factor->curvature_defect():0;
    if(defect>0&&atom_defect_factor!=stats.factor_attempts) {
      // A recursive source search can reuse this factor; it is not another
      // measured repair batch and must not advance the plateau counter.
      atom_defect_factor=stats.factor_attempts;
      if(atom_last_defect>0)atom_stagnant_batches=defect>=atom_last_defect?atom_stagnant_batches+1:0;
      atom_last_defect=defect;
      if(!atom_adaptation_active&&atom_stagnant_batches>=st.lazy_newton_atom_guided_after) {
        atom_adaptation_active=true;stats.atom_adaptation_activated=true;++stats.atom_plateau_events;
        if(st.live_curvature){CostScope display_cost(costs,NewtonCost::Diagnostics);std::cout<<"[ATOM-PLATEAU] k="<<stats.accepted_steps<<" defect="<<defect
          <<" stagnant_batches="<<atom_stagnant_batches<<std::endl;}
      }
    }
    std::vector<AtomWitness> witnesses;
    if(st.lazy_newton_atom_guided&&atom_adaptation_active) {
      if(atom_guide_iteration!=stats.accepted_steps){atom_guides.clear();atom_guide_iteration=stats.accepted_steps;}
      for(auto& w:atom_guides) {
        NewtonAtom dense;dense.support.resize(nr);std::iota(dense.support.begin(),dense.support.end(),0);dense.values=w.direction;
        const auto ray=atom_rayleigh(dense);w.curvature=w.initial=(std::isfinite(ray.first)&&ray.first<-ray.second)?ray.first:0.0;
      }
      atom_guides.erase(std::remove_if(atom_guides.begin(),atom_guides.end(),[](const AtomWitness& w){return !std::isfinite(w.curvature)||w.curvature>=0;}),atom_guides.end());
      if(atom_guides.empty())atom_guides=tangent_witnesses();
      witnesses=atom_guides;
    }
    std::vector<Vec> original_witnesses;
    for(const auto& w:witnesses) {
      Vec v(n,0.0);for(int j=0;j<n;++j)for(int k=tbegin[j];k<tbegin[j+1];++k)v[j]+=tvalue[k]*w.direction[tcolumn[k]];
      original_witnesses.push_back(std::move(v));
    }
    struct Source {const Quadratic* q;int family,origin;double weight,score;};
    std::vector<Source> sources;
    auto visit=[&](const Quadratic& q,int family,int origin,double weight) {
      if(weight==0.0||q.hessian.empty()||atoms->cached.count(&q))return;
      double score=0.0;for(auto e:q.hessian)score=std::max(score,std::abs(weight*q.scale*e.v));
      if(!original_witnesses.empty()) {
        score=0.0;
        for(const auto& w:original_witnesses) {
          double curvature=0.0;for(auto e:q.hessian)curvature+=(e.i==e.j?1.0:2.0)*e.v*w[e.i]*w[e.j];
          score+=std::max(0.0,-weight*q.scale*curvature);
        }
      }
      sources.push_back({&q,family,origin,weight,score});
    };
    visit(objective,0,-1,1.0);
    for(int i=0;i<mi;++i)if(active[i])visit(inequalities[i],1,i,z[i]);
    for(int e=0;e<me;++e)visit(equalities[e],2,e,lambda[e]);
    // Analyse only a growing batch of likely sources. The score is a cheap
    // selection heuristic, never a curvature certificate. Acceptance still
    // requires the complete KKT inertia and expanded direction residual.
    const int source_batch=std::min((int)sources.size(),32<<std::min(7,retry));
    std::partial_sort(sources.begin(),sources.begin()+source_batch,sources.end(),
      [](const Source& a,const Source& b){if(a.score!=b.score)return a.score>b.score;
        return a.family!=b.family?a.family<b.family:a.origin<b.origin;});
    for(int j=0;j<source_batch;++j){const auto& a=sources[j];atoms->cache(*a.q,a.family,a.origin,a.weight,stats,st.lazy_newton_atom_energy_sparsify);}
    auto weight=[&](int family,int origin) {return family==0?1.0:family==1?z[origin]:lambda[origin];};
    bool added=atoms->activate(weight,tbegin,tcolumn,tvalue,retained,kkt,base_values,
        st.lazy_newton_regularization_initial,retry,stats,
        [&](const NewtonAtom& a){return atom_rayleigh(a);},st.lazy_newton_atom_rayleigh_gate,
        !witnesses.empty()?&witnesses:nullptr);
    if(st.lazy_newton_atom_rayleigh_gate&&!added&&source_batch<(int)sources.size()&&retry<12) {
      stats.atom_analysis_seconds+=seconds(timer);
      return add_atoms(retry+1);
    }
    if(!added) {
      // Numerical completion: positive coordinate atoms on the deficient
      // rows of the REDUCED Lagrangian. No equality-dual shift or rho*I.
      Vec projected(kkt.size(),0.0);
      for(auto m:hmap)projected[m.k]+=m.multiplier*hvalue[m.h]*tvalue[m.a]*tvalue[m.b];
      added=atoms->complete(projected,kkt,retained,st.lazy_newton_regularization_initial,
                            base_values,stats);
    }
    stats.atom_analysis_seconds+=seconds(timer);
    if(added) {
      ++stats.atom_activation_batches;
      if(st.live_curvature){CostScope display_cost(costs,NewtonCost::Diagnostics);std::cout<<"[ATOMS-ADD] k="<<stats.accepted_steps
        <<" active="<<atoms->active_count()<<" cached="<<atoms->pool.size()
        <<" trace="<<atoms->trace()<<" retry="<<retry<<std::endl;}
    }
    return added;
  }
  bool numeric_pivot_failure(const std::string& reason) const {
    return (reason.find("inertia mismatch")!=std::string::npos &&
       factor->last_negative_pivots()>=0 && factor->last_negative_pivots()<dim-nr) ||
      (st.linear_backend=="cholesk" && reason.find("zero pivot / non-factorizable KKT")!=std::string::npos) ||
      (st.linear_backend=="ldl" && reason.find("numeric factorization failed at pivot")!=std::string::npos) ||
      (st.linear_backend=="mumps" && (reason.find("INFOG(1)=-10")!=std::string::npos ||
        reason.find("pivot_delay_workspace_exhaustion=1")!=std::string::npos)) ||
      (st.linear_backend=="ma27" && reason.find("factor INFO(1)=3")!=std::string::npos);
  }
  bool bounded_pivot_recovery(double first) {
    if(st.dca_engine!="majorants" ||
       (st.linear_backend!="ldl" && st.linear_backend!="mumps" && st.linear_backend!="ma27"))return false;
    const std::string cause=message;
    const double cap=std::min({1e-8,st.kkt_rho_x_max,st.kkt_delta_dual_max});
    for(double shift=first;shift<=cap*(1+1e-12);shift*=10.0) {
      pivot_rescue=std::make_unique<detail::BoundedPivotRecovery>(factor_settings,shift,message);
      ++stats.factor_attempts;++stats.backend_regularization_attempts;
      const bool ok=pivot_rescue->factor(dim,nr,kkt,message);
      stats.maximum_backend_primal_regularization=std::max(stats.maximum_backend_primal_regularization,shift);
      stats.maximum_backend_dual_regularization=std::max(stats.maximum_backend_dual_regularization,shift);
      if(st.live_display_level)std::cout<<"[KKT-BOUNDED-PIVOT-RECOVERY] factor_ok="<<ok
        <<" shift="<<shift<<" backend="<<st.linear_backend<<" unperturbed_audit_required=1 cause="<<cause<<std::endl;
      if(ok){current_factor=pivot_rescue.get();return true;}
      ++stats.backend_factor_failures;++stats.backend_regularized_factor_failures;
    }
    current_factor=factor.get();
    // The perturbed matrix's inertia is not an inertia diagnosis of the
    // original failed factor. Preserve the original numerical failure so
    // the outer curvature ladder is never triggered by this rejected probe.
    message=cause+"; bounded pivot recovery exhausted";
    return false;
  }
  bool recover_numeric_pivot_quality() {
    // Raising the threshold delays still more pivots after a proven delayed
    // fill exhaustion. Go directly to bounded unperturbed refinement instead.
    if(message.find("pivot_delay_workspace_exhaustion=1")!=std::string::npos)return false;
    if(st.dca_engine!="majorants"||
        (st.linear_backend!="mumps"&&st.linear_backend!="ma27")||
        !numeric_pivot_failure(message))return false;
    const std::string cause=message;
    for(int retry=0;retry<3;++retry) {
      const double before=factor->pivot_tolerance();
      const bool ok=factor->increase_quality(message);
      if(factor->pivot_tolerance()<=before)break;
      ++stats.factor_attempts;
      if(st.live_display_level>0)std::cout<<"[KKT-NUMERIC-PIVOT-QUALITY] before="<<before
        <<" after="<<factor->pivot_tolerance()<<" factor_ok="<<ok
        <<" same_matrix=1 original_audit_required=1"<<std::endl;
      if(ok){current_factor=factor.get();return true;}
      ++stats.backend_factor_failures;
    }
    // A rejected pivot probe is not a curvature certificate for the model.
    message=cause;return false;
  }
  // A private hybrid solve leaves its numeric factors at a private Newton
  // point even after the accepted outer anchor has been restored. A SOC can
  // explicitly factor the matrix at that anchor once, without changing the
  // fixed model diagonal or entering the curvature/representation retry ladder.
  // The ordinary factorization path (frozen_model=false) is unchanged.
  bool factorize(double proposed,bool frozen_model=false) {
    if(frozen_model&&(st.dca_engine!="majorants"||
        !(st.convexification_study=="HH"||st.convexification_study=="HS"||
          st.convexification_study=="SH"||st.convexification_study=="SS"||st.convexification_study=="H0")||
        st.lazy_newton_atomic_regularization||proposed!=regularization||
        !std::isfinite(proposed)))
      throw std::invalid_argument("frozen model factorization requires unchanged hybrid tau");
    CostScope cost(costs,NewtonCost::Factorization);
    regularization=st.lazy_newton_atomic_regularization?0.0:proposed;
    if(atom_refresh_pending&&atoms) {
      CostScope atom_cost(costs,NewtonCost::Atoms);const auto timer=Clock::now();
      if(!st.lazy_newton_lazy_curvature)atoms->clear_weights();
      atoms->retire_changed_sources([&](int family,int origin) {
        return family==0?1.0:family==1?(active[origin]?z[origin]:0.0):lambda[origin];
      },st.lazy_newton_atom_transport&&atom_adaptation_active,stats);
      if(atoms->active_count())++stats.curvature_reuses;
      stats.atom_update_seconds+=seconds(timer);atom_refresh_pending=false;
    }
    for(int retry=0;retry<15;++retry) {
      for(std::size_t j=0;j<kkt.size();++j)kkt[j].value=base_values[j];
      if(st.lazy_newton_atomic_regularization) {
        if(atoms) {CostScope atom_cost(costs,NewtonCost::Atoms);const auto timer=Clock::now();atoms->assemble(kkt,retained);const double elapsed=seconds(timer);stats.atom_update_seconds+=elapsed;stats.atom_assembly_seconds+=elapsed;}
      } else for(int j:diagonal)kkt[j].value+=regularization;
      const auto* pair_factor=st.dca_pair_elimination&&factor_settings.linear_backend!="cholesk"
          ?dynamic_cast<const detail::PairReducedFactorization*>(factor.get()):nullptr;
      const int underlying_before=pair_factor?pair_factor->underlying_factorizations():0;
      ++stats.factor_attempts;
      // Count the actual primary backend call, including retries. Model
      // construction or an exception before this call does not imply its use.
      if(st.factorial_refinement_active()&&st.convexification_study=="HH"&&
         st.factorial_trace_state&&st.factorial_trace_state->current_phase_metric_index>0)
        ++st.factorial_trace_state->primary_backend_factor_attempts;
      bool ok=factor->factor(dim,nr,kkt,message);current_factor=factor.get();
      const int primary_backend_attempts=factor->last_regularization_attempts();
      const double primary_backend_primal=factor->last_primal_regularization();
      const double primary_backend_dual=factor->last_dual_regularization();
      stats.backend_regularization_attempts+=primary_backend_attempts;
      stats.maximum_backend_primal_regularization=std::max(
          stats.maximum_backend_primal_regularization,primary_backend_primal);
      stats.maximum_backend_dual_regularization=std::max(
          stats.maximum_backend_dual_regularization,primary_backend_dual);
      if(!ok) {
        ++stats.backend_factor_failures;
        if(primary_backend_primal>0.0||primary_backend_dual>0.0)
          ++stats.backend_regularized_factor_failures;
        stats.maximum_failed_backend_primal_regularization=std::max(
            stats.maximum_failed_backend_primal_regularization,primary_backend_primal);
        stats.maximum_failed_backend_dual_regularization=std::max(
            stats.maximum_failed_backend_dual_regularization,primary_backend_dual);
      }
      if(pair_factor)stats.factor_attempts+=pair_factor->underlying_factorizations()-underlying_before-1;
      if(!ok&&st.live_curvature) {
        CostScope display_cost(costs,NewtonCost::Diagnostics);const auto display_timer=Clock::now();
        std::cout<<"[KKT-REJECT] k="<<stats.accepted_steps<<" retry="<<retry
                 <<" rho="<<regularization<<" reason="<<message<<std::endl;
        stats.diagnostic_seconds+=seconds(display_timer);
      }
      if(!ok&&st.verbose&&stats.factor_attempts<4)std::cout<<"[LAZY-NEWTON-FACTOR] "<<message<<"\n";
      // One explicit factor of this exact frozen model is sufficient for a
      // correction probe. A failure rejects SOC; it must not alter tau, atoms,
      // condensation or backend in pursuit of a different correction system.
      if(frozen_model&&!ok)return false;
      // Avoid interpreting lost precision in a huge Gram term as curvature.
      // The bordered representation has exactly the same Newton equations.
      if(!ok&&message.find("inertia mismatch")!=std::string::npos&&recover_condensation("inertia")) {
        ++stats.inertia_rejections;--retry;continue;
      }
      // A zero scalar LDL pivot does not establish wrong tangent inertia.
      // First try an algebraically equivalent sparse system; retain native
      // numerical pivoting whenever that attempt cannot establish a valid factor.
      if(!ok&&st.linear_backend=="ldl"&&message.find("inertia mismatch")==std::string::npos) {
        const std::string sparse_reason=message;++stats.sparse_numeric_failures;
        stats.last_sparse_failure_coordinate=factor->failed_pivot_coordinate();
        if(stats.first_sparse_failure_iteration<0)stats.first_sparse_failure_iteration=stats.accepted_steps;
        ok=normal_recovery(stats.last_sparse_failure_coordinate);
        if(!ok&&recover_condensation("factorization")){--retry;continue;}
        if(!ok&&(st.dca_engine!="majorants"||pivoted_sparse_fallback_enabled()))
          ok=native_factor("factorization",sparse_reason);
      }
      if(!ok&&!frozen_model)ok=recover_numeric_pivot_quality();
      // A numerical zero pivot in a condensed Gram matrix is no stronger
      // evidence of curvature with a pivoting backend than with LDL. Try
      // the algebraically equivalent bordered representation BEFORE adding
      // any numerical shift. Its direction still passes the expanded audit.
      if(!ok&&st.dca_engine=="majorants"&&numeric_pivot_failure(message)&&
          recover_condensation("numeric-pivot")){--retry;continue;}
      if(!ok && st.dca_engine=="majorants" && numeric_pivot_failure(message))
        ok=bounded_pivot_recovery(1e-12);
      if(ok) {
        const double backend_primal=current_factor->last_primal_regularization();
        const double backend_dual=current_factor->last_dual_regularization();
        if(backend_dual>0.0)++stats.dual_regularized_factors;
        stats.final_backend_primal_regularization=backend_primal;
        stats.final_backend_dual_regularization=backend_dual;
        if(!corrected())++stats.unshifted_factors;
        stats.maximum_regularization=std::max(stats.maximum_regularization,regularization);
        return true;
      }
      // In the majorant engine rho regularizes the AUXILIARY KKT only.
      // A backend/pivot failure is not evidence of negative curvature and
      // must not trigger a ladder of increasingly distorted directions.
      if(st.dca_engine=="majorants" && message.find("inertia mismatch")==std::string::npos)
        return false;
      ++stats.inertia_rejections;++stats.regularization_increases;
      if(st.lazy_newton_atomic_regularization) {
        if(!add_atoms(retry))break;
      } else regularization=regularization==0.0?st.lazy_newton_regularization_initial:regularization*10.0;
      if(!std::isfinite(regularization)||regularization>1e12)break;
    }
    if(st.linear_backend=="cholesk"&&pivoted_sparse_fallback_enabled()) {
      const std::string sparse_reason=message;
      ++stats.sparse_numeric_failures;
      stats.last_sparse_failure_coordinate=factor->failed_pivot_coordinate();
      if(stats.first_sparse_failure_iteration<0)
        stats.first_sparse_failure_iteration=stats.accepted_steps;
      if(normal_recovery(stats.last_sparse_failure_coordinate))return true;
      if(recover_condensation("factorization-fallback")&&factorize(regularization))return true;
      if(native_factor("factorization",sparse_reason))return true;
    }
    return false;
  }
  // A permit belongs to one already assembled private affine-repair system.
  // It freezes the chosen numeric factor during every solve/refinement while
  // retaining the unperturbed matrix and primal-dual point as audit authority.
  // Ordinary frozen-factor SOC calls never receive this permit.
  struct AffineRepairFactorPermit {
    const Workspace* owner;
    SymmetricFactorization* numeric_factor;
    int factor_attempts,dimension,primal_dimension,equality_dimension,borders;
    double tau;
    std::vector<int> retained;
    Vec point,slack,dual,equality_dual,base,hessian;
    std::vector<SymmetricTriplet> matrix;
    explicit AffineRepairFactorPermit(const Workspace& w)
        :owner(&w),numeric_factor(w.current_factor),factor_attempts(w.stats.factor_attempts),
         dimension(w.dim),primal_dimension(w.nr),equality_dimension(w.mr),borders(w.ndense),
         tau(w.regularization),retained(w.retained),point(w.x),slack(w.s),dual(w.z),
         equality_dual(w.lambda),base(w.base_values),hessian(w.hvalue),matrix(w.kkt) {}
    static bool same_values(const Vec& a,const Vec& b) {
      return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0);
    }
    bool same_model_and_point(const Workspace& w) const {
      if(owner!=&w||tau!=w.regularization||dimension!=w.dim||primal_dimension!=w.nr||
          equality_dimension!=w.mr||borders!=w.ndense||retained!=w.retained||
          !same_values(point,w.x)||!same_values(slack,w.s)||!same_values(dual,w.z)||
          !same_values(equality_dual,w.lambda)||!same_values(base,w.base_values)||
          !same_values(hessian,w.hvalue)||matrix.size()!=w.kkt.size())return false;
      for(std::size_t k=0;k<matrix.size();++k)
        if(matrix[k].i!=w.kkt[k].i||matrix[k].j!=w.kkt[k].j||
            std::memcmp(&matrix[k].value,&w.kkt[k].value,sizeof(double))!=0)return false;
      return true;
    }
    bool matches(const Workspace& w) const {
      return numeric_factor&&same_model_and_point(w)&&numeric_factor==w.current_factor&&
          factor_attempts==w.stats.factor_attempts&&
          (numeric_factor==w.factor.get()||numeric_factor==w.pivot_rescue.get());
    }
  };
  // Only this private repair may prepare a numerically recovered factor at
  // fixed tau. No condensation, atom, model-curvature or backend fallback is
  // admitted. The existing frozen-model/SOC policy above stays unchanged.
  bool factorize_affine_repair(double proposed) {
    if(factorize(proposed,true))return true;
    if(!numeric_pivot_failure(message))return false;
    const AffineRepairFactorPermit unchanged(*this);
    CostScope cost(costs,NewtonCost::Factorization);
    bool ok=recover_numeric_pivot_quality();
    if(!ok)ok=bounded_pivot_recovery(1e-12);
    if(!unchanged.same_model_and_point(*this))
      throw std::runtime_error("affine repair numerical recovery changed its fixed model or point");
    if(ok) {
      const double backend_primal=current_factor->last_primal_regularization();
      const double backend_dual=current_factor->last_dual_regularization();
      if(backend_dual>0.0)++stats.dual_regularized_factors;
      stats.final_backend_primal_regularization=backend_primal;
      stats.final_backend_dual_regularization=backend_dual;
      if(!corrected())++stats.unshifted_factors;
      stats.maximum_regularization=std::max(stats.maximum_regularization,regularization);
    }
    if(st.live_display_level>0)std::cout<<std::setprecision(17)
      <<"[HYBRID-AFFINE-REPAIR-NUMERIC] factor_ok="<<ok<<" tau="<<regularization
      <<" numeric_shift="<<(ok?current_factor->last_primal_regularization():0)
      <<" fixed_model=1 fixed_point=1 same_backend=1 unperturbed_audit_required=1"
      <<" model_margin_not_inferred_from_shifted_inertia=1"<<std::endl;
    return ok;
  }
  Vec hessian_product(const Vec& d) const {
    ++stats.lagrangian_hessian_products;
    Vec v(n,0.0);for(std::size_t k=0;k<hpattern.size();++k){auto [i,j]=hpattern[k];v[i]+=hvalue[k]*d[j];if(i!=j)v[j]+=hvalue[k]*d[i];}
    if(objective.model_active)for(const auto& a:objective.model_atoms)
      objective.add_model_atom(a,a.beta*objective.model_atom_product(a,d),v);
    if((st.convexification_study=="HH"||st.convexification_study=="HS"||st.convexification_study=="SH"||st.convexification_study=="SS"||st.convexification_study=="H0"))for(int i=0;i<mi;++i)if(active[i]&&inequalities[i].model_active)
      for(const auto& a:inequalities[i].model_atoms)
        inequalities[i].add_model_atom(a,z[i]*a.beta*inequalities[i].model_atom_product(a,d),v);
    return v;
  }
  // Reconstruct pivot multipliers and audit the SAME assembled auxiliary
  // equations together at extended precision. No preceding double audit or
  // duplicate pivot reconstruction is needed. The acceptance tolerance and
  // finite-state checks are unchanged; the scratch buffer is reused.
  double majorant_direction_residual_extended(Direction& d,const Vec& rc,
      const Vec& dual,const Vec& eq,const Vec& primal,Vec& ed,Vec& eh,Vec& ep,Vec& ec,
      double& dual_error,double& equality_error,double& primal_error,
      double& complementarity_error,bool allow_slack_balance=true) {
    ++stats.direction_residual_evaluations;
    ++stats.lagrangian_hessian_products;
    auto& wide=majorant_direction_wide;
    wide.assign(dual.size(),detail::AccurateProductSum{});
    for(int j=0;j<n;++j)wide[j].add(dual[j]);
    for(std::size_t k=0;k<hpattern.size();++k) {
      const auto [i,j]=hpattern[k];
      wide[i].add_product(hvalue[k],d.x[j]);
      if(i!=j)wide[j].add_product(hvalue[k],d.x[i]);
    }
    // Objective spectral atoms are represented by KKT borders. Audit their
    // uncondensed product as well, before reconstructing pivot multipliers.
    if(objective.model_active)for(const auto& a:objective.model_atoms) {
      long double product=0;
      for(int j=0;j<n;++j)product+=static_cast<long double>(a.direction[j])*d.x[j];
      for(int j=0;j<n;++j)wide[j].add(static_cast<long double>(a.beta)*a.direction[j]*product);
    }
    if((st.convexification_study=="HH"||st.convexification_study=="HS"||st.convexification_study=="SH"||st.convexification_study=="SS"||st.convexification_study=="H0"))for(int i=0;i<mi;++i)if(active[i]&&inequalities[i].model_active)
      for(const auto& a:inequalities[i].model_atoms) {
        long double product=0;
        for(int j=0;j<n;++j)product+=static_cast<long double>(a.direction[j])*d.x[j];
        for(int j=0;j<n;++j)wide[j].add(static_cast<long double>(z[i])*a.beta*a.direction[j]*product);
      }
    auto add_gradient=[&](const Quadratic& q,double multiplier) {
      q.note(&NewtonCalls::cached_gradient_accumulations);
      const int* columns=q.pooled_support?q.pooled_support:q.support.data();
      const double* values=q.pooled_gradient?q.pooled_gradient:q.gradient.data();
      for(std::size_t k=0;k<q.support.size();++k)
        wide[columns[k]].add_product(multiplier,values[k]);
    };
    auto product=[&](const Quadratic& q,double initial) {
      q.note(&NewtonCalls::jacobian_products);
      const int* columns=q.pooled_support?q.pooled_support:q.support.data();
      const double* values=q.pooled_gradient?q.pooled_gradient:q.gradient.data();
      detail::AccurateProductSum value(initial);
      for(std::size_t k=0;k<q.support.size();++k)
        value.add_product(values[k],d.x[columns[k]]);
      return value.value();
    };
    for(int i=0;i<mi;++i)if(active[i])add_gradient(inequalities[i],d.z[i]);
    for(int e:eq_retained)add_gradient(equalities[e],d.lambda[e]);
    // Each eliminated pivot occurs linearly in exactly one equality. Its
    // multiplier is recovered from this stationarity equation, so repeating
    // that recovery with accurate sums removes avoidable reconstruction
    // error without changing the reduced solution or the auxiliary model.
    for(int e=0;e<me;++e)if(pivot_variable[e]>=0)
      d.lambda[e]=static_cast<double>(-wide[pivot_variable[e]].value()/pivot_coefficient[e]);
    for(int e=0;e<me;++e)if(pivot_variable[e]>=0)
      add_gradient(equalities[e],d.lambda[e]);
    // Audit the unperturbed auxiliary equations. Backend shifts are only
    // a numerical preconditioner for the existing refinement loop.
    for(int j:retained)
      wide[j].add_product(regularization,d.x[j]);
    dual_error=equality_error=primal_error=complementarity_error=0;
    bool finite_residual=true;
    auto accumulate_error=[&](double value,double& error) {
      finite_residual=finite_residual&&std::isfinite(value);
      error=std::max(error,std::abs(value));
    };
    for(int j=0;j<n;++j) {
      ed[j]=static_cast<double>(wide[j].value());
      accumulate_error(ed[j],dual_error);
    }
    for(int e=0;e<me;++e) {
      long double value=product(equalities[e],eq[e]);
      if(equality_stabilization.enabled) {
        detail::AccurateProductSum stabilized(value);
        stabilized.add_product(-equality_stabilization.rho,d.lambda[e]);
        value=stabilized.value();
      }
      eh[e]=static_cast<double>(value);
      accumulate_error(eh[e],equality_error);
    }
    for(int i=0;i<mi;++i) {
      if(active[i]) {
        long double a=product(inequalities[i],primal[i]);
        if(inequality_algorithmic_rho(i)>0) {
          detail::AccurateProductSum stabilized(a);
          stabilized.add_product(-inequality_algorithmic_rho(i),d.z[i]);
          a=stabilized.value();
        }
        detail::AccurateProductSum bsum(rc[i]);bsum.add_product(s[i],d.z[i]);
        const long double b=bsum.value();
        // ds is reconstructed after condensation. Its nearest double to -a
        // need not minimize the residual of BOTH original row equations.
        // Keeping dx/dz/lambda fixed leaves stationarity/equalities unchanged;
        // this certified minimax rounding can only improve the expanded pair.
        if(allow_slack_balance) {
          if(source_primal_barrier(i))d.s[i]=balance_source_slack_direction(a,b,i,d.s[i]);
          else d.s[i]=hybrid_inner::balance_slack_direction(a,b,z[i],d.s[i]).value;
        }
        ep[i]=static_cast<double>(a+d.s[i]);
        if(source_primal_barrier(i))bsum.add(slack_newton_coefficient(i)*d.s[i]);
        else bsum.add_product(z[i],d.s[i]);
        ec[i]=static_cast<double>(bsum.value());
      }
      accumulate_error(ep[i],primal_error);
      // Preserve the historical aggregate precision contract on every block.
      // A separate per-source-row check audits unscaled Newton(eta-mu/s).
      accumulate_error(ec[i],complementarity_error);
    }
    if(!finite_residual||!finite(d.lambda))
      return std::numeric_limits<double>::infinity();
    return std::max({dual_error,equality_error,primal_error,complementarity_error});
  }
  // validate=false preserves its complete-direction legacy contract.
  // Backend solves and finite-state checks remain mandatory. Every direction
  // entering globalization uses the default strict expanded residual audit.
  Direction direction(const Vec& rc,const Vec& dual,const Vec& eq,const Vec& primal,
      int depth=0,bool validate=true,DirectionPurpose purpose=DirectionPurpose::Complete,
      bool frozen_factor=false,const AffineRepairFactorPermit* affine_repair_permit=nullptr) {
    CostScope cost(costs,NewtonCost::Direction);
    const bool majorant=st.dca_engine=="majorants";
    if(affine_repair_permit&&(!frozen_factor||!validate||!majorant||
        !affine_repair_permit->matches(*this)))
      throw std::runtime_error("affine repair frozen numeric factor permit is invalid");
    if(frozen_factor&&current_factor!=factor.get()&&!affine_repair_permit)
      throw std::runtime_error("frozen correction requires the configured linear backend");
    if(purpose==DirectionPurpose::ResidualCorrection&&(!majorant||depth<=0))
      throw std::invalid_argument("partial residual-correction data require an internal majorant refinement");
    if(depth>0)++stats.direction_corrections;
    Direction d;d.x.assign(n,0.0);d.z.assign(mi,0.0);d.s.assign(mi,0.0);
    d.lambda.assign(me,0.0);
    if(!validate)d.residual=std::numeric_limits<double>::infinity();
    Vec offset(n,0.0);
    bool nonzero_offset=!majorant;
    for(int e=0;e<me;++e)if(pivot_variable[e]>=0) {
      offset[pivot_variable[e]]=-eq[e]/pivot_coefficient[e];
      nonzero_offset=nonzero_offset||offset[pivot_variable[e]]!=0;
    }
    Vec b;
    if(nonzero_offset) {b=hessian_product(offset);for(int j=0;j<n;++j)b[j]+=dual[j];}
    else b=dual;
    Vec rhs(dim,0.0);
    for(int j=0;j<n;++j)for(int t=tbegin[j];t<tbegin[j+1];++t)rhs[tcolumn[t]]-=tvalue[t]*b[j];
    for(int a=0;a<mr;++a)rhs[nr+a]=-eq[eq_retained[a]]-
        (nonzero_offset?equalities[eq_retained[a]].directional(offset):0.0);
    for(int i=0;i<mi;++i)if(active[i]) {
      const double jo=nonzero_offset?inequalities[i].directional(offset):0.0;
      const auto& g=jr[i];
      if(g.dense>=0)rhs[nr+mr+g.dense]=source_primal_barrier(i)?
          static_cast<double>(static_cast<long double>(s[i])*rc[i]/source_barrier_mu-primal[i]-jo):
          rc[i]/z[i]-primal[i]-jo;
      else {
        const double v=source_primal_barrier(i)?
            static_cast<double>((static_cast<long double>(source_barrier_mu)*(primal[i]+jo)-
                static_cast<long double>(s[i])*rc[i])/source_barrier_denominator(i)):
            (inequality_algorithmic_rho(i)>0?
            static_cast<double>((static_cast<long double>(z[i])*(primal[i]+jo)-rc[i])/
                stabilized_slack_denominator(i)):(z[i]*(primal[i]+jo)-rc[i])/s[i]);
        for(std::size_t a=0;a<g.support.size();++a)rhs[g.support[a]]-=g.values[a]*v;
      }
    }
    ++stats.linear_solves;
    const bool using_recovery=current_factor==recovery.get(),using_native=current_factor==pivoted.get();
    const auto special_timer=(using_recovery||using_native)?Clock::now():Clock::time_point{};
    if(using_recovery) {
      const auto& a=ar[recovery_equation];
      const double v=(rhs[nr+recovery_equation]/recovery_scale)*recovery_shift;
      for(std::size_t i=0;i<a.support.size();++i)rhs[a.support[i]]+=v*recovery_normal[i];
    }
    bool solved;{CostScope backend_cost(costs,NewtonCost::LinearBackend);solved=finite(rhs)&&current_factor->solve(rhs,message)&&finite(rhs);}
    if(using_recovery){++stats.normal_recovery_linear_solves;stats.normal_recovery_solve_seconds+=seconds(special_timer);}
    if(using_native){++stats.native_linear_solves;stats.native_solve_seconds+=seconds(special_timer);}
    if(!solved) {
      if(frozen_factor)throw std::runtime_error("frozen-factor correction solve rejected: "+message);
      const std::string reason=message;
      if(current_factor==factor.get()&&current_factor->increase_quality(message)) {
        ++stats.factor_attempts;
        stats.backend_regularization_attempts+=current_factor->last_regularization_attempts();
        ++stats.regularization_increases;
        const double backend_primal=current_factor->last_primal_regularization();
        const double backend_dual=current_factor->last_dual_regularization();
        if(backend_dual>0.0)++stats.dual_regularized_factors;
        stats.maximum_backend_primal_regularization=std::max(
            stats.maximum_backend_primal_regularization,backend_primal);
        stats.maximum_backend_dual_regularization=std::max(
            stats.maximum_backend_dual_regularization,backend_dual);
        stats.final_backend_primal_regularization=backend_primal;
        stats.final_backend_dual_regularization=backend_dual;
        return direction(rc,dual,eq,primal,depth,validate,purpose);
      }
      if(recover_condensation("linear-solve")&&factorize(regularization))return direction(rc,dual,eq,primal,depth,validate,purpose);
      if(!using_native&&pivoted_sparse_fallback_enabled()&&
         native_factor("linear-solve",reason))return direction(rc,dual,eq,primal,depth,validate,purpose);
      // A backend residual failure can occur even when factor() succeeded.
      // Retry the same auxiliary matrix through bounded regularization; the
      // complete direction still has to satisfy its unperturbed equations.
      if(majorant && depth==0 && reason.find("backward error too large")!=std::string::npos) {
        const double next=current_factor==pivot_rescue.get()?
            10.0*pivot_rescue->last_primal_regularization():1e-12;
        if(bounded_pivot_recovery(next))return direction(rc,dual,eq,primal,depth,validate,purpose);
      }
      throw std::runtime_error("lazy-newton linear solve: "+reason);
    }
    for(int j=0;j<n;++j) {
      if(majorant) {
        // The reduced solve and its affine lift represent the SAME step.
        // Form its product-sum before rounding to a stored coordinate; a
        // sequence of double axpy operations can destroy small components
        // when a lifted row contains large cancelling terms.
        detail::AccurateProductSum lifted(offset[j]);
        for(int t=tbegin[j];t<tbegin[j+1];++t)
          lifted.add_product(tvalue[t],rhs[tcolumn[t]]);
        d.x[j]=static_cast<double>(lifted.value());
      } else {
        d.x[j]=offset[j];for(int t=tbegin[j];t<tbegin[j+1];++t)d.x[j]+=tvalue[t]*rhs[tcolumn[t]];
      }
    }
    for(int a=0;a<mr;++a)d.lambda[eq_retained[a]]=rhs[nr+a];
    int balanced_slack_rows=0;
    for(int i=0;i<mi;++i)if(active[i]) {
      if(majorant) {
        const auto& q=inequalities[i];
        q.note(&NewtonCalls::jacobian_products);
        const int* columns=q.pooled_support?q.pooled_support:q.support.data();
        const double* gradient=q.pooled_gradient?q.pooled_gradient:q.gradient.data();
        detail::AccurateProductSum row(primal[i]);
        for(std::size_t k=0;k<q.support.size();++k)
          row.add_product(gradient[k],d.x[columns[k]]);
        const long double a=row.value();
        // Use the unrounded row value in the condensed back substitution,
        // rather than multiply a previously rounded, potentially huge ds.
        d.z[i]=jr[i].dense>=0?rhs[nr+mr+jr[i].dense]:(source_primal_barrier(i)?
            static_cast<double>((static_cast<long double>(source_barrier_mu)*a-
                static_cast<long double>(s[i])*rc[i])/source_barrier_denominator(i)):
            static_cast<double>((static_cast<long double>(z[i])*a-rc[i])/
                (inequality_algorithmic_rho(i)>0?stabilized_slack_denominator(i):static_cast<long double>(s[i]))));
        if(inequality_algorithmic_rho(i)>0) {
          detail::AccurateProductSum slack(-a);
          slack.add_product(inequality_algorithmic_rho(i),d.z[i]);
          d.s[i]=static_cast<double>(slack.value());
        } else d.s[i]=static_cast<double>(-a);
        if(source_primal_barrier(i)) {
          // Residual corrections also need a represented ds satisfying the
          // true barrier row. Reconstructing only from the primal equation
          // can subtract O(1) terms to obtain an O(s^2) correction when s is
          // tiny. The same row-pair rounding used by the expanded audit is
          // required before returning an unaudited internal correction.
          detail::AccurateProductSum primal_row(a),barrier_row(rc[i]);
          primal_row.add_product(-inequality_algorithmic_rho(i),d.z[i]);
          barrier_row.add_product(s[i],d.z[i]);
          d.s[i]=balance_source_slack_direction(primal_row.value(),barrier_row.value(),i,d.s[i]);
        }
      } else {
        d.s[i]=-primal[i]-inequalities[i].directional(d.x);
        d.z[i]=jr[i].dense>=0?rhs[nr+mr+jr[i].dense]:(-rc[i]-z[i]*d.s[i])/s[i];
      }
      if(majorant&&st.study_controller=="certified"&&jr[i].dense>=0&&(st.convexification_study=="HH"||
          st.convexification_study=="HS"||st.convexification_study=="SH"||st.convexification_study=="SS")) {
        const auto& q=inequalities[i];
        const int* columns=q.pooled_support?q.pooled_support:q.support.data();
        const double* gradient=q.pooled_gradient?q.pooled_gradient:q.gradient.data();
        long double a=primal[i];
        for(std::size_t k=0;k<q.support.size();++k)
          a+=static_cast<long double>(gradient[k])*d.x[columns[k]];
        if(inequality_algorithmic_rho(i)>0)a-=static_cast<long double>(inequality_algorithmic_rho(i))*d.z[i];
        const long double b=static_cast<long double>(rc[i])+static_cast<long double>(s[i])*d.z[i];
        const auto balance=hybrid_inner::balance_slack_direction(a,b,z[i],d.s[i]);
        d.s[i]=balance.value;balanced_slack_rows+=balance.changed;
      }
    }
    Vec hd;
    // The fused majorant audit reconstructs the complete pivot multipliers.
    // Internal numerical corrections need retained multipliers only because
    // the parent reconstructs pivots from the combined direction afterwards.
    // Legacy/raw Complete calls still return all multipliers as before.
    if(!majorant||!validate||(depth>0&&purpose==DirectionPurpose::Complete)) {
      hd=hessian_product(d.x);b=hd;for(int j=0;j<n;++j)b[j]+=dual[j];
      for(int i=0;i<mi;++i)if(active[i])inequalities[i].add_gradient(d.z[i],b);
      for(int a:eq_retained)equalities[a].add_gradient(d.lambda[a],b);
      for(int e=0;e<me;++e)if(pivot_variable[e]>=0)d.lambda[e]=-b[pivot_variable[e]]/pivot_coefficient[e];
    }
    Vec majorant_ed,majorant_eh,majorant_ep,majorant_ec;
    double majorant_scale=1;
    if(majorant&&depth==0&&validate) {
      majorant_ed.resize(n);majorant_eh.resize(me);
      majorant_ep.assign(mi,0);majorant_ec.assign(mi,0);
      majorant_scale=std::max(1.0,inf_norm(dual));
      for(double value:eq)majorant_scale=std::max(majorant_scale,std::abs(value));
      for(int i=0;i<mi;++i)if(active[i])
        majorant_scale=std::max({majorant_scale,std::abs(primal[i]),std::abs(rc[i])});
    }
    // A correction is never an accepted direction. Its finite entries are
    // checked below; the caller audits the complete corrected direction in
    // original coordinates before acceptance. Avoid a discarded inner audit.
    std::array<double,8> refinement_history{};
    for(int refine=0;depth==0&&validate&&refine<8;++refine) {
      CostScope audit_cost(costs,NewtonCost::DirectionAudit);
      Vec legacy_ed,legacy_eh,legacy_ep,legacy_ec;
      Vec& ed=majorant?majorant_ed:legacy_ed;
      Vec& eh=majorant?majorant_eh:legacy_eh;
      Vec& ep=majorant?majorant_ep:legacy_ep;
      Vec& ec=majorant?majorant_ec:legacy_ec;
      double dual_error=0;
      double equality_error=0.0,primal_error=0.0,complementarity_error=0.0;
      double error=0,scale=majorant_scale;
      const double backend_primal=current_factor->last_primal_regularization();
      const double backend_dual=current_factor->last_dual_regularization();
      if(majorant) {
        error=majorant_direction_residual_extended(d,rc,dual,eq,primal,ed,eh,ep,ec,
            dual_error,equality_error,primal_error,complementarity_error);
      } else {
        // Keep nonmajorant arithmetic and its double-precision audit intact.
        ++stats.direction_residual_evaluations;
        ed=refine==0?std::move(hd):hessian_product(d.x);
        eh.assign(me,0.0);ep.assign(mi,0.0);ec.assign(mi,0.0);
        for(int j=0;j<n;++j)ed[j]+=dual[j];
        for(int e=0;e<me;++e)equalities[e].add_gradient(d.lambda[e],ed);
        for(int i=0;i<mi;++i)if(active[i])inequalities[i].add_gradient(d.z[i],ed);
        if(st.lazy_newton_atomic_regularization) {
          if(atoms&&atoms->active_count()) {CostScope atom_cost(costs,NewtonCost::Atoms);const auto timer=Clock::now();atoms->product(d.x,retained,ed);const double elapsed=seconds(timer);stats.atom_update_seconds+=elapsed;stats.atom_product_seconds+=elapsed;}
        } else for(int j:retained)ed[j]+=regularization*d.x[j];
        if(backend_primal!=0.0)for(int j:retained)ed[j]+=backend_primal*d.x[j];
        dual_error=inf_norm(ed);error=dual_error;scale=std::max(1.0,inf_norm(dual));
        for(int e=0;e<me;++e){eh[e]=eq[e]+equalities[e].directional(d.x);scale=std::max(scale,std::abs(eq[e]));}
        if(backend_dual!=0.0)for(int a=0;a<mr;++a)
          eh[eq_retained[a]]-=backend_dual*d.lambda[eq_retained[a]];
        equality_error=inf_norm(eh);error=std::max(error,equality_error);
        for(int i=0;i<mi;++i)if(active[i]) {
          ep[i]=primal[i]+inequalities[i].directional(d.x)+d.s[i];ec[i]=rc[i]+z[i]*d.s[i]+s[i]*d.z[i];
          primal_error=std::max(primal_error,std::abs(ep[i]));
          complementarity_error=std::max(complementarity_error,std::abs(ec[i]));
          error=std::max({error,std::abs(ep[i]),std::abs(ec[i])});
          scale=std::max({scale,std::abs(primal[i]),std::abs(rc[i])});
        }
      }
      d.residual=error/scale;
      if(majorant&&source_primal_barrier_mode)
        d.residual=std::max(d.residual,source_stationarity_direction_backward_error(d,rc));
      refinement_history[refine]=d.residual;
      if((st.convexification_study=="HH"||st.convexification_study=="HS"||st.convexification_study=="SH"||st.convexification_study=="SS"||st.convexification_study=="H0")&&st.live_display_level>0)
        std::cout<<"[HYBRID-BORDER-RESIDUAL] residual="<<std::setprecision(17)<<d.residual
          <<" threshold="<<st.kkt_backward_error_reject
          <<" accepted="<<(std::isfinite(d.residual)&&d.residual<=st.kkt_backward_error_reject)
          <<" refinement="<<refine<<" objective_atoms="<<hybrid_border_objective_atoms
          <<" balanced_slack_rows="<<balanced_slack_rows
          <<" inequality_atoms="<<hybrid_border_inequality_atoms<<std::endl;
      if(std::isfinite(d.residual)&&d.residual<=st.kkt_backward_error_reject)break;
      if(refine==7) {
        if(majorant&&st.live_display_level>0&&dual_error>0) {
          const int worst=static_cast<int>(std::max_element(ed.begin(),ed.end(),
              [](double a,double b){return std::abs(a)<std::abs(b);})-ed.begin());
          long double magnitude=0;
          for(int i=0;i<mi;++i)if(active[i]) {
            const auto& q=inequalities[i];
            for(std::size_t k=0;k<q.support.size();++k)if(q.support[k]==worst)
              magnitude+=std::abs(static_cast<long double>(d.z[i])*q.gradient[k]);
          }
          for(int e=0;e<me;++e) {
            const auto& q=equalities[e];
            for(std::size_t k=0;k<q.support.size();++k)if(q.support[k]==worst)
              magnitude+=std::abs(static_cast<long double>(d.lambda[e])*q.gradient[k]);
          }
          std::cout<<"[KKT-WORST-DUAL] coordinate="<<worst
            <<" residual="<<ed[worst]<<" rhs="<<dual[worst]
            <<" dx="<<d.x[worst]<<" pivot_equation="<<pivot_equation[worst]
            <<" multiplier_magnitude="<<double(magnitude)<<std::endl;
        }
        if(majorant&&st.live_display_level>0&&complementarity_error>0) {
          int worst=-1;
          for(int i=0;i<mi;++i)if(active[i]&&(worst<0||std::abs(ec[i])>std::abs(ec[worst])))worst=i;
          if(worst>=0)std::cout<<"[KKT-WORST-COMPLEMENTARITY] row="<<worst
            <<" dense="<<jr[worst].dense<<" rc="<<std::setprecision(17)<<rc[worst]
            <<" s="<<s[worst]<<" z="<<z[worst]<<" ds="<<d.s[worst]
            <<" dz="<<d.z[worst]<<" residual="<<ec[worst]<<std::endl;
        }
        if(frozen_factor)throw std::runtime_error("frozen-factor correction residual rejected");
        if(majorant&&st.live_display_level>0) {
          CostScope display_cost(costs,NewtonCost::Diagnostics);
          std::cout<<"[KKT-REFINEMENT-LIMIT] k="<<stats.accepted_steps<<" residuals=";
          for(double value:refinement_history)std::cout<<value<<",";
          std::cout<<std::endl;
        }
        if(recover_condensation("original-direction-residual")&&factorize(regularization))
          return direction(rc,dual,eq,primal,0,validate,purpose);
        if(majorant&&recover_condensation_precision(d,scale)) {
          if(!factorize(regularization))
            throw std::runtime_error("lazy-newton precision recovery factorization failed: "+message);
          Direction recovered=direction(rc,dual,eq,primal,0,validate,purpose);
          if(st.live_display_level>0||st.verbose) {
            CostScope display_cost(costs,NewtonCost::Diagnostics);
            std::cout<<"[KKT-ACCURACY-RECOVERED] k="<<stats.accepted_steps
                     <<" before="<<d.residual<<" after="<<recovered.residual
                     <<" tolerance="<<st.kkt_backward_error_reject
                     <<" backend="<<current_factor->name()<<std::endl;
          }
          return recovered;
        }
        if(majorant&&recover_precision_pivoting())
          return direction(rc,dual,eq,primal,0,validate,purpose);
        // A tiny normwise backend error does not certify these expanded
        // equations. Near dependent constraint rows can create enormous
        // cancelling dual terms, so binary64 refinement stalls although the
        // factorization itself succeeded. Reuse the existing bounded native
        // preconditioner in that case as well. Neither its diagonal shifts
        // nor its backward error enter the original-space acceptance test.
        // Each recursive attempt advances the finite 1e-12 .. 1e-8 ladder;
        // no change is made to the majorant, its proximal rho or the tolerance.
        if(majorant&&st.time_limit_seconds>0.0) {
          const double next=current_factor==pivot_rescue.get()?
              10.0*pivot_rescue->last_primal_regularization():1e-12;
          if(bounded_pivot_recovery(next)) {
            const double shift=current_factor->last_primal_regularization();
            stats.final_backend_primal_regularization=shift;
            stats.final_backend_dual_regularization=current_factor->last_dual_regularization();
            if(st.live_display_level>0||st.verbose)
              std::cout<<"[KKT-EXPANDED-PRECISION-RECOVERY] k="<<stats.accepted_steps
                <<" rejected_residual="<<d.residual<<" shift="<<shift
                <<" same_backend=1 original_audit_required=1"<<std::endl;
            return direction(rc,dual,eq,primal,0,validate,purpose);
          }
        }
        if(current_factor!=pivoted.get()&&pivoted_sparse_fallback_enabled()&&
           native_factor("original-direction-residual","expanded original Newton residual exceeded tolerance"))
          return direction(rc,dual,eq,primal,0,validate,purpose);
        std::ostringstream os;os<<"lazy-newton expanded Newton residual rejected: "
          <<std::scientific<<d.residual<<" (dual="<<dual_error
          <<", equality="<<equality_error<<", primal="<<primal_error
          <<", complementarity="<<complementarity_error<<", scale="<<scale
          <<", backend_berr="<<current_factor->last_backward_error()
          <<", rho_x="<<backend_primal<<", delta_d="<<backend_dual;
        if(majorant)os<<", audit_precision=compensated-long-double-fma";
        os<<")";
        throw std::runtime_error(os.str());
      }
      // Solve a correction with the same matrix and original-space residuals.
      Direction correction=direction(ec,ed,eh,ep,depth+1,validate,
          majorant?DirectionPurpose::ResidualCorrection:DirectionPurpose::Complete,frozen_factor,
          affine_repair_permit);
      axpy(1.0,correction.x,d.x);axpy(1.0,correction.lambda,d.lambda);
      axpy(1.0,correction.s,d.s);axpy(1.0,correction.z,d.z);
    }
    if(depth==0&&validate)stats.maximum_direction_residual=std::max(stats.maximum_direction_residual,d.residual);
    if(!finite(d.x)||!finite(d.lambda)||!finite(d.z)||!finite(d.s))throw std::runtime_error("nonfinite Newton direction");
    if(affine_repair_permit&&!affine_repair_permit->matches(*this))
      throw std::runtime_error("affine repair frozen numeric factor changed during correction");
    return d;
  }
  double fraction(const Direction& d,double tau,int part=0,double ceiling=1.0) const {
    double a=ceiling;
    for(int i=0;i<mi;++i)if(active[i]) {
      if(part!=2&&d.s[i]<0)a=std::min(a,-tau*s[i]/d.s[i]);
      if(part!=1&&requires_positive_dual(i)&&d.z[i]<0)a=std::min(a,-tau*z[i]/d.z[i]);
    }
    return a;
  }
  double theta(const Vec& point,const Vec& slack) const {
    double v=0.0;for(const auto& q:equalities)v+=std::abs(q.at(point));
    for(int i=0;i<mi;++i)if(active[i])v+=std::abs(inequalities[i].at(point)+slack[i]);
    return v;
  }
  double merit(const Vec& point,const Vec& slack,double mu) const {
    double v=objective.at(point);
    for(int i=0;i<mi;++i)if(active[i]) {
      if(!(slack[i]>0.0))return std::numeric_limits<double>::infinity();
      v-=mu*std::log(slack[i]);
    }
    return v+penalty*theta(point,slack);
  }
  double perturbed_residual(const Vec& point,const Vec& dual,const Vec& slack,const Vec& multiplier,double target) const {
    Vec gradient(n,0.0);objective.add_gradient_at(point,1.0,gradient);double residual=0.0;
    for(int e=0;e<me;++e){equalities[e].add_gradient_at(point,dual[e],gradient);residual=std::max(residual,std::abs(equalities[e].at(point)));}
    for(int i=0;i<mi;++i)if(active[i]) {
      inequalities[i].add_gradient_at(point,multiplier[i],gradient);
      residual=std::max({residual,std::abs(inequalities[i].at(point)+slack[i]),std::abs(slack[i]*multiplier[i]-target)});
    }
    return finite(gradient)&&std::isfinite(residual)?std::max(residual,inf_norm(gradient)):std::numeric_limits<double>::infinity();
  }
  double original_primal() const {
    double v=0.0;for(int e=0;e<me;++e)v=std::max(v,std::abs(h[e]/equalities[e].scale));
    for(int i=0;i<mi;++i)v=std::max(v,std::max(0.0,inequalities[i].value/inequalities[i].scale));
    return v;
  }
  bool certify(Result& r) const {
    CostScope cost(costs,NewtonCost::Certificate);
    return st.lazy_newton_sparse_certificate?certify_indexed_original(original,st,r,&original_linear_support):certify_dense_original(original,st,r);
  }
  void export_state(Result& r) const {
    export_state_at(r,x,s,z,lambda);
  }
  void export_state_at(Result& r,const Vec& px,const Vec& ps,
                       const Vec& pz,const Vec& plambda) const {
    if(st.termination_test=="ipopt") {
      r.termination_scaling_available=true;
      r.termination_objective_scale=objective.scale;
      r.termination_equality_scales.resize(me);
      for(int e=0;e<me;++e)r.termination_equality_scales[e]=equalities[e].scale;
      r.termination_inequality_scales.resize(original.inequalities.size());
      r.termination_inequality_slacks.assign(original.inequalities.size(),0.0);
      for(int i=0;i<mi;++i)if(inequality_kind[i]==0) {
        const auto& q=inequalities[i];const int origin=inequality_origin[i];
        r.termination_inequality_scales[origin]=q.scale;
        if(active[i]) {
          const double shift=q.model_active?std::max(0.0,q.at(px)-q.original_at(px)):0;
          r.termination_inequality_slacks[origin]=(ps[i]+shift)/q.scale;
        }
      }
    }
    r.x=fixed;for(int j=0;j<n;++j)r.x[free[j]]=px[j];
    r.certificate_equality_multipliers.resize(me);
    for(int e=0;e<me;++e)r.certificate_equality_multipliers[e]=plambda[e]*equalities[e].scale/objective.scale;
    r.certificate_inequality_multipliers.assign(original.inequalities.size(),0.0);
    r.certificate_lower_multipliers.assign(original.n,0.0);r.certificate_upper_multipliers.assign(original.n,0.0);
    for(int i=0;i<mi;++i) {
      double v=active[i]?pz[i]*inequalities[i].scale/objective.scale:0.0;
      if(inequality_kind[i]==0)r.certificate_inequality_multipliers[inequality_origin[i]]=v;
      else if(inequality_kind[i]<0)r.certificate_lower_multipliers[inequality_origin[i]]=v;
      else r.certificate_upper_multipliers[inequality_origin[i]]=v;
    }
  }
};
#include "lazy_newton_live.hpp"
} // namespace

bool certify_lazy_newton_original(const QCQPProblem& p,const Settings& st,Result& r) {
  return st.lazy_newton_sparse_certificate?certify_indexed_original(p,st,r,nullptr):certify_dense_original(p,st,r);
}

#include "dca_filter_engine.hpp"

Result solve_atomic_newton(const QCQPProblem& problem,const Settings& requested) {
  Settings st=requested;
  const int level=st.live_display_level<0?(st.live_curvature?4:st.live_iterations?1:0):st.live_display_level;
  if(level<0||level>4)throw std::runtime_error("display-level must be between 0 and 4");
  st.live_display_level=level;st.live_iterations=level>0;st.live_curvature=level>=4;st.verbose=false;

  problem.validate();
  for(double v:{st.lazy_newton_activation_margin,st.lazy_newton_initial_mu,st.lazy_newton_objective_gradient_target,st.lazy_newton_constraint_gradient_target,
                st.lazy_newton_regularization_initial,st.lazy_newton_condensation_ratio,st.time_limit_seconds,st.feasibility_tolerance,
                st.dual_infeasibility_tolerance,st.complementarity_tolerance,st.stationarity_tolerance})
    if(!(v>0.0)||!std::isfinite(v))throw std::runtime_error("lazy-newton parameters must be positive and finite");
  if(st.lazy_newton_atom_guided_after<0||st.lazy_newton_atom_guided_after>14)throw std::runtime_error("atom guided-after must be between 0 and 14");
  if(!(st.lazy_newton_atom_decay>0.0&&st.lazy_newton_atom_decay<1.0)||
     !(st.lazy_newton_regularization_decay>0.0&&st.lazy_newton_regularization_decay<1.0)||
     !(st.lazy_newton_fraction_to_boundary>0.0&&st.lazy_newton_fraction_to_boundary<1.0)||st.lazy_newton_dense_threshold<1)
    throw std::runtime_error("invalid lazy-newton regularization/fraction/threshold");
  if(st.max_outer_iterations<1)throw std::runtime_error("lazy-newton requires max-outer >= 1");
  if((st.live_iterations||st.live_curvature)&&(st.live_iteration_every<1||st.live_iteration_header_every<1))
    throw std::runtime_error("live iteration intervals must be positive");
  if(st.inner_backend=="ipopt")throw std::runtime_error("lazy-newton uses its own native primal-dual kernel; inner-backend=ipopt is incompatible");
  Result result;result.effective_outer_algorithm="lazy-newton";result.lazy_dca_equality_model="persistent-tangent-lagrangian";
  result.dca_dual_step=st.dca_dual_step;
  result.linear_solver_backend="lazy-newton/"+st.linear_backend;result.x=problem.x0;
  result.lazy_newton.live_iterations_enabled=st.live_iterations||st.live_curvature;
  result.lazy_newton.live_curvature_enabled=st.live_curvature;
  result.lazy_newton.packed_gradients_enabled=st.lazy_newton_packed_gradients;
  result.lazy_newton.predictor_corrector_requested=st.lazy_newton_predictor_corrector;
  result.lazy_newton.atomic_regularization_enabled=st.lazy_newton_atomic_regularization;
  result.lazy_newton.atom_decay=st.lazy_newton_atom_decay;
  result.lazy_newton.atom_transport_enabled=st.lazy_newton_atom_transport;
  result.lazy_newton.atom_step_decay_enabled=st.lazy_newton_atom_step_decay;
  result.lazy_newton.atom_guided_enabled=st.lazy_newton_atom_guided;
  result.lazy_newton.atom_guided_after=st.lazy_newton_atom_guided_after;
  result.lazy_newton.sparse_certificate_enabled=st.lazy_newton_sparse_certificate;
  result.lazy_newton.atom_energy_sparsify_enabled=st.lazy_newton_atom_energy_sparsify;
  result.lazy_newton.atom_rayleigh_gate_enabled=st.lazy_newton_atom_rayleigh_gate;
  auto start=Clock::now();detail::NewtonCostLedger costs(result.lazy_newton,start);
  result.lazy_newton.display_level=level;std::unique_ptr<Workspace> owner;
  bool all_fixed=true;for(int j=0;j<problem.n;++j)all_fixed=all_fixed&&std::isfinite(problem.lb[j])&&problem.lb[j]==problem.ub[j];
  if(all_fixed) {
    result.x=problem.lb;result.certificate_equality_multipliers.assign(problem.equalities.size(),0.0);
    result.certificate_inequality_multipliers.assign(problem.inequalities.size(),0.0);
    result.certificate_lower_multipliers.assign(problem.n,0.0);result.certificate_upper_multipliers.assign(problem.n,0.0);
    bool ok;{CostScope certificate_cost(costs,NewtonCost::Certificate);ok=certify_lazy_newton_original(problem,st,result);}
    result.status=ok?Status::Solved:Status::RestorationFailure;
    result.message=ok?"all coordinates fixed; original certificate satisfied":"fixed coordinates violate the original constraints";
    result.lazy_newton.fixed_variables=problem.n;
    if(level>0){CostScope display_cost(costs,NewtonCost::Diagnostics);
      std::cout<<(level<4?"\nEXIT: ":"[LAZY-NEWTON-END] ")<<result.message<<"; no free variables, no Newton step"<<std::endl;}
    result.elapsed_seconds=seconds(start);costs.finish();return result;
  }
  try {
    auto timer=Clock::now();{CostScope preparation(costs,NewtonCost::Preparation);owner=std::make_unique<Workspace>(problem,st,result.lazy_newton,costs);}
    auto& w=*owner;auto& stats=result.lazy_newton;stats.setup_seconds=seconds(timer);
    LiveNewton live(w,start);
    if(live.enabled)live.emit(live.state(0),"INIT",true);
    double proposed_regularization=0.0;int full_streak=0;
    // Near-feasible cold starts can have a tiny initial barrier but large
    // stationarity residuals. Their second-order complementarity product is
    // not a reliable nonlinear predictor. Keep the established centering
    // route unless adaptive initialization raised the barrier above its floor.
    bool corrector_enabled=st.lazy_newton_predictor_corrector&&stats.initial_barrier>st.lazy_newton_initial_mu;
    stats.predictor_corrector_enabled=corrector_enabled;
    stats.predictor_corrector_initialization_guard=st.lazy_newton_predictor_corrector&&!corrector_enabled;
    result.status=Status::MaxOuterIterations;result.message="lazy-newton iteration limit";
    for(int iteration=0;iteration<st.max_outer_iterations;) {
      if(seconds(start)>=st.time_limit_seconds){result.status=Status::TimeLimit;result.message="lazy-newton time limit";break;}
      timer=Clock::now();w.evaluate();stats.evaluation_seconds+=seconds(timer);
      double mu=0.0,comp=0.0;int count=0;
      for(int i=0;i<w.mi;++i)if(w.active[i]){double c=w.s[i]*w.z[i];mu+=c;comp=std::max(comp,c);++count;}
      mu=count?mu/count:0.0;
      const double primal=w.original_primal(),dual=inf_norm(w.rd)/w.objective.scale;
      std::optional<LazyNewtonIteration> live_row;
      if(live.enabled){live_row.emplace(live.state(iteration));live.begin_iteration(*live_row);}
      if(st.verbose)std::cout<<"[LAZY-NEWTON] k="<<iteration<<" f="<<std::setprecision(12)<<w.objective.value/w.objective.scale
          <<" primal="<<primal<<" dual="<<dual<<" comp="<<comp/w.objective.scale<<" mu="<<mu<<" rho="<<proposed_regularization<<" ws="<<count<<"\n";
      if(primal<=st.feasibility_tolerance&&dual<=st.dual_infeasibility_tolerance&&comp/w.objective.scale<=st.complementarity_tolerance) {
        timer=Clock::now();bool ok;{CostScope certificate_cost(costs,NewtonCost::Certificate);w.export_state(result);ok=w.certify(result);}stats.certificate_seconds+=seconds(timer);
        if(ok){result.status=Status::Solved;result.message="original QCQP KKT certificate satisfied by Lazy–Newton";break;}
      }
      if(st.lazy_newton_atomic_regularization&&w.atoms&&primal*primal<=st.feasibility_tolerance&&
         dual*dual<=st.dual_infeasibility_tolerance&&w.atoms->active_count()) {
        CostScope atom_cost(costs,NewtonCost::Atoms);w.atoms->clear_weights();++stats.regularization_decreases;
      }
      timer=Clock::now();w.update_matrix();stats.assembly_seconds+=seconds(timer);
      w.atom_refresh_pending=true;
      if(!st.lazy_newton_lazy_curvature)proposed_regularization=0.0;
      else if(proposed_regularization>0.0)++stats.curvature_reuses;
      if(live.enabled)live_row->regularization_reused=st.lazy_newton_lazy_curvature&&
          (proposed_regularization>0.0||(w.atoms&&w.atoms->active_count()>0));
      timer=Clock::now();bool factored=w.factorize(proposed_regularization);stats.factor_seconds+=seconds(timer);
      if(!factored){
        if(live.enabled){live_row->regularization=w.regularization;live.emit(*live_row,"FACTFAIL",true);}
        result.status=Status::NumericalError;result.message=w.message;break;
      }
      timer=Clock::now();Vec rc(w.mi,0.0);
      for(int i=0;i<w.mi;++i)if(w.active[i])rc[i]=w.s[i]*w.z[i];
      Direction affine=w.direction(rc,w.rd,w.h,w.rp),direction=affine;
      double target=0.0;
      if(count>0) {
        double alpha=w.fraction(affine,1.0,1),alpha_dual=w.fraction(affine,1.0,2),mu_aff=0.0;
        for(int i=0;i<w.mi;++i)if(w.active[i])mu_aff+=(w.s[i]+alpha*affine.s[i])*(w.z[i]+alpha_dual*affine.z[i]);
        mu_aff/=count;
        double sigma=std::clamp(std::pow(std::max(0.0,mu_aff)/std::max(mu,1e-300),3.0),1e-8,0.95);
        target=std::max(0.01*w.objective.scale*std::min(st.complementarity_tolerance,st.stationarity_tolerance),sigma*mu);
        for(int i=0;i<w.mi;++i)if(w.active[i])rc[i]=w.centered_complementarity_rhs(i,target)+(corrector_enabled?affine.s[i]*affine.z[i]:0.0);
        direction=w.direction(rc,w.rd,w.h,w.rp);
      }
      stats.solve_seconds+=seconds(timer);
      double alpha=0.0,alpha_primal_max=0.0,alpha_dual_max=0.0;
      double alpha_dual=0.0,alpha_lambda=0.0,theta=0.0,slope=0.0;
      Vec tx(w.n),ts(w.mi),tz(w.mi),tl(w.me);
      bool accepted=false,expanded=false;
      const double penalty_before_corrector=w.penalty;
      for(int corrector_attempt=0;corrector_attempt<2;++corrector_attempt) {
      if(live.enabled)live.direction(*live_row,direction);
      CostScope globalization_cost(costs,NewtonCost::Globalization);
      timer=Clock::now();alpha=w.fraction(direction,st.lazy_newton_fraction_to_boundary,1);
      alpha_primal_max=alpha;alpha_dual_max=w.fraction(direction,st.lazy_newton_fraction_to_boundary,2);
      theta=w.theta(w.x,w.s);
      slope=w.objective.directional(direction.x);
      for(int i=0;i<w.mi;++i)if(w.active[i])slope-=target*direction.s[i]/w.s[i];
      w.penalty=std::max(w.penalty,1.1*std::max(inf_norm(w.lambda),inf_norm(w.z)));
      if(theta>1e-14&&slope-w.penalty*theta>=-1e-8*theta)w.penalty=std::max(w.penalty,(slope+std::abs(slope)+1.0)/theta);
      slope-=w.penalty*theta;
      const double current=w.merit(w.x,w.s,target);
      for(int ls=0;ls<st.line_search_max_steps;++ls) {
        const double backtrack=alpha_primal_max>0.0?alpha/alpha_primal_max:0.0;
        const auto step=detail::primal_dual_step(
            alpha_primal_max,alpha_dual_max,backtrack,st.dca_dual_step);
        alpha_dual=step.bound_dual;alpha_lambda=step.equality_dual;
        for(int j=0;j<w.n;++j)tx[j]=w.x[j]+alpha*direction.x[j];
        for(int e=0;e<w.me;++e)tl[e]=w.lambda[e]+alpha_lambda*direction.lambda[e];
        bool positive=true;
        for(int i=0;i<w.mi;++i) {
          ts[i]=w.s[i]+alpha*direction.s[i];tz[i]=w.z[i]+alpha_dual*direction.z[i];
          if(w.active[i]&&(!(ts[i]>0.0)||!(tz[i]>0.0)))positive=false;
          const double original_trial_value=w.inequalities[i].at(tx);
          if(!std::isfinite(original_trial_value))positive=false;
          if(!w.active[i]&&std::isfinite(original_trial_value)&&original_trial_value>st.feasibility_tolerance*w.inequalities[i].scale) {
            w.active[i]=1;w.s[i]=std::max(0.01,-w.inequalities[i].value);w.z[i]=std::max(target,st.lazy_newton_initial_mu*0.01)/w.s[i];
            expanded=true;++stats.constraint_activations;
          }
        }
        if(expanded)break;
        const double trial=positive&&finite(tx)&&finite(tl)&&finite(tz)?w.merit(tx,ts,target):std::numeric_limits<double>::infinity();
        const double allowance=16*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(current));
        if(std::isfinite(trial)&&slope<0.0&&trial<=current+st.line_search_armijo*alpha*slope+allowance) {accepted=true;break;}
        // A feasible point can need a pure dual/centering update. In that
        // case the primal barrier merit is constant, so use a strictly
        // contracting original perturbed-KKT residual as the acceptance gate.
        if(std::isfinite(trial)&&alpha*inf_norm(direction.x)<=1e-12*(1.0+inf_norm(w.x))&&
           alpha*inf_norm(direction.s)<=1e-12*(1.0+inf_norm(w.s))) {
          const double before=w.perturbed_residual(w.x,w.lambda,w.s,w.z,target);
          const double after=w.perturbed_residual(tx,tl,ts,tz,target);
          if(before>0.0&&after<=(1.0-st.line_search_armijo*std::min(alpha,alpha_dual))*before) {
            accepted=true;++stats.dual_only_acceptances;break;
          }
        }
        alpha*=st.line_search_contraction;++stats.line_search_backtracks;
      }
      stats.globalization_seconds+=seconds(timer);globalization_cost.stop();
      if(accepted||expanded||!corrector_enabled||corrector_attempt!=0)break;
      // A second-order Mehrotra term need not be a descent direction for
      // this nonlinear merit. Retry the centering direction at the SAME
      // primal-dual point and with the same factor, tolerance and merit gate.
      // After a rejection keep centering for this solve: do not repeatedly
      // retry a nonlinear correction that the globalization has rejected.
      ++stats.predictor_corrector_fallbacks;
      corrector_enabled=false;w.penalty=penalty_before_corrector;
      timer=Clock::now();
      for(int i=0;i<w.mi;++i)if(w.active[i])rc[i]=w.centered_complementarity_rhs(i,target);
      direction=w.direction(rc,w.rd,w.h,w.rp);
      stats.solve_seconds+=seconds(timer);
      }
      if(live.enabled) {
        live_row->alpha_primal=alpha;live_row->alpha_dual=alpha_dual;
        live_row->step_inf=accepted?alpha*inf_norm(direction.x):0.0;
        live.emit(*live_row,expanded?"WS+":accepted?"STEP":"LSFAIL",expanded||!accepted);
      }
      if(expanded){++stats.working_set_retries;continue;}
      if(st.verbose)std::cout<<"[LAZY-NEWTON-STEP] alpha="<<alpha<<" alpha_dual="<<alpha_dual
        <<" alpha_lambda="<<alpha_lambda<<" theta="<<theta<<" slope="<<slope
        <<" penalty="<<w.penalty<<" target="<<target<<"\n";
      if(!accepted){result.status=Status::LineSearchFailure;result.message="lazy-newton original barrier-merit line search failed";break;}
      w.x.swap(tx);w.s.swap(ts);w.z.swap(tz);w.lambda.swap(tl);
      ++iteration;result.outer_iterations=iteration;result.inner_iterations=iteration;++stats.accepted_steps;
      if(!w.corrected())++stats.unshifted_accepted_steps;
      if(alpha>=0.99){++stats.full_steps;++full_streak;}else full_streak=0;
      stats.minimum_step=stats.accepted_steps==1?alpha:std::min(stats.minimum_step,alpha);
      result.step_inf=alpha*inf_norm(direction.x);
      if(st.lazy_newton_atomic_regularization&&w.atoms) {
        CostScope atom_cost(costs,NewtonCost::Atoms);const auto atom_timer=Clock::now();
        if(alpha>=0.05) {
          const double retention=st.lazy_newton_atom_step_decay&&w.atom_adaptation_active?std::pow(st.lazy_newton_atom_decay,std::min(1.0,alpha)):st.lazy_newton_atom_decay;
          w.atoms->decay(retention,full_streak>=2);
          ++stats.regularization_decreases;
          if(full_streak>=2)full_streak=0;
        }
        stats.atom_update_seconds+=seconds(atom_timer);
      }
      proposed_regularization=w.regularization;
      if(proposed_regularization>0.0&&alpha>=0.05) {
        proposed_regularization*=st.lazy_newton_regularization_decay;++stats.regularization_decreases;
        if(proposed_regularization<1e-10||full_streak>=2){proposed_regularization=0.0;full_streak=0;}
      }
    }
    timer=Clock::now();{CostScope certificate_cost(costs,NewtonCost::Certificate);w.export_state(result);
    if(result.status!=Status::Solved)w.certify(result);}
    stats.certificate_seconds+=seconds(timer);
    stats.final_regularization=w.regularization;stats.final_active_constraints=std::count(w.active.begin(),w.active.end(),1);
    stats.symbolic_analyses=w.factor->symbolic_analyses();stats.symbolic_reuses=w.factor->symbolic_reuses();
    stats.scatter_builds=w.factor->numeric_scatter_builds();stats.scatter_reuses=w.factor->numeric_scatter_reuses();
    if(live.enabled) {
      // A limit can leave values cached at x_(k-1), so refresh for the final
      // display only; all solving and the original certificate are complete.
      CostScope display_cost(costs,NewtonCost::Diagnostics);const auto display_timer=Clock::now();if(result.status!=Status::Solved)w.evaluate();stats.diagnostic_seconds+=seconds(display_timer);
      live.emit(live.state(stats.accepted_steps),result.status==Status::Solved?"SOLVED":"STOP",true);
      std::cout<<(level<4?"\nEXIT: ":"[LAZY-NEWTON-END] ")<<result.message<<std::endl;
    }
  } catch(const std::exception& e) {
    result.status=Status::NumericalError;result.message=e.what();
    if(owner){CostScope certificate_cost(costs,NewtonCost::Certificate);owner->export_state(result);owner->certify(result);}
    if(level>0){CostScope display_cost(costs,NewtonCost::Diagnostics);std::cout<<(level<4?"\nEXIT: ERROR: ":"[LAZY-NEWTON-END] ERROR: ")<<result.message<<std::endl;}
  }
  if(owner) {
    auto& w=*owner;auto& stats=result.lazy_newton;
    stats.final_regularization=w.regularization;
    if(w.atoms){CostScope display_cost(costs,NewtonCost::Diagnostics);const auto timer=Clock::now();w.atoms->diagnostics(stats,w.nr);stats.atom_update_seconds+=seconds(timer);}
    stats.final_active_constraints=std::count(w.active.begin(),w.active.end(),1);
    if(w.recovery){stats.normal_recovery_symbolic_analyses=w.recovery->symbolic_analyses();
      stats.normal_recovery_symbolic_reuses=w.recovery->symbolic_reuses();}
    if(w.factor){stats.symbolic_analyses=w.factor->symbolic_analyses();stats.symbolic_reuses=w.factor->symbolic_reuses();
      stats.scatter_builds=w.factor->numeric_scatter_builds();stats.scatter_reuses=w.factor->numeric_scatter_reuses();}
  }
  result.kkt_factorizations=result.lazy_newton.factor_attempts;
  result.elapsed_seconds=seconds(start);
  {CostScope cleanup(costs,NewtonCost::Cleanup);owner.reset();}costs.finish();return result;
}
} // namespace lazyipdca::qcqp
