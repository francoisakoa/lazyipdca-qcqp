#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include "lazyipdca_qcqp/settings.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

void require(bool condition,const std::string& message) {
  if(!condition) throw std::runtime_error(message);
}

bool bitwise_equal(double a,double b) {
  return std::memcmp(&a,&b,sizeof(double))==0;
}

bool bitwise_equal(const Vec& a,const Vec& b) {
  if(a.size()!=b.size()) return false;
  for(std::size_t i=0;i<a.size();++i)
    if(!bitwise_equal(a[i],b[i])) return false;
  return true;
}

ConvexQuadraticModel empty_model(int n,double constant) {
  ConvexQuadraticModel model;
  model.constant=constant;
  model.linear.assign(static_cast<std::size_t>(n),0.0);
  model.base_hessian.n=n;
  return model;
}

ConvexQuadraticModel bound_model(int n,int j,double sign,double constant) {
  ConvexQuadraticModel model=empty_model(n,constant);
  model.linear[static_cast<std::size_t>(j)]=sign;
  return model;
}

ConvexQuadraticModel quadratic_model(int n,int j,int k,double constant) {
  ConvexQuadraticModel model=empty_model(n,constant);
  model.linear[static_cast<std::size_t>(j)]=0.25;
  model.linear[static_cast<std::size_t>(k)]=-0.5;
  model.base_hessian.entries.push_back({j,j,1.5});
  model.base_hessian.entries.push_back({j,k,0.75});
  model.base_hessian.entries.push_back({k,k,2.0});
  return model;
}

ConvexSubproblem build_subproblem() {
  constexpr int n=12;
  ConvexSubproblem p;
  p.nvar=n;
  p.original_dimension=n;
  p.objective=empty_model(n,0.0);
  for(int j=0;j<n;++j) {
    p.objective.linear[static_cast<std::size_t>(j)]=0.1*(j+1);
    p.objective.base_hessian.entries.push_back({j,j,1.0+0.05*j});
  }
  p.equalities.nvar=n;
  SparseRow row;
  row.entries={{0,1.0},{1,-1.0},{5,0.5}};
  p.equalities.rows.push_back(row);
  p.equalities.rhs.push_back(0.125);
  for(int j=0;j<n;++j) {
    p.inequalities.push_back(bound_model(n,j,+1.0,-1.0));
    p.inequalities.push_back(bound_model(n,j,-1.0,-1.0));
  }
  p.inequalities.push_back(quadratic_model(n,2,7,-3.0));
  p.inequalities.push_back(quadratic_model(n,4,9,-2.5));
  p.prefer_feasible_centered_start=true;
  return p;
}

Settings settings(bool fastpath) {
  Settings st;
  st.verbose=false;
  st.linear_backend="native";
  st.inner_tolerance=1e-10;
  st.max_inner_iterations=200;
  st.lazyqp_model_support_fastpath=fastpath;
  return st;
}

void check_default_and_bitwise_parity() {
  Settings defaults;
  require(!defaults.lazyqp_model_support_fastpath,
          "support fast path must default to off");

  const ConvexSubproblem p=build_subproblem();
  const InnerResult reference=
      ConvexQCQPLazyQPKernel(settings(false)).solve(p);
  const InnerResult candidate=
      ConvexQCQPLazyQPKernel(settings(true)).solve(p);
  require(reference.success&&candidate.success,"parity solve failed");
  require(bitwise_equal(reference.y,candidate.y),"primal iterate changed");
  require(bitwise_equal(reference.equality_duals,candidate.equality_duals),
          "equality multipliers changed");
  require(bitwise_equal(reference.inequality_duals,candidate.inequality_duals),
          "inequality multipliers changed");
  require(bitwise_equal(reference.inequality_slacks,candidate.inequality_slacks),
          "inequality slacks changed");
  require(reference.iterations==candidate.iterations,
          "inner iteration count changed");
  require(reference.lazyqp_model_support_fastpath_requests==0 &&
          reference.lazyqp_model_support_value_evaluations==0 &&
          reference.lazyqp_model_support_gradient_evaluations==0,
          "generic path reported support work");
  require(candidate.lazyqp_model_support_fastpath_requests==1 &&
          candidate.lazyqp_model_support_fastpath_effective_subproblems==1 &&
          candidate.lazyqp_model_support_builds==1,
          "support fast path was not effectively wired");
  require(candidate.lazyqp_model_support_eligible_models==
              static_cast<int>(p.inequalities.size()) &&
          candidate.lazyqp_model_support_generic_fallback_models==0,
          "unexpected eligibility classification");
  require(candidate.lazyqp_model_support_value_evaluations>0 &&
          candidate.lazyqp_model_support_gradient_evaluations>0,
          "support kernels were never executed");
}

void check_dense_model_falls_back() {
  ConvexSubproblem p=build_subproblem();
  ConvexQuadraticModel dense=empty_model(p.nvar,-4.0);
  dense.shift_dimension=p.nvar;
  dense.scalar_shift=0.25;
  p.inequalities.push_back(std::move(dense));
  const InnerResult out=ConvexQCQPLazyQPKernel(settings(true)).solve(p);
  require(out.success,"mixed support/generic solve failed");
  require(out.lazyqp_model_support_fastpath_effective_subproblems==1,
          "eligible sparse models were not used");
  require(out.lazyqp_model_support_generic_fallback_models==1,
          "dense shift model did not use the generic fallback");
}

void check_malformed_models_fail_closed() {
  {
    ConvexSubproblem p=build_subproblem();
    p.inequalities.front().linear.pop_back();
    const InnerResult generic=
        ConvexQCQPLazyQPKernel(settings(false)).solve(p);
    const InnerResult fast=
        ConvexQCQPLazyQPKernel(settings(true)).solve(p);
    require(!generic.success&&!fast.success,
            "short inequality vector was accepted");
    require(generic.message==fast.message,
            "fast path changed malformed-dimension failure semantics");
  }
  {
    ConvexSubproblem p=build_subproblem();
    p.inequalities.front().base_hessian.entries.push_back(
        {0,p.nvar,1.0});
    const InnerResult out=ConvexQCQPLazyQPKernel(settings(true)).solve(p);
    require(!out.success,"out-of-range Hessian index was accepted");
  }
  {
    ConvexSubproblem p=build_subproblem();
    p.inequalities.front().shift_dimension=p.nvar+1;
    p.inequalities.front().scalar_shift=1.0;
    const InnerResult out=ConvexQCQPLazyQPKernel(settings(true)).solve(p);
    require(!out.success,"out-of-range dense shift was accepted");
  }
  {
    ConvexSubproblem p=build_subproblem();
    LowRankTerm term;
    term.beta=1.0;
    term.direction.assign(static_cast<std::size_t>(p.nvar-1),1.0);
    p.inequalities.front().low_rank.push_back(std::move(term));
    const InnerResult out=ConvexQCQPLazyQPKernel(settings(true)).solve(p);
    require(!out.success,"short low-rank direction was accepted");
  }
  {
    ConvexSubproblem p=build_subproblem();
    p.equalities.rows.front().entries.push_back({p.nvar,1.0});
    const InnerResult out=ConvexQCQPLazyQPKernel(settings(true)).solve(p);
    require(!out.success,"out-of-range equality index was accepted");
  }
}

} // namespace

int main() {
  try {
    check_default_and_bitwise_parity();
    check_dense_model_falls_back();
    check_malformed_models_fail_closed();
  } catch(const std::exception& error) {
    std::cerr<<"test_lazyqp_model_support_fastpath: "<<error.what()<<"\n";
    return 1;
  }
  std::cout<<"test_lazyqp_model_support_fastpath: OK\n";
  return 0;
}
