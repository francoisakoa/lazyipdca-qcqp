#include "lazyipdca_qcqp/linear_solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {
void require(bool ok,const std::string& message) {
  if(!ok)throw std::runtime_error(message);
}
Vec product(const std::vector<SymmetricTriplet>& matrix,const Vec& x) {
  std::vector<long double> sum(x.size(),0);
  for(auto e:matrix) {
    sum[e.i]+=static_cast<long double>(e.value)*x[e.j];
    if(e.i!=e.j)sum[e.j]+=static_cast<long double>(e.value)*x[e.i];
  }
  Vec result(x.size());
  for(std::size_t i=0;i<x.size();++i)result[i]=static_cast<double>(sum[i]);
  return result;
}
void check_original(const std::vector<SymmetricTriplet>& matrix,const Vec& rhs,
                    const Vec& x,const Vec& expected,const Vec& variable_scale) {
  std::vector<long double> residual(rhs.begin(),rhs.end()),magnitude(rhs.size());
  for(std::size_t i=0;i<rhs.size();++i)magnitude[i]=std::abs(rhs[i]);
  for(auto e:matrix) {
    const long double first=static_cast<long double>(e.value)*x[e.j];
    residual[e.i]-=first;magnitude[e.i]+=std::abs(first);
    if(e.i!=e.j) {
      const long double second=static_cast<long double>(e.value)*x[e.i];
      residual[e.j]-=second;magnitude[e.j]+=std::abs(second);
    }
  }
  for(std::size_t i=0;i<rhs.size();++i) {
    require(std::isfinite(x[i]),"nonfinite original-coordinate solution");
    require(std::abs(residual[i])<=2e-12L*std::max(1e-300L,magnitude[i]),
            "original rowwise residual rejected at row "+std::to_string(i));
    require(std::abs((x[i]-expected[i])*variable_scale[i])<=2e-9,
            "equilibration changed original solution coordinate "+std::to_string(i));
  }
}
}

int main() {
  try {
    if(!linear_backend_compiled("ldl")) {
      std::cout<<"[SKIP] SuiteSparse LDL+METIS not compiled\n";return 0;
    }
    Settings settings;
    require(!settings.kkt_equilibration,"LDL equilibration must remain opt-in");
    settings.linear_backend="ldl";
    settings.linear_metis=true;
    settings.kkt_adaptive_regularization=true;
    settings.kkt_preserve_structural_zeros=true;
    settings.kkt_iterative_refinement_steps=4;
    settings.kkt_backward_error_tolerance=1e-13;
    settings.kkt_backward_error_reject=1e-10;
    std::string message;
    auto baseline=make_symmetric_factorization(settings,message);
    require(static_cast<bool>(baseline),"baseline LDL factory: "+message);
    // Each factor borrows its own Settings for the entire factor lifetime.
    Settings equilibrated_settings=settings;
    equilibrated_settings.kkt_equilibration=true;
    auto factor=make_symmetric_factorization(equilibrated_settings,message);
    require(static_cast<bool>(factor),"equilibrated LDL factory: "+message);

    const Vec scale={1e-9,1e8,1e-5,1e6,1e-3,1e5};
    const Vec scaled_solution={0.75,-0.5,0.25,1.25,-0.75,0.5};
    Vec expected(6);
    for(int i=0;i<6;++i)expected[i]=scaled_solution[i]/scale[i];
    std::vector<SymmetricTriplet> matrix={
        {0,0,4},{1,1,5},{2,2,6},{3,3,7},{4,4,-3},{5,5,-4},
        {0,1,.1},{1,2,.2},{2,3,.15},{0,4,.3},{1,4,-.2},
        {2,5,.25},{3,5,.35},{4,5,.1}};
    for(auto& e:matrix)e.value*=scale[e.i]*scale[e.j];
    for(int update=0;update<3;++update) {
      if(update)matrix[2].value*=1.2;
      const Vec rhs=product(matrix,expected);
      require(factor->factor(6,4,matrix,message),"equilibrated factor: "+message);
      Vec answer=rhs;
      require(factor->solve(answer,message),"equilibrated solve: "+message);
      check_original(matrix,rhs,answer,expected,scale);
      require(baseline->factor(6,4,matrix,message),"baseline factor: "+message);
      answer=rhs;
      require(baseline->solve(answer,message),"baseline solve: "+message);
      check_original(matrix,rhs,answer,expected,scale);
      if(update) {
        require(factor->last_symbolic_reused(),"equilibration invalidated symbolic cache");
        require(factor->last_numeric_scatter_reused(),"equilibration invalidated numeric scatter");
      }
    }
    // Inertia rejection must survive the congruence, while a COMPLETE factor
    // remains available for a checked curvature probe in original coordinates.
    require(!factor->factor(6,6,matrix,message),"wrong inertia accepted after scaling");
    require(factor->curvature_defect()==2,"wrong curvature defect after scaling");
    Vec rhs=product(matrix,expected),answer=rhs;
    require(factor->solve_curvature_probe(answer,message),"curvature probe: "+message);
    check_original(matrix,rhs,answer,expected,scale);

    // A changed dimension must rebuild scales as well as symbolic storage.
    matrix={{0,0,1e-200},{1,1,-1e200}};
    const Vec tiny_scale={1e-100,1e100},tiny_expected={1e100,-1e-100};
    rhs=product(matrix,tiny_expected);answer=rhs;
    require(factor->factor(2,1,matrix,message),"extreme diagonal factor: "+message);
    require(factor->solve(answer,message),"extreme diagonal solve: "+message);
    check_original(matrix,rhs,answer,tiny_expected,tiny_scale);
    // Refuse to turn a nonzero into an exact zero by underflow in scaling.
    const std::vector<SymmetricTriplet> unscalable={
        {0,0,1e300},{0,1,1e-300},{1,1,-1e300}};
    require(!factor->factor(2,1,unscalable,message),"silent coefficient underflow accepted");
    require(message.find("erase")!=std::string::npos,"unexpected extreme-scaling failure: "+message);
    // A rejected scale must not contaminate the next numeric factorization.
    require(factor->factor(2,1,matrix,message),"refactor after rejected scale: "+message);
    answer=rhs;require(factor->solve(answer,message),"solve after rejected scale: "+message);
    check_original(matrix,rhs,answer,tiny_expected,tiny_scale);
    std::cout<<"[PASS] symmetric equilibration retains original solves, inertia, and caches\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
  }
}
