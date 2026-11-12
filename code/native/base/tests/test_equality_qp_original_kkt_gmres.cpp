#include "lazyipdca_qcqp/original_kkt_gmres.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {
double inf_norm_local(const Vec& x) {
  double value = 0.0;
  for (double entry : x) value = std::max(value, std::abs(entry));
  return value;
}
}  // namespace

int main() {
  try {
    constexpr int dimension = 5;
    const double matrix[dimension][dimension] = {
      {0.00022493313960010,0.00019405763736605,0.00102004745339937, 0.00708388554811141, 0.00025585560569825},
      {0.00019405763736605,0.00016746748587407,0.00088012770828845,-0.06775419489083805, 0.00015282729197326},
      {0.00102004745339937,0.00088012770828845,0.00462600890575227, 0.01035262800475593,-0.00096146689664057},
      {0.00708388554811141,-0.06775419489083805,0.01035262800475593,0.0,0.0},
      {0.00025585560569825,0.00015282729197326,-0.00096146689664057,0.0,0.0}
    };
    const double inverse_regularized[dimension][dimension] = {
      {80.62546146172899,11.999332999795486,23.362532224859237,3.437919133238298,488.93502835968030},
      {11.999332999795488,1.7858377736624245,3.4770010962207238,-14.614961069190391,-90.109136550990996},
      {23.362532224859240,3.4770010962207234,6.7696729256015775,-1.4082169535204243,-924.29011999116847},
      {3.4379191332382986,-14.614961069190390,-1.4082169535204243,-2.3284475645518006,-56.224854846680778},
      {488.93502835968036,-90.109136550990982,-924.29011999116847,-56.224854846680763,-14229.633072583847}
    };
    const Vec rhs={-0.00506482150846793,0.07492446090182754,
                   -0.00034325554814259,0.00785554306227220,
                   -0.00035192342367547};
    auto apply=[&](const Vec& x,Vec& y) {
      y.assign(dimension,0.0);
      for(int i=0;i<dimension;++i)
        for(int j=0;j<dimension;++j)
          y[static_cast<std::size_t>(i)]+=
              matrix[i][j]*x[static_cast<std::size_t>(j)];
    };
    auto precondition=[&](const Vec& x,Vec& y,std::string&) {
      y.assign(dimension,0.0);
      for(int i=0;i<dimension;++i)
        for(int j=0;j<dimension;++j)
          y[static_cast<std::size_t>(i)]+=
              inverse_regularized[i][j]*x[static_cast<std::size_t>(j)];
      return true;
    };

    Vec richardson;
    std::string why;
    precondition(rhs,richardson,why);
    for(int iteration=0;iteration<8;++iteration) {
      Vec applied;
      apply(richardson,applied);
      Vec residual(dimension,0.0);
      for(int i=0;i<dimension;++i)
        residual[static_cast<std::size_t>(i)]=
            rhs[static_cast<std::size_t>(i)]-applied[static_cast<std::size_t>(i)];
      Vec correction;
      precondition(residual,correction,why);
      for(int i=0;i<dimension;++i)
        richardson[static_cast<std::size_t>(i)]+=
            correction[static_cast<std::size_t>(i)];
    }
    Vec applied;
    apply(richardson,applied);
    Vec stalled_residual(dimension,0.0);
    for(int i=0;i<dimension;++i)
      stalled_residual[static_cast<std::size_t>(i)]=
          rhs[static_cast<std::size_t>(i)]-applied[static_cast<std::size_t>(i)];
    const double stalled=inf_norm_local(stalled_residual);
    if(!(stalled>1e-6))
      throw std::runtime_error("fixture no longer reproduces Richardson stagnation");

    Vec correction;
    const auto gmres=original_kkt_right_preconditioned_gmres(
        apply,precondition,stalled_residual,correction,32,1e-10);
    if(!gmres.converged)
      throw std::runtime_error("GMRES did not converge: "+gmres.message);
    for(int i=0;i<dimension;++i)
      richardson[static_cast<std::size_t>(i)]+=
          correction[static_cast<std::size_t>(i)];
    apply(richardson,applied);
    Vec final_residual(dimension,0.0);
    for(int i=0;i<dimension;++i)
      final_residual[static_cast<std::size_t>(i)]=
          rhs[static_cast<std::size_t>(i)]-applied[static_cast<std::size_t>(i)];
    const double final_value=inf_norm_local(final_residual);
    if(!(final_value<=1e-9) || !(gmres.residual_2<=1e-10))
      throw std::runtime_error("explicit original-KKT residual remains too large");

    std::cout << "[PASS] original-KKT GMRES removes large-shift bias"
              << " richardson8=" << stalled
              << " gmres_iter=" << gmres.iterations
              << " explicit_residual2=" << gmres.residual_2
              << " final_inf=" << final_value << "\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
