#pragma once
// Pure additional admission gate for the EXISTING next-model proximal
// proposal. Included beside the engine; the includer supplies <cmath>.
// No point, model, multiplier, acceptance test, tolerance or matrix is edited.
struct DcaGrShortStepProximal {
  static bool trigger(bool enabled,double alpha,int weak_dual_steps,
      double previous_physical_primal,double current_physical_primal) {
    return enabled&&std::isfinite(alpha)&&alpha>0&&alpha<1e-2&&
        weak_dual_steps>=3&&std::isfinite(previous_physical_primal)&&
        std::isfinite(current_physical_primal)&&previous_physical_primal>=0&&
        current_physical_primal>=0&&static_cast<long double>(current_physical_primal)>=
            .9L*previous_physical_primal;
  }
};
