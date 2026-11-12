#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lazyipdca::qcqp::detail {
// Evidence for proposing an OPTIONAL SOC at an already filter-accepted trial.
// All norms below are equality-only, in original physical units. The exact
// original path supplies both the affine prediction and nonlinear remainder;
// neither inequality slack residuals nor theta can establish this evidence.
// Admission changes no acceptance rule: a correction must still satisfy its
// frozen-factor, size, domain and original-filter checks, and improve both
// equality feasibility and theta compared with the already accepted source.
struct NonlinearTrialSocEvidence {
  long double anchor_inf=0,affine_inf=0,trial_inf=0,defect_inf=0;
  double tolerance=0;
  int rows=0;
  bool valid=true;
  explicit NonlinearTrialSocEvidence(double tol):tolerance(tol) {
    if(!std::isfinite(tol)||!(tol>0))
      throw std::invalid_argument("invalid nonlinear trial SOC tolerance");
  }
  void add_row(long double anchor,long double affine,long double trial,
               long double nonlinear_defect,double scale) {
    if(!std::isfinite(anchor)||!std::isfinite(affine)||!std::isfinite(trial)||
        !std::isfinite(nonlinear_defect)||!std::isfinite(scale)||!(scale>0)) {
      valid=false;return;
    }
    const long double a=std::abs(anchor)/scale,b=std::abs(affine)/scale,
        c=std::abs(trial)/scale,d=std::abs(nonlinear_defect)/scale;
    if(!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(c)||!std::isfinite(d)) {
      valid=false;return;
    }
    ++rows;anchor_inf=std::max(anchor_inf,a);affine_inf=std::max(affine_inf,b);
    trial_inf=std::max(trial_inf,c);defect_inf=std::max(defect_inf,d);
  }
  long double roundoff()const {
    return 64*std::numeric_limits<double>::epsilon()*
        std::max({1.0L,anchor_inf,affine_inf,trial_inf});
  }
  const char* reason()const {
    if(!valid)return "invalid-equality-evidence";
    if(rows==0)return "no-equalities";
    const long double floor=roundoff();
    if(!(trial_inf-std::max(anchor_inf,static_cast<long double>(tolerance))>floor))
      return "equality-not-worsened";
    if(!(trial_inf-affine_inf>floor))return "affine-prediction-explains-residual";
    if(!(defect_inf>floor))return "nonlinear-defect-below-floor";
    return "nonlinear-equality-feasibility-loss";
  }
  bool eligible()const {return valid&&rows>0&&
      trial_inf-std::max(anchor_inf,static_cast<long double>(tolerance))>roundoff()&&
      trial_inf-affine_inf>roundoff()&&defect_inf>roundoff();}
  bool improved_by(const NonlinearTrialSocEvidence& corrected)const {
    if(!valid||!corrected.valid||rows==0||rows!=corrected.rows)return false;
    const long double floor=64*std::numeric_limits<double>::epsilon()*
        std::max({1.0L,trial_inf,corrected.trial_inf});
    return trial_inf-corrected.trial_inf>floor;
  }
};
} // namespace lazyipdca::qcqp::detail
