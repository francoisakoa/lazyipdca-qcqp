#pragma once
#include <algorithm>
#include <cmath>
#include <string_view>

namespace lazyipdca::qcqp::detail {

// Rejection of a certificate, a linear solve and a trial point are different
// events. Only an explicit inertia diagnostic permits a curvature search.
enum class DcaAggregateRejection {
  none, inertia, backend, numerical_direction, domain, filter, kkt, globalization
};
inline const char* dca_aggregate_rejection_name(DcaAggregateRejection kind) {
  switch(kind) {
    case DcaAggregateRejection::none:return "none";
    case DcaAggregateRejection::inertia:return "inertia";
    case DcaAggregateRejection::backend:return "backend";
    case DcaAggregateRejection::numerical_direction:return "numerical_direction";
    case DcaAggregateRejection::domain:return "domain";
    case DcaAggregateRejection::filter:return "filter";
    case DcaAggregateRejection::kkt:return "kkt";
    case DcaAggregateRejection::globalization:return "globalization";
  }
  return "globalization";
}
inline DcaAggregateRejection dca_aggregate_factor_rejection(std::string_view message) {
  return message.find("inertia mismatch")!=std::string_view::npos ?
      DcaAggregateRejection::inertia:DcaAggregateRejection::backend;
}
inline DcaAggregateRejection dca_aggregate_point_rejection(std::string_view reason) {
  if(reason=="none")return DcaAggregateRejection::none;
  if(reason=="aggregate-curvature-certificate")return DcaAggregateRejection::inertia;
  if(reason=="backend-unavailable" || reason=="aggregate-backend-factorization")
    return DcaAggregateRejection::backend;
  if(reason=="precise-original-residual-invalid")return DcaAggregateRejection::numerical_direction;
  if(reason=="multiplier-or-slack-step" || reason=="nonpositive-local-state" ||
     reason=="inactive-original-row")return DcaAggregateRejection::domain;
  if(reason=="original-filter")return DcaAggregateRejection::filter;
  if(reason=="insufficient-kkt-contraction" || reason=="rho-probe-no-current-kkt-gain" ||
     reason=="precise-error-no-contraction")return DcaAggregateRejection::kkt;
  return DcaAggregateRejection::globalization;
}
inline bool dca_aggregate_retry_permitted(DcaAggregateRejection kind,bool separated) {
  return !separated || (kind!=DcaAggregateRejection::backend &&
                       kind!=DcaAggregateRejection::numerical_direction);
}
struct DcaPositiveRhoMemory {
  double value=0.0;
  // A zero certificate does not erase positive curvature history. Shifts
  // introduced by line-search or numerical recovery never become curvature
  // evidence for the next anchor.
  void certificate(double rho,bool curvature_search) {
    if(curvature_search && rho>0 && std::isfinite(rho))value=rho;
  }
  double seed(double initial,double decay,double cap) const {
    return std::min(cap,std::max(initial,value*decay));
  }
};
} // namespace lazyipdca::qcqp::detail
