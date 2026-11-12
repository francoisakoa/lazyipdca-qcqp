#pragma once
#include <cmath>
namespace lazyipdca::qcqp {
// A native kernel's stricter absolute convergence flag is diagnostic. The
// study's freshly recomputed FULL-model finite-accuracy certificate is the
// acceptance authority, including a complete last-iteration endpoint.
inline bool native_model_certificate_authorized(bool full_finite_domain_state,
    bool full_model_certificate,int steps,int minimum_steps,double elapsed,double deadline) {
 return full_finite_domain_state&&full_model_certificate&&steps>=minimum_steps&&
     std::isfinite(elapsed)&&elapsed>=0&&elapsed<deadline;
}
}
