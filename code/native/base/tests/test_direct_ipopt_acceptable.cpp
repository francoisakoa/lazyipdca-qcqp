#include "lazyipdca_qcqp/direct_ipopt.hpp"

#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

DirectIpoptResult acceptable_endpoint() {
  DirectIpoptResult result;
  result.available = true;
  result.ipopt_termination_accepted = true;
  result.ipopt_desired_convergence = false;
  result.feasibility = {1.0e-8, 0.0, 0.0, 1.0e-8};
  result.dual_infeasibility_inf = 4.9e-6;
  result.complementarity_inf = 1.1e-6;
  result.ipopt_internal_scaled_nlp_error = 4.0e-8;
  return result;
}
} // namespace

int main() {
  DirectIpoptSettings settings;

  auto acceptable = acceptable_endpoint();
  classify_direct_ipopt_result(settings, acceptable);
  require(acceptable.success, "guarded acceptable endpoint was rejected");
  require(acceptable.status == "solved_acceptable",
          "acceptable endpoint was mislabeled");
  require(acceptable.acceptable_convergence,
          "acceptable convergence flag was not set");
  require(!acceptable.ipopt_desired_convergence,
          "acceptable endpoint was confused with strict convergence");

  auto disabled = acceptable_endpoint();
  settings.acceptable_iterations = 0;
  classify_direct_ipopt_result(settings, disabled);
  require(!disabled.success && disabled.status == "kkt_failure",
          "acceptable-iter=0 did not disable acceptable success");

  settings.acceptable_iterations = 15;
  auto bad_scaled = acceptable_endpoint();
  bad_scaled.ipopt_internal_scaled_nlp_error = 2.0e-6;
  classify_direct_ipopt_result(settings, bad_scaled);
  require(!bad_scaled.success,
          "endpoint above Ipopt acceptable_tol was accepted");

  auto bad_absolute = acceptable_endpoint();
  bad_absolute.complementarity_inf = 2.0e-2;
  classify_direct_ipopt_result(settings, bad_absolute);
  require(!bad_absolute.success &&
              !bad_absolute.acceptable_absolute_limits_pass,
          "endpoint above original complementarity ceiling was accepted");

  auto strict = acceptable_endpoint();
  strict.ipopt_desired_convergence = true;
  classify_direct_ipopt_result(settings, strict);
  require(strict.success && strict.status == "solved",
          "strict endpoint was not preferred over acceptable status");
}
