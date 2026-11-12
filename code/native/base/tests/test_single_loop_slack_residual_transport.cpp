#include "lazyipdca_qcqp/single_loop_slack_transport.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    const double old_phi = 2.75;
    const double old_slack = 4.5;
    const double old_residual = old_phi + old_slack;
    const double new_phi = -3.125;
    const SingleLoopSlackRemap transported =
        remap_single_loop_slack_residual(old_residual, new_phi, 1e-4);
    require(transported.valid && transported.residual_preserved,
            "a positive exact remap was not preserved");
    require(std::abs((new_phi + transported.slack) - old_residual) <=
                8.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::abs(old_residual)),
            "the transported primal-slack residual changed");

    // Even a very small positive exact slack must survive verbatim.  The
    // interior push is not a replacement for a valid transported residual.
    const double tiny = 1e-12;
    const SingleLoopSlackRemap tiny_positive =
        remap_single_loop_slack_residual(1.0 + tiny, 1.0, 1e-4);
    require(tiny_positive.valid && tiny_positive.residual_preserved &&
                tiny_positive.slack > 0.0 &&
                tiny_positive.slack < 1e-6,
            "a positive transported slack was overwritten by the push");

    const SingleLoopSlackRemap repaired =
        remap_single_loop_slack_residual(1.0, 1.5, 1e-4);
    require(repaired.valid && !repaired.residual_preserved &&
                repaired.slack == 1e-4,
            "a non-positive exact slack was not repaired");

    const SingleLoopSlackRemap invalid =
        remap_single_loop_slack_residual(
            std::numeric_limits<double>::infinity(), 0.0, 1e-4);
    require(!invalid.valid, "a non-finite residual was accepted");

    std::cout << "[PASS] direct single-loop slack residual transport\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
