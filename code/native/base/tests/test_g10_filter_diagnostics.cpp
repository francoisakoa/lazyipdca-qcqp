#include "../src/dca_filter_contract.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

using lazyipdca::qcqp::detail::OriginalDcaFilter;
using lazyipdca::qcqp::detail::OriginalFilterDecision;
using lazyipdca::qcqp::detail::OriginalFilterPoint;

namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

OriginalFilterDecision check(const OriginalDcaFilter &filter, OriginalFilterPoint from,
                            OriginalFilterPoint to, double predicted, bool accepted,
                            const char *reason) {
  const auto count = filter.size();
  const auto decision = filter.diagnose(from, to, predicted, 1e-6);
  require(decision.accepted == accepted, "unexpected acceptance");
  require(std::strcmp(decision.reason, reason) == 0, "wrong diagnostic reason");
  require(filter.acceptable(from, to, predicted, 1e-6) == accepted,
          "bool API and diagnostic API disagree");
  require(filter.size() == count, "diagnostic changed filter history");
  return decision;
}
} // namespace

int main() {
  try {
    OriginalDcaFilter filter(1.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (auto point : {OriginalFilterPoint{nan, 0, 1}, OriginalFilterPoint{1, inf, 1},
                       OriginalFilterPoint{1, 0, nan}}) {
      const auto decision = check(filter, {2, 1, 1}, point, 1, false, "nonfinite");
      require(!decision.finite && !decision.history_checked, "nonfinite trial checked history");
    }

    const auto no_progress = check(filter, {1, 0, 1}, {1, 0, 1}, 0, false, "local-progress");
    require(no_progress.finite && !no_progress.feasibility && !no_progress.objective &&
                !no_progress.terminal && !no_progress.history_checked,
            "unchanged point has an incorrect progress flag");

    const auto feasibility = check(filter, {2, 1, 1}, {3, 0.5, 1}, 0, true, "accepted");
    require(feasibility.feasibility && !feasibility.objective && !feasibility.terminal &&
                feasibility.history_checked && !feasibility.history_bypassed &&
                feasibility.feasibility_theta_limit == 0.999,
            "feasibility decision or threshold incorrect");
    check(filter, {2, 1e-7, 1}, {3, 5e-8, 1}, 0, false, "local-progress");

    const auto objective = check(filter, {2, 0.5, 1}, {1.9999, 0.5, 1}, 1, true, "accepted");
    require(objective.objective && !objective.feasibility && !objective.terminal &&
                objective.objective_limit == 1.9999 && objective.theta_cap == 1,
            "objective boundary decision incorrect");
    check(filter, {2, 2, 1}, {1, 2, 1}, 1, false, "local-progress");

    filter.insert({1, 0.2, 1});
    filter.insert({2, 0.1, 1});
    require(filter.size() == 2, "expected two nondominated historical entries");
    const auto history = check(filter, {3, 1, 1}, {2.5, 0.15, 1}, 1, false, "filter-history");
    require(history.feasibility && history.objective && !history.terminal &&
                history.history_checked && !history.history_bypassed && history.history_index == 1 &&
                std::abs(history.history_theta_limit - 0.09999) < 1e-15 &&
                std::abs(history.history_objective_limit - 1.99999) < 1e-15,
            "historical rejection did not identify the first blocking entry");

    // Existing G10 behavior accepts this terminal trial although the newly
    // inserted entry dominates it. Report the bypass; do not silently call it
    // a successful history check or alter the witness's acceptance policy.
    filter.insert({2, 0, 1});
    const auto terminal = check(filter, {3, 5e-7, 1}, {3, 5e-7, 0.5}, 0, true, "accepted");
    require(terminal.terminal && terminal.history_bypassed && !terminal.history_checked &&
                terminal.history_index == -1 && terminal.terminal_kkt_limit == 0.999,
            "terminal history bypass was changed or hidden");

    const auto lagrangian = check(filter, {1, 0, 0.1, 1},
                                 {1.00000001, 1e-9, 0.01, 0.99999999}, 0, true, "accepted");
    require(lagrangian.terminal && lagrangian.history_bypassed,
            "terminal Lagrangian safeguard was changed");
    std::cout << "G10 filter diagnostic scenarios passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
