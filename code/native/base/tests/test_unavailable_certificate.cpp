#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

int main() {
  try {
    Result r;
    if(!std::isinf(r.model_stationarity_inf) ||
       !std::isinf(r.dual_infeasibility_inf) ||
       !std::isinf(r.complementarity_inf) ||
       !std::isinf(r.scaled_nlp_error))
      throw std::runtime_error("unavailable certificate fields must be nonfinite");
    r.lazyqp_inexact_forcing_model_evaluations=11;
    r.lazyqp_kkt_triplet_layout_builds=12;
    r.lazyqp_kkt_triplet_layout_reuses=13;
    r.lazyqp_kkt_triplet_layout_invalidations=14;
    r.kkt_numeric_scatter_builds=15;
    r.kkt_numeric_scatter_reuses=16;
    r.kkt_numeric_scatter_invalidations=17;
    const std::string j=result_json(r,"UNAVAILABLE");
    if(j.find("\"model_stationarity_inf\": null")==std::string::npos ||
       j.find("\"dual_infeasibility_inf\": null")==std::string::npos ||
       j.find("\"complementarity_inf\": null")==std::string::npos ||
       j.find("\"scaled_nlp_error\": null")==std::string::npos ||
       j.find("\"lazyqp_inexact_forcing_model_evaluations\": 11")==
           std::string::npos ||
       j.find("\"lazyqp_kkt_triplet_layout_builds\": 12")==
           std::string::npos ||
       j.find("\"lazyqp_kkt_triplet_layout_reuses\": 13")==
           std::string::npos ||
       j.find("\"lazyqp_kkt_triplet_layout_invalidations\": 14")==
           std::string::npos ||
       j.find("\"kkt_numeric_scatter_builds\": 15")==std::string::npos ||
       j.find("\"kkt_numeric_scatter_reuses\": 16")==std::string::npos ||
       j.find("\"kkt_numeric_scatter_invalidations\": 17")==
           std::string::npos)
      throw std::runtime_error("certificate or telemetry JSON serialization failed");
    std::cout << "[PASS] unavailable certificate and public telemetry serialize\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
