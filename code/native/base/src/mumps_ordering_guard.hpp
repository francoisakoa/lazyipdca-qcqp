#pragma once

#include <string>

namespace lazyipdca::qcqp::detail {

// MUMPS may silently replace ICNTL(7)=5 when METIS is unavailable. Only
// INFOG(7) and INFOG(32), returned by a successful analysis, certify the
// ordering and analysis that were actually selected.
inline bool mumps_requested_ordering_verified(bool require_metis,
    int actual_ordering, int actual_analysis, std::string& message) {
  if(!require_metis || (actual_ordering==5 && actual_analysis==1)) return true;
  message="MUMPS requested METIS ordering was not honored: INFOG(7)="+
      std::to_string(actual_ordering)+" INFOG(32)="+
      std::to_string(actual_analysis)+"; require METIS=5, sequential analysis=1";
  return false;
}

} // namespace lazyipdca::qcqp::detail
