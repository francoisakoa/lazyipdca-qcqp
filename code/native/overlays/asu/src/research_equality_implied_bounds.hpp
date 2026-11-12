#pragma once
#include <stdexcept>
#include <string>
namespace lazyipdca::qcqp::research {
inline bool quadratic_equality_implied_bounds_enabled(
    bool master,const std::string& engine,const std::string& mode,
    const std::string& policy) {
  if(policy!="profile"&&policy!="keep"&&policy!="remove")
    throw std::invalid_argument("research-equality-implied-bounds expects profile|keep|remove");
  const bool historical=mode=="HH"||mode=="H0"||mode=="HS"||mode=="SH";
  return master&&engine=="majorants"&&
      (policy=="remove"||(policy=="profile"&&historical));
}
}
