#pragma once
#include <algorithm>
#include <map>
#include <vector>

namespace lazyipdca::qcqp::detail {
// Workspace-owned T' u kernel. The original atom and CSR pattern of T are
// immutable. Numerical zeros are structural entries and must be retained.
struct NewtonAtomProjection {
  struct Entry {int output,input;double coefficient;};
  std::vector<Entry> entries;
  bool ready=false;
  int updates=0;
  void update(const std::vector<int>& original_support,const std::vector<double>& original_values,
      const std::vector<int>& begin,const std::vector<int>& column,const std::vector<double>& value,
      std::vector<int>& support,std::vector<double>& values) {
    if(!ready) {
      std::map<int,int> outputs;
      for(int j:original_support)
        for(int t=begin[j];t<begin[j+1];++t)outputs.emplace(column[t],0);
      support.clear();values.assign(outputs.size(),0.0);
      for(auto& p:outputs){p.second=(int)support.size();support.push_back(p.first);}
      // Same accumulation order as the uncached projection, even when
      // several source coordinates cancel in one reduced coordinate.
      for(std::size_t i=0;i<original_support.size();++i) {
        int j=original_support[i];
        for(int t=begin[j];t<begin[j+1];++t)
          entries.push_back({outputs.at(column[t]),t,original_values[i]});
      }
      ready=true;
    }
    std::fill(values.begin(),values.end(),0.0);
    for(auto p:entries)values[p.output]+=p.coefficient*value[p.input];
    ++updates;
  }
};
}
