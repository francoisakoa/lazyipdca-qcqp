#pragma once
#include "model.hpp"
#include "result.hpp"
#include <vector>

namespace lazyipdca::qcqp {
struct Settings;
// A coordinate t with zero objective/other-row incidence and one row
// q(y)-a*t^2<=0, a>0, 0<=t<infinity, is an exact free epigraph leaf.
// Projecting it removes no restriction on y; lift sets its row/bound duals0.
class FreeEpigraphPresolve {
 public:
  struct Leaf {int variable=-1,row=-1;double diagonal=0;};
  explicit FreeEpigraphPresolve(const QCQPProblem&);
  const QCQPProblem& reduced() const {return reduced_;}
  const QCQPProblem& original() const {return original_;}
  const FreeEpigraphStatistics& statistics() const {return stats_;}
  const std::vector<Leaf>& leaves() const {return leaves_;}
  bool changed() const {return !leaves_.empty();}
  void lift(Result&) const;
 private:
  const QCQPProblem& original_;
  QCQPProblem reduced_;
  FreeEpigraphStatistics stats_;
  std::vector<Leaf> leaves_;
  std::vector<int> kept_variables_,kept_inequalities_;
};
Result solve_with_free_epigraph_presolve(const QCQPProblem&,const Settings&);
}
