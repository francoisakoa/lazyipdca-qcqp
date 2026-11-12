#pragma once
#include "lazyipdca_qcqp/linear_solver.hpp"

namespace lazyipdca::qcqp::detail {
// This is a preconditioner for refinement of the unperturbed equations.
// Its caller must audit those equations before accepting any direction.
// Instantiated only after a numerical factorization failure, never on the
// successful path. It retains the requested backend and ordering.
class BoundedPivotRecovery final : public SymmetricFactorization {
  Settings settings_;
  std::unique_ptr<SymmetricFactorization> kernel_;
  double shift_=0;
 public:
  BoundedPivotRecovery(const Settings& requested,double shift,std::string& message)
      : settings_(requested),shift_(shift) {
    Settings& s=settings_;
    s.kkt_adaptive_regularization=true;
    // Zero/tiny saddle pivots can cause pathological delayed fill before
    // MUMPS reaches a zero-pivot diagnosis. Match the shifted preconditioner
    // numerically; acceptance still concerns the caller's unshifted matrix.
    if(s.linear_backend=="mumps")s.mumps_symmetric_matching=true;
    s.kkt_primal_regularization=s.kkt_dual_regularization=0;
    kernel_=make_symmetric_factorization(s,message);
  }
  bool factor(int n,int primal,const std::vector<SymmetricTriplet>& entries,
              std::string& message) override {
    if(!kernel_)return false;
    auto shifted=entries;
    for(int j=0;j<n;++j) {
      if(j<primal)shifted.push_back({j,j,shift_});
      else if(settings_.kkt_dual_regularization_end<0||j<settings_.kkt_dual_regularization_end)
        shifted.push_back({j,j,-shift_});
    }
    return kernel_->factor(n,primal,shifted,message);
  }
  bool solve(Vec& rhs,std::string& message) const override {return kernel_->solve(rhs,message);}
  double last_backward_error() const override {return kernel_->last_backward_error();}
  double last_initial_backward_error() const override {return kernel_->last_initial_backward_error();}
  double last_residual_inf() const override {return kernel_->last_residual_inf();}
  int last_refinement_steps() const override {return kernel_->last_refinement_steps();}
  int last_positive_pivots() const override {return kernel_->last_positive_pivots();}
  int last_negative_pivots() const override {return kernel_->last_negative_pivots();}
  int last_zero_pivots() const override {return kernel_->last_zero_pivots();}
  int symbolic_analyses() const override {return kernel_->symbolic_analyses();}
  int symbolic_reuses() const override {return kernel_->symbolic_reuses();}
  int symbolic_invalidations() const override {return kernel_->symbolic_invalidations();}
  double last_primal_regularization() const override {return shift_;}
  double last_dual_regularization() const override {return shift_;}
  int last_regularization_attempts() const override {return 1;}
  const char* name() const override {return kernel_->name();}
};
}
