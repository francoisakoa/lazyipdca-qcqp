#include "../src/nonlinear_trial_soc_policy.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using lazyipdca::qcqp::detail::NonlinearTrialSocEvidence;
int main() {
  int checks=0;
  auto check=[&](bool pass,const char* why){++checks;if(!pass)throw std::runtime_error(why);};
  // An exact equality h(x)=x^2-1 at x=1, direction d=1: the complete
  // accepted source trial x=2 has h=3, affine prediction 2 and remainder 1.
  // The policy is intentionally independent of alpha/small-step history.
  NonlinearTrialSocEvidence full(1e-6);
  full.add_row(0.,2.,3.,1.,1.);
  check(full.eligible(),"full-step nonlinear equality loss was missed");
  check(std::string(full.reason())=="nonlinear-equality-feasibility-loss","wrong admission reason");
  NonlinearTrialSocEvidence corrected(1e-6);
  corrected.add_row(0.,.4,.44,.04,1.);
  check(full.improved_by(corrected),"corrected equality improvement was rejected");
  check(!corrected.improved_by(full),"worse equality was accepted as improvement");

  // Actual R6 HH case_2 iteration 17->18 reports these equality maxima and
  // nonlinear defect at alpha=1. Use an affine residual of roundoff size;
  // the full CLI test separately exercises coefficients and the filter.
  NonlinearTrialSocEvidence trace(1e-6);
  trace.add_row(.044259989964466236L,1e-16L,.083451192088228682L,
      .083451192088228585L,1.);
  check(trace.eligible(),"recorded full-step equality drift did not admit a proposal");

  NonlinearTrialSocEvidence affine(1e-6),improving(1e-6),feasible(1e-6),roundoff(1e-6),empty(1e-6);
  affine.add_row(0.,3.,3.,0.,1.);
  improving.add_row(3.,0.,1.,1.,1.);
  feasible.add_row(0.,0.,1e-7,1e-7,1.);
  roundoff.add_row(1.,0.,std::nextafter(1.,2.),1.,1.);
  check(!affine.eligible(),"affine residual authorized nonlinear SOC");
  check(!improving.eligible(),"improving equality authorized an extra accepted-trial proposal");
  check(!feasible.eligible(),"final-feasible equality authorized an extra proposal");
  check(!roundoff.eligible(),"roundoff-only equality growth authorized SOC");
  check(!empty.eligible(),"no-equality model authorized SOC");
  check(!full.improved_by(empty),"missing corrected equality was accepted");

  // Evidence must be interpreted in original units, independent of internal
  // scaling, and use equality norms rather than a mixed theta/slack norm.
  NonlinearTrialSocEvidence scaled(1e-6),mixed_rows(1e-6);
  scaled.add_row(0.,2e-9L,3e-9L,1e-9L,1e-9);
  check(scaled.eligible()&&std::abs(scaled.trial_inf-full.trial_inf)<1e-14,
      "constraint scaling changed nonlinear admission");
  mixed_rows.add_row(100.,100.,100.,0.,1.);
  mixed_rows.add_row(0.,0.,1.,1.,1.);
  check(!mixed_rows.eligible(),"unchanged global equality norm counted as deterioration");

  for(double invalid:{0.,-1.,std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()}) {
    NonlinearTrialSocEvidence bad(1e-6);bad.add_row(0.,0.,1.,1.,invalid);
    check(!bad.eligible(),"invalid internal constraint scale authorized SOC");
  }
  for(long double invalid:{std::numeric_limits<long double>::infinity(),
      std::numeric_limits<long double>::quiet_NaN()}) {
    for(int field=0;field<4;++field) {
      long double row[]={0.,0.,1.,1.};row[field]=invalid;
      NonlinearTrialSocEvidence bad(1e-6);bad.add_row(row[0],row[1],row[2],row[3],1.);
      check(!bad.eligible(),"nonfinite equality evidence authorized SOC");
    }
  }
  // A failed correction must not invalidate source acceptance: admission and
  // corrected equality improvement are distinct predicates. Engine tests
  // verify the actual candidate/slack fallback and filter accounting.
  check(full.eligible()&&!full.improved_by(full),"source/proposal verdicts were conflated");
  std::cout<<"nonlinear_trial_soc_policy PASS checks="<<checks
      <<" full_step=1 physical_units=1 affine_excluded=1 actual_trace_drift=1"
      <<" unchanged_filter_required=1 source_fallback_separate=1\n";
}
