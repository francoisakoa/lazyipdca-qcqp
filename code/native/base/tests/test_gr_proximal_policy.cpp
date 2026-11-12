#include "../src/gr_proximal_policy.hpp"
#include "../src/failure_recovery_policy.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace detail=lazyipdca::qcqp::detail;
static int checks=0;
static void require(bool value,const char* message) {
  ++checks;if(!value)throw std::runtime_error(message);
}

int main() {
  try {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    // Exact historical parity and isolation from every non-GR route.
    for(bool gr:{false,true})for(double old:{0.,.001,1.,1e11}) {
      const double proposal=detail::small_step_proximal_proposal(old,.04,1e3,1e-8);
      const auto h=detail::gr_globalization_proximal_growth(gr,"historical",old,proposal);
      require(h.effective==proposal&&!h.blocked,"historical proposal changed");
      const auto f=detail::gr_globalization_proximal_growth(false,"frozen",old,proposal);
      require(f.effective==proposal&&!f.blocked,"non-GR proposal changed");
    }
    const auto no_growth=detail::gr_globalization_proximal_growth(true,"frozen",.01,.001);
    require(no_growth.effective==.001&&!no_growth.blocked,"a downward proposal was frozen");
    const auto unchanged=detail::gr_globalization_proximal_growth(true,"frozen",.01,.01);
    require(unchanged.effective==.01&&!unchanged.blocked,"an unchanged proposal was changed");
    // The same historical tenfold escalation is visible but never committed.
    double historical=.001,frozen=.001;
    int applied_frozen_retries=0;
    for(int i=0;i<20;++i) {
      const double hp=detail::small_step_proximal_proposal(historical,.04,1e3,1e-8);
      historical=detail::gr_globalization_proximal_growth(true,"historical",historical,hp).effective;
      const double fp=detail::small_step_proximal_proposal(frozen,.04,1e3,1e-8);
      const auto decision=detail::gr_globalization_proximal_growth(true,"frozen",frozen,fp);
      require(decision.blocked&&decision.effective==.001,"frozen memory amplified");
      // This is the engine's existing strict-increase admission gate. A
      // blocked LS proposal cannot consume a retry or spin at one anchor.
      if(std::isfinite(decision.effective)&&decision.effective>frozen&&decision.effective<=1e12)
        ++applied_frozen_retries;
      frozen=decision.effective;
    }
    require(historical>1e6,"historical feedback example did not amplify");
    require(frozen==.001&&applied_frozen_retries==0,"frozen failure admits endless retries");
    // NaN/Inf must not be converted into finite successful retry proposals.
    for(double proposal:{nan,inf,-inf}) {
      auto d=detail::gr_globalization_proximal_growth(true,"frozen",.001,proposal);
      require(!d.blocked&&!std::isfinite(d.effective),"invalid proposal became admissible");
    }
    require(detail::gr_globalization_proximal_growth(true,"frozen",0.,1e-8).effective==0.,
        "zero remembered proximal grew under frozen policy");
    bool rejected=false;
    try{detail::validate_gr_proximal_policy("unknown");}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unknown policy was accepted");
    // The independent lower-proximal probe admission and proposal retain
    // their original feasibility, finite-budget and certified-floor gates.
    require(detail::lower_proximal_probe_eligible(false,false,false,1e-7,1e-6,20,1.,10.),
        "valid lower-proximal probe changed");
    require(!detail::lower_proximal_probe_eligible(false,false,false,1e-7,1e-6,20,10.,10.),
        "expired lower-proximal probe admitted");
    require(!detail::lower_proximal_probe_eligible(false,false,false,1e-7,1e-6,1,1.,10.),
        "exhausted iteration budget admitted");
    require(detail::lower_proximal_proposal(10.,.1)==1.,"lower proposal changed");
    std::cout<<"gr_proximal_policy PASS checks="<<checks<<"\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"FAIL "<<error.what()<<" checks="<<checks<<"\n";return 1;
  }
}
