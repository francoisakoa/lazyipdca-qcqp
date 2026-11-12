#pragma once
#include "lazyipdca_qcqp/result.hpp"
#include <chrono>
namespace lazyipdca::qcqp::detail {
class NewtonCostLedger {
  using Clock=std::chrono::steady_clock;
  LazyNewtonStatistics& stats;
  Clock::time_point start,last;
  NewtonCost current=NewtonCost::CoreOther;
  bool finished=false;
  void tick(){auto now=Clock::now();stats.cost_seconds[(int)current]+=std::chrono::duration<double>(now-last).count();last=now;}
public:
  // Redirect once at phase boundaries, not once per scalar evaluation.
  NewtonCalls* active_calls;
  NewtonCostLedger(LazyNewtonStatistics& s,Clock::time_point t):stats(s),start(t),last(t),active_calls(s.calls[1].data()){}
  int phase() const {return current==NewtonCost::Preparation?0:current==NewtonCost::Diagnostics?2:current==NewtonCost::Certificate?3:1;}
  struct Scope {
    NewtonCostLedger* owner;NewtonCost before;
    Scope(NewtonCostLedger& c,NewtonCost next):owner(&c),before(c.current){
      c.tick();
      // Constructor and presentation evaluations stay peripheral, even when
      // they reuse methods normally called by the numerical kernel.
      if(before!=NewtonCost::Preparation&&before!=NewtonCost::Diagnostics)c.current=next;
      c.active_calls=c.stats.calls[c.phase()].data();
    }
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
    ~Scope(){stop();}
    void stop(){if(owner){owner->tick();owner->current=before;owner->active_calls=owner->stats.calls[owner->phase()].data();owner=nullptr;}}
  };
  void finish(){if(finished)return;tick();finished=true;
    stats.solver_wall_seconds=std::chrono::duration<double>(last-start).count();
    stats.core_solve_seconds=0;
    for(int i=(int)NewtonCost::Evaluation;i<=(int)NewtonCost::CoreOther;++i)stats.core_solve_seconds+=stats.cost_seconds[i];
  }
};
}
