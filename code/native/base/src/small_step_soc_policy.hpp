#pragma once
#include <cmath>
#include <algorithm>
#include <deque>
#include <limits>
#include <stdexcept>

namespace lazyipdca::qcqp::detail {
// A bounded history of ACCEPTED original-filter steps. The source path stays
// unchanged until repeated small steps demonstrate a globalization stall.
// Once that evidence exists, SOC remains available; it must still pass the
// unchanged direction audit, domain/size checks and original filter.
class SmallStepSocPolicy {
  int patience_=0,streak_=0,accepted_=0;
  double threshold_=.125;
  bool latched_=false;
public:
  SmallStepSocPolicy(int patience,double threshold):patience_(patience),threshold_(threshold) {
    if(patience<0||patience>1000000||!std::isfinite(threshold)||!(threshold>0&&threshold<=1))
      throw std::invalid_argument("invalid small-step SOC controls");
  }
  bool enabled() const {return patience_==0||latched_;}
  bool latched() const {return latched_;}
  int streak() const {return streak_;}
  int accepted() const {return accepted_;}
  bool observe_accepted(double alpha) {
    if(!std::isfinite(alpha)||!(alpha>0&&alpha<=1))
      throw std::invalid_argument("invalid accepted SOC step");
    if(accepted_<1000000)++accepted_;
    if(patience_==0||latched_)return false;
    if(alpha<=threshold_) {if(streak_<patience_)++streak_;}
    else streak_=0;
    if(streak_>=patience_){latched_=true;return true;}
    return false;
  }
};

// Optional admission based on a bounded window, independent of the historical
// consecutive-step latch above. Moderately larger steps do not erase recent
// evidence when the average accepted step remains small, original theta has
// worsened over the same window, and the CURRENT accepted step has a material
// nonlinear equality defect. This only enables transactional SOC proposals;
// it does not accept a direction, alter the filter, or relax a KKT tolerance.
// patience==0 disables this additional policy, leaving historical admission
// entirely to SmallStepSocPolicy and its existing callers.
class NonlinearStepSocPolicy {
  struct Entry {double alpha,from_theta;};
  int patience_=0,accepted_=0;
  double threshold_=.125;
  std::deque<Entry> window_;
  long double alpha_sum_=0;
  bool latched_=false;
  const char* reason_="window-incomplete";
public:
  NonlinearStepSocPolicy(int patience,double threshold)
      :patience_(patience),threshold_(threshold) {
    if(patience<0||patience>1000000||!std::isfinite(threshold)||!(threshold>0&&threshold<=1))
      throw std::invalid_argument("invalid nonlinear-step SOC controls");
    if(patience==0)reason_="disabled";
  }
  bool enabled()const{return latched_;}
  bool latched()const{return latched_;}
  int accepted()const{return accepted_;}
  int window_size()const{return static_cast<int>(window_.size());}
  double mean_alpha()const{return window_.empty()?0:double(alpha_sum_/window_.size());}
  const char* reason()const{return reason_;}
  void clear_window() {
    window_.clear();alpha_sum_=0;
    if(!latched_)reason_=patience_==0?"disabled":"window-incomplete";
  }
  bool observe_accepted(double alpha,double from_theta,double to_theta,
                        double nonlinear_defect,double defect_floor) {
    if(!std::isfinite(alpha)||!(alpha>0&&alpha<=1)||
        !std::isfinite(from_theta)||from_theta<0||!std::isfinite(to_theta)||to_theta<0||
        !std::isfinite(nonlinear_defect)||nonlinear_defect<0||
        !std::isfinite(defect_floor)||defect_floor<0)
      throw std::invalid_argument("invalid nonlinear-step SOC observation");
    if(accepted_<1000000)++accepted_;
    if(patience_==0||latched_)return false;
    window_.push_back({alpha,from_theta});alpha_sum_+=alpha;
    if(window_.size()>static_cast<std::size_t>(patience_)) {
      alpha_sum_-=window_.front().alpha;window_.pop_front();
    }
    if(window_.size()<static_cast<std::size_t>(patience_)) {
      reason_="window-incomplete";return false;
    }
    if(alpha_sum_>static_cast<long double>(threshold_)*patience_) {
      reason_="window-not-small";return false;
    }
    const double anchor=window_.front().from_theta;
    const double roundoff=10*std::numeric_limits<double>::epsilon()*std::max(anchor,to_theta);
    if(!(to_theta-anchor>roundoff)) {reason_="theta-progress";return false;}
    if(!(nonlinear_defect>defect_floor)) {reason_="defect-below-floor";return false;}
    latched_=true;reason_="short-window-nonlinear-feasibility-loss";return true;
  }
};
}
