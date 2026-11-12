#pragma once
// Observational work trace.  It never changes a model, a stopping test or a
// backend request.  Timers are subsets of the enclosing restoration wall time.
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <type_traits>

namespace lazyipdca::qcqp {

struct NativeRestorationTrace {
  std::uint64_t factor_api_calls=0,backend_solve_api_calls=0;
  std::uint64_t dense_factor_solve_calls=0,iterative_system_attempts=0,cg_iterations=0;
  std::uint64_t spectral_calls=0,quality_requests=0,curvature_solve_calls=0;
  std::uint64_t backend_regularization_attempts=0,backend_refinement_steps=0;
  double factor_api_seconds=0,backend_solve_api_seconds=0;
  double dense_factor_solve_seconds=0,iterative_system_seconds=0,spectral_seconds=0;
  double quality_seconds=0,curvature_solve_seconds=0;
  std::uint64_t factorization_calls() const { return factor_api_calls+dense_factor_solve_calls; }
  std::uint64_t linear_system_calls() const {
    return backend_solve_api_calls+dense_factor_solve_calls+iterative_system_attempts+curvature_solve_calls;
  }
};

inline void add_native_restoration_count(int& target,std::uint64_t increment) {
  if(target<0||increment>std::uint64_t(std::numeric_limits<int>::max()-target))
    throw std::overflow_error("native restoration interface counter exceeds result capacity");
  target+=static_cast<int>(increment);
}

inline thread_local NativeRestorationTrace* active_native_restoration_trace=nullptr;
inline bool native_restoration_trace_active(){return active_native_restoration_trace!=nullptr;}

class NativeRestorationTraceScope {
  NativeRestorationTrace* previous_;
 public:
  explicit NativeRestorationTraceScope(NativeRestorationTrace& trace)
      :previous_(active_native_restoration_trace){active_native_restoration_trace=&trace;}
  ~NativeRestorationTraceScope(){active_native_restoration_trace=previous_;}
  NativeRestorationTraceScope(const NativeRestorationTraceScope&)=delete;
  NativeRestorationTraceScope& operator=(const NativeRestorationTraceScope&)=delete;
};

// One merge for a normal return OR for an exception after attempted work.
// The exception is never converted into solver success. The merger executes
// outside the try block on the normal path, so a merger exception cannot
// accidentally debit the same phase twice.
template<class Function,class Merger>
auto run_counted_native_restoration(NativeRestorationTrace& work,
                                    Function&& function,Merger&& merger) {
  using Return=decltype(function());
  static_assert(!std::is_void<Return>::value&&!std::is_reference<Return>::value,
                "restoration trace expects a value result");
  auto invoke=[&]() {
    NativeRestorationTraceScope counted(work);
    try {return function();}
    catch(...) {merger(work);throw;}
  };
  auto result=invoke();
  merger(work);
  return result;
}

enum class NativeRestorationOperation { Factor, Solve, Dense, Iterative, Spectral, Quality, Curvature };
class NativeRestorationCall {
  using Clock=std::chrono::steady_clock;
  NativeRestorationTrace* trace_;
  NativeRestorationOperation operation_;
  Clock::time_point start_;
 public:
  explicit NativeRestorationCall(NativeRestorationOperation operation)
      :trace_(active_native_restoration_trace),operation_(operation),start_(trace_?Clock::now():Clock::time_point{}) {
    if(!trace_)return;
    switch(operation_){
      case NativeRestorationOperation::Factor:++trace_->factor_api_calls;break;
      case NativeRestorationOperation::Solve:++trace_->backend_solve_api_calls;break;
      case NativeRestorationOperation::Dense:++trace_->dense_factor_solve_calls;break;
      case NativeRestorationOperation::Iterative:++trace_->iterative_system_attempts;break;
      case NativeRestorationOperation::Spectral:++trace_->spectral_calls;break;
      case NativeRestorationOperation::Quality:++trace_->quality_requests;break;
      case NativeRestorationOperation::Curvature:++trace_->curvature_solve_calls;break;
    }
  }
  ~NativeRestorationCall(){
    if(!trace_)return;
    const double elapsed=std::chrono::duration<double>(Clock::now()-start_).count();
    switch(operation_){
      case NativeRestorationOperation::Factor:trace_->factor_api_seconds+=elapsed;break;
      case NativeRestorationOperation::Solve:trace_->backend_solve_api_seconds+=elapsed;break;
      case NativeRestorationOperation::Dense:trace_->dense_factor_solve_seconds+=elapsed;break;
      case NativeRestorationOperation::Iterative:trace_->iterative_system_seconds+=elapsed;break;
      case NativeRestorationOperation::Spectral:trace_->spectral_seconds+=elapsed;break;
      case NativeRestorationOperation::Quality:trace_->quality_seconds+=elapsed;break;
      case NativeRestorationOperation::Curvature:trace_->curvature_solve_seconds+=elapsed;break;
    }
  }
  NativeRestorationTrace* trace() const{return trace_;}
  NativeRestorationCall(const NativeRestorationCall&)=delete;
};

class NativeRestorationPcgCall {
  NativeRestorationCall call_{NativeRestorationOperation::Iterative};
 public:
  void iteration(){if(call_.trace())++call_.trace()->cg_iterations;}
};

class NativeRestorationFactor final:public SymmetricFactorization {
  std::unique_ptr<SymmetricFactorization> delegate_;
 public:
  explicit NativeRestorationFactor(std::unique_ptr<SymmetricFactorization> p):delegate_(std::move(p)){}
  bool factor(int n,int primal,const std::vector<SymmetricTriplet>& entries,std::string& message)override{
    NativeRestorationCall call(NativeRestorationOperation::Factor);
    const bool ok=delegate_->factor(n,primal,entries,message);
    if(call.trace())call.trace()->backend_regularization_attempts+=std::max(0,delegate_->last_regularization_attempts());
    return ok;
  }
  bool solve(Vec& rhs,std::string& message)const override{
    NativeRestorationCall call(NativeRestorationOperation::Solve);
    const bool ok=delegate_->solve(rhs,message);
    if(call.trace())call.trace()->backend_refinement_steps+=std::max(0,delegate_->last_refinement_steps());
    return ok;
  }
  bool increase_quality(std::string& message)override{
    NativeRestorationCall call(NativeRestorationOperation::Quality);return delegate_->increase_quality(message);
  }
  bool solve_curvature_probe(Vec& rhs,std::string& message)const override{
    NativeRestorationCall call(NativeRestorationOperation::Curvature);return delegate_->solve_curvature_probe(rhs,message);
  }
#define NATIVE_TRACE_FORWARD(type,name) type name() const override{return delegate_->name();}
  NATIVE_TRACE_FORWARD(double,pivot_tolerance)
  NATIVE_TRACE_FORWARD(int,curvature_defect)
  NATIVE_TRACE_FORWARD(int,failed_pivot_coordinate)
  NATIVE_TRACE_FORWARD(double,last_backward_error)
  NATIVE_TRACE_FORWARD(double,last_initial_backward_error)
  NATIVE_TRACE_FORWARD(double,last_residual_inf)
  NATIVE_TRACE_FORWARD(int,last_refinement_steps)
  NATIVE_TRACE_FORWARD(int,symbolic_analyses)
  NATIVE_TRACE_FORWARD(int,symbolic_reuses)
  NATIVE_TRACE_FORWARD(int,symbolic_invalidations)
  NATIVE_TRACE_FORWARD(double,symbolic_seconds)
  NATIVE_TRACE_FORWARD(double,numeric_seconds)
  NATIVE_TRACE_FORWARD(double,assembly_seconds)
  NATIVE_TRACE_FORWARD(double,linear_solve_seconds)
  NATIVE_TRACE_FORWARD(bool,last_symbolic_reused)
  NATIVE_TRACE_FORWARD(bool,has_symbolic_cache)
  NATIVE_TRACE_FORWARD(int,numeric_scatter_builds)
  NATIVE_TRACE_FORWARD(int,numeric_scatter_reuses)
  NATIVE_TRACE_FORWARD(int,numeric_scatter_invalidations)
  NATIVE_TRACE_FORWARD(bool,last_numeric_scatter_reused)
  NATIVE_TRACE_FORWARD(int,last_positive_pivots)
  NATIVE_TRACE_FORWARD(int,last_negative_pivots)
  NATIVE_TRACE_FORWARD(int,last_zero_pivots)
  NATIVE_TRACE_FORWARD(double,minimum_abs_pivot)
  NATIVE_TRACE_FORWARD(double,maximum_abs_pivot)
  NATIVE_TRACE_FORWARD(int,minimum_abs_pivot_coordinate)
  NATIVE_TRACE_FORWARD(double,last_primal_regularization)
  NATIVE_TRACE_FORWARD(double,last_dual_regularization)
  NATIVE_TRACE_FORWARD(int,last_regularization_attempts)
  NATIVE_TRACE_FORWARD(const char*,name)
#undef NATIVE_TRACE_FORWARD
};

inline std::unique_ptr<SymmetricFactorization> trace_native_restoration_factor(
    std::unique_ptr<SymmetricFactorization> factor) {
  // All restoration factors are constructed inside their enclosing trace.
  // Outside that scope the factory returns the historical concrete object.
  if(factor&&native_restoration_trace_active())return std::make_unique<NativeRestorationFactor>(std::move(factor));
  return factor;
}

inline void write_native_restoration_trace(std::ostream& out,const NativeRestorationTrace& w,const char* phase){
  out<<"[NATIVE-RESTORATION-WORK] {\"phase\":\""<<phase<<"\",\"factor_api_calls\":"<<w.factor_api_calls
      <<",\"backend_solve_api_calls\":"<<w.backend_solve_api_calls<<",\"dense_factor_solve_calls\":"<<w.dense_factor_solve_calls
      <<",\"iterative_system_attempts\":"<<w.iterative_system_attempts<<",\"cg_iterations\":"<<w.cg_iterations
      <<",\"spectral_calls\":"<<w.spectral_calls<<",\"quality_requests\":"<<w.quality_requests
      <<",\"curvature_solve_calls\":"<<w.curvature_solve_calls<<",\"backend_regularization_attempts\":"<<w.backend_regularization_attempts
      <<",\"backend_refinement_steps\":"<<w.backend_refinement_steps<<",\"factor_api_seconds\":"<<w.factor_api_seconds
      <<",\"backend_solve_api_seconds\":"<<w.backend_solve_api_seconds<<",\"dense_factor_solve_seconds\":"<<w.dense_factor_solve_seconds
      <<",\"iterative_system_seconds\":"<<w.iterative_system_seconds<<",\"spectral_seconds\":"<<w.spectral_seconds
      <<",\"quality_seconds\":"<<w.quality_seconds<<",\"curvature_solve_seconds\":"<<w.curvature_solve_seconds
      <<",\"timers_subset_of_restoration\":true,\"backend_internal_numeric_calls_not_inferred\":true}"<<std::endl;
}
} // namespace lazyipdca::qcqp
