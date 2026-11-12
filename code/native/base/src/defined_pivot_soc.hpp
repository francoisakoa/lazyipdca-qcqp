#pragma once
// Pure local SOC geometry. The includer supplies algorithm, cmath, limits,
// string and vector. Rows expose ORIGINAL linear/hessian/signed_square and
// their positive internal scale. No Workspace state or KKT factor is changed.
struct DcaDefinedPivotSoc {
  struct Pivot {int row=-1,variable=-1;long double coefficient_internal=0;};
  int n=0,equality_count=0,curved_equalities=0;
  bool supported=false;
  std::vector<Pivot> pivots;
  std::vector<int> uncovered_rows;
  std::string reason="unprepared";

  template<class Rows>
  static DcaDefinedPivotSoc build(int dimension,const Rows& equalities) {
    DcaDefinedPivotSoc plan;plan.n=dimension;plan.equality_count=static_cast<int>(equalities.size());
    if(dimension<0){plan.reason="invalid-dimension";return plan;}
    std::vector<int> linear_occurrences(dimension,0);
    std::vector<unsigned char> in_any_hessian(dimension,0),curved(equalities.size(),0);
    for(int e=0;e<plan.equality_count;++e) {
      const auto& q=equalities[e];
      if(!std::isfinite(q.scale)||!(q.scale>0)) {plan.reason="invalid-equality-scale";return plan;}
      for(auto a:q.linear) {
        if(a.first<0||a.first>=dimension||!std::isfinite(a.second)) {plan.reason="invalid-linear-coefficient";return plan;}
        if(a.second!=0)++linear_occurrences[a.first];
      }
      for(auto h:q.hessian) {
        if(h.i<0||h.j<0||h.i>=dimension||h.j>=dimension||!std::isfinite(h.v)) {plan.reason="invalid-Hessian-coefficient";return plan;}
        if(h.v!=0){curved[e]=1;in_any_hessian[h.i]=in_any_hessian[h.j]=1;}
      }
      for(auto a:q.signed_square)if(a.second!=0) {plan.reason="non-QCQP-native-term";return plan;}
    }
    for(int e=0;e<plan.equality_count;++e)if(curved[e]) {
      ++plan.curved_equalities;Pivot selected;selected.row=e;
      const auto& q=equalities[e];
      for(auto a:q.linear)if(a.second!=0&&linear_occurrences[a.first]==1&&!in_any_hessian[a.first]) {
        const long double c=static_cast<long double>(q.scale)*a.second;
        if(!std::isfinite(c)||c==0)continue;
        if(selected.variable<0||std::abs(c)>std::abs(selected.coefficient_internal)||
            (std::abs(c)==std::abs(selected.coefficient_internal)&&a.first<selected.variable))
          selected={e,a.first,c};
      }
      if(selected.variable<0)plan.uncovered_rows.push_back(e);
      else plan.pivots.push_back(selected);
    }
    plan.supported=plan.uncovered_rows.empty();
    plan.reason=plan.supported?"all-curved-equalities-have-private-linear-pivots":"curved-equality-without-private-linear-pivot";
    return plan;
  }

  // Defects have the SAME internal row scaling as coefficient_internal.
  // Output is transactional. This is a geometric equality-normal SOC;
  // it is deliberately not presented as a complete Newton-KKT solution.
  template<class Defects>
  bool apply(const Defects& defects,std::vector<double>& delta_x,std::string* rejection=nullptr) const {
    auto reject=[&](const char* message){if(rejection)*rejection=message;return false;};
    if(!supported)return reject("unsupported-equality-geometry");
    if(static_cast<int>(defects.size())!=equality_count)return reject("equality-defect-dimension");
    std::vector<unsigned char> selected(equality_count,0);
    for(const auto& pivot:pivots)selected[pivot.row]=1;
    for(int e=0;e<equality_count;++e) {
      const long double value=static_cast<long double>(defects[e]);
      if(!std::isfinite(value))return reject("nonfinite-equality-defect");
      if(!selected[e]&&value!=0)return reject("nonzero-affine-equality-defect");
    }
    std::vector<double> candidate(n,0);
    for(const auto& pivot:pivots) {
      const long double value=-static_cast<long double>(defects[pivot.row])/pivot.coefficient_internal;
      if(!std::isfinite(value)||std::abs(value)>std::numeric_limits<double>::max())return reject("SOC-coordinate-overflow");
      const double rounded=static_cast<double>(value);
      if(!std::isfinite(rounded)||(value!=0&&rounded==0))return reject("SOC-coordinate-underflow-or-nonfinite");
      candidate[pivot.variable]=rounded;
    }
    delta_x.swap(candidate);if(rejection)rejection->clear();return true;
  }
};
