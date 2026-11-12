#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "defined_pivot_soc.hpp"
struct Hessian {int i,j;double v;};
struct Row {std::vector<std::pair<int,double>> linear;std::vector<Hessian> hessian;std::vector<std::pair<int,double>> signed_square;double scale=1;};
static int checks=0;
static void require(bool value,const char* message) {++checks;if(!value)throw std::runtime_error(message);}
static Row product(int pivot,int i=0,int j=1,double coefficient=1,double scale=1) {
  Row q;q.linear={{pivot,coefficient}};q.hessian={{i,j,-1}};q.scale=scale;return q;
}
int main() {
  try {
    std::vector<Row> rows={product(2),Row{{{0,1},{1,1}},{},{},1}};
    auto plan=DcaDefinedPivotSoc::build(3,rows);
    require(plan.supported&&plan.curved_equalities==1&&plan.pivots.size()==1,"basic defined product not recognized");
    require(plan.pivots[0].row==0&&plan.pivots[0].variable==2&&plan.pivots[0].coefficient_internal==1,"basic pivot metadata");
    std::vector<double> delta={999};std::string rejection="old";
    require(plan.apply(std::vector<long double>{.25L,0},delta,&rejection),"basic correction rejected");
    require(delta==std::vector<double>({0,0,-.25})&&rejection.empty(),"basic correction or transaction");
    require(plan.apply(std::vector<double>{0,0},delta)&&delta==std::vector<double>({0,0,0}),"zero defect handling");

    // Original row scaling enters both numerator and denominator. Cache the
    // immutable coefficient, not an anchor-dependent gradient.
    rows[0]=product(2,0,1,-4,2.5);plan=DcaDefinedPivotSoc::build(3,rows);
    require(plan.pivots[0].coefficient_internal==-10,"scaled constant pivot coefficient");
    require(plan.apply(std::vector<long double>{3.75L,0},delta)&&delta[2]==.375,"scaled defect correction");
    auto cache=plan;rows[0].scale=17;
    require(cache.apply(std::vector<long double>{3.75L,0},delta)&&delta[2]==.375,"plan did not own immutable coefficient cache");

    // All curved equations must be covered, with disjoint pivots absent from
    // every Hessian. These exact dyadic coefficients make J*delta=-defect.
    rows={product(2),product(3,0,1,-2,.25)};
    plan=DcaDefinedPivotSoc::build(4,rows);
    require(plan.supported&&plan.curved_equalities==2&&plan.pivots.size()==2,"multiple private pivots");
    require(plan.apply(std::vector<long double>{.5L,.125L},delta),"multiple correction rejected");
    require(delta==std::vector<double>({0,0,-.5,.25}),"multiple correction values");
    for(const auto& q:rows)for(auto h:q.hessian)require(delta[h.i]==0&&delta[h.j]==0,"correction enters Hessian support");
    require(rows[0].scale*delta[2]==-.5&&rows[1].scale*(-2)*delta[3]==-.125,"normal identity changed");

    rows={product(2),Row{{{2,1}},{},{},1}};plan=DcaDefinedPivotSoc::build(3,rows);
    require(!plan.supported&&plan.uncovered_rows==std::vector<int>({0}),"pivot used in affine equality accepted");
    rows={product(2),product(3,2,1)};plan=DcaDefinedPivotSoc::build(4,rows);
    require(!plan.supported&&plan.uncovered_rows==std::vector<int>({0}),"pivot in another Hessian accepted");
    rows={Row{{},{{0,1,1}},{},1}};plan=DcaDefinedPivotSoc::build(2,rows);
    require(!plan.supported,"unrecognized curved equality accepted");
    rows={product(2),Row{{},{{2,2,0}},{},1}};plan=DcaDefinedPivotSoc::build(3,rows);
    require(plan.supported&&plan.curved_equalities==1,"zero Hessian entry blocks safe pivot");
    rows={product(2)};rows[0].linear.push_back({2,1});
    require(!DcaDefinedPivotSoc::build(3,rows).supported,"duplicate unknown coefficient representation accepted");

    // Deterministic largest coefficient and then coordinate index.
    rows={product(2)};rows[0].linear={{3,2},{2,-2},{4,1}};
    plan=DcaDefinedPivotSoc::build(5,rows);
    require(plan.supported&&plan.pivots[0].variable==2,"coefficient tie not deterministic");
    rows[0].linear.push_back({5,3});plan=DcaDefinedPivotSoc::build(6,rows);
    require(plan.pivots[0].variable==5,"largest coefficient not chosen");

    // Reject malformed or unsupported models, without a fallback optimizer.
    rows={product(2)};rows[0].scale=0;require(!DcaDefinedPivotSoc::build(3,rows).supported,"zero row scale");
    rows[0].scale=NAN;require(!DcaDefinedPivotSoc::build(3,rows).supported,"NaN row scale");
    rows={product(2)};rows[0].linear[0].second=INFINITY;require(!DcaDefinedPivotSoc::build(3,rows).supported,"infinite linear coefficient");
    rows={product(3)};require(!DcaDefinedPivotSoc::build(3,rows).supported,"linear coordinate out of range");
    rows={product(2)};rows[0].hessian[0].i=3;require(!DcaDefinedPivotSoc::build(3,rows).supported,"Hessian coordinate out of range");
    rows={product(2)};rows[0].hessian[0].v=NAN;require(!DcaDefinedPivotSoc::build(3,rows).supported,"NaN Hessian");
    rows={product(2)};rows[0].signed_square={{0,1}};require(!DcaDefinedPivotSoc::build(3,rows).supported,"native non-QCQP term");

    // Every apply failure is transactional and finite range is explicit.
    auto fails=[&](const DcaDefinedPivotSoc& p,const std::vector<long double>& defect) {
      delta={123,456};rejection.clear();const bool ok=p.apply(defect,delta,&rejection);
      require(!ok&&!rejection.empty()&&delta==std::vector<double>({123,456}),"failed correction changed output or lacked reason");
    };
    rows={product(2),Row{{{0,1}},{},{},1}};plan=DcaDefinedPivotSoc::build(3,rows);
    fails(plan,{1});fails(plan,{1,1});fails(plan,{NAN,0});fails(plan,{INFINITY,0});
    rows={product(2,0,1,std::numeric_limits<double>::denorm_min())};plan=DcaDefinedPivotSoc::build(3,rows);
    fails(plan,{std::numeric_limits<double>::max()});
    rows={product(2,0,1,std::numeric_limits<double>::max())};plan=DcaDefinedPivotSoc::build(3,rows);
    fails(plan,{std::numeric_limits<double>::denorm_min()});
    rows={Row{{},{{0,1,1}},{},1}};fails(DcaDefinedPivotSoc::build(2,rows),{0});
    plan=DcaDefinedPivotSoc::build(2,std::vector<Row>{});
    require(plan.supported&&plan.apply(std::vector<double>{},delta)&&delta==std::vector<double>({0,0}),"empty geometry");
    std::cout<<"defined_pivot_soc_actual_helper PASS checks="<<checks<<"\n";
    return 0;
  }catch(const std::exception& error) {std::cerr<<"FAIL "<<error.what()<<" checks="<<checks<<"\n";return 1;}
}
