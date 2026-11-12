#include "lazyipdca_qcqp/dca_filter.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
QuadraticFunction quad(Vec g,std::vector<SymTriplet> h,double c=0){QuadraticFunction q;q.linear=g;q.constant=c;q.hessian.n=g.size();q.hessian.entries=h;return q;}
int main(){try{
  for(bool elimination:{false,true})for(bool outside:{false,true}){
    QCQPProblem p;p.n=3;p.lb=Vec(3,-2);p.ub=Vec(3,2);p.x0={0.999,0.01,0};
    p.objective=quad({-2,0,0},{{0,0,-1},{1,1,2},{2,2,1}});
    p.equalities.push_back(quad({0,0,1},{}));
    p.inequalities.push_back(quad({0,0,0},{{0,0,2},{1,1,2}},-1));
    // A remaining curved inequality must keep its convex majorant.
    p.inequalities.push_back(quad({0,0,0},{{1,1,2}},-0.25));
    if(outside){
      p.x0={1.001,0.01,0};
      p.objective=quad({0,0,0},{{0,0,2},{1,1,4},{2,2,-1}});
      p.inequalities[0]=quad({0,0,0},{{0,0,-2},{1,1,-2}},1);
    }
    Settings s;s.linear_backend="native";s.dca_engine="filter";s.live_display_level=0;
    s.dca_local_active=true;s.dca_pair_elimination=elimination;s.dca_local_trigger=1e6;
    s.max_outer_iterations=200;s.feasibility_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
    auto r=solve_filter_dca(p,s);
    std::cout<<r.message<<" local="<<r.dca_filter.local_acceptances<<" x="<<r.x[0]<<"\n";
    if(r.status!=Status::Solved||std::abs(r.x[0]-1)>1e-5||std::abs(r.x[1])>1e-4||!r.dca_filter.local_acceptances)
      throw std::runtime_error("local DCA failed analytical constrained solution");
  }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
