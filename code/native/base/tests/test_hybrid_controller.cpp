#include "lazyipdca_qcqp/convexification_study.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;
void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(){
 check(study::certified_objective_base_lower(2,{{0,0,200},{1,1,200}})==200,"strong objective model curvature ignored");
 check(study::certified_objective_base_lower(2,{{0,0,1},{0,1,2},{1,1,1}})==-1,"insufficient objective curvature overstated");
 check(study::certified_objective_base_lower(3,{{0,0,200},{1,1,200}})==0,"ambient zero model curvature omitted");
 const double outward=study::certified_objective_base_lower(2,{{0,0,.3},{0,1,.1},{1,1,.3}});
 check(study::Rational(outward)<=study::Rational(.3)-study::Rational(.1),"objective curvature lower bound rounded inward");
 check(study::certified_proximal_margin_floor(1,2)==0,"strictly convex model received an unnecessary proximal floor");
 check(study::certified_objective_margin_sum(2,0)==2,"intrinsic model margin discarded");
 const auto floor=study::certified_proximal_margin_floor(.3,.1);
 const auto exact_floor=study::Rational(.3)-study::Rational(.1);
 check(study::Rational(floor)>=exact_floor,"proximal floor rounded below its certified deficit");
 check(study::Rational(std::nextafter(floor,-INFINITY))<exact_floor,"proximal floor is not the minimal representable certified value");
 const auto margin=study::certified_objective_margin_sum(.1,.3);
 const auto exact_margin=study::Rational(.1)+study::Rational(.3);
 check(study::Rational(margin)<=exact_margin,"summed model margin rounded above the certificate");
 check(study::Rational(std::nextafter(margin,INFINITY))>exact_margin,"summed model margin is unnecessarily loose");
 const auto scaled_margin=study::certified_objective_margin_sum(.1,.3,.7);
 check(study::Rational(scaled_margin)<=(study::Rational(.1)+study::Rational(.3))/study::Rational(.7),"objective normalization rounded the margin upward");
 // An affine elimination has identity rows on retained coordinates. Here
 // x=(u,7u-5v,v) and Q=diag(2,3,4). Its reduced quadratic dominates the
 // certified (2+tau)*I despite the additional eliminated-coordinate energy.
 const double test_tau=.1,retained_margin=study::certified_objective_margin_sum(2,test_tau);
 for(int i=-8;i<=8;++i)for(int j=-8;j<=8;++j) {
   const study::Rational u=study::Rational(i)/4,v=study::Rational(j)/4;
   const auto quadratic=2*u*u+3*(7*u-5*v)*(7*u-5*v)+4*v*v+study::Rational(test_tau)*(u*u+v*v);
   check(quadratic>=study::Rational(retained_margin)*(u*u+v*v),"retained-coordinate margin exceeds the exact reduced quadratic");
 }
 bool invalid_margin=false;try{study::certified_objective_margin_sum(-1,1);}catch(const std::invalid_argument&){invalid_margin=true;}
 check(invalid_margin,"invalid negative curvature certificate accepted");
 auto c=study::construct_hybrid(2,{{0,0,-4},{1,1,1}});
 check(c.rho==4&&c.hybrid_atoms.empty(),"initial metric not scalar");
 check(study::hybrid_enrich(c,2,2,2,.5),"first atom not admitted");
 check(c.hybrid_atoms.size()==1&&c.rho>1.9&&c.rho<2.1,"residual did not decrease with atom");
 check(study::hybrid_certificate(c,c.rho),"first model not certified");
 check(study::hybrid_enrich(c,2,2,2,.5),"second atom not admitted");
 check(c.hybrid_atoms.size()==2&&c.rho<1.1&&c.rho>.9,"second residual not decreased");
 auto old=c;
 check(!study::hybrid_enrich(c,2,2,2,.5),"atom budget not enforced");
 check(!study::hybrid_downweight(c,2,0,.5),"unsafe downweight accepted");
 check(c.rho==old.rho&&c.hybrid_atoms[0].weight==old.hybrid_atoms[0].weight,"failed trial mutated accepted metric");
 c.rho=4;
 check(study::hybrid_downweight(c,2,0,.5),"safe downweight rejected");
 check(study::hybrid_certificate(c,c.rho),"downweighted model not PSD");
 auto convex=study::construct_hybrid(2,{{0,0,1},{0,1,2},{1,1,4}});
 check(convex.rho==0&&convex.hybrid_atoms.empty(),"PSD function corrected");
 std::vector<SymTriplet> large;for(int i=0;i<20;++i)large.push_back({i,i,i==0?-2.0:1.0});
 auto big=study::construct_hybrid(20,large);check(big.rho==2,"large initial scalar");
 check(study::hybrid_enrich(big,20,2,2,.5),"large sparse atom missing");
 check(big.rho==1&&big.hybrid_atoms[0].indices.size()==1,"large certified residual incorrect");
 check(study::hybrid_certificate(big,big.rho),"large fullspace certificate failed");
 auto no=study::construct_hybrid(2,{{0,0,-1},{0,1,2},{1,1,1}});
 check(!study::hybrid_enrich(no,2,0,2,.5),"zero atom ablation not respected");
 auto refined=no;refined.rho=study::hybrid_certified_residual(refined,refined.rho);
 check(refined.rho<no.rho&&refined.hybrid_atoms.empty(),"H0 scalar refinement missing");
 // Exercise the same role selector and constructors used by production HS/SH.
 for(const std::string mode:{"HS","SH"}) {
  check(study::support_study_mode(mode),"mixed mode missed H3 numerical path");
  for(bool objective:{true,false}) {
   const bool hybrid=study::correction_family(mode,objective)=="hybrid";
   auto role=hybrid?study::construct_hybrid(4,{{1,1,-1},{1,3,2},{3,3,1}}):
     study::construct_support_scalar(4,{{1,1,-1},{1,3,2},{3,3,1}});
   check(role.support_scalar&&role.support==std::vector<int>({1,3}),"mixed mode scalar scope incorrect");
   const double before=role.rho;
   role.rho=study::hybrid_certified_residual(role,role.rho);
   check(role.rho<before&&study::hybrid_certificate(role,role.rho),"mixed role scalar refinement missing");
   const auto accepted=role;
   const bool enriched=study::hybrid_enrich(role,4,8,4,.5);
   if(hybrid) {
    check(enriched&&!role.hybrid_atoms.empty(),"mixed H role failed to activate atoms");
    check(study::hybrid_certificate(role,role.rho),"mixed H model not certified");
   } else {
    check(!enriched&&role.hybrid_atoms.empty()&&role.rho==accepted.rho,"mixed S role received atoms");
    check(!study::hybrid_downweight(role,4,0,.5),"mixed S role accepted downweight");
   }
  }
 }
 check(!study::support_study_mode("SS")&&study::correction_family("SA",true)=="scalar"&&
   study::correction_family("SA",false)=="atoms","legacy role routing changed");
 std::cout<<"hybrid_controller_certificates PASS\n";
}
