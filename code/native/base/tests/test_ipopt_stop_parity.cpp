#include "lazyipdca_qcqp/acceptable.hpp"
#include "IpIpoptApplication.hpp"
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptData.hpp"
#include "IpIteratesVector.hpp"
#include "IpDenseVector.hpp"
#include "IpTNLP.hpp"
#include <iostream>
#include <stdexcept>
using Ipopt::Index;using Ipopt::Number;
using namespace lazyipdca::qcqp;
class Probe final: public Ipopt::TNLP {
 public:
  Settings settings;
  bool square;int samples=0,mapped_samples=0;double worst=0;std::string failure;
  explicit Probe(bool sq,double sm):square(sq){settings.nlp_error_scaling_max=sm;settings.termination_test="ipopt";}
  bool get_nlp_info(Index& n,Index& m,Index& j,Index& h,IndexStyleEnum& style)override{
    n=3;m=square?3:2;j=3;h=3;style=C_STYLE;return true;
  }
  bool get_bounds_info(Index,Number* l,Number* u,Index m,Number* gl,Number* gu)override{
    for(int i=0;i<3;++i){l[i]=-2;u[i]=2;}
    for(int i=0;i<m;++i){gl[i]=(square||i==0)?0:-1e20;gu[i]=0;}return true;
  }
  bool get_starting_point(Index,bool,Number* x,bool iz,Number* zl,Number* zu,Index m,bool il,Number* lam)override{
    x[0]=.3;x[1]=.7;x[2]=.3;
    if(iz)for(int i=0;i<3;++i){zl[i]=1;zu[i]=1;}
    if(il)for(int i=0;i<m;++i)lam[i]=i==0?20:5;
    return true;
  }
  bool get_scaling_parameters(Number& sf,bool& ux,Index,Number*,bool& ug,Index m,Number* g)override{
    sf=1e-4;ux=false;ug=true;for(int i=0;i<m;++i)g[i]=i==0?.02:.03;return true;
  }
  bool eval_f(Index,const Number* x,bool,Number& f)override{
    f=1e5*((x[0]-.2)*(x[0]-.2)+2*(x[1]-.6)*(x[1]-.6)+(x[2]+.1)*(x[2]+.1));return true;
  }
  bool eval_grad_f(Index,const Number* x,bool,Number* g)override{
    g[0]=2e5*(x[0]-.2);g[1]=4e5*(x[1]-.6);g[2]=2e5*(x[2]+.1);return true;
  }
  bool eval_g(Index,const Number* x,bool,Index,Number* g)override{
    if(square){g[0]=x[0]-.2;g[1]=x[1]-.6;g[2]=x[2]+.1;}
    else{g[0]=x[0]+x[1]-1;g[1]=x[2]*x[2]-.25;}return true;
  }
  bool eval_jac_g(Index,const Number* x,bool,Index,Index,Index* rows,Index* cols,Number* values)override{
    if(!values){rows[0]=0;rows[1]=square?1:0;rows[2]=square?2:1;for(int i=0;i<3;++i)cols[i]=i;}
    else{values[0]=1;values[1]=1;values[2]=square?1:2*x[2];}return true;
  }
  bool eval_h(Index,const Number*,bool,Number sf,Index,const Number* lam,bool,Index,Index* rows,Index* cols,Number* v)override{
    if(!v)for(int i=0;i<3;++i)rows[i]=cols[i]=i;
    else{v[0]=sf*2e5;v[1]=sf*4e5;v[2]=sf*2e5+(square?0:2*lam[1]);}return true;
  }
  bool intermediate_callback(Ipopt::AlgorithmMode,Index iteration,Number,Number,Number,Number,Number,Number,Number,Number,Index,
       const Ipopt::IpoptData* data,Ipopt::IpoptCalculatedQuantities* cq)override{
    try{
      auto cur=data->curr();IpoptStopSnapshot v;
      v.internal_dual=cq->curr_dual_infeasibility(Ipopt::NORM_MAX);
      v.internal_primal=cq->curr_nlp_constraint_violation(Ipopt::NORM_MAX);
      v.internal_compl=cq->curr_complementarity(0.,Ipopt::NORM_MAX);
      v.unscaled_dual=cq->unscaled_curr_dual_infeasibility(Ipopt::NORM_MAX);
      v.unscaled_primal=cq->unscaled_curr_nlp_constraint_violation(Ipopt::NORM_MAX);
      v.unscaled_compl=cq->unscaled_curr_complementarity(0.,Ipopt::NORM_MAX);
      v.y_l1=cur->y_c()->Asum()+cur->y_d()->Asum();
      v.z_l1=cur->z_L()->Asum()+cur->z_U()->Asum()+cur->v_L()->Asum()+cur->v_U()->Asum();
      v.compl_dimension=cur->z_L()->Dim()+cur->z_U()->Dim()+cur->v_L()->Dim()+cur->v_U()->Dim();
      v.dual_dimension=v.compl_dimension+cur->y_c()->Dim()+cur->y_d()->Dim();
      v.free_variables=cur->x()->Dim();v.equality_dimension=cur->y_c()->Dim();
      const auto d=ipopt_stop_decision(v,settings);const double native=cq->curr_nlp_error();
      const double error=std::abs(native-d.metrics.overall_nlp_error)/std::max(1.,std::abs(native));
      worst=std::max(worst,error);++samples;
      if(error>2e-14)throw std::runtime_error("native Ipopt curr_nlp_error mismatch");
      if(!square && iteration==0){
        Result r;double xx[3],zl[3],zu[3],gg[2],ll[2];
        if(!get_curr_iterate(data,cq,false,3,xx,zl,zu,2,gg,ll))throw std::runtime_error("get_curr_iterate failed");
        if(cq->curr_grad_lag_s()->Amax()>1e-9)throw std::runtime_error("initial y_d != v_U; canonical mapping not applicable");
        auto sv=dynamic_cast<const Ipopt::DenseVector*>(Ipopt::GetRawPtr(cur->s()));
        if(!sv)throw std::runtime_error("expected dense native slack vector");
        QCQPProblem p;p.n=3;p.lb={-2,-2,-2};p.ub={2,2,2};p.x0={.3,.7,.3};
        p.objective.linear={-4e4,-24e4,2e4};p.objective.constant=77000;p.objective.hessian.n=3;
        p.objective.hessian.entries={{0,0,2e5},{1,1,4e5},{2,2,2e5}};
        QuadraticFunction eq;eq.linear={1,1,0};eq.constant=-1;eq.hessian.n=3;p.equalities={eq};
        QuadraticFunction iq;iq.linear={0,0,0};iq.constant=-.25;iq.hessian.n=3;iq.hessian.entries={{2,2,2}};p.inequalities={iq};
        r.x.assign(xx,xx+3);r.certificate_equality_multipliers={ll[0]};r.certificate_inequality_multipliers={ll[1]};
        r.certificate_lower_multipliers.assign(zl,zl+3);r.certificate_upper_multipliers.assign(zu,zu+3);
        r.termination_scaling_available=true;r.termination_objective_scale=1e-4;
        r.termination_equality_scales={.02};r.termination_inequality_scales={.03};r.termination_inequality_slacks={-sv->Values()[0]/.03};
        detail::audit_original_termination(p,settings,r);
        if(!r.termination_audit_valid||std::abs(r.scaled_nlp_error-native)>1e-10*std::max(1.,native))
          throw std::runtime_error("original-to-internal audit differs from native Ipopt");
        if(std::abs(r.termination_compl_inf-v.unscaled_compl)>1e-9*std::max(1.,v.unscaled_compl))
          throw std::runtime_error("slack complementarity mapping mismatch");
        ++mapped_samples;
      }
      return true;
    }catch(const std::exception& e){failure=e.what();return false;}
  }
  void finalize_solution(Ipopt::SolverReturn,Index,const Number*,const Number*,const Number*,Index,const Number*,const Number*,Number,
      const Ipopt::IpoptData*,Ipopt::IpoptCalculatedQuantities*)override{}
};
int main(){
  int samples=0,mapped=0;
  for(bool square:{false,true})for(double sm:{1.,100.}){
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app=IpoptApplicationFactory();
    app->Options()->SetStringValue("linear_solver","mumps");
    app->Options()->SetStringValue("nlp_scaling_method","user-scaling");
    app->Options()->SetStringValue("warm_start_init_point","yes");
    app->Options()->SetNumericValue("bound_relax_factor",0);
    app->Options()->SetNumericValue("s_max",sm);
    app->Options()->SetIntegerValue("max_iter",100);
    app->Options()->SetIntegerValue("print_level",0);
    app->Options()->SetStringValue("sb","yes");
    if(app->Initialize()!=Ipopt::Solve_Succeeded)throw std::runtime_error("Ipopt init failed");
    Ipopt::SmartPtr<Probe> p=new Probe(square,sm);
    const auto status=app->OptimizeTNLP(p);
    if(!p->failure.empty())throw std::runtime_error(p->failure);
    if(p->samples<2||(status!=Ipopt::Solve_Succeeded&&status!=Ipopt::Solved_To_Acceptable_Level))throw std::runtime_error("Ipopt parity solve failed");
    samples+=p->samples;mapped+=p->mapped_samples;
    std::cout<<"square="<<square<<" s_max="<<sm<<" samples="<<p->samples<<" max_relative_error="<<p->worst<<"\n";
  }
  std::cout<<"IPOPT_NATIVE_PARITY_PASS samples="<<samples<<" mapped_original_endpoints="<<mapped<<"\n";
}
