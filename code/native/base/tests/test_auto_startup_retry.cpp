#include "lazyipdca_qcqp/majorant_startup_retry.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using lazyipdca::qcqp::detail::AutoStartupRetry;
void ck(bool b,const char* m){if(!b)throw std::runtime_error(m);}
bool sample(AutoStartupRetry& a,bool eligible=true,double alpha=1e-5,double primal=0.03,
            double dual=9000,double comp=0.009,double mu=1e-4,int left=2000,double seconds=300){
 return a.observe(eligible,alpha,primal,dual,comp,mu,1e-4,1e-6,1e-6,left,seconds);
}
int main(){try{
 AutoStartupRetry a;
 for(int i=0;i<63;++i)ck(!sample(a),"premature retry");
 ck(sample(a),"64 stalled tiny steps must trigger");a.mark_used();
 for(int i=0;i<100;++i)ck(!sample(a),"second retry forbidden");
 AutoStartupRetry progress;
 for(int i=0;i<63;++i)ck(!sample(progress),"window");
 ck(!sample(progress,true,1e-5,0.001),"primal progress preserves trajectory");
 ck(progress.streak()==0,"primal progress resets window");
 for(int i=0;i<63;++i)sample(progress);
 ck(!sample(progress,true,1e-5,0.03,10),"dual progress preserves trajectory");
 for(int k=0;k<8;++k){AutoStartupRetry z;for(int i=0;i<100;++i){bool b=false;
 switch(k){
 case 0:b=sample(z,false);break;
 case 1:b=sample(z,true,0.1);break;
 case 2:b=sample(z,true,1e-5,1e-7);break;
 case 3:b=sample(z,true,1e-5,0.03,9000,0.009,2e-5);break;
 case 4:b=sample(z,true,1e-5,0.03,9000,0.009,1e-4,63);break;
 case 5:b=sample(z,true,1e-5,0.03,9000,0.009,1e-4,2000,0);break;
 case 6:b=sample(z,true,1e-5,0.03,std::numeric_limits<double>::quiet_NaN());break;
 case 7:b=sample(z,true,1e-5,0.03,0.2,0.1);break;
 }ck(!b,"guarded case should not restart");}}
 AutoStartupRetry rejected;for(int i=0;i<63;++i)sample(rejected);rejected.clear_window();ck(!sample(rejected),"rejection breaks consecutive window");
 std::cout<<"PASS auto-startup retry: 64 steps, progress, finite state, one attempt, unchanged budget\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
