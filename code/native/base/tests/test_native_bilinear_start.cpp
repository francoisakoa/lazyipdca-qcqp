#include "../src/native_bilinear_start.hpp"
#include "../src/frozen_start_json.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
using namespace lazyipdca::qcqp;
static QuadraticFunction affine(int n,double c,std::initializer_list<std::pair<int,double>> a){QuadraticFunction q;q.constant=c;q.linear.assign(n,0);q.hessian.n=n;for(auto e:a)q.linear[e.first]=e.second;return q;}
int main(int argc,char** argv){try{int checks=0;auto require=[&](bool a,const char* why){++checks;if(!a)throw std::runtime_error(why);};
  auto deadline=[](){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()+30;};
  QCQPProblem p;p.n=6;p.x0={.2,.8,0,0,0,0};p.lb.assign(6,0);p.ub={1,1,10,10,10,10};p.objective=affine(6,0,{{2,-1},{3,-2}});
  p.equalities.push_back(affine(6,-1,{{0,1},{1,1}}));
  auto prod=affine(6,0,{{4,1}});prod.hessian.entries.push_back({0,2,-1});p.equalities.push_back(prod);
  prod=affine(6,0,{{5,1}});prod.hessian.entries.push_back({1,3,-1});p.equalities.push_back(prod);
  p.inequalities.push_back(affine(6,0,{{2,1},{3,-1}}));p.inequalities.push_back(affine(6,-8,{{2,1},{3,1}}));
  auto seed=detail::native_bilinear_start(p,deadline());require(seed.admitted&&seed.success,"supported simplex/product model rejected");require(seed.final_feasibility<=1e-8,"constructed point infeasible");require(seed.objective<0,"descent ray not activated");require(p.x0[0]==.2&&p.x0[1]==.8,"original frozen start modified");require(seed.x[0]==.5&&seed.x[1]==.5,"uniform simplex wrong");
  require(!detail::native_bilinear_start(p,0).success,"expired deadline accepted");
  auto bad=p;bad.objective.hessian.entries.push_back({2,3,1});require(!detail::native_bilinear_start(bad,deadline()).admitted,"nonaffine objective accepted");
  bad=p;bad.equalities.push_back(affine(6,-1,{{0,1},{1,1}}));require(!detail::native_bilinear_start(bad,deadline()).admitted,"overlapping simplex accepted");
  bad=p;bad.equalities[1].hessian.entries[0].v=1;require(!detail::native_bilinear_start(bad,deadline()).admitted,"negative product accepted");
  bad=p;bad.inequalities.push_back(affine(6,1,{{2,-1}}));require(!detail::native_bilinear_start(bad,deadline()).admitted,"unsupported positive IQ constant accepted");
  for(int a=1;a+1<argc;a+=2){auto original=read_qcq_compact(argv[a]);auto r=detail::native_bilinear_start(original,deadline());require(r.admitted&&r.success,"real model initializer failed");std::ifstream reference(argv[a+1]);std::stringstream buffer;buffer<<reference.rdbuf();std::string json=buffer.str();auto key=json.find("\"x\"");if(key==std::string::npos)throw std::runtime_error("missing reference vector");json.replace(key,3,"\"x0\"");auto expected=detail::FrozenStartJson(json,original.n).read();double difference=0;for(int j=0;j<original.n;++j)difference=std::max(difference,std::abs(expected[j]-r.x[j]));require(difference<=1e-10*std::max(1.,inf_norm(expected)),"C++ ray construction disagrees with Python reference");std::cout<<std::setprecision(17)<<"REFERENCE_PASS n="<<original.n<<" difference="<<difference<<" physical_feasibility="<<r.final_feasibility<<" objective="<<r.objective<<'\n';}
  std::cout<<"NATIVE_BILINEAR_START_PASS checks="<<checks<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
