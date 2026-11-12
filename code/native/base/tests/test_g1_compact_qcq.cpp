#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
using namespace lazyipdca::qcqp;
void check(bool v,const char* message){if(!v)throw std::runtime_error(message);}
template<class F>void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad,"invalid compact input accepted");}
int main(){
 auto path=std::filesystem::temp_directory_path()/("compact_qcq_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".qcq");
 try{
  std::ostringstream text;text<<"LAZYIPDCA_QCQP_V1 name fixture n +40 x0 ";for(int i=0;i<40;++i)text<<"0.25 ";text<<"lb ";for(int i=0;i<40;++i)text<<"-INF ";text<<"ub ";for(int i=0;i<40;++i)text<<"INF ";
  text<<"objective quadratic name objective constant 0.125 linear ";
  for(int i=0;i<40;++i)text<<(i==2?"-0 ":i==19?"1e-200 ":i==38?"-2.5\t":"0 ");
  text<<"hessian_nnz 3 0 0 2 38 19 -0.75 1 1 1 end_quadratic equalities 0 inequalities 0 end_problem\n";
  const auto source=text.str();auto write=[&](const std::string& v){std::ofstream out(path);out<<v;};write(source);
  auto dense=read_qcq(path.string()),compact=read_qcq_compact(path.string());
  check(compact.objective.linear.empty()&&compact.objective.compact_linear.size()==2,"input not compact or coefficient lost");
  check(compact.objective.compact_linear[0].second==1e-200,"tiny nonzero pruned");
  check(dense.x0==compact.x0&&dense.lb==compact.lb&&dense.ub==compact.ub,"vectors changed");
  for(int j=0;j<4;++j){Vec x(40,0.25*j);check(dense.objective.value(x)==compact.objective.value(x),"value differs");check(dense.objective.gradient(x)==compact.objective.gradient(x),"gradient differs");}
  auto copy=compact.objective;copy.materialize_linear();check(copy.linear==dense.objective.linear&&!copy.has_compact_linear(),"materialization differs");
  copy=compact.objective;copy.linear.assign(40,0.0);check(!copy.has_compact_linear(),"stale sparse values override dense replacement");
  rejects([&]{compact.objective.value(Vec(39));});
  copy=compact.objective;copy.compact_linear.push_back({19,1});rejects([&]{copy.validate(40);});
  copy=compact.objective;copy.compact_linear={{40,1}};rejects([&]{copy.validate(40);});
  copy=compact.objective;copy.compact_linear={{0,INFINITY}};rejects([&]{copy.validate(40);});
  std::ostringstream sparse;sparse<<"LAZYIPDCA_QCQP_V2 name fixture_v2 n 40 x0 ";
  for(int i=0;i<40;++i)sparse<<"0.25 ";sparse<<"lb ";
  for(int i=0;i<40;++i)sparse<<"-INF ";sparse<<"ub ";
  for(int i=0;i<40;++i)sparse<<"INF ";
  sparse<<"objective quadratic name objective constant 0.125 linear_nnz 2 19 1e-200 38 -2.5 "
        <<"hessian_nnz 3 0 0 2 38 19 -0.75 1 1 1 end_quadratic "
        <<"equalities 0 inequalities 0 end_problem\n";
  write(sparse.str());auto shared_v2=read_qcq(path.string());
  check(shared_v2.objective.value(Vec(40,0.5))==compact.objective.value(Vec(40,0.5)),
        "shared QCQ reader changed V2 objective");
  check(shared_v2.objective.has_compact_linear(),"shared QCQ reader did not accept sparse V2");
  write(source);
  for(const auto& change:std::vector<std::pair<std::string,std::string>>{{"1e-200","NAN"},{"hessian_nnz 3","hessian_nnz -3"},{"38 19 -0.75","40 19 -0.75"},{"n +40","n 0"},{"equalities 0","equalities -1"},{"inequalities 0","inequalities -1"},{"1e-200","1e-200junk"}}){auto s=source;s.replace(s.find(change.first),change.first.size(),change.second);write(s);rejects([&]{read_qcq_compact(path.string());});}
  write(source.substr(0,source.size()/2));rejects([&]{read_qcq_compact(path.string());});write("");rejects([&]{read_qcq_compact(path.string());});
  QCQPProblem p;p.n=1;p.x0={0};p.lb={-5};p.ub={5};p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,2}};p.objective.compact_linear_dimension=1;p.objective.compact_linear={{0,-2}};
  Settings st;st.verbose=false;st.linear_backend="native";st.outer_algorithm="lazy-newton";st.ln_engine="atomic";
  auto r=Solver(st).solve(p);check(r.status==Status::Solved&&std::abs(r.x[0]-1)<1e-5,"compact Newton solve failed");
  st.outer_algorithm="lazy-dca";auto legacy=Solver(st).solve(p);check(legacy.status==Status::Solved,"compact legacy fallback failed");
  std::filesystem::remove(path);std::cout<<"PASS exact compact coefficients, derivatives, malformed input, ownership and legacy fallback\n";return 0;
 }catch(const std::exception& e){std::filesystem::remove(path);std::cerr<<e.what()<<"\n";return 1;}
}
