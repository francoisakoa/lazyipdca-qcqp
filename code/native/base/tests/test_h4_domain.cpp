#include "lazyipdca_qcqp/h4_domain.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
bool same_bits(double left,double right) {
  return std::memcmp(&left,&right,sizeof(double))==0;
}
bool same_bits(const Vec& left,const Vec& right) {
  if(left.size()!=right.size())return false;
  for(std::size_t j=0;j<left.size();++j)if(!same_bits(left[j],right[j]))return false;
  return true;
}
void write_model(const std::filesystem::path& path,const QCQPProblem& p,int format) {
  std::ofstream out(path);out<<std::setprecision(17);
  auto vector=[&](const char* name,const Vec& values){out<<name;for(double value:values)out<<' '<<value;out<<'\n';};
  auto quadratic=[&](const QuadraticFunction& q){
    out<<"quadratic\nname f\nconstant "<<q.constant<<'\n';
    Vec linear(p.n,0);q.for_each_linear_nonzero([&](int j,double value){linear[j]=value;});
    if(format==1)vector("linear",linear);
    else {
      std::size_t count=0;for(double value:linear)count+=value!=0;
      out<<"linear_nnz "<<count<<'\n';
      for(int j=0;j<p.n;++j)if(linear[j]!=0)out<<j<<' '<<linear[j]<<'\n';
    }
    out<<"hessian_nnz "<<q.hessian.entries.size()<<'\n';
    for(auto e:q.hessian.entries)out<<e.i<<' '<<e.j<<' '<<e.v<<'\n';
    out<<"end_quadratic\n";
  };
  out<<"LAZYIPDCA_QCQP_V"<<format<<"\nname domain_contract\nn "<<p.n<<'\n';
  vector("x0",p.x0);vector("lb",p.lb);vector("ub",p.ub);
  out<<"objective\n";quadratic(p.objective);
  out<<"equalities "<<p.equalities.size()<<'\n';for(const auto& q:p.equalities)quadratic(q);
  out<<"inequalities "<<p.inequalities.size()<<'\n';for(const auto& q:p.inequalities)quadratic(q);
  out<<"end_problem\n";
}
std::string bytes(const std::filesystem::path& path) {
  std::ifstream input(path,std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),{});
}
QCQPProblem scalar_model(double lower,double upper) {
  QCQPProblem p;p.n=1;p.x0={-0.0};p.lb={lower};p.ub={upper};
  p.objective.compact_linear_dimension=1;p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,2}};
  return p;
}
QuadraticFunction balance(int u,int v,int a,int b,double r,double t) {
  QuadraticFunction q;q.compact_linear_dimension=4;q.hessian.n=4;
  auto add=[&](int i,int j,double value){if(i>j)std::swap(i,j);q.hessian.entries.push_back({i,j,value});};
  add(u,u,2*r);add(v,v,2*r);add(u,a,-r);add(v,b,-r);add(v,a,t);add(u,b,-t);
  std::sort(q.hessian.entries.begin(),q.hessian.entries.end(),[](auto x,auto y){return std::pair{x.i,x.j}<std::pair{y.i,y.j};});
  return q;
}
}
int main() {
  int checks=0;
  auto require=[&](bool b,const char* m){++checks;if(!b)throw std::runtime_error(m);};
  auto good=[](const QCQPProblem& p){try{study::validate_h4_domain(p);return true;}catch(const std::invalid_argument&){return false;}};
  const double inf=std::numeric_limits<double>::infinity();
  // Exercise the actual legacy reader/presolver contract before testing the
  // guard. Both readers canonicalize absent sides before any rank presolve;
  // finite sides, including huge values and signed zero, remain bit-exact.
  const auto work=std::filesystem::temp_directory_path()/
      ("h4_domain_"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(work);
  struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove_all(path,e);}} cleanup{work};
  for(int format:{1,2})for(const auto& sides:std::vector<std::pair<std::string,std::string>>{
      {"-inf","inf"},{"inf","inf"},{"-inf","-inf"},{"inf","-inf"},
      {"inf","47.5"},{"-2","-inf"},{"-1e300","1e300"},
      {"-2","inf"},{"-inf","3"},{"2","2"},{"-0","0"}}) {
    const auto path=work/("domain_v"+std::to_string(format)+".qcq");
    const double lower=std::stod(sides.first),upper=std::stod(sides.second);
    write_model(path,scalar_model(lower,upper),format);
    const auto original_bytes=bytes(path);
    for(bool compact:{false,true}) {
      const auto parsed=compact?read_qcq_compact(path.string()):read_qcq(path.string());
      require(same_bits(parsed.lb[0],std::isinf(lower)?-inf:lower)&&
          same_bits(parsed.ub[0],std::isinf(upper)?inf:upper),
          "reader failed bound sentinel canonicalization or changed a finite side");
      require(same_bits(parsed.x0[0],-0.0),"reader changed the original start bits");
      require(bytes(path)==original_bytes,"reader rewrote the frozen input file");
      require(good(parsed),"H4 rejected a domain accepted by the real QCQP reader");
      ConstraintPresolve presolve(parsed,true);
      const auto& reduced=presolve.reduced();
      require(good(reduced),"H4 rejected the real rank-presolve domain");
      require(reduced.lb[0]==parsed.lb[0]&&reduced.ub[0]==parsed.ub[0]&&
          reduced.x0[0]==parsed.x0[0]&&reduced.inequalities.empty(),
          "domain path introduced a surrogate box or changed the start");
      if(!std::isfinite(parsed.lb[0])&&!std::isfinite(parsed.ub[0]))
        require(feasibility(parsed,Vec{1e150}).bound_inf==0,
            "legacy infinite sentinels became finite/fixed bounds");
    }
  }
  // Reject malformed actual bound sides; canonicalization must not hide NaNs
  // or an empty finite interval.
  for(int format:{1,2})for(const auto sides:std::vector<std::pair<double,double>>{
      {NAN,inf},{-inf,NAN},{2,1}}) {
    const auto path=work/("invalid_v"+std::to_string(format)+".qcq");
    write_model(path,scalar_model(sides.first,sides.second),format);
    for(bool compact:{false,true}) {
      bool rejected=false;
      try{(void)(compact?read_qcq_compact(path.string()):read_qcq(path.string()));}
      catch(const std::exception&){rejected=true;}
      require(rejected,"reader accepted NaN bounds or reversed finite bounds");
    }
  }
  // Noncanonical absent sides must never manufacture the nonzero proof used
  // by leaf contraction or dominate a genuine finite side during bound merge.
  for(int format:{1,2})for(bool finite_nonzero:{false,true}) {
    QCQPProblem canonical;canonical.n=4;canonical.x0={1,0,1,0};
    canonical.lb=Vec(4,-inf);canonical.ub=Vec(4,inf);
    canonical.objective.compact_linear_dimension=4;canonical.objective.hessian.n=4;
    canonical.equalities={balance(2,3,0,1,2,3),balance(2,3,0,1,3,-2)};
    if(finite_nonzero){canonical.lb[2]=.5;canonical.ub[0]=3;canonical.ub[2]=2;}
    auto historical=canonical;
    for(double& lower:historical.lb)if(std::isinf(lower))lower=inf;
    for(double& upper:historical.ub)if(std::isinf(upper))upper=-inf;
    const auto canonical_path=work/("canonical_v"+std::to_string(format)+".qcq");
    const auto historical_path=work/("historical_v"+std::to_string(format)+".qcq");
    write_model(canonical_path,canonical,format);write_model(historical_path,historical,format);
    for(bool compact:{false,true}) {
      const auto clean=compact?read_qcq_compact(canonical_path.string()):read_qcq(canonical_path.string());
      const auto old=compact?read_qcq_compact(historical_path.string()):read_qcq(historical_path.string());
      ConstraintPresolve clean_reduction(clean),old_reduction(old);
      const auto& a=clean_reduction.reduced();const auto& b=old_reduction.reduced();
      require(a.n==b.n&&same_bits(a.lb,b.lb)&&same_bits(a.ub,b.ub)&&same_bits(a.x0,b.x0)&&
          clean_reduction.variable_map()==old_reduction.variable_map()&&
          clean_reduction.statistics().leaf_pairs==old_reduction.statistics().leaf_pairs&&
          clean_reduction.statistics().nonzero_rejected==old_reduction.statistics().nonzero_rejected,
          "legacy bound signs changed rank-presolve semantics");
      const auto clean_reduced_path=work/"clean_reduced.qcq";
      const auto old_reduced_path=work/"old_reduced.qcq";
      write_model(clean_reduced_path,a,format);write_model(old_reduced_path,b,format);
      require(bytes(clean_reduced_path)==bytes(old_reduced_path),
          "canonical and legacy readers produced different reduced polynomials");
      if(finite_nonzero)
        require(b.n==2&&b.lb[0]==.5&&b.ub[0]==2,
            "absent bound dominated the strongest finite merged side");
      else
        require(b.n==4&&old_reduction.statistics().leaf_pairs==0,
            "absent bound manufactured a nonzero proof for leaf contraction");
    }
  }
  QCQPProblem p;p.n=4;p.x0={0,0,0,2};p.lb={-inf,-3,-inf,2};p.ub={inf,inf,4,2};
  require(good(p),"free, one-sided and finite fixed domains rejected");
  require(std::isinf(p.lb[0])&&std::isinf(p.ub[0])&&p.x0[0]==0,"domain validator changed a free variable");
  auto huge=p;huge.lb[0]=-1e300;huge.ub[0]=1e300;require(good(huge),"finite large bounds treated as infinity");
  auto outside=p;outside.x0[1]=-4;require(good(outside),"finite infeasible start must remain eligible for restoration");
  for(int side:{0,1}) {
    auto q=p;(side==0?q.lb[0]:q.ub[0])=std::numeric_limits<double>::quiet_NaN();
    require(!good(q),"NaN bound side accepted");
  }
  auto crossed_inf=p;crossed_inf.lb[0]=inf;crossed_inf.ub[0]=-inf;
  require(!good(crossed_inf),"crossed infinite bounds accepted");
  auto crossed_finite=p;crossed_finite.lb[0]=inf;crossed_finite.ub[0]=1;
  require(!good(crossed_finite),"crossed infinite/finite bounds accepted");
  for(double bad:{inf,-inf,std::numeric_limits<double>::quiet_NaN()}) {
    auto q=p;q.x0[0]=bad;require(!good(q),"nonfinite start accepted");
  }
  auto inconsistent=p;inconsistent.lb[2]=5;require(!good(inconsistent),"reversed bounds accepted");
  auto dim=p;dim.ub.pop_back();require(!good(dim),"truncated bound vector accepted");
  std::cout<<"h4_domain PASS checks="<<checks<<" real_V1_V2_readers=1 real_rank_presolve=1 no_artificial_bound=1 one_sided=1 free=1 fixed=1 legacy_infinite_sentinels=1 canonical_absent_sides=1 frozen_bytes_preserved=1 finite_bounds_bit_exact=1\n";
}
