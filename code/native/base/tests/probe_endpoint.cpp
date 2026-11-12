#include "../src/atomic_newton.cpp"
#include "lazyipdca_qcqp/io.hpp"
using namespace lazyipdca::qcqp;
Vec read_vector(std::istream& f){std::uint64_t n=0;f.read((char*)&n,8);if(n>100000000)throw std::runtime_error("bad vector size");Vec v(n);f.read((char*)v.data(),n*8);if(!f||!finite(v))throw std::runtime_error("bad vector data");return v;}
int main(int argc,char**argv){try{
 if(argc<4)throw std::runtime_error("probe_endpoint input.qcq vectors.bin output.json [mumps|cholesk] [budget] [tangent:0|1] [slack_floor] [steps]");
 auto p=read_qcq(argv[1]);std::ifstream f(argv[2],std::ios::binary);auto x=read_vector(f),eq=read_vector(f),iq=read_vector(f),lo=read_vector(f),up=read_vector(f);
 if(x.size()!=size_t(p.n)||eq.size()!=p.equalities.size()||iq.size()!=p.inequalities.size()||lo.size()!=x.size()||up.size()!=x.size())throw std::runtime_error("shape mismatch");
 Settings s;s.termination_test="ipopt";s.outer_algorithm="lazy-dca";s.dca_engine="filter";
 s.linear_backend=argc>4?argv[4]:"mumps";s.dca_filter_curvature_backend="cholesk";
 s.dca_aggregate=true;s.dca_aggregate_backend="mumps";s.dca_aggregate_adaptive_pivot=true;s.dca_aggregate_adaptive_dual=true;
 s.lazy_newton_atomic_regularization=false;s.lazy_newton_lazy_constraints=false;s.kkt_backward_error_reject=1e-7;
 s.stationarity_tolerance=s.feasibility_tolerance=s.complementarity_tolerance=s.dual_infeasibility_tolerance=1e-6;
 s.dca_krylov_max_iterations=argc>5?std::stoi(argv[5]):24;s.dca_filter_tangent_curvature=argc>6?std::stoi(argv[6]):false;
 s.dca_krylov_predictor_corrector=argc>11?std::stoi(argv[11]):true;s.live_display_level=4;s.time_limit_seconds=240;s.dca_filter_packed_atoms=true;s.dca_filter_active_evaluation=true;s.dca_filter_block_ritz=true;
 const double floor=argc>7?std::stod(argv[7]):1e-14;const int steps=argc>8?std::stoi(argv[8]):1;
 Result r;auto start=Clock::now();detail::NewtonCostLedger costs(r.lazy_newton,start);Workspace w(p,s,r.lazy_newton,costs);
 for(int j=0;j<w.n;++j)w.x[j]=x[w.free[j]];
 for(int e=0;e<w.me;++e)w.lambda[e]=eq[e]*w.objective.scale/w.equalities[e].scale;
 for(int i=0;i<w.mi;++i){const auto& q=w.inequalities[i];int origin=w.inequality_origin[i];double raw=w.inequality_kind[i]==0?iq[origin]:w.inequality_kind[i]<0?lo[origin]:up[origin];
  w.z[i]=raw*w.objective.scale/q.scale;w.active[i]=raw>0;w.s[i]=std::max(floor,-q.original_at(w.x));}
 w.evaluate();w.update_geometry();DcaModels models(w,r.dca_filter);DcaAggregate aggregate(s);detail::OriginalDcaFilter filter(1,1e-6);auto outer=dca_original(w,w.x,w.lambda,w.z);
 std::cout<<std::scientific<<std::setprecision(12)<<"[ENDPOINT] scope=reconstructed-slacks floor="<<floor<<" raw_dual="<<outer.dual<<" normalized="<<outer.normalized_kkt<<" sf="<<w.objective.scale<<" primal="<<outer.primal<<" terminal="<<detail::dca_stationary_phase(outer.primal,outer.normalized_kkt,1e-6,1e-6)<<" n="<<w.n<<" nr="<<w.nr<<" mr="<<w.mr<<" mi="<<w.mi<<"\n"<<std::flush;
 for(int it=0;it<steps;++it){
  bool ok=false;
  const Vec rhos=argc>9?Vec{0.,std::stod(argv[9])}:Vec{0.,0.02,0.1,1.,10.,40.};
  for(double rho:rhos){
   DcaAggregate::Candidate candidate;bool valid=aggregate.attempt_one(w,models,outer,filter,it+1,rho,&candidate,true,false);
   std::cout<<"[PROBE] rho="<<rho<<" certified="<<aggregate.last_attempt_certified<<" accepted="<<valid<<" reason="<<aggregate.last_rejection<<" candidate="<<aggregate.last_candidate_kkt<<" residual="<<aggregate.last_backward_error<<"\n"<<std::flush;
   if(valid){w.x=std::move(candidate.x);w.lambda=std::move(candidate.lambda);w.z=std::move(candidate.z);w.s=std::move(candidate.s);outer=candidate.point;ok=true;break;}
  }
  if(!ok){
   DcaKrylov ak;Direction as;as.x.assign(w.n,0);as.lambda.assign(w.me,0);as.s.assign(w.mi,0);as.z.assign(w.mi,0);double aa=0;
   ok=ak.attempt(w,r.dca_filter,outer,filter,as,as,1e-12,it,start,aa,outer,true,&aggregate);
   std::cout<<"[AGGREGATE-ROOT] ok="<<ok<<" reason="<<r.dca_filter.krylov_last_rejection<<" residual="<<ak.last_residual<<" kkt="<<(ok?ak.proposal.point.kkt:outer.kkt)<<" time="<<seconds(start)<<"\n"<<std::flush;
   if(ok){w.x=std::move(ak.proposal.x);w.lambda=std::move(ak.proposal.lambda);w.z=std::move(ak.proposal.z);w.s=std::move(ak.proposal.s);outer=ak.proposal.point;dca_disable_models(w);r.outer_iterations=it+1;continue;}
   if(argc>10 && std::stoi(argv[10]))break;
   std::cout<<"[MODEL] begin time="<<seconds(start)<<"\n"<<std::flush;models.build();w.evaluate();w.update_matrix();
   if(!w.factorize(0))throw std::runtime_error("majorant factor "+w.message);
   std::cout<<"[MODEL] factored time="<<seconds(start)<<" rho="<<models.lagrangian.rho<<"\n"<<std::flush;
   DcaKrylov k;Direction seed;seed.x.assign(w.n,0);seed.lambda.assign(w.me,0);seed.s.assign(w.mi,0);seed.z.assign(w.mi,0);double alpha=0;
   ok=k.attempt(w,r.dca_filter,outer,filter,seed,seed,1e-12,it,start,alpha,outer,true);
   std::cout<<"[ROOT] ok="<<ok<<" reason="<<r.dca_filter.krylov_last_rejection<<" residual="<<k.last_residual<<" kkt="<<(ok?k.proposal.point.kkt:outer.kkt)<<" time="<<seconds(start)<<"\n"<<std::flush;
   if(ok){w.x=std::move(k.proposal.x);w.lambda=std::move(k.proposal.lambda);w.z=std::move(k.proposal.z);w.s=std::move(k.proposal.s);outer=k.proposal.point;dca_disable_models(w);}
  }
  if(!ok)break;
  r.outer_iterations=it+1;
 }
 w.export_state(r);w.certify(r);r.status=Status::MaxOuterIterations;r.message="Endpoint diagnostic with reconstructed slacks; not a full solver run";r.termination_test="ipopt";
 Settings stop=s;stop.stationarity_tolerance=1e-8;stop.dual_infeasibility_tolerance=1;stop.feasibility_tolerance=stop.complementarity_tolerance=1e-4;
 detail::audit_original_termination(p,stop,r);write_result_json(r,p.name,argv[3]);
 std::cout<<"[DONE] raw_dual="<<r.dual_infeasibility_inf<<" E="<<r.scaled_nlp_error<<" seconds="<<seconds(start)<<"\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
