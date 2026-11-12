#include "lazyipdca_qcqp/convexification_study.hpp"
#include "lazyipdca_qcqp/hybrid_spectrum.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;
using namespace lazyipdca::qcqp::study;

int main() {
  int checks=0;auto check=[&](bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);};
  std::vector<SymTriplet> q;for(int i=0;i<17;++i)q.push_back({i,i,i==0?-2.:1.});
  auto c=construct_hybrid(17,q);HybridAtom atom;atom.weight=2;atom.indices={0};atom.values={1};
  c.hybrid_atoms={atom};c.rho=0;hybrid_assemble_trace(c);
  check(hybrid_certificate(c,0),"exactly cancelled source was not PSD");
  const auto first_hits=c.certificate_memo->hits;
  check(hybrid_certificate(c,0)&&c.certificate_memo->hits==first_hits+1,"positive proof was recomputed");
  auto original=c;auto trial=c;
  check(!hybrid_downweight(trial,17,0,.5),"unsafe attenuation was accepted");
  const auto before=c.certificate_memo->downweight_hits;
  check(!hybrid_downweight(c,17,0,.5)&&c.certificate_memo->downweight_hits==before+1,"rejected rolled-back trial proof was lost");
  check(c.hybrid_atoms[0].weight==2&&c.rho==0&&c.hessian[0].v==original.hessian[0].v,"failed trial altered installed metric");
  c.hybrid_atoms[0].strikes=50;
  const auto strikes_hits=c.certificate_memo->downweight_hits;
  check(!hybrid_downweight(c,17,0,.5)&&c.certificate_memo->downweight_hits==strikes_hits+1,"activity metadata invalidated matrix proof");
  auto changed=c;changed.hybrid_atoms[0].weight=4;hybrid_assemble_trace(changed);
  check(hybrid_downweight(changed,17,0,.5)&&changed.hybrid_atoms[0].weight==2,"atom-weight mutation reused stale rejection");
  auto vector_changed=c;vector_changed.hybrid_atoms[0].values[0]=.5;hybrid_assemble_trace(vector_changed);
  check(!hybrid_certificate(vector_changed,0),"atom-vector mutation reused stale positive proof");
  auto source_changed=c;source_changed.original[0].v=-3;hybrid_assemble_trace(source_changed);
  check(!hybrid_certificate(source_changed,0),"original Hessian mutation reused stale proof");
  check(!hybrid_certificate(c,0,.25),"margin mutation reused stale proof");
  check(!hybrid_certificate(c,-1),"invalid scalar reused a positive proof");
  auto support_changed=c;support_changed.support.erase(support_changed.support.begin());
  check(hybrid_query_key(support_changed,0,0)!=hybrid_query_key(c,0,0),"support mutation was absent from proof key");
  auto gram=hybrid_exact_gram(c);const auto gh=c.certificate_memo->gram_hits;
  check(hybrid_exact_gram(c)==gram&&c.certificate_memo->gram_hits==gh+1,"exact Gram snapshot was not reused");
  auto changed_gram=c;changed_gram.hybrid_atoms[0].weight=3;
  check(hybrid_exact_gram(changed_gram).at({0,0})==Rational(3),"Gram snapshot survived atom mutation");
  check(hybrid_exact_gram(c).at({0,0})==Rational(2),"trial mutation poisoned parent Gram snapshot");

  // On the <=16 path all rounded arrays participate in the certificate.
  auto small=construct_hybrid(2,{{0,0,-1},{1,1,1}});small.rho=1;hybrid_assemble_trace(small);
  check(hybrid_certificate(small,1),"small baseline certificate failed");
  auto invalid_stored=small;invalid_stored.hessian[0].v=-100;
  check(!hybrid_certificate(invalid_stored,1),"rounded model mutation reused a stale small proof");
  auto invalid_correction=small;invalid_correction.correction={{0,0,-100}};
  check(!hybrid_certificate(invalid_correction,1),"rounded correction mutation reused a stale small proof");
  auto convex=construct_hybrid(17,{{0,0,1},{1,1,2}});
  check(convex.rho==0&&convex.hybrid_atoms.empty()&&hybrid_certificate(convex,0),"convex source received a correction");
  auto bad_convex=convex;bad_convex.original[0].v=-1;hybrid_assemble_trace(bad_convex);
  check(!hybrid_certificate(bad_convex,0),"convex C=0 proof survived source mutation");
  HybridCertificateMemo bounded;
  for(int i=0;i<300;++i)hybrid_remember_query(bounded,std::to_string(i),false);
  check(bounded.entries.size()<=HybridCertificateMemo::max_entries&&bounded.evictions>0,"query memo entry budget was not enforced");
  check(bounded.bytes<=HybridCertificateMemo::max_bytes,"query memo byte budget exceeded");
  const auto size=bounded.entries.size();hybrid_remember_query(bounded,std::string(HybridCertificateMemo::max_bytes+1,'x'),false);
  check(bounded.entries.size()==size,"oversized key was retained");
  auto oversized=c;oversized.original.resize(HybridCertificateMemo::max_bytes/(2*sizeof(int)+sizeof(double))+1);
  check(!hybrid_cache_key_bounded(oversized),"oversized source key was not bypassed before construction");
  bool reusable=false;check(!hybrid_certificate_uncached(c,0,.25,reusable)&&reusable,"deterministic rejected query was not reusable");
  SparsePsdBudget budget;budget.max_seconds=1;budget.max_updates=10;
  SparsePsdProof partial;partial.reason="pivot-not-strictly-positive";partial.seconds=1;
  check(!hybrid_interval_answer_reusable(partial,budget),"budget-limited first pivot became a reusable refusal");
  partial.seconds=.1;partial.updates=10;
  check(!hybrid_interval_answer_reusable(partial,budget),"update-limited first pivot became a reusable refusal");
  partial.updates=0;partial.reason="entry-budget";
  check(!hybrid_interval_answer_reusable(partial,budget),"entry-budget failure became a reusable refusal");
  partial.reason="pivot-not-strictly-positive";
  check(hybrid_interval_answer_reusable(partial,budget),"complete deterministic pivot rejection was not reusable");
  // A later exact positive proof is stronger information than an earlier
  // deterministic inconclusive answer. Its installation clears refusals.
  hybrid_remember_query(*c.certificate_memo,"old-inconclusive-refusal",false);
  const auto invalidated=c.certificate_memo->refusals_invalidated;
  hybrid_store_interval_certificate(c,0,0,128,true);
  check(c.certificate_memo->refusals_invalidated>invalidated,"new positive proof retained obsolete refusals");
  hybrid_remember_query(*c.certificate_memo,hybrid_query_key(c,0,0),false);
  check(hybrid_certificate(c,0),"obsolete refusal overrode current positive interval proof");
  auto large=construct_hybrid(129,{{0,0,-1},{128,128,1}});large.support.resize(129);std::iota(large.support.begin(),large.support.end(),0);
  large.certificate_memo.reset();bool large_reusable=false;
  check(hybrid_certificate(large,large.rho,0,&large_reusable)&&!large.certificate_memo,"large-support query constructed a new memo key");
  for(int n:{2,4,17})for(int variant=0;variant<8;++variant) {
    std::vector<SymTriplet> matrix;for(int i=0;i<n;++i)matrix.push_back({i,i,i==0?-2.:1.});
    auto cached=construct_hybrid(n,matrix);HybridAtom a;a.indices={0};a.values={.5};a.weight=variant*.5;
    cached.hybrid_atoms={a};hybrid_assemble_trace(cached);
    const double rho=variant*.25,margin=(variant%2)*.125;
    auto fresh=cached;fresh.implicit_interval_certificate.reset();fresh.exact_gram_snapshot.reset();
    bool complete=false;const bool expected=hybrid_certificate_uncached(fresh,rho,margin,complete);
    check(hybrid_certificate(cached,rho,margin)==expected,"memoized and fresh certifier disagree");
    check(hybrid_certificate(cached,rho,margin)==expected,"replayed certifier changed its answer");
  }
  auto spectrum_equal=[](const HybridSpectrum& a,const HybridSpectrum& b){return a.lambda_min==b.lambda_min&&a.vectors==b.vectors&&a.rayleigh==b.rayleigh&&a.raw_candidates==b.raw_candidates&&a.orthogonalized_candidates==b.orthogonalized_candidates&&a.dependent_dropped==b.dependent_dropped&&a.nonnegative_projected_dropped==b.nonnegative_projected_dropped;};
  for(int n:{17,64}) {
    std::vector<SymTriplet> matrix;for(int i=0;i<n;++i)matrix.push_back({i,i,i==0?-2.:1.});
    auto metric=construct_hybrid(n,matrix);
    const auto fresh=hybrid_spectrum_uncached(metric,8,24,2,1e-8);
    check(spectrum_equal(hybrid_spectrum(metric,8,24,2,1e-8),fresh),"cached spectrum differs from cold bounded search");
    const auto hits=metric.certificate_memo->spectrum_hits;auto copy=metric;copy.rho*=.5;
    check(spectrum_equal(hybrid_spectrum(copy,8,24,2,1e-8),fresh)&&metric.certificate_memo->spectrum_hits==hits+1,"rho-only change reran Q+atoms spectrum");
    const auto misses=metric.certificate_memo->spectrum_misses;
    hybrid_spectrum(metric,3,24,2,1e-8);
    check(metric.certificate_memo->spectrum_misses==misses+1,"spectral search setting mutation reused stale pool");
    metric.original[0].v=2;
    const auto empty=hybrid_spectrum(metric,8,24,2,1e-8);
    check(empty.vectors.empty()&&spectrum_equal(empty,hybrid_spectrum_uncached(metric,8,24,2,1e-8)),"source mutation did not refresh negative pool");
    const auto empty_hits=metric.certificate_memo->spectrum_hits;
    check(hybrid_spectrum(metric,8,24,2,1e-8).vectors.empty()&&metric.certificate_memo->spectrum_hits==empty_hits+1,"empty deterministic negative pool was not cached");
    metric.hybrid_atoms.push_back({1,{0},{1},0});
    check(spectrum_equal(hybrid_spectrum(metric,8,24,2,1e-8),hybrid_spectrum_uncached(metric,8,24,2,1e-8)),"atom mutation reused stale spectral pool");
  }
  std::cout<<"hybrid_query_cache PASS checks="<<checks<<" content_invalidation=1 rollback=1 exact_Gram=1 bounded=1 convex_C0=1\n";
}
