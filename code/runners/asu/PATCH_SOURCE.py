#!/usr/bin/env python3
"""Apply one generic research switch in a NEW source tree only."""
from pathlib import Path
import argparse,difflib,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
HEADER=r'''#pragma once
#include <stdexcept>
#include <string>
namespace lazyipdca::qcqp::research {
inline bool quadratic_equality_implied_bounds_enabled(
    bool master,const std::string& engine,const std::string& mode,
    const std::string& policy) {
  if(policy!="profile"&&policy!="keep"&&policy!="remove")
    throw std::invalid_argument("research-equality-implied-bounds expects profile|keep|remove");
  const bool historical=mode=="HH"||mode=="H0"||mode=="HS"||mode=="SH";
  return master&&engine=="majorants"&&
      (policy=="remove"||(policy=="profile"&&historical));
}
}
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_map(p):return {str(x.relative_to(p)):sha(x)for x in sorted(p.rglob('*'))if x.is_file()}
def patch(out):
    ref=HERE/'native/reference_source';out=Path(out).resolve()
    if out.exists():raise ValueError('Dossier de sources neuf requis : '+str(out))
    if out==ref or ref in out.parents or out in ref.parents:raise ValueError('Sortie séparée requise')
    expected={k.split('/source/',1)[-1]:v for k,v in json.loads((HERE/'BASE_CONTEXT.json').read_text())['plan_header']['source_sha256'].items()}
    if len(expected)!=823 or source_map(ref)!=expected:raise ValueError('Les 823 sources de référence ont changé')
    shutil.copytree(ref,out)
    def change(file,old,new):
        p=out/file;s=p.read_text()
        if s.count(old)!=1:raise ValueError('Ancre de patch ambiguë : '+file)
        p.write_text(s.replace(old,new))
    change('include/lazyipdca_qcqp/settings.hpp',
      '  bool dca_rank_implied_bounds=true;',
      '  std::string research_equality_implied_bounds="profile"; // independent quadratic-equality bound rule only\n  bool dca_rank_implied_bounds=true;')
    change('src/main.cpp',
      '      else if(a=="--dca-rank-presolve-report")st.dca_rank_presolve_report_file=need(a.c_str());',
      '''      else if(a=="--research-equality-implied-bounds") {
        st.research_equality_implied_bounds=need(a.c_str());
        const auto& v=st.research_equality_implied_bounds;
        if(v!="profile"&&v!="keep"&&v!="remove")throw std::invalid_argument(a+" expects profile|keep|remove");
      }
      else if(a=="--dca-rank-presolve-report")st.dca_rank_presolve_report_file=need(a.c_str());''')
    change('src/main.cpp',
      '    << "  --dca-rank-presolve-report FILE  JSON of verified reductions\\n"',
      '    << "  --research-equality-implied-bounds profile|keep|remove  research override for quadratic-equality implied box sides\\n"\n    << "  --dca-rank-presolve-report FILE  JSON of verified reductions\\n"')
    change('src/rank_presolve.cpp',
      '#include "lazyipdca_qcqp/rank_presolve.hpp"',
      '#include "lazyipdca_qcqp/rank_presolve.hpp"\n#include "research_equality_implied_bounds.hpp"')
    change('src/rank_presolve.cpp',
      '''  const bool equality_bounds=st.dca_rank_implied_bounds&&st.dca_engine=="majorants"&&
      (st.convexification_study=="HH"||st.convexification_study=="H0"||
       st.convexification_study=="HS"||st.convexification_study=="SH");''',
      '''  const bool equality_bounds=research::quadratic_equality_implied_bounds_enabled(
      st.dca_rank_implied_bounds,st.dca_engine,st.convexification_study,
      st.research_equality_implied_bounds);''')
    change('src/rank_presolve.cpp',
      "    f<<reduction.report_json()<<'\\n';",
      '''    const std::string report=reduction.report_json();
    if(report.empty()||report.back()!='}')throw std::runtime_error("invalid native presolve report");
    f<<report.substr(0,report.size()-1)
      <<",\\\"research_equality_implied_bounds\\\":\\\""<<st.research_equality_implied_bounds<<"\\\""
      <<",\\\"quadratic_equality_rule_enabled\\\":"<<(equality_bounds?"true":"false")
      <<",\\\"affine_equality_rule_enabled\\\":"<<(affine_equality_bounds?"true":"false")
      <<",\\\"radial_inequality_master_enabled\\\":"<<(st.dca_rank_implied_bounds?"true":"false")<<"}\\n";''')
    (out/'src/research_equality_implied_bounds.hpp').write_text(HEADER)
    differences=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(out);q=ref/rel
        if not q.exists()or p.read_bytes()!=q.read_bytes():differences.append(str(rel))
    assert set(differences)=={'include/lazyipdca_qcqp/settings.hpp','src/main.cpp','src/rank_presolve.cpp','src/research_equality_implied_bounds.hpp'}
    evidence=dict(schema='quadratic-equality-bound-policy-patch-v1',changed_files=differences,
      source_reference_sha256=source_map(ref),source_research_sha256=source_map(out),
      only_numerical_intervention='gate of the existing proof-based quadratic-equality implied-bound removal',
      untouched=['arithmetic proof','primal frozen start','original model','affine-equality bound rule','radial-inequality bound rule','globalization','curvature policies','restoration','linear solver'],
      default_profile_preserves_historical_gate=True,research_variant=True,optimizer_invocations=0)
    (out.parent/'PATCH_RECEIPT.json').write_text(json.dumps(evidence,indent=2)+'\n')
    chunks=[]
    for rel in differences:
        a=(ref/rel).read_text().splitlines(True)if(ref/rel).exists()else[];b=(out/rel).read_text().splitlines(True)
        chunks.extend(difflib.unified_diff(a,b,fromfile='reference/'+rel,tofile='research/'+rel))
    (out.parent/'BOUND_REMOVAL.patch').write_text(''.join(chunks))
    return evidence
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    print(json.dumps(patch(a.out),indent=2))
