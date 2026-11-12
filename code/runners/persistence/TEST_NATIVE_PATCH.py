#!/usr/bin/env python3
"""Compile identity/scope checks and validate exact patch anchors; no QCQP solve."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess
import PATCH_PERSISTENCE as patcher

CPP=r'''#include "research_curvature_support.hpp"
#include <cassert>
#include <iostream>
using lazyipdca::qcqp::research::Sha256;
int main(){
  for(const std::string& s:{std::string(),std::string("abc"),std::string(100000,'a')}){Sha256 h;h.raw(s);std::cout<<h.finish()<<"\n";}
  Sha256 h;h.integer(3);h.real(0.0);h.real(-0.0);h.real(1.5);std::cout<<h.finish()<<"\n";
  int accepted=0,rejected=0;
  for(const std::string mode:{"reuse","rebuild","bad"})
  for(bool active:{false,true})for(const std::string engine:{"majorants","auto","filter"})
  for(const std::string study:{"off","SS","HH"})for(bool objective:{false,true}){
    const bool good=mode!="bad"&&(!active||((engine=="majorants"||engine=="auto")&&study=="off"&&!objective));
    try{lazyipdca::qcqp::research::validate_cache_mode(mode,active,engine,study,objective);assert(good);++accepted;}
    catch(const std::invalid_argument&){assert(!good);++rejected;}
  }
  std::cout<<accepted<<" "<<rejected<<"\n";
}'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('New validation directory required')
    a.out.mkdir(parents=True);source=a.out/'research_source';receipt=patcher.patch(source)
    cpp=a.out/'test_identity.cpp';cpp.write_text(CPP);binary=a.out/'test_identity'
    command=['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(source/'src'),str(cpp),'-o',str(binary)]
    r=subprocess.run(command,capture_output=True,text=True);(a.out/'compile.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError(r.stderr)
    lines=subprocess.check_output([str(binary)],text=True).splitlines()
    expected=[hashlib.sha256(x).hexdigest()for x in [b'',b'abc',b'a'*100000]]
    expected.append(hashlib.sha256(struct.pack('<Qddd',3,0.,-0.,1.5)).hexdigest())
    assert lines[:4]==expected
    accepted,rejected=map(int,lines[4].split());assert accepted+rejected==108
    # Policy/scope and double initialization have native integration tests
    # separately; this suite establishes byte identity and generic refusal.
    result=dict(schema='persistence-native-patch-local-unit-v1',passed=True,
      sha256_standard_vectors=3,binary64_signed_zero_identity_checked=True,
      generic_scope_checks=accepted+rejected,scope_accepted=accepted,scope_rejected=rejected,
      changed_source_files=receipt['changed_files'],baseline_source_unchanged=True,
      optimizer_invocations=0,scope='compiled support/hash/scope checks; separate native integration receipt records actual QCQP tests')
    (a.out/'VALIDATION_NATIVE_PATCH.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
