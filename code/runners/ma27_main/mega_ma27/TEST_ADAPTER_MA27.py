#!/usr/bin/env python3
"""Test the source adaptation and actual pure workspace/accessor code.
No HSL library is fabricated. No MA27 numerical factorization is claimed.
"""
from pathlib import Path
import hashlib,json,subprocess,tempfile,unittest
import ADAPTER_MA27 as a
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE=ROOT/'factorial/qualified_source'
if not BASE.is_dir():BASE=ROOT/'article_revision_R4/base/LazyQCQP_article_20261108_R3_sources/reproduction/qualified_native_source'
RESEARCH=ROOT/'factorial/source'
if not RESEARCH.is_dir():RESEARCH=ROOT/'article_revision_R4/factorial/source'

class AdapterTests(unittest.TestCase):
 def test_known_qualified_and_research_copies_are_exactly_scoped(self):
  for source in (BASE,RESEARCH):
   original=a.source_map(source)
   with tempfile.TemporaryDirectory()as d:
    out=Path(d)/'ma27';r=a.prepare(source,out)
    self.assertEqual(a.source_map(source),original)
    self.assertEqual(set(original),set(r['adapted_source_sha256']))
    changed={p for p in original if original[p]!=r['adapted_source_sha256'][p]}
    self.assertEqual(changed,set(a.PATCH_PATHS))
    self.assertFalse(r['production_authorized']);self.assertFalse(r['local_MA27_qualified'])
    self.assertFalse(r['hidden_mumps_substitution'])
    self.assertEqual(r['primary_backend'],'ma27')
 def test_unknown_source_refused_before_creating_output(self):
  with tempfile.TemporaryDirectory()as d:
   source=Path(d)/'unknown';source.mkdir();(source/'file.cpp').write_text('unrecognized')
   out=Path(d)/'out'
   with self.assertRaises(ValueError):a.prepare(source,out)
   self.assertFalse(out.exists())
 def test_existing_output_refused(self):
  with tempfile.TemporaryDirectory()as d:
   out=Path(d)/'out';out.mkdir();(out/'keep').write_text('unchanged')
   with self.assertRaises(ValueError):a.prepare(BASE,out)
   self.assertEqual((out/'keep').read_text(),'unchanged')
 def test_in_place_mutation_refused(self):
  with self.assertRaises(ValueError):a.prepare(BASE,BASE/'adapted')
 def test_existing_MUMPS_factory_and_no_shift_contract_retained(self):
  original=(BASE/'src/linear_solver.cpp').read_text();adapted,_=a.adapted_text('src/linear_solver.cpp',original)
  mumps='''#ifdef LAZYIPDCA_HAVE_MUMPS
  if(settings.linear_backend=="mumps"&&settings.linear_metis&&
      maximum_workspace_mb>0&&maximum_workspace_mb<=128)
    return trace_native_restoration_factor(std::make_unique<MumpsMetisFactor>(settings,maximum_workspace_mb));
#endif'''
  self.assertIn(mumps,original);self.assertIn(mumps,adapted)
  for source in (BASE,RESEARCH):
   text=(source/'src/dca_convexification_study_models.hpp').read_text();modified,_=a.adapted_text('src/dca_convexification_study_models.hpp',text)
   for line in ('cert.kkt_adaptive_regularization=false;cert.cholesk_adaptive_regularization=false;',
                'cert.kkt_primal_regularization=cert.kkt_dual_regularization=0;',
                'if(ok&&factor->last_positive_pivots()==n&&factor->last_negative_pivots()==0)'):
    self.assertIn(line,modified)
 def test_complete_inertia_only_and_bounded_failure_reuse_is_blocked(self):
  text,_=a.adapted_text('src/hsl_ma27_factorization.hpp',(BASE/'src/hsl_ma27_factorization.hpp').read_text())
  self.assertLess(text.index('reported_negative_=-1;'),text.index('if(!numeric(negative,message))return false;'))
  self.assertLess(text.index('if(negative<0||negative>n_)'),text.index('reported_negative_=negative;'))
  numeric=text[text.index('bool numeric(int& negative,std::string& message)final {'):]
  self.assertLess(numeric.index('if(workspace_failed_)'),numeric.index('std::copy(values_.begin()'))
  self.assertIn('workspace_failed_=false;',text[text.index('void clear_numeric()final'):])
 def test_actual_workspace_and_inertia_accessor_code_compiles_and_obeys_caps(self):
  source,_=a.adapted_text('src/hsl_ma27_factorization.hpp',(BASE/'src/hsl_ma27_factorization.hpp').read_text())
  self.assertIn(a.WORKSPACE_METHODS,source)
  accessor_lines='\n'.join(line.strip().replace('final','')for line in source.splitlines()if 'int last_'in line and '_pivots()'in line)
  self.assertEqual(len(accessor_lines.splitlines()),3)
  program=r'''#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstddef>
using Vec=std::vector<double>;
struct Harness {
 int workspace_cap_mb_=0;bool workspace_failed_=false;
 int n_=7,reported_negative_=-1;
 Vec a_;std::vector<int> iw_;
'''+a.WORKSPACE_METHODS+accessor_lines+r'''
};
int main(){
 int checks=0;auto expect=[&](bool good,const char* name){if(!good)throw std::runtime_error(name);++checks;};
 std::string why;
 Harness ordinary;ordinary.a_={1,2,3};ordinary.iw_={7,8};
 expect(ordinary.resize_integer_workspace(5,why),"ordinary integer growth");
 expect(ordinary.a_==Vec({1,2,3})&&ordinary.iw_[0]==7&&ordinary.iw_[1]==8,"ordinary data retained");
 expect(ordinary.resize_real_workspace(8,why)&&ordinary.a_[1]==2,"ordinary real growth preserves historical behavior");
 Harness bounded;bounded.workspace_cap_mb_=1;
 expect(bounded.resize_integer_workspace(100,why),"bounded initial integers");
 for(int i=0;i<100;++i)bounded.iw_[i]=i+11;
 expect(bounded.resize_real_workspace(100,why),"bounded initial reals");
 expect(bounded.resize_integer_workspace(500,why),"bounded integer growth");
 expect(bounded.a_.empty()&&bounded.a_.capacity()==0,"discard obsolete real factor input before integer replacement");
 expect(bounded.iw_[99]==110,"preserve analyzed integer data");
 expect(bounded.resize_real_workspace(200,why),"recreate real input before MA27 reload");
 expect(!bounded.resize_real_workspace(1000000,why)&&bounded.workspace_failed_,"real workspace cap fails closed");
 expect(bounded.iw_[99]==110,"integer analysis preserved after real failure");
 Harness copy;copy.workspace_cap_mb_=1;
 expect(copy.resize_integer_workspace(200000,why),"integer workspace under cap");
 copy.iw_[0]=73;
 expect(!copy.resize_integer_workspace(250000,why)&&copy.workspace_failed_,"integer replacement overlap counted");
 expect(copy.iw_[0]==73&&copy.iw_.size()==200000,"rejected replacement leaves old analyzed integers intact");
 Harness boundary;boundary.workspace_cap_mb_=1;
 expect(boundary.resize_integer_workspace(250000,why),"one-megabyte integer budget");
 expect(boundary.resize_real_workspace(6000,why),"real+integer total just below cap");
 expect(!boundary.resize_real_workspace(6100,why),"real+integer total over cap rejected");
 Harness invalid;invalid.workspace_cap_mb_=1;
 expect(!invalid.resize_integer_workspace(-1,why)&&invalid.workspace_failed_,"negative integer size rejected");
 expect(!invalid.resize_real_workspace(-1,why),"negative real size rejected");
 Harness inertia;
 expect(inertia.last_negative_pivots()==-1&&inertia.last_positive_pivots()==-1&&inertia.last_zero_pivots()==-1,"unknown inertia remains unknown");
 inertia.reported_negative_=0;
 expect(inertia.last_negative_pivots()==0&&inertia.last_positive_pivots()==7&&inertia.last_zero_pivots()==0,"positive nonsingular inertia diagnostics");
 inertia.reported_negative_=2;
 expect(inertia.last_negative_pivots()==2&&inertia.last_positive_pivots()==5&&inertia.last_zero_pivots()==0,"negative nonsingular inertia diagnostics");
 inertia.reported_negative_=-1;
 expect(inertia.last_negative_pivots()==-1&&inertia.last_zero_pivots()==-1,"reset prevents stale inertia");
 for(std::string backend:{"mumps","ma27","ldl","native"})for(bool metis:{false,true})for(bool ipopt:{false,true}){
  const bool old_reject=backend!="mumps"||!metis||ipopt;
  const bool new_reject=(backend!="mumps"&&backend!="ma27")||!metis||ipopt;
  expect(backend=="ma27"?new_reject==(!metis||ipopt):old_reject==new_reject,"MUMPS/other backend eligibility preserved; explicit MA27 admitted");
 }
 std::cout<<checks<<" pure backend support checks passed; no HSL solver invoked\n";
}
'''
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'test.cpp';b=Path(d)/'test';f.write_text(program)
   compile=subprocess.run(['g++','-std=c++17','-O2',str(f),'-o',str(b)],capture_output=True,text=True)
   self.assertEqual(compile.returncode,0,compile.stdout+compile.stderr)
   run=subprocess.run([str(b)],capture_output=True,text=True)
   self.assertEqual(run.returncode,0,run.stdout+run.stderr)
   self.assertIn('no HSL solver invoked',run.stdout)
   (HERE/'MA27_PURE_SUPPORT_TEST.txt').write_text(run.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
