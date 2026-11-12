#!/usr/bin/env python3
import importlib.util,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('m',Path(__file__).with_name('ANALYSER_OBJECTIFS_M_DEFAUT.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def sample():
 rows=[]
 for cid,dataset,regime in [('a','beam_structural','M'),('b','other','M'),('c','other','I')]:
  for cfg in m.CONFIGS:
   rows.append(dict(case_id=cid,config=cfg,regime=regime,dataset=dataset,family=dataset,input_sha256='input-'+cid,start_sha256='start-'+cid,classification='strict',primal=1e-8,dual=1e-8,complementarity=1e-8,objective=2. if cfg in m.LAZY else 1.))
 return rows
class Tests(unittest.TestCase):
 def test_counts_and_thresholds(self):
  idx,cases,_=m.parse(sample(),False);rows,pairs,res,cohorts=m.analyse(idx,cases);self.assertEqual(len(rows),72);self.assertEqual(cohorts['M_all'],['a','b']);self.assertEqual(len(pairs),96)
  r=rows[0];self.assertEqual(r['signed_gap_N'],2);self.assertEqual(r['signed_gap_median'],.5);self.assertEqual(r['higher_1em3'],2)
 def test_hash_mismatch_rejected(self):
  r=sample();r[1]['start_sha256']='different'
  with self.assertRaises(ValueError):m.parse(r,False)
 def test_duplicate_rejected(self):
  r=sample()
  with self.assertRaises(ValueError):m.parse(r+[r[0]],False)
 def test_strict_original_not_internal(self):
  r=sample();r[0]['primal']=.1
  with self.assertRaises(ValueError):m.parse(r,False)
 def test_nonstrict_not_objective_cohort(self):
  r=sample();r[0].update(classification='nonstrict',primal=.1);idx,cases,_=m.parse(r,False);rows,*_=m.analyse(idx,cases);self.assertEqual(rows[0]['signed_gap_N'],1)
 def test_signed_negative_favors_left(self):
  r=sample();r[0]['objective']=0.;idx,cases,_=m.parse(r,False);rows,*_=m.analyse(idx,cases);self.assertEqual(rows[0]['lower_1em3'],1);self.assertEqual(rows[0]['higher_1em3'],1)
 def test_type7_quartiles(self):self.assertEqual(m.quantile([0,2,4,6],.25),1.5)
if __name__=='__main__':unittest.main(verbosity=2)
