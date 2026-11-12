import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import benchmark_filter_dca_g10 as runner

class Contract(unittest.TestCase):
    def options(self):
        return SimpleNamespace(max_iter=300,time_limit=180,display_level=0,live_every=1,
            live_header_every=20,inner_max=8,max_atoms=500,atom_batch_size=2,atom_event_every=1,
            atom_events=True,research_profile='aggregate',rho_refinements=3,initial_restoration=False,
            normal_weighted_atoms=False,no_model_reuse=False,no_extrapolation=False,
            aggregate_rho_max=100,aggregate_backend='auto')
    def command(self,mode):
        return runner.lazy_command(self.options(),Path('solver'),Path('same.qcq'),Path('out.json'),mode)
    def test_convex_model_controls_and_unmodified_tolerances(self):
        command=self.command('filter-dca')
        for flag in ['--dca-aggregate','--no-lazy-newton-lazy-constraints','--dca-sparse-fallback']:
            self.assertIn(flag,command)
        self.assertNotIn('--dca-krylov',command)
        for flag in ['--feas-tol','--opt-tol','--dual-inf-tol','--compl-inf-tol']:
            self.assertEqual(command[command.index(flag)+1],'1e-6')
        self.assertEqual(command[command.index('--kkt-berr-reject')+1],'1e-7')
    def test_fixed_g7_baseline_and_single_switch_ablation(self):
        baseline=self.command('baseline')
        self.assertIn('--dca-local-active',baseline)
        self.assertNotIn('--dca-aggregate',baseline)
        self.assertNotIn('--no-lazy-newton-lazy-constraints',baseline)
        self.assertEqual(self.command('unaccelerated'),self.command('filter-dca')+['--no-dca-aggregate'])
    def test_target_misses_do_not_hide_correctness_failures(self):
        self.assertEqual(runner.campaign_exit_code({'all_correct':True,'target_met':True},True),0)
        self.assertEqual(runner.campaign_exit_code({'all_correct':True,'target_met':False},True),2)
        self.assertEqual(runner.campaign_exit_code({'all_correct':False,'target_met':True},True),1)
    def test_identity_and_package_entry_points(self):
        identity=json.loads((ROOT/'release_identity_g10.json').read_text())
        self.assertEqual(identity['api_version'],4340)
        header=(ROOT/'include/lazyipdca_qcqp/version.hpp').read_text()
        for token in (identity['version'],identity['build_id'],str(identity['api_version'])):
            self.assertIn(token,header)
        spec=importlib.util.spec_from_file_location('package_g10',ROOT/'tools/package_g10.py')
        packager=importlib.util.module_from_spec(spec);spec.loader.exec_module(packager)
        for name in ['tools/build_g10.sh','tools/run_g10.sh','tools/install_and_test_g10.sh','src/dca_aggregate.hpp']:
            self.assertIn(name,packager.REQUIRED)
            self.assertTrue((ROOT/name).is_file())

if __name__=='__main__':unittest.main()
