#!/usr/bin/env python3
"""File-only frozen import proofs; no optimizer or predecessor result is read."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_inventory import FrozenLocator,materialize,path_values,select_metadata
from MEGA import ordered_cases


class ImportTests(unittest.TestCase):
    def test_old_frozen_tree_supplies_raw_bytes_without_bundled_catalogue(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'new';root.mkdir();old=base/'old';old.mkdir()
            raw=old/'data/frozen/X/input.qcq';raw.parent.mkdir(parents=True)
            data=b'raw frozen source including signed infinity and raw x0\n';raw.write_bytes(data)
            canonical=old/'canonical_inputs/X.qcq';canonical.parent.mkdir()
            canonical.write_bytes(b'exact alternative serialization with a different file hash')
            (old/'PREPARATION.json').write_text(json.dumps({'cases':[{'id':'X','input':str(canonical)}]}))
            expected=hashlib.sha256(data).hexdigest()
            source=FrozenLocator(root,[old]).locate('X','input',expected,{})
            target=Path(materialize(root,'X','input',expected,source,[]))
            self.assertEqual(target.read_bytes(),data)
            self.assertFalse((root/'data/catalogue').exists())
            self.assertFalse((root/'data/bundled_frozen').exists())

    def test_source_locator_and_independent_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'new';root.mkdir();old=base/'old';old.mkdir()
            source=old/'raw.qcq';data=b'LAZYIPDCA_QCQP_V1\nname frozen\nn 2\nx0 -0 1\n'
            source.write_bytes(data);expected=hashlib.sha256(data).hexdigest()
            (old/'PREPARATION.json').write_text(json.dumps({'cases':[{'id':'CUTEst__X','input':str(source)}]}))
            locator=FrozenLocator(root,[old]);p=locator.locate('CUTEst__X','input',expected,{})
            imports=[];target=Path(materialize(root,'CUTEst__X','input',expected,p,imports))
            source.unlink()
            self.assertEqual(target.read_bytes(),data)
            self.assertEqual(imports[0]['sha256'],expected)
            self.assertTrue(target.is_relative_to(root))
            self.assertFalse(target.is_symlink())
            self.assertEqual(path_values({'endpoint':'forbidden','result_file':'forbidden','aliases':[{'input':'allowed'}]},'input'),['allowed'])

    def test_mismatched_bytes_never_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.qcq';source.write_bytes(b'bad')
            locator=FrozenLocator(root,[])
            with self.assertRaises(FileNotFoundError):locator.locate('X','input','0'*64,{'input':str(source)})
            expected=hashlib.sha256(b'bad').hexdigest();target=root/'data/frozen/X/input.qcq';target.parent.mkdir(parents=True);target.write_bytes(b'foreign')
            with self.assertRaises(ValueError):materialize(root,'X','input',expected,source,[])
            self.assertEqual(target.read_bytes(),b'foreign')

    def test_owned_complete_temporary_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';data=b'exact';source.write_bytes(data);expected=hashlib.sha256(data).hexdigest()
            temp=root/'data/frozen/X/input.qcq.importing';temp.parent.mkdir(parents=True);temp.write_bytes(data)
            target=Path(materialize(root,'X','input',expected,source,[]))
            self.assertEqual(target.read_bytes(),data);self.assertFalse(temp.exists())

    def test_destination_parent_symlink_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'root';root.mkdir();outside=base/'outside';outside.mkdir()
            (root/'data').symlink_to(outside,target_is_directory=True)
            source=base/'source';source.write_bytes(b'exact');expected=hashlib.sha256(b'exact').hexdigest()
            with self.assertRaises(ValueError):materialize(root,'X','input',expected,source,[])
            self.assertEqual(list(outside.iterdir()),[])

    def test_explicit_case_root_and_migrated_project_mirror(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'new';root.mkdir();mirror=base/'mirror';mirror.mkdir()
            p=mirror/'X/input.qcq';p.parent.mkdir();p.write_bytes(b'first');expected=hashlib.sha256(b'first').hexdigest()
            self.assertEqual(FrozenLocator(root,[mirror]).locate('X','input',expected,{}),p.resolve())
            q=mirror/'legacy/Y/input.qcq';q.parent.mkdir(parents=True);q.write_bytes(b'second');expected=hashlib.sha256(b'second').hexdigest()
            row={'input':'/data/projects/IPDCA_QCQP/legacy/Y/input.qcq'}
            self.assertEqual(FrozenLocator(root,[mirror]).locate('Y','input',expected,row),q.resolve())

    def test_exact_full_non_dey_metadata(self):
        root=Path(__file__).resolve().parents[1]
        rows,selection=select_metadata(root);cases,datasets,_=ordered_cases(rows)
        self.assertEqual(len(cases),4078);self.assertEqual(len(cases)*8,32624)
        self.assertEqual(len(selection['excluded_case_ids']),50)
        counts={d['id']:d['N'] for d in datasets}
        self.assertNotIn('pooling_dey',counts)
        self.assertEqual([d['id'] for d in datasets[:3]],['CUTEst','multicast_beamforming','PGLIB'])
        self.assertEqual([counts[k] for k in ['CUTEst','multicast_beamforming','PGLIB','beam_safety']],[136,129,198,3])
        self.assertEqual([counts[k] for k in ['pooling_luedtke','pooling_MINLPLIB','pooling_reduced']],[360,73,2])


if __name__=='__main__':unittest.main(verbosity=2)
