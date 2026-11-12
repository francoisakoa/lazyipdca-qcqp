#!/usr/bin/env python3
"""Adversarial build-proof checks, without compiling or invoking an optimizer."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
import BUILD_PERSISTANCE as build


def fixture():
    result=[]
    for role in ('reference','research'):
        source='/source/'+role;output='/build/'+role
        result.append(dict(source=source,binary=output+'/cli',homogeneous_compilation=True,
            hsl_adapter_compiled=True,backends='ma27 mumps',compiler_version='g++ identical',
            dependencies={'/lib/mumps.so':'abc','/lib/metis.so':'def'},source_sha256={'main.cpp':role},
            translation_units=[dict(name='main',returncode=0,
                command=['g++','-O3','-std=c++17','-I'+source+'/include','-c',source+'/src/main.cpp','-o',output+'/main.o'])],
            link_command=['g++',output+'/main.o','/lib/mumps.so','/lib/metis.so','-o',output+'/cli']))
    receipt=dict(source_reference_sha256=result[0]['source_sha256'],source_research_sha256=result[1]['source_sha256'])
    return *result,receipt


class BuildTests(unittest.TestCase):
    def test_same_toolchain_with_isolated_paths_passes(self):
        build.validate_build_pair(*fixture())

    def test_flag_and_link_drift_rejected(self):
        reference,research,receipt=fixture()
        changed=copy.deepcopy(research);changed['translation_units'][0]['command'].insert(1,'-ffast-math')
        with self.assertRaises(ValueError):build.validate_build_pair(reference,changed,receipt)
        changed=copy.deepcopy(research);changed['link_command'].insert(1,'-lipopt')
        with self.assertRaises(ValueError):build.validate_build_pair(reference,changed,receipt)

    def test_library_backend_and_source_drift_rejected(self):
        reference,research,receipt=fixture()
        for key,value in [('backends','mumps'),('hsl_adapter_compiled',False),
                          ('dependencies',{'/lib/mumps.so':'changed'}),('source_sha256',{'main.cpp':'changed'})]:
            changed=copy.deepcopy(research);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                build.validate_build_pair(reference,changed,receipt)

    def test_missing_or_failed_translation_unit_rejected(self):
        reference,research,receipt=fixture()
        changed=copy.deepcopy(research);changed['translation_units']=[]
        with self.assertRaises(ValueError):build.validate_build_pair(reference,changed,receipt)
        changed=copy.deepcopy(research);changed['translation_units'][0]['returncode']=1
        with self.assertRaises(ValueError):build.validate_build_pair(reference,changed,receipt)

    def test_source_and_installation_outputs_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';installation=root/'installation'
            for output in (source,source/'subdirectory',installation,installation/'results',root):
                with self.subTest(output=str(output)),self.assertRaises(ValueError):
                    build.require_isolated(output,[source,installation])
            build.require_isolated(root/'isolated_output',[source,installation])

    def test_mutated_dependency_pin_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'library.so';path.write_bytes(b'original')
            pins={str(path):build.sha(path)};build.verify_pins(pins)
            path.write_bytes(b'modified')
            with self.assertRaises(ValueError):build.verify_pins(pins)


if __name__=='__main__':unittest.main()
