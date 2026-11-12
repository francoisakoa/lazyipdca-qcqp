#!/usr/bin/env python3
"""Independently check the native state digest against Python hashlib."""
from pathlib import Path
import hashlib,json,struct,subprocess,tempfile
HERE=Path(__file__).resolve().parent
def main():
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d);cpp=folder/'sha.cpp';exe=folder/'sha'
        cpp.write_text('#include <iostream>\n#include "factorial_trace.hpp"\nint main(){using namespace lazyipdca::qcqp::detail; FactorialSHA256 h;h.vector({0.0,-0.0,1.0,0x0.0000000000001p-1022,1e300});h.vector({});h.vector({-1.0,2.0});h.vector({0.1});std::cout<<h.finish()<<"\\n";}\n')
        subprocess.run(['g++','-std=c++17','-I'+str(HERE/'source/include'),'-I'+str(HERE/'source/src'),str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
        actual=subprocess.check_output([str(exe)],text=True).strip();h=hashlib.sha256()
        for vector in [[0.0,-0.0,1.0,5e-324,1e300],[],[-1.0,2.0],[0.1]]:
            h.update(struct.pack('<Q',len(vector)))
            for value in vector:h.update(struct.pack('<d',value))
        expected=h.hexdigest()
        if actual!=expected:raise AssertionError((actual,expected))
        result={'schema':'factorial-state-hash-independent-check-v1','passed':True,'native':actual,'python_hashlib':expected,
                'encoding':'length-prefixed unsigned64 little-endian lengths, then exact binary64 little-endian values for x,s,z,lambda',
                'covers':['empty vector','negative zero','positive zero','smallest positive subnormal','large finite value','negative value'],
                'source_sha256':hashlib.sha256((HERE/'source/src/factorial_trace.hpp').read_bytes()).hexdigest(),'corpus_gate_passed':False}
        (HERE/'TRACE_HASH_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
if __name__=='__main__':main()
