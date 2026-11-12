#!/usr/bin/env python3
"""Compile and test the actual original-constant guard extracted from the engine.

No optimizer is called. This deliberately exercises binary64 coefficients much
smaller than any solver tolerance and objectives constant only on a fixed box.
"""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--cxx',default='c++');a=p.parse_args()
engine=a.source/'src/dca_inequality_majorant_engine.hpp';text=engine.read_text();start=text.index('bool original_constant_objective=problem.objective.signed_square.empty()&&');end=text.index('  auto audit_original=[&]() {',start);guard=text[start:end]
cpp='#include "lazyipdca_qcqp/model.hpp"\n#include <algorithm>\n#include <iostream>\n#include <stdexcept>\nusing namespace lazyipdca::qcqp;\nbool actual_guard(const QCQPProblem& problem){\n'+guard+'\nreturn original_constant_objective;\n}\nint main(){\n int checks=0;auto require=[&](bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);};\n QCQPProblem p;p.n=2;p.x0={0,0};p.lb={0,0};p.ub={1,1};p.objective.linear={0,0};p.objective.hessian.n=2;\n require(actual_guard(p),"zero objective excluded");\n p.objective.constant=-7;require(actual_guard(p),"negative constant objective excluded");\n p.objective.constant=9;require(actual_guard(p),"positive constant objective excluded");\n p.objective.linear[0]=1e-300;require(!actual_guard(p),"tiny affine coefficient treated as zero");\n p.lb[0]=p.ub[0]=0;require(!actual_guard(p),"fixed-box affine objective confused with original constant");\n p.objective.linear[0]=0;p.objective.hessian.entries={{0,0,1e-300}};\n require(!actual_guard(p),"tiny Hessian coefficient treated as zero");\n p.objective.hessian.entries={{0,1,-1e-300}};require(!actual_guard(p),"tiny bilinear coefficient treated as zero");\n p.objective.hessian.entries={{0,1,0}};require(actual_guard(p),"structural zero coefficient excluded");\n p.objective.signed_square={{0,1}};require(!actual_guard(p),"piecewise source objective admitted");\n std::cout<<"constant_objective_actual_guard PASS checks="<<checks<<" optimizer_calls=0\\n";\n}\n'
with tempfile.TemporaryDirectory(prefix='lazy_constant_guard_')as folder:
 folder=Path(folder);source=folder/'guard.cpp';binary=folder/'guard';source.write_text(cpp)
 compile=subprocess.run([a.cxx,'-std=c++17','-O2','-I'+str(a.source/'include'),str(source),'-o',str(binary)],capture_output=True,text=True)
 if compile.returncode:print(compile.stdout+compile.stderr);raise SystemExit(compile.returncode)
 run=subprocess.run([str(binary)],capture_output=True,text=True)
 print(run.stdout+run.stderr,end='');print(json.dumps({'engine_sha256':hashlib.sha256(engine.read_bytes()).hexdigest(),'actual_guard_extracted':True,'optimizer_calls':0,'returncode':run.returncode}));raise SystemExit(run.returncode)
