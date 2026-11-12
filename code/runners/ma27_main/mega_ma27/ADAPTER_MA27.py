#!/usr/bin/env python3
"""Isolate the declared MA27 backend support adaptation of known native sources.

No published source tree is changed in place. This admits the existing MA27
factorization for the same native policies; it does not change a QCQP model,
curvature correction, tolerance, pivot decision or original certificate test.
The real HSL library and all MA27 numerical trajectories must be qualified on
the target machine. No HSL code or numerical test double is distributed.
"""
from pathlib import Path
import argparse,difflib,hashlib,json,shutil,sys

EXPECTED={
 'ca51c188c9e0c3c1ea708457947aaff571e564f14157eb3fdf0b367c32333fd0':('qualified',823),
 '58bfdda4101d7aaf17b06fab9fc251151f300789aa6e8c0cd24c1ffb3729e2a2':('factorial',825),
}
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def source_map(source):return {str(p.relative_to(source)):sha(p)for p in sorted(source.rglob('*'))if p.is_file()}
def digest(value):return sha_bytes(json.dumps(value,sort_keys=True,separators=(',',':')).encode())

def replace_once(text,old,new,path,operations,label):
    if text.count(old)!=1:raise ValueError('Unsupported MA27 patch site '+path+' / '+label+': '+str(text.count(old)))
    operations.append(label);return text.replace(old,new,1)

WORKSPACE_METHODS=r'''
  // Bounded restoration caps the numerical real/integer MA27 work arrays.
  // It does not claim to cap symbolic auxiliary arrays or total process RSS.
  // Integer growth preserves the analyzed factor data; real factor input is
  // reloaded from values_ before every numeric attempt, so it can be released.
  bool workspace_allowed(std::size_t real_count,std::size_t integer_count,
                         std::string& message) {
    if(workspace_cap_mb_<=0)return true;
    const long double bytes=sizeof(double)*static_cast<long double>(real_count)+
        sizeof(int)*static_cast<long double>(integer_count);
    if(bytes<=static_cast<long double>(workspace_cap_mb_)*1024*1024)return true;
    workspace_failed_=true;
    message="MA27 bounded numerical A/IW workspace exceeds "+std::to_string(workspace_cap_mb_)+" MB";
    return false;
  }
  bool resize_integer_workspace(int count,std::string& message) {
    if(workspace_cap_mb_<=0){iw_.resize(count);return true;}
    if(count<0){workspace_failed_=true;message="invalid MA27 bounded integer workspace";return false;}
    const auto wanted=static_cast<std::size_t>(count);
    if(wanted<=iw_.capacity()){iw_.resize(wanted);return true;}
    // During replacement the previous integer array remains live. Account
    // for BOTH arrays before allocating; free the obsolete real input first.
    Vec().swap(a_);
    if(!workspace_allowed(0,iw_.capacity()+wanted,message))return false;
    std::vector<int> next;
    try{next.resize(wanted);}catch(const std::exception& error){
      workspace_failed_=true;message=std::string("MA27 bounded integer allocation failed: ")+error.what();return false;
    }
    if(!workspace_allowed(0,iw_.capacity()+next.capacity(),message))return false;
    std::copy_n(iw_.begin(),std::min(iw_.size(),wanted),next.begin());
    iw_.swap(next);
    return true;
  }
  bool resize_real_workspace(int count,std::string& message) {
    if(workspace_cap_mb_<=0){a_.resize(count);return true;}
    if(count<0){workspace_failed_=true;message="invalid MA27 bounded real workspace";return false;}
    const auto wanted=static_cast<std::size_t>(count);
    if(wanted<=a_.capacity()){a_.resize(wanted);return true;}
    // No numeric factor input survives a growth retry: values_ is reloaded
    // before the next call to MA27BD. Releasing avoids an uncounted copy peak.
    Vec().swap(a_);
    if(!workspace_allowed(wanted,iw_.capacity(),message))return false;
    Vec next;
    try{next.resize(wanted);}catch(const std::exception& error){
      workspace_failed_=true;message=std::string("MA27 bounded real allocation failed: ")+error.what();return false;
    }
    if(!workspace_allowed(next.capacity(),iw_.capacity(),message))return false;
    a_.swap(next);
    return true;
  }
'''

def adapted_text(path,text):
    operations=[]
    old='st.linear_backend!="mumps"||!st.linear_metis||st.inner_backend=="ipopt"'
    new='(st.linear_backend!="mumps"&&st.linear_backend!="ma27")||!st.linear_metis||st.inner_backend=="ipopt"'
    if path in ('src/main.cpp','src/dca_inequality_majorant_engine.hpp',
                'include/lazyipdca_qcqp/standard_auto.hpp','include/lazyipdca_qcqp/majorant_equality_stabilization.hpp'):
        text=replace_once(text,old,new,path,operations,'admit existing MA27+METIS under unchanged native policy checks')
        messages={
          'src/main.cpp':('convexification study requires native dca-engine majorants/auto and MUMPS+METIS','convexification study requires native dca-engine majorants/auto and MUMPS+METIS or MA27+METIS'),
          'src/dca_inequality_majorant_engine.hpp':('study requires native MUMPS+METIS','study requires native MUMPS+METIS or MA27+METIS'),
          'include/lazyipdca_qcqp/standard_auto.hpp':('native-MUMPS-METIS-required','native-MUMPS-or-MA27-METIS-required'),
          'include/lazyipdca_qcqp/majorant_equality_stabilization.hpp':('equality stabilization requires native MUMPS+METIS','equality stabilization requires native MUMPS+METIS or MA27+METIS'),
        }
        a,b=messages[path];text=replace_once(text,a,b,path,operations,'backend-specific refusal message')
        if path=='src/main.cpp':
            text=replace_once(text,'HH/HS/SH/SS require direct bordered MUMPS; optional pair elimination is unsupported',
              'HH/HS/SH/SS require a direct bordered native solver; optional pair elimination is unsupported',path,operations,'correct generic direct-solver description')
    elif path=='src/dca_convexification_study_models.hpp':
        text=replace_once(text,'cert.linear_backend="mumps";cert.linear_metis=true;',
          'cert.linear_backend=w.factor_settings.linear_backend=="ma27"?"ma27":"mumps";cert.linear_metis=true;',path,operations,'use requested MA27 for large support inertia classification')
        text=replace_once(text,'c.original_convex=c.classification_known=true;c.certificate="unregularized-MUMPS-positive-inertia";',
          'c.original_convex=c.classification_known=true;c.certificate=cert.linear_backend=="ma27"?"unregularized-MA27-positive-inertia":"unregularized-MUMPS-positive-inertia";',path,operations,'label the actual curvature certificate backend')
    elif path=='src/hsl_ma27_factorization.hpp':
        text=replace_once(text,'ready_=probe_ready_=false;defect_=0;',
          'ready_=probe_ready_=false;defect_=0;reported_negative_=-1;',path,operations,'invalidate reported inertia before every factor attempt')
        text=replace_once(text,'if(negative<0||negative>n_)throw std::runtime_error("HSL returned invalid inertia");',
          'if(negative<0||negative>n_)throw std::runtime_error("HSL returned invalid inertia");\n      reported_negative_=negative;',path,operations,'retain existing complete-factor INFO(15) inertia only')
        marker='  int curvature_defect() const final{return defect_;}'
        getters='''  int last_negative_pivots()const final{return reported_negative_;}
  int last_positive_pivots()const final{return reported_negative_>=0?n_-reported_negative_:-1;}
  int last_zero_pivots()const final{return reported_negative_>=0?0:-1;}
'''+marker
        text=replace_once(text,marker,getters,path,operations,'expose existing nonsingular MA27 inertia through common interface')
        text=replace_once(text,'int defect_=0,analyses_=0,reuses_=0,invalidations_=0;',
          'int defect_=0,reported_negative_=-1,analyses_=0,reuses_=0,invalidations_=0;',path,operations,'store diagnostic inertia separately from Newton acceptance')
        text=replace_once(text,'explicit Ma27Factor(const Settings& s):HslFactorBase(s) {',
          '''explicit Ma27Factor(const Settings& s,int maximum_workspace_mb=0):HslFactorBase(s),workspace_cap_mb_(maximum_workspace_mb) {
    if(workspace_cap_mb_<0||workspace_cap_mb_>128)throw std::invalid_argument("MA27 bounded workspace cap must be in 1..128 MB or zero for the ordinary path");''',path,operations,'optional cap for existing bounded restoration only')
        text=replace_once(text,' private:\n  bool metis_order(std::string& message) {',
          ' private:\n  int workspace_cap_mb_=0;bool workspace_failed_=false;\n'+WORKSPACE_METHODS+'\n  bool metis_order(std::string& message) {',path,operations,'bounded real/integer workspace resize guards')
        text=replace_once(text,'liw_=hsl_size(2LL*(2LL*ne_+3LL*n_+1));iw_.resize(liw_);',
          'liw_=hsl_size(2LL*(2LL*ne_+3LL*n_+1));if(!resize_integer_workspace(liw_,message))return false;',path,operations,'cap initial analysis integer workspace')
        text=replace_once(text,'a_.resize(la_);iw_.resize(liw_);return true;',
          '''if(workspace_cap_mb_>0){
      if(!resize_integer_workspace(liw_,message)||!resize_real_workspace(la_,message))return false;
    }else{a_.resize(la_);iw_.resize(liw_);}
    return true;''',path,operations,'cap completed symbolic workspace before numerical factorization')
        text=replace_once(text,'if(info_[0]==-3){liw_=hsl_grow(std::max(liw_,info_[1]),st_.hsl_memory_factor);iw_.resize(liw_);}\n      else{la_=hsl_grow(std::max(la_,info_[1]),st_.hsl_memory_factor);a_.resize(la_);}',
          '''if(info_[0]==-3){
        liw_=hsl_grow(std::max(liw_,info_[1]),st_.hsl_memory_factor);
        if(workspace_cap_mb_>0){
          if(!resize_integer_workspace(liw_,message)||!resize_real_workspace(la_,message))return false;
        }else iw_.resize(liw_);
      }else{
        la_=hsl_grow(std::max(la_,info_[1]),st_.hsl_memory_factor);
        if(workspace_cap_mb_>0){if(!resize_real_workspace(la_,message))return false;}
        else a_.resize(la_);
      }''',path,operations,'keep existing MA27 memory retry ladder inside the optional cap')
        text=replace_once(text,'bool numeric(int& negative,std::string& message)final {',
          'bool numeric(int& negative,std::string& message)final {\n    if(workspace_failed_){message="MA27 bounded workspace unavailable after failed allocation";return false;}',path,operations,'refuse reuse of a bounded factor after a workspace rejection')
        text=replace_once(text,'void clear_numeric()final{iw_.clear();keep_.clear();a_.clear();}',
          'void clear_numeric()final{iw_.clear();keep_.clear();a_.clear();if(workspace_cap_mb_>0){std::vector<int>().swap(iw_);Vec().swap(a_);workspace_failed_=false;}}',path,operations,'release obsolete bounded workspace before a new pattern')
    elif path=='src/linear_solver.cpp':
        marker='''#ifdef LAZYIPDCA_HAVE_MUMPS
  if(settings.linear_backend=="mumps"&&settings.linear_metis&&
      maximum_workspace_mb>0&&maximum_workspace_mb<=128)'''
        replacement='''#ifdef LAZYIPDCA_HAVE_HSL
  if(settings.linear_backend=="ma27"&&settings.linear_metis&&
      maximum_workspace_mb>0&&maximum_workspace_mb<=128) {
    try{return trace_native_restoration_factor(std::make_unique<Ma27Factor>(settings,maximum_workspace_mb));}
    catch(const std::exception& error){message=error.what();return {};}
  }
#endif
'''+marker
        text=replace_once(text,marker,replacement,path,operations,'MA27 bounded restoration factory without a backend substitute')
        text=replace_once(text,'bounded sparse restoration requires MUMPS+METIS and workspace cap in 1..128 MB',
          'bounded sparse restoration requires MUMPS+METIS or MA27+METIS and workspace cap in 1..128 MB',path,operations,'declare supported bounded sparse backends')
    elif path=='src/positive_square_epigraph_restoration.hpp':
        text=replace_once(text,'requested.linear_backend!="mumps"||!requested.linear_metis||',
          '(requested.linear_backend!="mumps"&&requested.linear_backend!="ma27")||!requested.linear_metis||',path,operations,'preserve positive-square feasibility repair on bounded MA27')
    elif path=='src/restoration.cpp':
        text=replace_once(text,'st.linear_backend!="mumps"||!st.linear_metis||p.n<512||',
          '(st.linear_backend!="mumps"&&st.linear_backend!="ma27")||!st.linear_metis||p.n<512||',path,operations,'preserve existing bounded sparse restoration eligibility on MA27')
    elif path=='src/solver.cpp':
        text=replace_once(text,'(st_.linear_backend == "mumps" ? "mumps_sym_indef_metis" : "native_dense_lapack")',
          '(st_.linear_backend == "mumps" ? "mumps_sym_indef_metis" : (st_.linear_backend == "ma27" ? (st_.linear_metis ? "ma27_sym_indef_metis" : "ma27_sym_indef_internal") : "native_dense_lapack"))',path,operations,'label the explicitly requested MA27 backend truthfully')
    else:raise ValueError('Unknown MA27 adaptation path: '+path)
    return text,operations

PATCH_PATHS=(
 'src/main.cpp','src/dca_inequality_majorant_engine.hpp',
 'include/lazyipdca_qcqp/standard_auto.hpp','include/lazyipdca_qcqp/majorant_equality_stabilization.hpp',
 'src/dca_convexification_study_models.hpp','src/hsl_ma27_factorization.hpp',
 'src/linear_solver.cpp','src/positive_square_epigraph_restoration.hpp','src/restoration.cpp','src/solver.cpp',
)
def prepare(source,out):
    source=Path(source).resolve();out=Path(out).resolve()
    if not source.is_dir():raise ValueError('Original native source absent')
    if out==source or out.is_relative_to(source):raise ValueError('MA27 adaptation must use a separate source tree')
    original=source_map(source);signature=digest(original)
    if signature not in EXPECTED:raise ValueError('Source tree is not one of the exact qualified/factorial snapshots')
    kind,count=EXPECTED[signature]
    if len(original)!=count:raise ValueError('Original source file count differs')
    if out.exists():raise ValueError('MA27 adapted output must be fresh')
    planned={};changes=[];diff=[]
    for relative in PATCH_PATHS:
        old=(source/relative).read_text();new,operations=adapted_text(relative,old)
        if old==new:raise ValueError('Empty MA27 adaptation: '+relative)
        planned[relative]=new
        changes.append(dict(path=relative,before_sha256=original[relative],after_sha256=sha_bytes(new.encode()),operations=operations))
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),'qualified/'+relative,'ma27/'+relative))
    shutil.copytree(source,out)
    for relative,text in planned.items():(out/relative).write_text(text)
    adapted=source_map(out)
    if source_map(source)!=original:raise ValueError('Original source changed during isolated adaptation')
    if set(original)!=set(adapted):raise ValueError('Unexpected source file additions/removals')
    if {p for p in original if original[p]!=adapted[p]}!=set(PATCH_PATHS):raise ValueError('Adaptation changed an unexpected source file')
    patchpath=out.parent/(out.name+'.MA27_ADAPTATION.patch');patchpath.write_text(''.join(diff))
    receipt=dict(schema='lazy-ma27-backend-support-adaptation-v1',original_kind=kind,
      original_source=str(source),adapted_source=str(out),original_source_sha256=original,adapted_source_sha256=adapted,
      original_source_signature=signature,adapted_source_signature=digest(adapted),changed_files=changes,
      adapter_file=str(Path(__file__).resolve()),adapter_sha256=sha(Path(__file__)),
      patch=str(patchpath),patch_sha256=sha(patchpath),primary_backend='ma27',
      auxiliary_backend='ma27 for large-support inertia and bounded sparse restoration; existing dense spectral work unchanged',
      hidden_mumps_substitution=False,pivot_decisions_changed=False,original_certificate_tests_changed=False,
      models_inputs_starts_changed=False,original_tree_unchanged=True,
      bounded_workspace_scope='MA27 numerical A/IW arrays, including accounted integer replacement copies; excludes symbolic auxiliaries and total process RSS',
      requires_real_HSL_target_qualification=True,local_MA27_qualified=False,production_authorized=False)
    receiptpath=out.parent/(out.name+'.MA27_ADAPTATION.json')
    receiptpath.write_text(json.dumps(receipt,indent=2)+'\n')
    receipt['receipt_path']=str(receiptpath);receipt['receipt_sha256']=sha(receiptpath)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=prepare(a.source,a.out)
    print(json.dumps({k:r[k]for k in ('schema','original_kind','adapted_source','primary_backend','requires_real_HSL_target_qualification','production_authorized','receipt_path')},indent=2))
if __name__=='__main__':
    try:main()
    except Exception as error:print('ADAPTATION MA27 REFUSÉE — '+str(error),file=sys.stderr);raise SystemExit(2)
