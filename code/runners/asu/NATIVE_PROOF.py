"""Validate reported exact removals; never turn a gate into an actual removal."""
def validate_case_report(report,case):
    dimensions=case['dimensions']
    for key,expected in [('original_variables',dimensions['n']),('original_equalities',dimensions['equalities_total']),('original_inequalities',dimensions['inequalities_total'])]:
        if report.get(key)!=expected:raise ValueError('Native report dimensions differ from frozen case: '+key)
    if dimensions['linear_equalities']!=0 or dimensions['fixed_variables']!=0:
        raise ValueError('This study is restricted to the declared ASU quadratic-equality scope')
    if report.get('implied_bound_sides',0)>dimensions['lower_bounds']+dimensions['upper_bounds']:
        raise ValueError('Reported removed sides exceed original box')


def validate_side_report(report,keep=False):
    for key in ['original_variables','reduced_variables','original_equalities','original_inequalities','reduced_inequalities',
                'implied_bound_sides','equality_implied_bound_sides']:
        if type(report.get(key))is not int or report[key]<0:raise ValueError('Native integer missing/invalid: '+key)
    if report['schema']!='exact-rank-presolve-v1':raise ValueError('Native presolve schema differs')
    all_sides=[];counts={}
    for key,count_key,origin_key in [('equality_implied_bounds','equality_implied_bound_sides','original_equalities'),
                                    ('implied_bounds',None,'reduced_inequalities')]:
        array=report.get(key)
        if not isinstance(array,list):raise ValueError('Native removal array absent: '+key)
        if count_key and len(array)!=report[count_key]:raise ValueError('Native removal array/count differ')
        seen=set()
        for item in array:
            if not isinstance(item,list)or len(item)!=3 or any(type(x)is not int for x in item):raise ValueError('Malformed native bound proof')
            variable,side,origin=item
            if not 0<=variable<report['reduced_variables']or side not in(-1,1)or not 0<=origin<report[origin_key]:raise ValueError('Native bound proof indices out of range')
            if(variable,side)in seen or(variable,side)in all_sides:raise ValueError('Duplicate native removed bound side')
            seen.add((variable,side));all_sides.append((variable,side))
        counts[key]=len(array)
    if len(all_sides)!=report['implied_bound_sides']:raise ValueError('Native total bound-side count differs')
    # This kit's frozen503-case contract has no original affine equality or
    # fixed coordinate. Thus the combined equality counter is quadratic-only.
    if keep and report['equality_implied_bound_sides']!=0:raise ValueError('Conservation has actual equality-implied removals')
    return dict(equality_sides=report['equality_implied_bound_sides'],total_sides=len(all_sides),
                equality_proofs=report['equality_implied_bounds'],radial_proofs=report['implied_bounds'])


def endpoint_core(value):
    """Exactbinary64 core, including all work counters, excluding only clocks."""
    if isinstance(value,dict):
        return {k:endpoint_core(v)for k,v in value.items()if not k.endswith('_seconds')}
    if isinstance(value,list):return [endpoint_core(v)for v in value]
    if type(value)is float:return {'binary64':value.hex()}
    return value


def rank_core(value):
    observational={'research_equality_implied_bounds','quadratic_equality_rule_enabled',
                   'affine_equality_rule_enabled','radial_inequality_master_enabled'}
    return endpoint_core({k:v for k,v in value.items()if k not in observational})
