from pathlib import Path
from io import StringIO
import sys,random
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import qcq_model_r3p23 as q

def parse(tokens,n,compact):
 reader=q._LineReader(StringIO('linear '+' '.join(tokens)+'\n'),Path('fixture'))
 if compact:return q._read_sparse_linear(reader,n)
 return tuple((i,v) for i,v in enumerate(q._read_dense_vector(reader,'linear',n,allow_infinity=False)) if v!=0)
rng=random.Random(871)
for count in (0,1,15,16,17,100,25086):
 tokens=[rng.choice(('0','0','0','0','-0','+0','0.000','-2.5','1e-320','-1e-200','1e+100')) for _ in range(count)]
 assert parse(tokens,count,True)==parse(tokens,count,False)
for token in ('NaN','INF','-inf','0x','1e1000','1e-200junk'):
 for compact in (False,True):
  try:parse(['0']*32+[token],33,compact)
  except ValueError:pass
  else:raise AssertionError((token,compact))
for compact in (False,True):
 for n in (2,4):
  try:parse(['0','0','0'],n,compact)
  except ValueError:pass
  else:raise AssertionError(('length',n,compact))
print('PASS sparse Python reader matches retained dense reader for zeros, tiny values, malformed tokens and lengths')
