import numpy as np, sys, time, json
from multiprocessing import Pool
from scipy.stats import rankdata
CR='fmts/camera_ready'
sys.path.insert(0,CR)
from exact_signflip import exact_sum_p
z=np.load(CR+'/calib_d4_data.npz'); cl=z['gmean|cl']
_,inv=np.unique(cl,return_inverse=True); n=np.bincount(inv).astype(float); G=len(n)
D={k:z[k] for k in z.files if k.startswith('gmean|') and not k.endswith('cl')}
pairs=[(i,j) for i in range(10) for j in range(i+1,10)]
def run(a):
    k,(i,j)=a; d=D['gmean|%d_%d'%(i,j)]; m0=np.bincount(inv,weights=d)/n
    ii,jj=np.triu_indices(G); th=np.median((m0[ii]+m0[jj])/2)
    rng=np.random.default_rng([424242,k]); r=0; R=2000
    for _ in range(R):
        x=m0[rng.integers(0,G,G)]-th
        r+= exact_sum_p(np.sign(x)*rankdata(np.abs(x)))<0.05
    return 100*r/R
if __name__=='__main__':
    t=time.time()
    with Pool(9) as p: per=p.map(run,list(enumerate(pairs)))
    per=np.array(per); o=np.argsort(-per)
    print('pooled %.2f%%'%per.mean(),'| pairs >7.5%%: %d'%(per>7.5).sum(),'| >10%%: %d'%(per>10).sum(),'| worst five:',[('%d-%d'%(pairs[q][0]+1,pairs[q][1]+1),round(float(per[q]),2)) for q in o[:5]],'| min %.2f'%per.min(),'| %ds'%(time.time()-t))
    json.dump(dict(per_pair=per.tolist(),reps=2000,seed=424242),open(CR+'/gmean_level_recheck_main.json','w'))
