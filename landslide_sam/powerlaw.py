"""Frozen Clauset continuous-area MLE and reference-cutoff KS scan."""
import numpy as np

def mle(a,xmin):
    t=np.asarray(a)[np.asarray(a)>=xmin];den=np.log(t/xmin).sum()
    return (1+len(t)/den if len(t)>1 and den>0 else np.nan),len(t)

def fit_xmin(a):
    a=np.sort(a[a>0]); best=None
    for xmin in np.unique(a):
        left=np.searchsorted(a,xmin,side='left');t=a[left:];n=len(t)
        if n<30: break
        beta,_=mle(t,xmin)
        if not np.isfinite(beta): continue
        unique,cnt=np.unique(t,return_counts=True);f=1-(unique/xmin)**(1-beta)
        right=np.cumsum(cnt)/n;leftcdf=(np.cumsum(cnt)-cnt)/n
        ks=float(max(np.max(abs(f-right)),np.max(abs(f-leftcdf))))
        row=(float(xmin),float(beta),ks,n)
        if best is None or ks<best[2]: best=row
        yield row,best
