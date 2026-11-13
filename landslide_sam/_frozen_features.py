"""Frozen feature arithmetic. See docs/SOURCE_PROVENANCE.csv; no GT inputs."""

import numpy as np

from scipy import ndimage

ONE8=np.ones((3,3),bool)

RING=15

GSD={'gvlm':.59,'bijie':.8,'jz':None}

_DS='jz'

def ring_sub(m, r):
    """mask 外扩 r 像元的棋盘环带, 只在 mask 外接框 ±r 内算。返回 (框, 环带布尔)。"""
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    H, W = m.shape
    y0 = max(int(ys.min()) - r, 0); y1 = min(int(ys.max()) + r + 1, H)
    x0 = max(int(xs.min()) - r, 0); x1 = min(int(xs.max()) + r + 1, W)
    sub = m[y0:y1, x0:x1]
    d = ndimage.distance_transform_cdt(~sub, metric='chessboard')
    return y0, y1, x0, x1, (d <= r) & ~sub

def signed_dist(m, cy, cx):
    """点到候选边界的有符号棋盘距离（内为正, 外为负）。"""
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return 0.0
    H, W = m.shape
    y0 = max(min(int(ys.min()), cy) - 1, 0); y1 = min(max(int(ys.max()), cy) + 2, H)
    x0 = max(min(int(xs.min()), cx) - 1, 0); x1 = min(max(int(xs.max()), cx) + 2, W)
    sub = m[y0:y1, x0:x1]
    if sub[cy - y0, cx - x0]:
        d = ndimage.distance_transform_cdt(sub, metric='chessboard')[cy - y0, cx - x0]
    else:
        d = -ndimage.distance_transform_cdt(~sub, metric='chessboard')[cy - y0, cx - x0]
    return float(d)

def diff_in_ring(m, x, r=RING, how='mean'):
    """x 在候选内的统计量 − x 在候选外 r 像元环带内的同一个统计量。"""
    if not m.any():
        return 0.0 if how == 'mean' else -np.inf
    ins = x[m]
    rs = ring_sub(m, r)
    if rs is None:
        return 0.0 if how == 'mean' else -np.inf
    y0, y1, x0, x1, rg = rs
    v = x[y0:y1, x0:x1][rg]
    if not v.size:
        return 0.0 if how == 'mean' else -np.inf
    fn = np.mean if how == 'mean' else np.median
    return float(fn(ins) - fn(v))

def cand_feats(m, score, logit, gray, sat, mag, c1, pt, win_px, areas, rank_k):
    """一个候选的全部特征（pt = 该候选用的提示点, 供 dcent 用）。"""
    f = {}
    area = int(np.count_nonzero(m))
    f['area'] = area
    f['sam'] = float(score)
    f['arank'] = rank_k
    if area == 0:
        for k in ('stab', 'ar', 'aratio', 'comp', 'dcent', 'br', 'sat', 'sob',
                  'grad', 'logarea'):
            f[k] = 0.0
        f['logarea'] = None
        if c1 is not None:
            f['chg'] = -np.inf
        return f
    hi = logit > 1.0
    lo = logit > -1.0
    u = int(np.count_nonzero(lo))
    f['stab'] = (int(np.count_nonzero(hi)) / u) if u else 0.0
    f['ar'] = area / win_px
    f['logarea'] = float(np.log10(area * (GSD[_DS] ** 2 if GSD.get(_DS) else 1.0)))
    oth = (float(np.sum(areas)) - area) / (len(areas) - 1.0)
    f['aratio'] = (area / oth) if oth > 0 else 0.0
    b = m.astype(np.uint8)
    per = int(np.abs(np.diff(b, axis=0)).sum() + np.abs(np.diff(b, axis=1)).sum())
    f['comp'] = (4.0 * np.pi * area / (per * per)) if per else 0.0
    eq = np.sqrt(area / np.pi)
    f['dcent'] = (signed_dist(m, pt[0], pt[1]) / eq) if eq > 0 else 0.0
    f['br'] = diff_in_ring(m, gray)
    f['sat'] = diff_in_ring(m, sat)
    f['sob'] = diff_in_ring(m, mag)
    edge = m & ~ndimage.binary_erosion(m, ONE8)
    f['grad'] = float(mag[edge].mean()) if edge.any() else 0.0
    f['chg'] = diff_in_ring(m, c1, how='median') if c1 is not None else None
    return f

def signed_fast(m,cy,cx):
 if not m.any():return 0.
 H,W=m.shape;inside=bool(m[cy,cx]);r=1
 while True:
  y0=max(cy-r,0);y1=min(cy+r+1,H);x0=max(cx-r,0);x1=min(cx+r+1,W)
  q=m[y0:y1,x0:x1];target=~q if inside else q
  if target.any():
   yy,xx=np.nonzero(target);d=int(np.minimum.reduce(np.maximum(np.abs(yy+y0-cy),np.abs(xx+x0-cx))))
   return float(d if inside else -d)
  if y0==0 and x0==0 and y1==H and x1==W:return -1. if inside else 0.
  r*=2

def feature(m,sc,stab,rgb,pt,areas,rank,side,planes):
 gray,sat,mag=planes;area=int(np.count_nonzero(m))
 # Identical legacy feature arithmetic (including legacy compactness implementation).
 f=dict(sam=float(sc),stab=float(stab),arank=float(rank),logscale=float(np.log2(side)),ar=area/m.size)
 if not area:
  f.update({k:0. for k in ['aratio','comp','dcent','br','sat','sob','grad']});f['logarea']=None;return f
 oth=(sum(areas)-area)/2.;f['aratio']=area/oth if oth>0 else 0.
 b=m.astype(np.uint8);per=int(np.abs(np.diff(b,axis=0)).sum()+np.abs(np.diff(b,axis=1)).sum())
 f['comp']=4.*np.pi*area/(per*per) if per else 0.
 f['dcent']=signed_fast(m,*pt)/np.sqrt(area/np.pi)
 rs=ring_sub(m,RING)
 for name,plane in [('br',gray),('sat',sat),('sob',mag)]:
  if rs is None:f[name]=0.;continue
  y0,y1,x0,x1,rg=rs;v=plane[y0:y1,x0:x1][rg]
  f[name]=float(np.mean(plane[m])-np.mean(v)) if v.size else 0.
 edge=m&~ndimage.binary_erosion(m,ONE8);f['grad']=float(mag[edge].mean()) if edge.any() else 0.
 f['logarea']=float(np.log10(area*.59**2));return f

def planes(rgb):
 f=rgb.astype(np.float32);gray=f.mean(2);mx=f.max(2);mn=f.min(2);sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1.),0.)
 mag=np.hypot(ndimage.sobel(gray,axis=1,mode='nearest'),ndimage.sobel(gray,axis=0,mode='nearest')).astype(np.float32)
 return gray,sat,mag

def mask_iou(a,wa,b,wb,area_a=None,area_b=None):
 y,x,h,w=wa;v,u,hh,ww=wb;y0=max(y,v);x0=max(x,u);y1=min(y+h,v+hh);x1=min(x+w,u+ww)
 inter=int(np.count_nonzero(a[y0-y:y1-y,x0-x:x1-x]&b[y0-v:y1-v,x0-u:x1-u])) if y1>y0 and x1>x0 else 0
 aa=int(a.sum()) if area_a is None else int(area_a);bb=int(b.sum()) if area_b is None else int(area_b)
 return inter/(aa+bb-inter) if aa+bb-inter else 0.
