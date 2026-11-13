"""Recompute Table 2, Table 3, S3.10/S3.9 intervals and abstract claims from frozen inputs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from landslide_sam.evaluation import paired_statistics, percentile_interval
from landslide_sam.powerlaw import mle

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['GVLM_v1_0', 'GVLM_v1_1', 'CAS', 'jz', 'bijie']


def csvread(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def save(path, records):
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


class Reproduction:
    def __init__(self, data, out):
        self.data, self.out = data, out
        self.expected = json.loads((ROOT/'configs/manuscript_expected.json').read_text())
        self.intervals = json.loads((ROOT/'configs/manuscript_intervals.json').read_text())
        self.results = []
        self.main = {}

    def check(self, location, statistic, got, expected, source, decimals=1, exact=False, unit='percent'):
        if isinstance(got, str):
            passed = got == str(expected)
            delta = 'MISSING'
            shown = got
        else:
            shown = float(f'{got:.{decimals}f}')
            delta = float(got)-float(expected)
            if exact:
                passed = got == expected
            elif unit == 'IoU_percentage_points':
                passed = abs(delta) <= .1
            else:
                # Other quantities must reproduce the value at the manuscript's printed precision.
                passed = shown == float(expected)
        row = dict(manuscript_location=location, statistic=statistic, recomputed=got,
                   manuscript_value=expected, displayed_recomputed=shown, difference=delta,
                   unit=unit, status='PASS' if passed else 'FAIL', frozen_source=source)
        self.results.append(row)
        if not passed:
            save(self.out/'comparison.csv', self.results)
            raise RuntimeError('STOP: manuscript mismatch: '+json.dumps(row))

    def table2(self):
        method_rows = {'R0':4,'training_fixed':5,'gbt':6,'upper':8,'box':9}
        counts = json.loads((ROOT/'configs/manuscript_counts.json').read_text())
        for di, d in enumerate(DATASETS):
            path = self.data/'frozen'/d/'evaluation_arrays.npz'
            with np.load(path, allow_pickle=False) as z:
                keep = z['units'] != 'Saint_Sebastian_Brazil' if d == 'GVLM_v1_1' else np.ones(len(z['units']), bool)
                methods = list(z['methods'])
                clusters = z['events'][keep] if d not in ['jz','bijie'] else z['units'][keep]
                point, boots = paired_statistics(z['T'][0,keep], z['PP'][0,keep], z['GG'][keep],
                    z['OS'][0,keep], z['OC'][keep], clusters, repeats=10000, seed=0)
                self.main[d] = (point, boots, methods)
                self.check(counts[d]['location'], d+' query objects', int(z['OC'][keep].sum()),
                           counts[d]['count'], f'frozen/{d}/evaluation_arrays.npz:OC,units', exact=True, unit='count')
                for method, ri in method_rows.items():
                    for metric, ci in [('iou', di+2),('object_iou', di+7)]:
                        target = next(r for r in self.expected if (r['table'],r['row'],r['column']) == (2,ri,ci))
                        self.check(target['location'], d+'/'+method+'/'+metric,
                            point[metric][methods.index(method)]*100, target['value'],
                            f'frozen/{d}/evaluation_arrays.npz:T,PP,GG,OS,OC; P1; {method}',
                            unit='IoU_percentage_points')
                for r in self.intervals:
                    if r['dataset'] != d or 'S3.10' not in r['manuscript_location']:
                        continue
                    key = 'equal_iou' if r['metric']=='equal' else r['metric']
                    if r['method']=='gbt-R0':
                        sample = boots[key][:,methods.index('gbt')] - boots[key][:,methods.index('R0')]
                    else:
                        sample = boots[key][:,methods.index(r['method'])]
                    ci = percentile_interval(sample)*100
                    for j, end in enumerate(['low','high']):
                        self.check(r['manuscript_location']+f', extracted line {r["extracted_line"]}',
                            d+'/'+r['method']+'/'+key+'/'+end, ci[j], float(r['published_'+end]),
                            f'frozen/{d}/evaluation_arrays.npz; paired bootstrap B=10000, seed=0',
                            unit='IoU_percentage_points')

    def table3(self):
        records = csvread(self.data/'area_frequency/per_object_areas.csv')
        power = csvread(self.data/'area_frequency/af_powerlaw.csv')
        names = ['reference','R0','selector','click_plus_box_k20','all_box']
        for di,d in enumerate(DATASETS):
            rr = [r for r in records if r['dataset']==d]
            refkey = 'ref_area_px' if d=='jz' else 'ref_area_m2'
            suffix = '_px' if d=='jz' else ''
            areas = np.array([[float(r[refkey])]+[float(r['area_'+name+suffix]) for name in names[1:]] for r in rr])
            frozen = next(r for r in power if r['dataset']==d and r['method']=='reference')
            xmin = float(frozen['xmin'])
            clusters = np.array([r['unit'] if d in ['jz','bijie'] else r['event'] for r in rr])
            groups = np.unique(clusters)
            gid = np.searchsorted(groups, clusters)
            draws = np.random.default_rng(0).integers(0,len(groups),size=(10000,len(groups)))
            weights = np.array([np.bincount(v,minlength=len(groups)) for v in draws],float)
            for j,name in enumerate(names):
                beta, nt = mle(areas[:,j],xmin)
                eligible = len(rr)>=50 and nt>=30 and np.isfinite(beta)
                ri, column = di+2, j+2
                targets = [r for r in self.expected if (r['table'],r['row'],r['column'])==(3,ri,column)]
                if targets:
                    t=targets[0]
                    self.check(t['location'],d+'/'+name+'/beta',beta if eligible else 'INSUFFICIENT',
                        t['value'],f'area_frequency/per_object_areas.csv; dataset={d}; reference xmin={xmin}',
                        decimals=t['decimals'],unit='beta')
                elif eligible:
                    raise RuntimeError('STOP: manuscript marks missing beta but eligible tail exists: '+d+'/'+name)
                if j:
                    t=next(r for r in self.expected if (r['table'],r['row'],r['column'])==(3,ri,j+6))
                    ratio=areas[:,j].sum()/areas[:,0].sum()
                    self.check(t['location'],d+'/'+name+'/summed_area_ratio',ratio,t['value'],
                        f'area_frequency/per_object_areas.csv; dataset={d}; all per-click areas',
                        decimals=t['decimals'],unit='area_ratio')
                # The fixed-rule beta appears only in the supplement and is handled below.
                if eligible:
                    self.beta_interval(d,name,areas[:,j],xmin,gid,weights)
            fixed=np.array([float(r['area_fixed_rule'+suffix]) for r in rr])
            self.beta_interval(d,'fixed_rule',fixed,xmin,gid,weights)

    def beta_interval(self,d,name,values,xmin,gid,weights):
        target=[r for r in self.intervals if r['dataset']==d and r['method']==name and r['metric']=='beta']
        if not target:return
        mask=values>=xmin
        logs=np.zeros(len(values));logs[mask]=np.log(values[mask]/xmin)
        bc=weights@np.bincount(gid,weights=mask,minlength=weights.shape[1])
        bl=weights@np.bincount(gid,weights=logs,minlength=weights.shape[1])
        valid=(bc>=2)&(bl>0)
        ci=np.quantile(1+bc[valid]/bl[valid],[.025,.975])
        for j,end in enumerate(['low','high']):
            r=target[0]
            self.check(r['manuscript_location']+f', extracted line {r["extracted_line"]}',
                d+'/'+name+'/beta/'+end,ci[j],float(r['published_'+end]),
                f'area_frequency/per_object_areas.csv; conditional reference xmin={xmin}; B=10000 seed=0',
                decimals=2,unit='beta')

    def abstract(self):
        if not self.main:self.table2()
        gains=[]
        for d,(p,b,m) in self.main.items():
            gain=(p['object_iou'][m.index('gbt')]-p['object_iou'][m.index('R0')])*100
            gains.append(gain)
        self.check('Abstract, sentence 5; DOCX block 21','minimum selector-minus-R0 per-object gain',min(gains),2.5,
                   'frozen/*/evaluation_arrays.npz; P1 seed0',unit='IoU_percentage_points')
        self.check('Abstract, sentence 5; DOCX block 21','maximum selector-minus-R0 per-object gain',max(gains),15.1,
                   'frozen/*/evaluation_arrays.npz; P1 seed0',unit='IoU_percentage_points')
        p,_,m=self.main['GVLM_v1_1']
        for name,target in [('training_fixed',41.7),('gbt',46.1)]:
            self.check('Abstract, sentence 5; DOCX block 21','external whole-image '+name,p['iou'][m.index(name)]*100,target,
                       'frozen/GVLM_v1_1/evaluation_arrays.npz; Saint excluded',unit='IoU_percentage_points')
        with np.load(self.data/'decomposition/part2_GVLM_v1_0_P1_evaluation.npz') as z:
            methods=list(z['methods'])
            point,boot=paired_statistics(z['T'],z['PP'],z['GG'],z['OS'],z['OC'],z['units'],10000,0)
            j,k=methods.index('b_R0'),methods.index('c_R0')
            delta=(point['iou'][j]-point['iou'][k])*100
            self.check('Sect.3.1 paragraph 1; DOCX block 97','reference-sized window inflation',delta,19.6,
                       'decomposition/part2_GVLM_v1_0_P1_evaluation.npz; b_R0-c_R0',unit='IoU_percentage_points')
            self.check('Abstract, sentence 3; DOCX block 21','about 20 points (rounded)',round(delta),20,
                       'decomposition/part2_GVLM_v1_0_P1_evaluation.npz; b_R0-c_R0',exact=True,unit='rounded_points')
        boxes=csvread(self.data/'human/reference_vs_human_boxes.csv')
        rr=[r for r in boxes if r['round']=='2' and r['in_requested_paired_cohort']=='True']
        self.check('Sect.3.1 paragraph 2; DOCX block 100','paired Round2 boxes',len(rr),88,
                   'human/reference_vs_human_boxes.csv',exact=True,unit='count')
        gap=np.mean([float(r['reference_box_SAM_iou'])-float(r['human_box_SAM_iou']) for r in rr])*100
        self.check('Abstract, sentence 3; DOCX block 21','about 18 points (rounded)',round(gap),18,
                   'human/reference_vs_human_boxes.csv; Round2 requested paired cohort',exact=True,unit='rounded_points')
        pairs=csvread(self.data/'human/paired_per_object.csv')
        costs=csvread(self.data/'human/cost_per_object.csv')
        click_savings=[];mixed_savings=[]
        for pid in ['p001','p005','p006']:
            p=[r for r in pairs if r['scope']==pid and r['object_set']=='main']
            c=[r for r in costs if r['scope']==pid and r['object_set']=='main']
            self.check('Sect.2.2.5 and Fig.5; DOCX blocks 90,120',pid+' paired objects',len(p),88,
                       'human/paired_per_object.csv; main/'+pid,exact=True,unit='count')
            ids={r['object_id'] for r in sorted(c,key=lambda r:(float(r['confidence']),r['object_id']))[:len(c)*20//100]}
            boxed={r['object_id'] for r in c if r['add_box']=='True'}
            if ids!=boxed:raise RuntimeError('STOP: frozen confidence ranking mismatch '+pid)
            self.check('Sect.2.2.3 paragraph 1; DOCX block 75',pid+' boxes at k=20%',len(boxed),17,
                       'human/cost_per_object.csv; main/'+pid,exact=True,unit='count')
            click=np.mean([float(r['click_seconds']) for r in p]);box=np.mean([float(r['box_seconds']) for r in p])
            mixed=sum(float(r['click_seconds'])+(float(r['box_seconds']) if r['object_id'] in ids else 0) for r in c)/len(c)
            click_savings.append(100*(1-click/box));mixed_savings.append(100*(1-mixed/box))
        for label,values,targets in [('click_time_saving',click_savings,[37,48]),('mixed_time_saving',mixed_savings,[17,28])]:
            for edge,got,target in [('minimum',min(values),targets[0]),('maximum',max(values),targets[1])]:
                self.check('Abstract, sentence 8; Sect.3.5 paragraph 1; DOCX blocks 21,118',label+'/'+edge,
                    round(got),target,'human/paired_per_object.csv; human/cost_per_object.csv; ratio of means',
                    exact=True,unit='rounded_percent')
        save(self.out/'time_savings_unrounded.csv',[dict(participant_id=pid,click_savings_percent=c,mixed_savings_percent=m)
            for pid,c,m in zip(['p001','p005','p006'],click_savings,mixed_savings)])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('target',choices=['table2','table3','abstract','all'])
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();job=Reproduction(a.data,a.output)
    for name in (['table2','table3','abstract'] if a.target=='all' else [a.target]):
        getattr(job,name)()
    save(a.output/'comparison.csv',job.results)
    summary=dict(status='PASS',checks=len(job.results),elapsed_seconds=time.perf_counter()-start,
                 main_bootstrap_repeats=10000,bootstrap_seed=0)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()
