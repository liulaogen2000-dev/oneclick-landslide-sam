"""Verify every payload file against the relative-path SHA256 manifest."""
import argparse,csv,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    records=list(csv.DictReader((a.data/'MANIFEST.csv').open()))
    for r in records:
        path=a.data/r['relative_path']
        if Path(r['relative_path']).is_absolute() or '..' in Path(r['relative_path']).parts:raise ValueError('Invalid manifest path')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        if path.stat().st_size!=int(r['size_bytes']) or h.hexdigest()!=r['sha256']:raise RuntimeError('STOP: package checksum mismatch '+r['relative_path'])
    print(f'PASS: {len(records)} payload files; MANIFEST.csv excludes itself')

if __name__=='__main__':main()
