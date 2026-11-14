"""Download official monthly TLC yellow taxi data and retain JFK records.

Run only when downloading large public files is desired; existing 2025-03 source is reused.
"""
import calendar, hashlib, json, os, tempfile, urllib.request
from pathlib import Path
from datetime import date,datetime
from collections import Counter
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
FILTERED=OUT/'monthly_jfk'
BASE='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-{:02d}.parquet'

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 FILTERED.mkdir(exist_ok=True)
 reports=[]
 for month in range(1,13):
  dest=FILTERED/f'jfk_yellow_2025-{month:02d}.parquet'
  meta=FILTERED/f'jfk_yellow_2025-{month:02d}.json'
  url=BASE.format(month)
  if dest.exists() and meta.exists():
   old=json.loads(meta.read_text(encoding='utf8'))
   if old.get('filtered_sha256')==sha(dest):
    reports.append(old);print(f'{month:02d}/12 using validated cached output',flush=True);continue
  if month==3:
   src=ROOT/'data/yellow_tripdata_2025-03.parquet';temporary=False
   if not src.is_file(): raise FileNotFoundError(src)
  else:
   temp=tempfile.NamedTemporaryFile(prefix=f'tlc_2025_{month:02d}_',suffix='.parquet',delete=False);src=Path(temp.name);temp.close();temporary=True
   print(f'{month:02d}/12 downloading official data...',flush=True)
   try: urllib.request.urlretrieve(url,str(src))
   except Exception:
    src.unlink(missing_ok=True);raise
  try:
   source_hash=sha(src)
   table=pq.read_table(src,filters=[('PULocationID','=',132)])
   dt=table['tpep_pickup_datetime']
   start=datetime(2025,month,1);end=datetime(2026,1,1) if month==12 else datetime(2025,month+1,1)
   valid=pc.and_kleene(pc.greater_equal(dt,start),pc.less(dt,end))
   clean=table.filter(pc.fill_null(valid,False)).sort_by([('tpep_pickup_datetime','ascending')])
   date_counts=Counter(str(t.date()) for t in clean['tpep_pickup_datetime'].to_pylist() if t is not None)
   missing=[date(2025,month,d).isoformat() for d in range(1,calendar.monthrange(2025,month)[1]+1) if not date_counts[date(2025,month,d).isoformat()]]
   if missing: raise ValueError(f'Month {month}: days with no JFK rows: {missing}')
   pq.write_table(clean,dest,compression='zstd')
   report={'month':f'2025-{month:02d}','source_url':url,'source_sha256':source_hash,'source_bytes':src.stat().st_size,'jfk_raw_rows':table.num_rows,'jfk_in_month_rows':clean.num_rows,'out_of_month_or_null':table.num_rows-clean.num_rows,'filtered_sha256':sha(dest),'rows_by_date':dict(sorted(date_counts.items()))}
   meta.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');reports.append(report)
   print(f'{month:02d}/12 OK: {clean.num_rows} JFK pickups',flush=True)
  finally:
   if temporary:src.unlink(missing_ok=True)
 (OUT/'collection_summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf8')
 print('DONE: 12 months, total pickups',sum(x['jfk_in_month_rows'] for x in reports),flush=True)
if __name__=='__main__':main()
