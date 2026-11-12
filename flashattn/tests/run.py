#!/usr/bin/env python3
"""Public CLI. All generated artifacts belong to an immutable source/config run."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runners'))
def main():
 p=argparse.ArgumentParser()
 p.add_argument('command',choices=['inspect','all','resume','report','archive'])
 p.add_argument('--config',default=str(Path(__file__).resolve().parent/'configs/full.json'))
 p.add_argument('--run');p.add_argument('--run-root');p.add_argument('--name')
 a=p.parse_args()
 from engine import inspect_environment,create_run,execute,report,archive
 if a.command=='inspect':
  import json
  print(json.dumps(inspect_environment(),indent=2));return
 if a.command=='all':
  run=create_run(Path(a.config),Path(a.run_root) if a.run_root else None,a.name)
  print('RUN_DIRECTORY',run,flush=True);execute(run);return
 if not a.run:p.error('--run is required')
 run=Path(a.run).resolve()
 if a.command=='resume':execute(run)
 elif a.command=='report':report(run)
 else:archive(run)
if __name__=='__main__':main()
