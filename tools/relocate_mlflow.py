"""Repair local MLflow artifact locations when the project folder is moved."""
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

def relocate(root):
    marker=root/'artifacts/project-location.json'
    current=root.resolve().as_posix()
    roots=set()
    if marker.exists():
        roots.add(json.loads(marker.read_text())['root'])
    columns={'experiments':['artifact_location'],'runs':['artifact_uri'],'model_versions':['source','storage_location'],'logged_models':['artifact_location']}
    database=root/'artifacts/mlflow.db'
    if database.exists():
        with sqlite3.connect(database) as db:
            tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table,fields in columns.items():
                if table not in tables:
                    continue
                for field in fields:
                    for (value,) in db.execute(f'SELECT {field} FROM {table} WHERE {field} IS NOT NULL'):
                        normalized=str(value).replace('\\','/')
                        index=normalized.lower().find('/mlruns')
                        if index >= 0:
                            prefix=normalized[:index].removeprefix('file:')
                            if prefix not in {'', '.'}:
                                roots.add(prefix.lstrip('/'))
            for old in roots:
                if old and old!=current:
                    for table,fields in columns.items():
                        if table not in tables:
                            continue
                        for field in fields:
                            for before,after in [(old,current),(old.replace('/','\\'),current.replace('/','\\')),(quote(old,safe='/:'),quote(current,safe='/:'))]:
                                db.execute(f'UPDATE {table} SET {field}=replace({field},?,?) WHERE instr({field},?)>0',(before,after,before))
    marker.write_text(json.dumps({'root':current},indent=2))

if __name__=='__main__': relocate(Path(__file__).resolve().parents[1])
