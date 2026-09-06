"""Repair only local artifact-location columns when the submitted folder is moved."""
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

def relocate(root):
    marker=root/'artifacts/project-location.json'
    current=root.resolve().as_posix()
    if marker.exists():
        old=json.loads(marker.read_text())['root']
        if old!=current:
            columns={'experiments':['artifact_location'],'runs':['artifact_uri'],'model_versions':['source','storage_location'],'logged_models':['artifact_location']}
            with sqlite3.connect(root/'artifacts/mlflow.db') as db:
                for table,fields in columns.items():
                    for field in fields:
                        for before,after in [(old,current),(old.replace('/','\\'),current.replace('/','\\')),(quote(old,safe='/:'),quote(current,safe='/:'))]:
                            db.execute(f'UPDATE {table} SET {field}=replace({field},?,?) WHERE instr({field},?)>0',(before,after,before))
    marker.write_text(json.dumps({'root':current},indent=2))

if __name__=='__main__': relocate(Path(__file__).resolve().parents[1])
