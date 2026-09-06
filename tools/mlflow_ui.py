import os
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.append(str(root/'.deps'))
from relocate_mlflow import relocate
relocate(root)
from mlflow.cli import cli
os.chdir(root)
os.environ['PYTHONPATH']=str(root)+os.pathsep+os.environ.get('PYTHONPATH','')
cli(['server','--backend-store-uri','sqlite:///'+(root/'artifacts/mlflow.db').as_posix(),'--host','127.0.0.1','--port','5000'])
