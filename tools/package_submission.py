"""Package source, generated data and evidence; omit dependencies, secrets and live audit."""
from pathlib import Path
import sqlite3
import zipfile
root=Path(__file__).resolve().parents[1]
out=root/'outputs'; out.mkdir(exist_ok=True)
target=out/'ai-factory-submission.zip'
files=['app.py','train.py','requirements.txt','requirements-tested.txt','requirements-local-llm.txt','README.md','start.ps1','sitecustomize.py','.gitignore']
folders=['factory','data','docs','tools','tests','.streamlit','artifacts','mlruns','outputs']
candidates=[root/p for p in files]
for folder in folders:
    candidates.extend(p for p in (root/folder).rglob('*') if p.is_file())
# SQLite online backup gives a consistent snapshot even with the tracking viewer open.
backup=root/'work/mlflow-package.db'; backup.parent.mkdir(exist_ok=True)
with sqlite3.connect(root/'artifacts/mlflow.db') as source,sqlite3.connect(backup) as destination: source.backup(destination)
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in candidates:
        if p==target or p.suffix=='.zip' or '__pycache__' in p.parts or p.name in ['decisions.sqlite','browser_check.cjs'] or p.name.endswith(('-wal','-shm')): continue
        archive.write(backup if p==root/'artifacts/mlflow.db' else p,p.relative_to(root).as_posix())
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert all(name in z.namelist() for name in ['app.py','artifacts/cnn.pt','artifacts/maintenance.joblib','outputs/presentation.pdf','outputs/sample-incident.pdf'])
print(f'{target.name}: {target.stat().st_size/1024/1024:.1f} MB; archive integrity verified')
