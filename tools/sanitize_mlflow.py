"""Make checked-in MLflow locations portable and remove local machine paths."""
import sqlite3
from pathlib import Path


def sanitize(root: Path) -> int:
    database = root / "artifacts" / "mlflow.db"
    columns = {
        "experiments": ["artifact_location"],
        "runs": ["artifact_uri"],
        "model_versions": ["source", "storage_location"],
        "logged_models": ["artifact_location"],
    }
    changed = 0
    with sqlite3.connect(database) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, fields in columns.items():
            if table not in tables:
                continue
            for field in fields:
                rows = db.execute(f"SELECT rowid, {field} FROM {table} WHERE {field} IS NOT NULL").fetchall()
                for rowid, value in rows:
                    normalized = str(value).replace("\\", "/")
                    index = normalized.lower().find("/mlruns")
                    if index < 0:
                        continue
                    portable = "file:." + normalized[index:]
                    if portable != value:
                        db.execute(f"UPDATE {table} SET {field}=? WHERE rowid=?", (portable, rowid))
                        changed += 1
    return changed


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    print(f"Sanitized {sanitize(project_root)} MLflow database locations.")
