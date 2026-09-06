from pathlib import Path
from factory.local_llm import generate_local_explanation

def test_missing_weights_are_not_presented_as_generation(tmp_path,monkeypatch):
    monkeypatch.setattr('factory.local_llm.MODEL_DIR',tmp_path)
    result=generate_local_explanation({})
    assert not result['available'] and result['mode']=='local_unavailable'
