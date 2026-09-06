"""Download the optional Apache-2.0 Qwen model into this project, without an API key."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'work/huggingface')
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',revision='7ae557604adf67be50417f59c2c2f167def9a775',local_dir=str(ROOT/'local_models/qwen-0.5b'),allow_patterns=['*.json','*.safetensors','*.txt','LICENSE','README.md'],token=False,max_workers=2)
import json
metadata=ROOT/'local_models/qwen-0.5b/.cache/huggingface/download/config.json.metadata'
revision=metadata.read_text().splitlines()[0] if metadata.exists() else 'main'
(ROOT/'artifacts/local_llm_manifest.json').write_text(json.dumps({'model':'Qwen/Qwen2.5-0.5B-Instruct','revision':revision,'license':'Apache-2.0','source':'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct','weights_in_submission_archive':False},indent=2))
print('Local model downloaded. Select Local Qwen in the app.')
