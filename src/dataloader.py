from huggingface_hub import snapshot_download
import os
from utils import get_current_dir

resultspath = get_current_dir().parent / "data" / "results"
os.makedirs(resultspath, exist_ok=True)

dataset = snapshot_download(
    repo_id="sueszli/whatsapp-lens",
    repo_type="dataset",
    local_dir=resultspath,
    allow_patterns="*.csv",
    token=True,
)
