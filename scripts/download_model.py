from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen3-0.6B"
local_dir = Path("models/Qwen3-0.6B")

local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
)

print(f"Model downloaded to: {local_dir}")