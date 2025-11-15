import os
from huggingface_hub import create_repo, upload_folder, whoami

REPO_ID = "yltyadi/multilabel-emotion-detection-bert"
FOLDER_TO_UPLOAD = "enc_emotions"
TOKEN = os.getenv("HF_TOKEN", None)

try:
    me = whoami(token=TOKEN) if TOKEN else whoami()
    print("Authenticated as:", me.get("name") or me.get("orgs", ["unknown"])[0])
except Exception as e:
    raise SystemExit(
        "Not authenticated. Run login() in Python or set HUGGINGFACE_HUB_TOKEN.\n"
        f"Details: {e}"
    )

upload_folder(
    repo_id=REPO_ID,
    folder_path=FOLDER_TO_UPLOAD,
    path_in_repo="",
    commit_message="Upload best models and thresholds",
    ignore_patterns=[
        "**/checkpoint-*",
        "**/*.csv",
        "**/*.jsonl",
        "**/.ipynb_checkpoints",
        "**/__pycache__",
    ],
    token=TOKEN,
)
print("✅ Upload complete")