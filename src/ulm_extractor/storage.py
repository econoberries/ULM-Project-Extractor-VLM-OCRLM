"""Atomic writes and fixed document identities."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="._tmp_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)

def key_for(record):
    identity = [str(record[k]) for k in ("dataset_name", "id", "image_path")]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24]
