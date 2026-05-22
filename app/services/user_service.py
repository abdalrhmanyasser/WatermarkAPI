import os
from pathlib import Path
from typing import Optional

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
USER_SECRET_FILE = DATA_ROOT / "user_secrets.txt"


def ensure_data_root() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not USER_SECRET_FILE.exists():
        USER_SECRET_FILE.write_text("user_id|secret_file|secret_payload\n", encoding="utf-8")
    return DATA_ROOT


def append_user_secret(user_id: str, secret_file: str, secret_payload: str) -> None:
    ensure_data_root()
    line = f"{user_id}|{secret_file}|{secret_payload}\n"
    USER_SECRET_FILE.open("a", encoding="utf-8").write(line)


def get_user_secrets(user_id: Optional[str] = None) -> list[str]:
    ensure_data_root()
    lines = USER_SECRET_FILE.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[1:]:
        if not line.strip():
            continue
        if user_id is None or line.startswith(f"{user_id}|"):
            entries.append(line)
    return entries
