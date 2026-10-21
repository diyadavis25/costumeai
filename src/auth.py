"""Local username/password accounts (no email needed).

* Passwords are never stored: only a salted PBKDF2-SHA256 hash (200k iterations).
* Accounts live in data/users.json.
* 5 wrong passwords lock that account for 60 seconds (saved on disk, so a refresh does not reset it).
"""
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time

from .config import DATA_DIR

USERS_PATH = DATA_DIR / "users.json"
ITERATIONS = 200_000
MAX_FAILS = 5
LOCK_SECONDS = 60
_DUMMY = None


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS).hex()


def _load() -> dict:
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _save(users: dict) -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=USERS_PATH.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_PATH)  # atomic: a crash never leaves a half-written file


# ---------------------------------------------------------------- validation
def validate_username(username: str):
    """Return an error string, or None when valid."""
    u = (username or "").strip()
    if not u:
        return "Enter a username."
    if not 3 <= len(u) <= 30:
        return "Username must be 3-30 characters."
    return None


def validate_password(password: str, username: str = ""):
    """Return an error string, or None when valid."""
    if not password:
        return "Enter a password."
    if len(password) < 4:
        return "Password must be at least 4 characters."
    return None


# ---------------------------------------------------------------- accounts
def register(username: str, password: str, confirm: str):
    """Create an account. Returns (ok, message)."""
    username = " ".join((username or "").split())  # trim and tidy spaces
    err = validate_username(username) or validate_password(password)
    if err:
        return False, err
    if password != confirm:
        return False, "Passwords do not match."
    users = _load()
    if username.lower() in users:
        return False, "That username is already taken."
    salt = secrets.token_bytes(16)
    users[username.lower()] = {"display": username, "salt": salt.hex(), "hash": _hash(password, salt),
                               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "fails": 0, "locked_until": 0}
    _save(users)
    return True, "Account created. You can log in now."


def login(username: str, password: str):
    """Check credentials. Returns (ok, message, display_name)."""
    global _DUMMY
    if not (username or "").strip() or not password:
        return False, "Enter your username and password.", None
    users = _load()
    key = " ".join(username.split()).lower()
    user = users.get(key)
    if user is None:
        if _DUMMY is None:  # equal work for unknown users so timing does not reveal who exists
            _DUMMY = _hash("dummy", b"0" * 16)
        _hash(password, b"0" * 16)
        return False, "Invalid username or password.", None
    now = time.time()
    if user.get("locked_until", 0) > now:
        wait = int(user["locked_until"] - now) + 1
        return False, f"Too many failed attempts. Try again in {wait} seconds.", None
    ok = hmac.compare_digest(_hash(password, bytes.fromhex(user["salt"])), user["hash"])
    if ok:
        user["fails"], user["locked_until"] = 0, 0
        _save(users)
        return True, "Welcome back!", user["display"]
    user["fails"] = user.get("fails", 0) + 1
    if user["fails"] >= MAX_FAILS:
        user["fails"], user["locked_until"] = 0, now + LOCK_SECONDS
        _save(users)
        return False, f"Too many failed attempts. Locked for {LOCK_SECONDS} seconds.", None
    _save(users)
    left = MAX_FAILS - user["fails"]
    return False, f"Invalid username or password. {left} attempt(s) left.", None