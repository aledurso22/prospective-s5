"""B38c byte-level data. enwik8 is the declared target; a tiny local corpus is
used ONLY for smoke tests and is never silently substituted -- the config records
which dataset was used and `require_enwik8` makes substitution an error."""
import hashlib
import os
import urllib.request
import numpy as np

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def enwik8_path(root):
    return os.path.join(root, "enwik8")


def stage_enwik8(root, download=False):
    """Return path to enwik8, downloading into `root` if allowed. No hard-coded
    machine paths: `root` comes from the config or $B38C_DATA_DIR."""
    os.makedirs(root, exist_ok=True)
    p = enwik8_path(root)
    if os.path.exists(p):
        return p
    if not download:
        return None
    zp = p + ".zip"
    urllib.request.urlretrieve(ENWIK8_URL, zp)
    import zipfile
    with zipfile.ZipFile(zp) as z:
        z.extract("enwik8", root)
    return p


def load_bytes(cfg):
    """Returns (train, val, test) uint8 arrays and a provenance dict."""
    root = cfg.get("data_dir") or os.environ.get("B38C_DATA_DIR", "./data")
    name = cfg.get("dataset", "enwik8")
    if name == "enwik8":
        p = stage_enwik8(root, cfg.get("download", False))
        if p is None:
            if cfg.get("require_enwik8", True):
                raise FileNotFoundError(
                    f"enwik8 not found under {root}. Set data_dir / $B38C_DATA_DIR, "
                    f"or run with download=true, or set dataset='smoke' explicitly.")
            name = "smoke"
        else:
            raw = np.frombuffer(open(p, "rb").read(), dtype=np.uint8)
            n = cfg.get("n_bytes") or len(raw)
            raw = raw[:n]
            src = dict(dataset="enwik8", path=p, n_bytes=int(len(raw)),
                       sha256=hashlib.sha256(raw.tobytes()).hexdigest()[:16])
    if name == "smoke":
        # tiny, fully local, reproducible: this repo's own committed text
        here = os.path.dirname(os.path.abspath(__file__))
        buf = b""
        for fn in sorted(os.listdir(here)):
            if fn.endswith((".py", ".md")):
                buf += open(os.path.join(here, fn), "rb").read()
        raw = np.frombuffer((buf * 4)[:cfg.get("n_bytes", 200000)], dtype=np.uint8)
        src = dict(dataset="smoke", path=here, n_bytes=int(len(raw)),
                   sha256=hashlib.sha256(raw.tobytes()).hexdigest()[:16])
    n = len(raw)
    a, b = int(n * 0.90), int(n * 0.95)
    return raw[:a], raw[a:b], raw[b:], src


def batches(arr, B, T, seed, n_steps):
    """Non-overlapping (input, target) byte windows; target is the next byte."""
    rng = np.random.RandomState(seed)
    hi = len(arr) - T - 2
    for _ in range(n_steps):
        i = rng.randint(0, hi, size=B)
        x = np.stack([arr[j:j + T] for j in i]).astype(np.int32)
        y = np.stack([arr[j + 1:j + T + 1] for j in i]).astype(np.int32)
        yield x, y


def fixed_eval_set(arr, B, T, seed, n_batches):
    return list(batches(arr, B, T, seed, n_batches))
