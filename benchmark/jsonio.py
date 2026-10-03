"""JSON for the benchmark files: deterministic, diff-friendly, exact.

Floats are written with Python's shortest round-trip representation, so a
value read back is bit-identical to the one written. Lists of plain values
go on one line; dicts and nested lists are indented by one space per level.
NaN and infinity are refused. Standard library only, so the OpenMC side
(Python 3.13) and the mcslab side (Python 3.9) produce identical text.
"""
import hashlib
import json


def _plain(v):
    """A JSON scalar from a Python or numpy scalar."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    # numpy scalars, without importing numpy
    kind = type(v).__name__
    if kind.startswith("bool"):
        return bool(v)
    if kind.startswith(("int", "uint")):
        return int(v)
    if kind.startswith("float"):
        return float(v)
    raise TypeError(f"not a JSON scalar: {v!r} ({kind})")


def _encode(obj, level):
    pad = " " * level
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{pad} {json.dumps(str(k))}: {_encode(v, level + 1)}"
                 for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, (list, tuple)) or type(obj).__name__ == "ndarray":
        seq = obj.tolist() if type(obj).__name__ == "ndarray" else list(obj)
        if all(not isinstance(v, (dict, list, tuple)) for v in seq):
            return "[" + ", ".join(json.dumps(_plain(v), allow_nan=False) for v in seq) + "]"
        items = [f"{pad} {_encode(v, level + 1)}" for v in seq]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(_plain(obj), allow_nan=False)


def dumps(obj) -> str:
    return _encode(obj, 0) + "\n"


def write(obj, path) -> str:
    """Write obj to path; returns the text's sha256."""
    text = dumps(obj)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
