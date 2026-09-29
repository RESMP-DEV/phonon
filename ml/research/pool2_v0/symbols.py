"""pool2 step 1b: public class/function names from installed packages on gpubox."""
from __future__ import annotations
import json, os, pkgutil
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUT = Path("/data/phonon_pool2_v0/raw/symbols.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

TORCH_SUBS = [
    "torch.nn", "torch.nn.functional", "torch.nn.init", "torch.nn.utils", "torch.cuda",
    "torch.cuda.amp", "torch.distributed", "torch.distributed.tensor", "torch.optim",
    "torch.optim.lr_scheduler", "torch.linalg", "torch.fft", "torch.special", "torch.sparse",
    "torch.autograd", "torch.autograd.functional", "torch.utils.data", "torch.utils.checkpoint",
    "torch.jit", "torch.profiler", "torch.amp", "torch.export", "torch.compiler", "torch.func",
    "torch.nested", "torch.random", "torch.testing", "torch.masked", "torch.signal",
    "torch.backends.cuda", "torch.backends.cudnn", "torch.multiprocessing",
]
NUMPY_SUBS = ["numpy.linalg", "numpy.random", "numpy.fft", "numpy.char", "numpy.ma",
              "numpy.polynomial", "numpy.testing", "numpy.strings"]
TOP = ["torch", "transformers", "numpy", "peft", "datasets", "jiwer", "trl", "accelerate",
       "safetensors", "tokenizers", "huggingface_hub", "scipy", "sklearn", "mlx_lm"]

rows: list[dict] = []
seen: set[str] = set()


def emit(name: str, source: str, mod: str, is_module: bool = False) -> None:
    if not name or name.startswith("_") or len(name) < 3 or len(name) > 60:
        return
    key = name.lower()
    if key in seen:
        return
    seen.add(key)
    rows.append({"term": name, "source": source, "module": mod,
                 "sym_kind": "module" if is_module else "symbol"})


def walk(modname: str, source: str) -> int:
    try:
        m = __import__(modname, fromlist=["*"])
    except Exception as exc:
        print(f"  skip {modname}: {type(exc).__name__}: {exc}", flush=True)
        return 0
    n0 = len(rows)
    names = getattr(m, "__all__", None) or dir(m)
    for nm in names:
        if not isinstance(nm, str):
            continue
        emit(nm, source, modname)
    # documented submodules one level down (names only)
    p = getattr(m, "__path__", None)
    if p is not None:
        try:
            for mi in pkgutil.iter_modules(p):
                if not mi.name.startswith("_"):
                    emit(mi.name, source, modname, is_module=True)
        except Exception:
            pass
    print(f"  {modname}: +{len(rows)-n0}", flush=True)
    return len(rows) - n0


for mod in TOP:
    walk(mod, f"pkg:{mod.split('.')[0]}")
for mod in TORCH_SUBS:
    walk(mod, "pkg:torch")
for mod in NUMPY_SUBS:
    walk(mod, "pkg:numpy")

# transformers model class names: the lazy top-level already exposes them, but walk
# transformers.models submodule names too (they are spoken as module names).
try:
    import transformers.models as TM
    for mi in pkgutil.iter_modules(TM.__path__):
        if not mi.name.startswith("_"):
            emit(mi.name, "pkg:transformers", "transformers.models", is_module=True)
    print(f"  transformers.models: total {len(rows)}", flush=True)
except Exception as exc:
    print(f"  skip transformers.models: {exc}", flush=True)

with OUT.open("w", encoding="utf-8") as h:
    for r in rows:
        h.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"SYMBOLS done rows={len(rows)} -> {OUT}", flush=True)
