"""Read CUTEst names without PyCUTEst's fixed-width allocation assumption.

CUTEst 2.7.1's public C bridge writes 11 bytes (ten characters plus NUL)
per name. PyCUTEst 1.8.2 allocates ten, so invoking its native name methods
can corrupt memory. This module calls only the public ``cutest_cint_*``
bridge of the *already initialized* extension, with eleven-byte storage.
It does not call raw Fortran routines, rebuild modules, or change caches.
"""
from __future__ import annotations

import ctypes
import importlib.machinery
from pathlib import Path
import threading


class CUTEstNamesError(RuntimeError):
    pass


_LOCK = threading.RLock()
_HANDLES = {}
_CINT_SYMBOLS = {
    "varnames": ("cutest_cint_varnames_", "cutest_cint_varnames"),
    "connames": ("cutest_cint_connames_", "cutest_cint_connames"),
}


def _native_library(prob, kind):
    wrapper = getattr(prob, "_module", None)
    if wrapper is None:
        raise CUTEstNamesError("Native CUTEst module unavailable; supply names as a list instead of calling an unchecked native wrapper")
    provider = getattr(wrapper, kind, None)
    native = getattr(provider, "__self__", None)
    filename = getattr(native, "__file__", None)
    if filename is None:
        # Some wrappers expose the compiled module itself rather than reexports.
        filename = getattr(wrapper, "__file__", None)
    if not isinstance(filename, str) or not any(filename.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
        raise CUTEstNamesError("Cannot identify the initialized CUTEst extension for " + kind)
    path = Path(filename).resolve()
    if not path.is_file():
        raise CUTEstNamesError("CUTEst extension no longer exists: " + str(path))
    key = str(path)
    if key not in _HANDLES:
        # PyDLL retains the GIL while the non-thread-safe initialized CUTEst
        # state is read. Loading this exact path reuses the existing handle.
        _HANDLES[key] = ctypes.PyDLL(key)
    return _HANDLES[key], key


def _decode_names(raw, count, stride):
    names = []
    for index in range(count):
        field = raw[index * stride:(index + 1) * stride]
        if stride == 11:
            if field[-1:] != b"\0":
                raise CUTEstNamesError("Missing NUL terminator in CUTEst C name bridge")
            field = field[:-1]
        if b"\0" in field:
            raise CUTEstNamesError("Unexpected embedded NUL in CUTEst name")
        try:
            name = field.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise CUTEstNamesError("CUTEst returned a non-ASCII name") from exc
        if not name or any(ord(c) < 32 or ord(c) > 126 for c in name):
            raise CUTEstNamesError("CUTEst returned an empty or invalid name")
        names.append(name)
    return names


def _read_c_names(library, count, kind):
    symbol = next((name for name in _CINT_SYMBOLS[kind] if hasattr(library, name)), None)
    if symbol is None:
        raise CUTEstNamesError("Public CUTEst C name bridge missing; raw Fortran name routines are deliberately not called")
    function = getattr(library, symbol)
    function.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]
    function.restype = None
    # 32-bit integers are the supported PyCUTEst C ABI. Deliberately do not
    # guess integer-64 or hidden Fortran string-length calling conventions.
    if count < 0 or count > 10_000_000:
        raise CUTEstNamesError("CUTEst name count outside supported range")
    size = 11 * count
    sentinel = b"\xa5"
    buffer = ctypes.create_string_buffer(sentinel * (size + 64), size + 64)
    status, dimension = ctypes.c_int(-1234567), ctypes.c_int(count)
    function(ctypes.byref(status), ctypes.byref(dimension), ctypes.cast(buffer, ctypes.c_void_p))
    if status.value != 0 or dimension.value != count:
        raise CUTEstNamesError("CUTEst name bridge failed: status=" + str(status.value))
    raw = buffer.raw
    if raw[size:] != sentinel * 64:
        raise CUTEstNamesError("Unsupported CUTEst name width: guard bytes changed")
    if all(raw[11 * index + 10] == 0 for index in range(count)):
        stride = 11
    elif raw[10 * count:] == sentinel * (count + 64):
        stride = 10
    else:
        raise CUTEstNamesError("Unsupported CUTEst C name layout; expected stride 10 or 11")
    return _decode_names(raw, count, stride), {"symbol": symbol, "stride": stride, "native_count": count}


def safe_names(prob, kind="varnames", *, return_info=False):
    """Return stripped names in the current PyCUTEst variable/constraint order.

    Static Python lists/tuples are supported for independently verified model
    providers and tests. Callable native wrappers are never invoked. Fixed
    variables are removed using PyCUTEst's explicit ``idx_free`` mapping.
    """
    if kind not in _CINT_SYMBOLS:
        raise ValueError("kind must be varnames or connames")
    count = int(prob.n if kind == "varnames" else prob.m)
    if count == 0:
        result = ([], {"provider": "empty", "native_count": 0})
        return result if return_info else result[0]
    value = getattr(prob, kind, None)
    if isinstance(value, (list, tuple)):
        names = [str(name).strip() for name in value]
        if len(names) != count or any(not name for name in names):
            raise CUTEstNamesError("Invalid static CUTEst names")
        result = (names, {"provider": "static", "native_count": count})
        return result if return_info else result[0]
    full_count = int(getattr(prob, "n_full", count)) if kind == "varnames" else count
    with _LOCK:
        library, filename = _native_library(prob, kind)
        names, info = _read_c_names(library, full_count, kind)
    info.update(provider="public_c_bridge", extension=filename)
    if kind == "varnames" and full_count != count:
        indices = getattr(prob, "idx_free", None)
        if indices is None or len(indices) != count:
            raise CUTEstNamesError("Fixed-variable mapping missing or inconsistent")
        indices = [int(index) for index in indices]
        if len(set(indices)) != count or any(index < 0 or index >= full_count for index in indices):
            raise CUTEstNamesError("Invalid free-variable mapping")
        names = [names[index] for index in indices]
        info["fixed_variables_removed"] = full_count - count
    if len(names) != count:
        raise CUTEstNamesError("Returned CUTEst name count differs from problem dimension")
    return (names, info) if return_info else names


def varnames(prob):
    return safe_names(prob, "varnames")


def connames(prob):
    return safe_names(prob, "connames")


safe_cutest_names = safe_names
