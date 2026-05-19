import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


N_SATZILLA_FEATURES = 48
SATFEATPY_DIR = os.environ.get("LANGSAT_SATFEATPY_DIR", "").strip()
FEATURE_CACHE_DIR = os.environ.get("LANGSAT_FEATURE_CACHE_DIR", "").strip()
SATFEATPY_FULL_LOCAL_SEARCH = os.environ.get("LANGSAT_SATFEATPY_FULL_LOCAL_SEARCH", "1") == "1"
_BACKEND_NOTICE_PRINTED: set[str] = set()
BACKEND_USAGE = {"satfeatpy": 0, "cache": 0}
CACHE_VERSION = "satfeat-v2"


SATZILLA_FEATURE_ORDER = [
    "c",
    "v",
    "clauses_vars_ratio",
    "vcg_var_mean",
    "vcg_var_coeff",
    "vcg_var_min",
    "vcg_var_max",
    "vcg_var_entropy",
    "vcg_clause_mean",
    "vcg_clause_coeff",
    "vcg_clause_min",
    "vcg_clause_max",
    "vcg_clause_entropy",
    "vg_mean",
    "vg_coeff",
    "vg_min",
    "vg_max",
    "pnc_ratio_mean",
    "pnc_ratio_coeff",
    "pnc_ratio_entropy",
    "pnv_ratio_mean",
    "pnv_ratio_coeff",
    "pnv_ratio_min",
    "pnv_ratio_max",
    "pnv_ratio_entropy",
    "binary_ratio",
    "ternary+",
    "ternary_ratio",
    "hc_fraction",
    "hc_var_mean",
    "hc_var_coeff",
    "hc_var_min",
    "hc_var_max",
    "hc_var_entropy",
    "unit_props_at_depth_1",
    "unit_props_at_depth_4",
    "unit_props_at_depth_16",
    "unit_props_at_depth_64",
    "unit_props_at_depth_256",
    "mean_depth_to_contradiction_over_vars",
    "estimate_log_number_nodes_over_vars",
    "saps_BestSolution_Mean",
    "saps_FirstLocalMinStep_Median",
    "saps_FirstLocalMinStep_Q.10",
    "saps_FirstLocalMinStep_Q.90",
    "saps_BestAvgImprovement_Mean",
    "saps_FirstLocalMinRatio_Mean",
    "saps_EstACL_Mean",
]
LOCAL_SEARCH_FEATURES = SATZILLA_FEATURE_ORDER[41:]


def extract_sat_features(filepath: str, n_features: int = N_SATZILLA_FEATURES) -> np.ndarray:
    cached = _read_cache(filepath, "satfeatpy", n_features)
    if cached is not None:
        return cached

    arr = _normalize(_extract_with_satfeatpy(filepath, n_features), n_features)
    BACKEND_USAGE["satfeatpy"] += 1
    _notice_once("satfeatpy", "[Features] Using SATfeatPy/SATzilla-style global features.")
    _write_cache(filepath, "satfeatpy", n_features, arr)
    return arr


def _extract_with_satfeatpy(filepath: str, n_features: int) -> np.ndarray:
    satfeat_root = os.environ.get("LANGSAT_SATFEATPY_DIR", SATFEATPY_DIR).strip()
    if not satfeat_root:
        raise RuntimeError("LANGSAT_SATFEATPY_DIR is not set")

    satfeat_dir = Path(satfeat_root)
    if not satfeat_dir.exists():
        raise FileNotFoundError(f"SATfeatPy directory not found: {satfeat_dir}")

    satfeat_path = str(satfeat_dir)
    if satfeat_path not in sys.path:
        sys.path.insert(0, satfeat_path)

    from sat_instance.sat_instance import SATInstance as SATFeatInstance

    normalized_path = _normalized_dimacs_copy(filepath)
    cwd = os.getcwd()
    try:
        os.chdir(satfeat_path)
        sat = SATFeatInstance(normalized_path, preprocess=False)
        if getattr(sat, "solved", False):
            return np.zeros(n_features, dtype=np.float32)

        sat.gen_basic_features()
        sat.gen_dpll_probing_features()
        if SATFEATPY_FULL_LOCAL_SEARCH:
            try:
                sat.gen_local_search_probing_features()
            except Exception as exc:
                raise RuntimeError(
                    "SATfeatPy full local-search probing failed. Strict paper "
                    "reproduction needs the SATzilla local-search features "
                    f"{LOCAL_SEARCH_FEATURES}; install/configure ubcsat or set "
                    "LANGSAT_SATFEATPY_FULL_LOCAL_SEARCH=0 for a partial-feature "
                    "diagnostic run."
                ) from exc

        features = sat.features_dict
        if SATFEATPY_FULL_LOCAL_SEARCH:
            missing = [name for name in SATZILLA_FEATURE_ORDER if name not in features]
            if missing:
                raise RuntimeError(
                    "SATfeatPy did not return the full 48 SATzilla feature set. "
                    f"Missing: {missing}"
                )
        values = [_safe_feature_value(features.get(name, 0.0)) for name in SATZILLA_FEATURE_ORDER]
        return np.array(values, dtype=np.float32)
    finally:
        os.chdir(cwd)
        try:
            os.remove(normalized_path)
        except Exception:
            pass


def _normalized_dimacs_copy(filepath: str) -> str:
    """SATfeatPy is strict about whitespace in DIMACS headers."""
    fd, out_path = tempfile.mkstemp(prefix="langsat_satfeat_", suffix=".cnf")
    with os.fdopen(fd, "w", encoding="utf-8") as out, open(filepath, encoding="utf-8") as src:
        for raw in src:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "c":
                continue
            if parts[0] == "p" and len(parts) >= 4:
                out.write(f"p cnf {int(parts[2])} {int(parts[3])}\n")
                continue
            out.write(" ".join(parts) + "\n")
    return out_path


def _normalize(arr: np.ndarray, n_features: int) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    if len(arr) >= n_features:
        arr = arr[:n_features]
    else:
        arr = np.pad(arr, (0, n_features - len(arr)))
    arr = np.clip(arr, -1e6, 1e6)
    scale = np.max(np.abs(arr))
    if scale > 0:
        arr = arr / (scale + 1e-8)
    return arr.astype(np.float32)


def _read_cache(filepath: str, source: str, n_features: int) -> np.ndarray | None:
    cache_path = _cache_path(filepath, source, n_features)
    if not cache_path or not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text())
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != CACHE_VERSION or payload.get("source") != source:
            return None
        BACKEND_USAGE["cache"] += 1
        return _normalize(np.array(payload["features"], dtype=np.float32), n_features)
    except Exception:
        return None


def _write_cache(filepath: str, source: str, n_features: int, arr: np.ndarray):
    cache_path = _cache_path(filepath, source, n_features)
    if not cache_path:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "source": source,
            "features": arr.astype(float).tolist(),
        }
        cache_path.write_text(json.dumps(payload))
    except Exception:
        pass


def _cache_path(filepath: str, source: str, n_features: int) -> Path | None:
    if not FEATURE_CACHE_DIR:
        return None
    resolved = str(Path(filepath).resolve())
    key = hashlib.sha1(
        f"{CACHE_VERSION}|{resolved}|{source}|{n_features}|{SATFEATPY_FULL_LOCAL_SEARCH}".encode()
    ).hexdigest()
    return Path(FEATURE_CACHE_DIR) / f"{key}.json"


def _safe_feature_value(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value


def _notice_once(key: str, message: str):
    if key not in _BACKEND_NOTICE_PRINTED:
        print(message)
        _BACKEND_NOTICE_PRINTED.add(key)
