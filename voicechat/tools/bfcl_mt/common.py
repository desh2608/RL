"""Shared helpers for BFCL-style multi-turn training data: NEW tasks on the gorilla v1.3 multi-turn backends
(data/bfcl-src/gorilla-bfcl-v1.3, Apache-2.0). The BFCL test items are read only for the contamination filter.

Task record (tasks.jsonl, written by gen_tasks.py):
  {"tid": "g000123", "involved_classes": [...], "initial_config": {...},
   "turns": [{"user": str, "calls": ["cd(folder='x')", ...], "results": [str, ...]}, ...],
   "miss_param": {"turn": t, "vague": str, "followup": str, "value": str} | absent}
Entry / answer pairs in BFCL test format (make_variants.py) feed rollout.py, which runs the official FC handler loop
and multi_turn_checker. Stdlib + openai + mpmath only.
"""
import copy
import inspect
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
GORILLA = os.path.join(PROJECT, "data", "bfcl-src", "gorilla-bfcl-v1.3")
sys.path.insert(0, GORILLA)
sys.path.append(os.path.join(HERE, "shims"))  # used only when the real package is missing

from bfcl_eval.constants.category_mapping import MULTI_TURN_FUNC_DOC_FILE_MAPPING  # noqa: E402
from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as MTU  # noqa: E402
from bfcl_eval.utils import load_file  # noqa: E402

CLASSES = list(MULTI_TURN_FUNC_DOC_FILE_MAPPING)
STATELESS = set(MTU.STATELESS_CLASSES)
DATA = os.path.join(GORILLA, "bfcl_eval", "data")
MT_CATS = ("base", "miss_func", "miss_param", "long_context")
# BFCL v3 multi_turn_base: class combos and turn counts (2026-10-11 count); new tasks follow the same mix.
COMBOS = [(("TradingBot",), 20), (("VehicleControlAPI",), 19), (("TicketAPI", "TravelAPI"), 15),
          (("TwitterAPI", "VehicleControlAPI"), 14), (("GorillaFileSystem",), 13), (("MessageAPI", "TradingBot"), 13),
          (("TravelAPI",), 13), (("GorillaFileSystem", "TwitterAPI"), 12), (("GorillaFileSystem", "MathAPI"), 12),
          (("TravelAPI", "TwitterAPI"), 10), (("MessageAPI", "TravelAPI"), 10), (("MessageAPI", "VehicleControlAPI"), 9),
          (("TicketAPI", "TradingBot"), 9), (("GorillaFileSystem", "MessageAPI"), 8), (("MathAPI", "VehicleControlAPI"), 6),
          (("GorillaFileSystem", "TicketAPI"), 5), (("MathAPI", "TradingBot"), 5), (("TradingBot", "TwitterAPI"), 3),
          (("TicketAPI", "VehicleControlAPI"), 2), (("MathAPI", "TravelAPI"), 2)]
TURNS = [(2, 38), (3, 48), (4, 51), (5, 44), (6, 14)]


def func_docs(cls):
    return copy.deepcopy(load_file(MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls]))


def class_schema_source(cls):
    """DEFAULT_STATE + _load_scenario source of a backend class: the state schema without any test-set state."""
    import importlib
    mod = importlib.import_module(MTU.CLASS_FILE_PATH_MAPPING[cls])
    out = []
    if hasattr(mod, "DEFAULT_STATE"):
        out.append("DEFAULT_STATE = " + json.dumps(getattr(mod, "DEFAULT_STATE"), default=str)[:3000])
    klass = getattr(mod, cls)
    if hasattr(klass, "_load_scenario"):
        out.append(inspect.getsource(klass._load_scenario))
    return "\n".join(out)


def _instance_name(model_name, entry_id, cls):
    # same spelling as multi_turn_utils.execute_multi_turn_func_call
    return "%s_%s_%s_instance" % (model_name.replace("-", "_").replace(".", "_").replace("/", "_"), entry_id, cls.lower())


def reset_instances(model_name, entry_id, classes):
    """Drop the module-global backend instances gorilla keeps per (model, entry id, class)."""
    ns = vars(MTU)
    for c in classes:
        for suffix in ("", "_eval"):
            ns.pop(_instance_name(model_name + suffix, entry_id, c), None)


def is_safe_call(call):
    """gorilla eval()s call strings: allow only bare_name(literal args / kwargs) (no attributes, no expressions)."""
    import ast
    try:
        node = ast.parse(str(call).strip(), mode="eval").body
    except Exception:
        return False
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):  # BFCL calls are bare names
        return False
    try:
        for x in list(node.args) + [k.value for k in node.keywords]:
            ast.literal_eval(x)
    except Exception:
        return False
    return all(k.arg for k in node.keywords)


_ORIG_EXEC = MTU.execute_multi_turn_func_call


def safe_execute_multi_turn_func_call(func_call_list, initial_config, involved_classes, model_name, test_entry_id,
                                      long_context=False, is_evaL_run=False):
    if all(is_safe_call(c) for c in func_call_list):
        return _ORIG_EXEC(func_call_list, initial_config, involved_classes, model_name, test_entry_id,
                          long_context=long_context, is_evaL_run=is_evaL_run)
    results = []
    for c in func_call_list:
        if is_safe_call(c):
            r, _ = _ORIG_EXEC([c], initial_config, involved_classes, model_name, test_entry_id,
                              long_context=long_context, is_evaL_run=is_evaL_run)
            results.extend(r)
        else:
            results.append("Error during execution: call rejected (only literal arguments are allowed)")
    _, inst = _ORIG_EXEC([], initial_config, involved_classes, model_name, test_entry_id,
                         long_context=long_context, is_evaL_run=is_evaL_run)
    return results, inst


def install_safe_exec(*modules):
    """Point gorilla modules that imported execute_multi_turn_func_call by name at the guarded version."""
    for m in (MTU,) + modules:
        if hasattr(m, "execute_multi_turn_func_call"):
            m.execute_multi_turn_func_call = safe_execute_multi_turn_func_call


install_safe_exec()


def run_calls(calls, initial_config, classes, model_name, entry_id, long_context=False):
    """Execute python-call strings on the (persistent) instances for (model_name, entry_id); returns result strings."""
    results, _ = MTU.execute_multi_turn_func_call(calls, initial_config, classes, model_name, entry_id,
                                                  long_context=long_context, is_evaL_run=False)
    return [str(r) for r in results]


def is_error(result):
    r = str(result)
    if r.startswith("Error during execution"):
        return True
    try:
        d = json.loads(r)
    except Exception:
        try:
            import ast
            d = ast.literal_eval(r)
        except Exception:
            return False
    return isinstance(d, dict) and "error" in d


CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def call_name(call):
    m = CALL_RE.match(call)
    return m.group(1) if m else None


def call_kwargs(call):
    """Keyword argument values of one python-call string (literal values only)."""
    import ast
    try:
        node = ast.parse(call.strip(), mode="eval").body
        return {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
    except Exception:
        return {}


def test_questions():
    """All BFCL v3 multi-turn test user messages (contamination filter only)."""
    out = []
    for cat in MT_CATS:
        for line in open(os.path.join(DATA, "BFCL_v3_multi_turn_%s.json" % cat)):
            for turn in json.loads(line)["question"]:
                for m in turn:
                    if m.get("role") == "user" and m.get("content"):
                        out.append(m["content"])
    return sorted(set(out))


def shingles(text, n=3):
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


class ContaminationIndex:
    """Max word-3-gram Jaccard of a text against all BFCL multi-turn test user messages."""

    def __init__(self):
        self.items = [shingles(q) for q in test_questions()]
        self.inv = {}
        for i, s in enumerate(self.items):
            for g in s:
                self.inv.setdefault(g, []).append(i)

    def max_sim(self, text):
        s = shingles(text)
        cand = {}
        for g in s:
            for i in self.inv.get(g, ()):
                cand[i] = cand.get(i, 0) + 1
        best = 0.0
        for i, inter in cand.items():
            best = max(best, inter / float(len(s | self.items[i]) or 1))
        return best


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
