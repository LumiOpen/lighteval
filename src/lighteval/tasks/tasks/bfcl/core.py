"""BFCL non-live grading core — pure logic, no inspect_ai / lighteval deps.

Kept dependency-light so it can be unit-tested with a plain interpreter. It reuses
BFCL's real ``ast_checker`` from the ``bfcl-eval`` package and vendors BFCL's
Python AST decoder (stdlib ``ast`` only — BFCL's ``model_handler.utils`` cannot be
imported without pulling in tenacity + tree_sitter).

Requires BFCL's ``ast_checker`` from the ``bfcl_eval`` package. Note that
``bfcl-eval`` pins ``numpy==1.26.4``, which conflicts with lighteval's ``numpy>=2``,
so it usually cannot be pip-installed into the same environment. The recommended
path is to point ``BFCL_EVAL_ROOT`` at a gorilla
``berkeley-function-call-leaderboard`` checkout, which is prepended to ``sys.path``.
"""

import ast
import json
import os
import sys
import types


# bfcl_eval is imported lazily (see _load_bfcl). Importing it at module-import time
# would break registry loading of ALL tasks when bfcl-eval is not installed, so the
# dependency is only resolved when a BFCL task actually runs.
_BFCL = None


def _load_bfcl():
    """Lazily import BFCL's ast_checker/enums/prompt templates (memoized)."""
    global _BFCL
    if _BFCL is not None:
        return _BFCL

    # Optional: point at a gorilla source checkout instead of the installed package.
    bfcl_root = os.environ.get("BFCL_EVAL_ROOT")
    if bfcl_root and bfcl_root not in sys.path:
        sys.path.insert(0, bfcl_root)

    # ``bfcl_eval.constants.model_config`` eagerly imports every API handler (which we
    # don't need). ``ast_checker`` only touches it via ``convert_func_name`` for the
    # ``underscore_to_dot`` flag, so we stub it: any model resolves to
    # ``underscore_to_dot=False`` and function names pass through unchanged.
    if "bfcl_eval.constants.model_config" not in sys.modules:

        class _AnyModelCfg(dict):
            def __getitem__(self, key):
                return types.SimpleNamespace(underscore_to_dot=False)

            def __contains__(self, key):
                return True

        stub = types.ModuleType("bfcl_eval.constants.model_config")
        stub.MODEL_CONFIG_MAPPING = _AnyModelCfg()
        sys.modules["bfcl_eval.constants.model_config"] = stub

    try:
        from bfcl_eval.constants.default_prompts import (
            OUTPUT_FORMAT_MAPPING,
            PARAM_TYPE_MAPPING,
            PROMPT_STYLE_TEMPLATES,
            PROMPT_TEMPLATE_MAPPING,
        )
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    except ImportError as exc:
        raise ImportError(
            "The BFCL tasks require BFCL's ast_checker from the 'bfcl_eval' package. "
            "bfcl-eval pins numpy==1.26.4, which conflicts with lighteval's numpy>=2, so "
            "it usually cannot be pip-installed into the same environment. Point "
            "BFCL_EVAL_ROOT at a gorilla 'berkeley-function-call-leaderboard' checkout "
            "(prepended to sys.path), or install bfcl-eval in an environment whose numpy "
            "it can satisfy."
        ) from exc

    _BFCL = {
        "ast_checker": ast_checker,
        "Language": Language,
        # Per-category language passed to ast_checker for correct type coercion.
        "category_language": {
            "simple_java": Language.JAVA,
            "simple_javascript": Language.JAVASCRIPT,
        },
        "OUTPUT_FORMAT_MAPPING": OUTPUT_FORMAT_MAPPING,
        "PARAM_TYPE_MAPPING": PARAM_TYPE_MAPPING,
        "PROMPT_STYLE_TEMPLATES": PROMPT_STYLE_TEMPLATES,
        "PROMPT_TEMPLATE_MAPPING": PROMPT_TEMPLATE_MAPPING,
    }
    return _BFCL


# --------------------------------------------------------------------------
# Vendored BFCL Python AST decoder (bfcl_eval/model_handler/utils.py).
# --------------------------------------------------------------------------
def _resolve_ast_by_type(value):  # noqa: C901 - vendored verbatim from BFCL
    if isinstance(value, ast.Constant):
        return "..." if value.value is Ellipsis else value.value
    if isinstance(value, ast.UnaryOp):
        return -value.operand.value
    if isinstance(value, ast.List):
        return [_resolve_ast_by_type(v) for v in value.elts]
    if isinstance(value, ast.Dict):
        return {_resolve_ast_by_type(k): _resolve_ast_by_type(v) for k, v in zip(value.keys, value.values)}
    if isinstance(value, ast.NameConstant):
        return value.value
    if isinstance(value, ast.BinOp):
        return eval(ast.unparse(value))
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        return ast.unparse(value) if len(value.keywords) == 0 else _resolve_ast_call(value)
    if isinstance(value, ast.Tuple):
        return tuple(_resolve_ast_by_type(v) for v in value.elts)
    if isinstance(value, ast.Ellipsis):
        return "..."
    if isinstance(value, ast.Subscript):
        return ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    raise Exception(f"Unsupported AST type: {type(value)}")


def _resolve_ast_call(elem):
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {arg.arg: _resolve_ast_by_type(arg.value) for arg in elem.keywords}
    return {func_name: args_dict}


def decode_python_ast(result):
    """BFCL default_decode_ast_prompting (Python branch) -> list[{func: {args}}]."""
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    cleaned = result.strip().strip("'")
    parsed = ast.parse(cleaned, mode="eval")
    extracted = []
    if isinstance(parsed.body, ast.Call):
        extracted.append(_resolve_ast_call(parsed.body))
    else:
        for elem in parsed.body.elts:
            assert isinstance(elem, ast.Call)
            extracted.append(_resolve_ast_call(elem))
    return extracted


def decode_robust(text):
    """Decode a model response into BFCL call dicts, tolerating reasoning
    prefixes/suffixes (e.g. "Reasoning trace:\\n[call]", "Response:\\n[call]").

    Tries BFCL's default decode on the whole string first (canonical behavior),
    then falls back to the bracketed "[ ... ]" call span. Only accepts python
    function-call syntax (JSON-style ``[{"name": ...}]`` is rejected by the Call
    assertion), preserving parity with BFCL's python return format.
    """
    if not text:
        return None
    try:
        return decode_python_ast(text)
    except Exception:
        pass
    close = text.rfind("]")
    for open_idx in (i for i, ch in enumerate(text) if ch == "["):
        if close <= open_idx:
            break
        try:
            calls = decode_python_ast(text[open_idx : close + 1])
            if calls:
                return calls
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------
# BFCL default python-classic system prompt (formulate_system_prompt for
# "ret_fmt=python&tool_call_tag=False&func_doc_fmt=json&prompt_fmt=plaintext&style=classic").
# --------------------------------------------------------------------------
def build_system_prompt(functions):
    bfcl = _load_bfcl()
    style = bfcl["PROMPT_STYLE_TEMPLATES"]["classic"]
    template = bfcl["PROMPT_TEMPLATE_MAPPING"]["plaintext"]
    tool_call_format = style["tool_call_no_tag"].format(
        output_format=bfcl["OUTPUT_FORMAT_MAPPING"]["python"],
        param_types=bfcl["PARAM_TYPE_MAPPING"]["python"],
    )
    available_tools = style["available_tools"].format(format="json", functions=json.dumps(functions, indent=4))
    return template.format(
        persona=style["persona"],
        task=style["task"],
        tool_call_format=tool_call_format,
        multiturn_behavior=style["multiturn_behavior"],
        available_tools=available_tools,
    )


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------
def grade(text, category, function, ground_truth):
    """Return ``(score in {0.0, 1.0}, explanation)``."""
    if text and "</think>" in text:
        text = text.split("</think>")[-1]

    decoded = decode_robust(text or "")

    if category == "irrelevance":
        ok = not decoded
        return (1.0 if ok else 0.0, f"irrelevance: emitted_call={bool(decoded)}")

    if decoded is None:
        return (0.0, "decode_failed")
    if not ground_truth:
        return (0.0, "missing_ground_truth")

    bfcl = _load_bfcl()
    language = bfcl["category_language"].get(category, bfcl["Language"].PYTHON)
    try:
        result = bfcl["ast_checker"](function, decoded, ground_truth, language, category, "bfcl")
    except Exception as exc:
        return (0.0, f"checker_error:{exc}")
    ok = bool(result.get("valid"))
    return (1.0 if ok else 0.0, "" if ok else str(result.get("error", ""))[:200])
