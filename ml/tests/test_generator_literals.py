"""
Every simulation constant of the cohort generator lives in ml/simulation/assumptions.yaml.

The generator source may contain only:
- 0 and 1;
- integer subscripts (indices), round() digit counts, `parents[...]` and `str.split(..., 1)`;
- the named non-simulation constants in NAMED_ALLOWLIST below (numerical solver settings, an
  overflow guard, a unit conversion, the month-step divisor of the trend, and the CLI defaults).
Any other numeric literal fails this test with its line number.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GENERATOR = Path(__file__).resolve().parents[2] / "ml" / "data_pipeline" / "generate_synthetic_indian.py"

# (value, context) -> reason. Context is the enclosing assignment target, call or function name.
NAMED_ALLOWLIST: Dict[Tuple[float, str], str] = {
    (20.0, "solve_intercept_for_base_rate"): "brentq search bracket for the intercept",
    (50.0, "solve_intercept_for_base_rate"): "widened brentq bracket when the first one fails",
    (1e-6, "solve_intercept_for_base_rate"): "brentq tolerance",
    (25.0, "solve_intercept_for_base_rate"): "logit clip guarding exp() overflow",
    (25.0, "dropout_probability"): "logit clip guarding exp() overflow",
    (100.0, "att_core1"): "latent attendance factor (fraction) to percent",
    (100.0, "att_core2"): "latent attendance factor (fraction) to percent",
    (100.0, "att_lab"): "latent attendance factor (fraction) to percent",
    (100.0, "att_elective"): "latent attendance factor (fraction) to percent",
    (100.0, "attendance_month_1"): "latent attendance factor (fraction) to percent",
    (100.0, "attendance_month_2"): "latent attendance factor (fraction) to percent",
    (100.0, "attendance_month_3"): "latent attendance factor (fraction) to percent",
    (100.0, "logger.info"): "dropout rate printed as a percent",
    (2.0, "attendance_3m_trend"): "two month-steps from month 1 to month 3",
    (2000, "n_students"): "CLI / signature default cohort size",
    (42, "seed"): "CLI / signature default seed",
}


def _parents(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else ""
        return f"{base}.{func.attr}" if base else func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _contexts(node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> List[str]:
    """Enclosing contexts, innermost first: keyword/parameter, assignment target, logger call, function."""
    found: List[str] = []
    cur = node
    while cur in parents:
        parent = parents[cur]
        if isinstance(parent, ast.keyword) and parent.arg:
            found.append(parent.arg)
        if isinstance(parent, ast.arguments):  # a parameter default: name it by its parameter
            params = parent.args[len(parent.args) - len(parent.defaults):]
            found += [param.arg for param, default in zip(params, parent.defaults) if default is cur]
        if isinstance(parent, (ast.Assign, ast.AnnAssign)):
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            found += [t.id for t in targets if isinstance(t, ast.Name)]
        if isinstance(parent, ast.Call) and _call_name(parent) == "logger.info":
            found.append("logger.info")
        if isinstance(parent, ast.FunctionDef):
            found.append(parent.name)
        cur = parent
    return found


def _structurally_allowed(node: ast.Constant, parents: Dict[ast.AST, ast.AST]) -> bool:
    if node.value in (0, 1):
        return True
    parent = parents.get(node)
    if isinstance(parent, ast.Subscript) and parent.slice is node and isinstance(node.value, int):
        return True  # index, including parents[2]
    if isinstance(parent, ast.Call):
        name = _call_name(parent)
        if name.endswith("round") and len(parent.args) > 1 and parent.args[1] is node:
            return True  # round() digits
        if name.endswith("split") and len(parent.args) > 1 and parent.args[1] is node:
            return True  # maxsplit
    return False


def find_disallowed_literals(source: str) -> List[Tuple[int, float, Optional[str]]]:
    tree = ast.parse(source)
    parents = _parents(tree)
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            continue
        if _structurally_allowed(node, parents):
            continue
        contexts = _contexts(node, parents)
        if any((node.value, ctx) in NAMED_ALLOWLIST for ctx in contexts):
            continue
        bad.append((node.lineno, node.value, contexts[0] if contexts else None))
    return sorted(bad)


def test_generator_has_no_unlisted_numeric_literals():
    bad = find_disallowed_literals(GENERATOR.read_text(encoding="utf-8"))
    assert not bad, (
        "Numeric literals in the generator must live in ml/simulation/assumptions.yaml "
        "(line, value, context): " + ", ".join(f"{GENERATOR.name}:{ln} {v!r} in {ctx}" for ln, v, ctx in bad)
    )


def test_scanner_flags_a_new_constant_and_allows_structural_ones():
    source = (
        "def generate(rng, assumptions, n_students=2000, seed=42):\n"
        "    noise = rng.normal(0, 3.7, size=n_students)\n"          # new constant -> flagged
        "    first = draws[0][1]\n"                                 # indices -> allowed
        "    value = round(noise.mean(), 2)\n"                      # round digits -> allowed
        "    att_core1 = factor * 100.0\n"                          # named allowlist -> allowed
        "    attendance_month_1 = factor * 100.0 + 4.0\n"           # 4.0 not listed -> flagged
    )
    assert [(ln, v) for ln, v, _ in find_disallowed_literals(source)] == [(2, 3.7), (6, 4.0)]
