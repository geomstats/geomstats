"""DHOOM encoder/decoder — vendored copy of the reference Python implementation.

DHOOM (Davis Human-readable Optimized Object Markup) is a compact, human-readable
serialization format for structured data. It encodes the same data model as JSON
but exploits fiber-bundle decomposition of homogeneous collections to eliminate
structural redundancy. See https://dhoom.dev for the format specification.

This module is vendored from the reference Python implementation at
https://github.com/nurdymuny/dhoom (package: ``dhoom``).
It is intended as an internal implementation detail of
``geomstats.datasets.utils.load_dhoom`` and ``save_dhoom`` and is not part
of the geomstats public API. Future releases may switch to ``dhoom`` as an
optional dependency (``pip install geomstats[dhoom]``).

License
-------
DHOOM is released under the MIT License.

    MIT License

    Copyright (c) 2026 Bee Rosa Davis / Davis Geometric

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

References
----------
.. [1] DHOOM format specification, v0.5. https://dhoom.dev
.. [2] Davis, B. R. (2024). *The Geometry of Sameness*. Amazon KDP.
.. [3] Davis, B. R. (2026). *The Double Cover Principle*. Zenodo.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

JsonValue = Any  # str | int | float | bool | None | list | dict


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DhoomError(Exception):
    def __init__(self, message: str, line: int | None = None):
        self.line = line
        if line is not None:
            super().__init__(f"Line {line}: {message}")
        else:
            super().__init__(message)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Modifier:
    type: str
    start: JsonValue = None
    step: int | None = None
    default_value: JsonValue = None
    target: str | None = None
    pool: list[str] | None = None
    expr: str | None = None
    constraint: str | None = None


@dataclass
class FieldDecl:
    name: str
    modifier: Modifier | None = None


@dataclass
class Fiber:
    name: str | None = None
    fields: list[FieldDecl] = field(default_factory=list)
    sparse: bool = False


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------


def coerce(s: str) -> JsonValue:
    if s == "T":
        return True
    if s == "F":
        return False
    if s == "null":
        return None
    if s == "":
        return ""
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return s


def value_to_dhoom(v: JsonValue) -> str:
    if v is True:
        return "T"
    if v is False:
        return "F"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if any(c in v for c in (",", ":", "\n", '"')):
            return '"' + v.replace('"', '""') + '"'
        return v
    return ""


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

_STRING_PATTERN = re.compile(r"^(.*\D)(\d+)$")


def _parse_string_pattern(s: str) -> tuple[str, int, int] | None:
    m = _STRING_PATTERN.match(s)
    if not m:
        return None
    return m.group(1), int(m.group(2)), len(m.group(2))


def _arithmetic_value(start: JsonValue, step: int, i: int) -> JsonValue:
    if isinstance(start, (int, float)) and not isinstance(start, bool):
        return start + step * i
    if isinstance(start, str):
        pat = _parse_string_pattern(start)
        if pat:
            prefix, num, width = pat
            return prefix + str(num + step * i).zfill(width)
        return start
    return start


# ---------------------------------------------------------------------------
# Fiber parser
# ---------------------------------------------------------------------------


def _parse_field_decl(token: str) -> FieldDecl:
    arrow_idx = token.find("->")
    if arrow_idx != -1:
        name = token[:arrow_idx]
        target = token[arrow_idx + 2 :]
        return FieldDecl(name=name, modifier=Modifier(type="morphism", target=target))

    hash_idx = token.find("#")
    if hash_idx != -1:
        name = token[:hash_idx]
        expr = token[hash_idx + 1 :]
        return FieldDecl(name=name, modifier=Modifier(type="computed", expr=expr))

    bang_idx = token.find("!")
    if bang_idx != -1:
        name = token[:bang_idx]
        constraint = token[bang_idx + 1 :]
        return FieldDecl(
            name=name, modifier=Modifier(type="constraint", constraint=constraint)
        )

    if token.endswith("&"):
        return FieldDecl(name=token[:-1], modifier=Modifier(type="interned"))

    if token.endswith("^"):
        return FieldDecl(name=token[:-1], modifier=Modifier(type="delta"))

    if token.endswith(">"):
        return FieldDecl(name=token[:-1], modifier=Modifier(type="nested"))

    at_idx = token.find("@")
    if at_idx != -1:
        name = token[:at_idx]
        rest = token[at_idx + 1 :]
        plus_idx = rest.find("+")
        if plus_idx != -1:
            start = coerce(rest[:plus_idx])
            step = int(rest[plus_idx + 1 :])
            return FieldDecl(
                name=name, modifier=Modifier(type="arithmetic", start=start, step=step)
            )
        return FieldDecl(
            name=name, modifier=Modifier(type="arithmetic", start=coerce(rest))
        )

    pipe_idx = token.find("|")
    if pipe_idx != -1:
        name = token[:pipe_idx]
        default_value = coerce(token[pipe_idx + 1 :])
        return FieldDecl(
            name=name, modifier=Modifier(type="default", default_value=default_value)
        )

    return FieldDecl(name=token)


def parse_fiber(input_str: str) -> Fiber:
    s = input_str.strip()
    brace_start = s.find("{")
    brace_end = s.rfind("}")
    if brace_start == -1 or brace_end == -1:
        raise DhoomError("Missing braces in fiber header")

    raw_name = s[:brace_start].strip() or None
    sparse = False
    name = None

    if raw_name:
        if raw_name.startswith("~"):
            sparse = True
            stripped = raw_name[1:].strip()
            name = stripped or None
        else:
            name = raw_name

    fields = [
        _parse_field_decl(t.strip())
        for t in s[brace_start + 1 : brace_end].split(",")
        if t.strip()
    ]
    return Fiber(name=name, fields=fields, sparse=sparse)


# ---------------------------------------------------------------------------
# Record field splitter (respects quotes)
# ---------------------------------------------------------------------------


def _split_record_fields(line: str) -> list[str]:
    fields: list[str] = []
    current: list[str] = []
    in_quotes = False
    i = 0
    while i < len(line):
        c = line[i]
        if in_quotes:
            if c == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    current.append('"')
                    i += 1
                else:
                    in_quotes = False
            else:
                current.append(c)
        elif c == '"':
            in_quotes = True
        elif c == ",":
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(c)
        i += 1
    fields.append("".join(current).strip())
    return fields


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def _find_header_end(input_str: str) -> int:
    brace = input_str.find("}")
    if brace == -1:
        return -1
    colon = input_str.find(":", brace + 1)
    if colon == -1:
        return -1
    return colon + 1


def _record_fields(fiber: Fiber) -> list[FieldDecl]:
    return [
        f
        for f in fiber.fields
        if not (f.modifier and f.modifier.type in ("arithmetic", "computed"))
    ]


def _decode_flat_records(body: str, fiber: Fiber) -> list[JsonValue]:
    rec_fields = _record_fields(fiber)
    records: list[JsonValue] = []
    ordinal = 0
    delta_accum: dict[str, float] = {}

    for line in body.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue

        raw = _split_record_fields(trimmed)
        obj: dict[str, JsonValue] = {}

        for fd in fiber.fields:
            if fd.modifier and fd.modifier.type == "arithmetic":
                obj[fd.name] = _arithmetic_value(
                    fd.modifier.start, fd.modifier.step or 1, ordinal
                )

        for j, rf in enumerate(rec_fields):
            if j < len(raw):
                val = raw[j]
                if val == "":
                    resolved = (
                        rf.modifier.default_value
                        if rf.modifier and rf.modifier.type == "default"
                        else ""
                    )
                elif val.startswith(":"):
                    resolved = coerce(val[1:])
                else:
                    resolved = coerce(val)

                if (
                    rf.modifier
                    and rf.modifier.type == "delta"
                    and isinstance(resolved, (int, float))
                    and not isinstance(resolved, bool)
                ):
                    if ordinal == 0:
                        delta_accum[rf.name] = resolved
                    else:
                        prev = delta_accum.get(rf.name, 0)
                        resolved = prev + resolved
                        delta_accum[rf.name] = resolved

                obj[rf.name] = resolved
            else:
                if rf.modifier and rf.modifier.type == "default":
                    obj[rf.name] = rf.modifier.default_value

        records.append(obj)
        ordinal += 1

    return records


def _decode_sparse_records(body: str, fiber: Fiber) -> list[JsonValue]:
    records: list[JsonValue] = []
    ordinal = 0

    for line in body.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue

        obj: dict[str, JsonValue] = {}

        for fd in fiber.fields:
            if fd.modifier and fd.modifier.type == "arithmetic":
                obj[fd.name] = _arithmetic_value(
                    fd.modifier.start, fd.modifier.step or 1, ordinal
                )

        pairs = _split_record_fields(trimmed)
        for pair in pairs:
            colon_idx = pair.find(":")
            if colon_idx == -1:
                continue
            field_name = pair[:colon_idx].strip()
            raw_value = pair[colon_idx + 1 :].strip()
            obj[field_name] = coerce(raw_value)

        for fd in fiber.fields:
            if fd.name not in obj:
                if fd.modifier and fd.modifier.type == "default":
                    obj[fd.name] = fd.modifier.default_value
                elif not (fd.modifier and fd.modifier.type == "arithmetic"):
                    obj[fd.name] = None

        records.append(obj)
        ordinal += 1

    return records


def _decode_nested_records(body: str, fiber: Fiber) -> list[JsonValue]:
    rec_fields = _record_fields(fiber)
    records: list[JsonValue] = []
    lines = body.split("\n")
    line_idx = 0
    ordinal = 0

    while line_idx < len(lines):
        trimmed = lines[line_idx].strip()
        if not trimmed:
            line_idx += 1
            continue

        obj: dict[str, JsonValue] = {}

        for fd in fiber.fields:
            if fd.modifier and fd.modifier.type == "arithmetic":
                obj[fd.name] = _arithmetic_value(
                    fd.modifier.start, fd.modifier.step or 1, ordinal
                )

        raw = _split_record_fields(trimmed)
        nested_fields: list[FieldDecl] = []
        rf_idx = 0

        for rf in rec_fields:
            if rf.modifier and rf.modifier.type == "nested":
                nested_fields.append(rf)
            else:
                if rf_idx < len(raw):
                    val = raw[rf_idx]
                    if val == "":
                        obj[rf.name] = (
                            rf.modifier.default_value
                            if rf.modifier and rf.modifier.type == "default"
                            else ""
                        )
                    elif val.startswith(":"):
                        obj[rf.name] = coerce(val[1:])
                    else:
                        obj[rf.name] = coerce(val)
                elif rf.modifier and rf.modifier.type == "default":
                    obj[rf.name] = rf.modifier.default_value
                rf_idx += 1

        line_idx += 1

        for _nf in nested_fields:
            nested_text = ""
            while line_idx < len(lines):
                line = lines[line_idx]
                if (
                    line != ""
                    and not line.startswith(" ")
                    and not line.startswith("\t")
                    and nested_text != ""
                ):
                    break
                if line.strip() == "" and nested_text == "":
                    line_idx += 1
                    continue
                if "}:\n" in nested_text and line.strip().startswith("{"):
                    break
                nested_text += line.strip() + "\n"
                line_idx += 1

            if nested_text.strip():
                result = _decode_bundle(nested_text.strip())
                obj[_nf.name] = result["value"]

        records.append(obj)
        ordinal += 1

    return records


_POOL_RE = re.compile(r"^&(\w[\w-]*)?\[(.+)\]$")


def _decode_bundle(input_str: str) -> dict:
    header_end = _find_header_end(input_str)
    if header_end == -1:
        raise DhoomError("Missing '}:' header terminator")

    header = input_str[: header_end - 1].strip()
    body = input_str[header_end:]
    fiber = parse_fiber(header)

    body_lines = body.split("\n")
    remaining_lines: list[str] = []
    for line in body_lines:
        m = _POOL_RE.match(line.strip())
        if m:
            pool_field = m.group(1) or ""
            pool_values = [v.strip() for v in m.group(2).split(",")]
            for fd in fiber.fields:
                if (
                    fd.name == pool_field
                    and fd.modifier
                    and fd.modifier.type == "interned"
                ):
                    fd.modifier.pool = pool_values
        else:
            remaining_lines.append(line)
    body = "\n".join(remaining_lines)

    rec_fields = _record_fields(fiber)
    has_nested = any(f.modifier and f.modifier.type == "nested" for f in rec_fields)

    if fiber.sparse:
        records = _decode_sparse_records(body, fiber)
    elif has_nested:
        records = _decode_nested_records(body, fiber)
    else:
        records = _decode_flat_records(body, fiber)

    for fd in fiber.fields:
        if fd.modifier and fd.modifier.type == "interned" and fd.modifier.pool:
            pool = fd.modifier.pool
            for rec in records:
                if isinstance(rec, dict) and fd.name in rec:
                    val = rec[fd.name]
                    if (
                        isinstance(val, int)
                        and not isinstance(val, bool)
                        and 0 <= val < len(pool)
                    ):
                        rec[fd.name] = pool[val]

    for fd in fiber.fields:
        if fd.modifier and fd.modifier.type == "computed" and fd.modifier.expr:
            expr = fd.modifier.expr
            m = re.match(r"^(\w[\w-]*)\s*([+\-*])\s*(\w[\w-]*)$", expr)
            if m:
                left_name, op, right_name = m.group(1), m.group(2), m.group(3)
                for rec in records:
                    if isinstance(rec, dict):
                        left_val = rec.get(left_name)
                        right_val = rec.get(right_name)
                        if (
                            isinstance(left_val, (int, float))
                            and not isinstance(left_val, bool)
                            and isinstance(right_val, (int, float))
                            and not isinstance(right_val, bool)
                        ):
                            if op == "+":
                                rec[fd.name] = left_val + right_val
                            elif op == "-":
                                rec[fd.name] = left_val - right_val
                            elif op == "*":
                                rec[fd.name] = left_val * right_val

    return {"name": fiber.name, "value": records}


def decode(input_str: str) -> JsonValue:
    """Decode a DHOOM string into a Python value."""
    s = input_str.strip()
    if not s:
        return None

    result = _decode_bundle(s)
    if result["name"]:
        return {result["name"]: result["value"]}
    return result["value"]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _detect_arithmetic(values: list[JsonValue]) -> dict | None:
    if len(values) < 2:
        return None

    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        step = values[1] - values[0]
        if all(values[i] - values[i - 1] == step for i in range(1, len(values))):
            return {"start": values[0], "step": step}

    if all(isinstance(v, str) for v in values):
        patterns = [_parse_string_pattern(v) for v in values]
        if all(p is not None for p in patterns):
            if all(p[0] == patterns[0][0] and p[2] == patterns[0][2] for p in patterns):
                step = patterns[1][1] - patterns[0][1]
                if all(
                    patterns[i][1] - patterns[i - 1][1] == step
                    for i in range(1, len(patterns))
                ):
                    return {"start": values[0], "step": step}

    return None


def _find_modal_default(values: list[JsonValue]) -> dict | None:
    if not values:
        return None
    counts: dict[str, dict] = {}
    for v in values:
        key = json.dumps(v)
        if key in counts:
            counts[key]["count"] += 1
        else:
            counts[key] = {"value": v, "count": 1}
    best = max(counts.values(), key=lambda x: x["count"])
    return best


def _json_equal(a: JsonValue, b: JsonValue) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _detect_delta(values: list[JsonValue]) -> bool:
    if len(values) < 3:
        return False
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return False
    nums = values
    deltas = [nums[0]] + [nums[i] - nums[i - 1] for i in range(1, len(nums))]
    abs_len = sum(len(str(v)) for v in nums)
    delta_len = sum(len(str(d)) for d in deltas)
    return delta_len < abs_len * 0.7


def _detect_interned(values: list[JsonValue]) -> list[str] | None:
    if len(values) < 3:
        return None
    if not all(isinstance(v, str) for v in values):
        return None
    distinct = list(dict.fromkeys(values))
    if len(distinct) < 2 or len(distinct) > math.ceil(len(values) / 3):
        return None
    raw_len = sum(len(v) for v in values)
    pool_len = sum(len(v) for v in distinct) + len(distinct) - 1
    index_len = len(values)
    if pool_len + index_len >= raw_len * 0.9:
        return None
    return distinct


def _detect_computed(
    key: str, values: list[JsonValue], all_keys: list[str], records: list[dict]
) -> str | None:
    if not values or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    ):
        return None
    for op in ("+", "-", "*"):
        for a in all_keys:
            if a == key:
                continue
            for b in all_keys:
                if b == key:
                    continue
                match = True
                for r in records:
                    av = r.get(a)
                    bv = r.get(b)
                    if not (isinstance(av, (int, float)) and not isinstance(av, bool)):
                        match = False
                        break
                    if not (isinstance(bv, (int, float)) and not isinstance(bv, bool)):
                        match = False
                        break
                    if op == "+":
                        expected = av + bv
                    elif op == "-":
                        expected = av - bv
                    else:
                        expected = av * bv
                    if r.get(key) != expected:
                        match = False
                        break
                if match:
                    return f"{a}{op}{b}"
    return None


def _encode_bundle(name: str, records: list[dict], indent: int) -> str:
    prefix = " " * indent

    if not records:
        return f"{prefix}{name}{{}}:\n"

    keys = list(records[0].keys())
    ordered_fields: list[FieldDecl] = []
    arithmetic_keys: set[str] = set()
    delta_keys: set[str] = set()
    default_keys: dict[str, JsonValue] = {}
    nested_keys: set[str] = set()
    variable_keys: list[str] = []
    interned_keys: dict[str, list[str]] = {}
    computed_keys: dict[str, str] = {}

    remaining_keys: list[str] = []
    for key in keys:
        values = [r[key] for r in records]

        if all(isinstance(v, list) for v in values):
            nested_keys.add(key)
            continue

        arith = _detect_arithmetic(values)
        if arith:
            arithmetic_keys.add(key)
            step = arith["step"]
            ordered_fields.append(
                FieldDecl(
                    name=key,
                    modifier=Modifier(
                        type="arithmetic",
                        start=arith["start"],
                        step=step if step != 1 else None,
                    ),
                )
            )
            continue

        remaining_keys.append(key)

    computed_to_remove: list[str] = []
    for key in remaining_keys:
        values = [r[key] for r in records]
        expr = _detect_computed(key, values, remaining_keys, records)
        if expr:
            computed_keys[key] = expr
            computed_to_remove.append(key)
    for key in computed_to_remove:
        remaining_keys.remove(key)

    for key in remaining_keys:
        values = [r[key] for r in records]

        if _detect_delta(values):
            delta_keys.add(key)
            continue

        pool = _detect_interned(values)
        if pool is not None:
            interned_keys[key] = pool
            continue

        modal = _find_modal_default(values)
        if modal and modal["count"] > len(records) / 2:
            default_keys[key] = modal["value"]
            continue

        variable_keys.append(key)

    if not variable_keys and not delta_keys and not nested_keys and not interned_keys:
        for key in keys:
            if key in arithmetic_keys:
                arithmetic_keys.discard(key)
                ordered_fields = [f for f in ordered_fields if f.name != key]
                variable_keys.append(key)
                break
            if key in default_keys:
                del default_keys[key]
                variable_keys.append(key)
                break
            if key in computed_keys:
                del computed_keys[key]
                variable_keys.append(key)
                break

    for key, expr in computed_keys.items():
        ordered_fields.append(
            FieldDecl(name=key, modifier=Modifier(type="computed", expr=expr))
        )

    for key in delta_keys:
        ordered_fields.append(FieldDecl(name=key, modifier=Modifier(type="delta")))

    for key in interned_keys:
        ordered_fields.append(
            FieldDecl(
                name=key, modifier=Modifier(type="interned", pool=interned_keys[key])
            )
        )

    for key in variable_keys:
        ordered_fields.append(FieldDecl(name=key))

    default_entries = []
    for key, val in default_keys.items():
        count = sum(1 for r in records if _json_equal(r[key], val))
        default_entries.append((key, val, count))
    default_entries.sort(key=lambda x: -x[2])
    for key, val, _ in default_entries:
        ordered_fields.append(
            FieldDecl(name=key, modifier=Modifier(type="default", default_value=val))
        )

    for key in nested_keys:
        ordered_fields.append(FieldDecl(name=key, modifier=Modifier(type="nested")))

    non_arith_keys = [
        k
        for k in keys
        if k not in arithmetic_keys and k not in nested_keys and k not in computed_keys
    ]
    use_sparse = False
    if len(non_arith_keys) >= 8:
        null_count = 0
        total_cells = 0
        for r in records:
            for k in non_arith_keys:
                total_cells += 1
                v = r.get(k)
                if v is None or v == "":
                    null_count += 1
        use_sparse = null_count > total_cells * 0.75

    sparse_prefix = "~" if use_sparse else ""
    parts = []
    for fd in ordered_fields:
        s = fd.name
        if fd.modifier:
            if fd.modifier.type == "arithmetic":
                s += f"@{value_to_dhoom(fd.modifier.start)}"
                if fd.modifier.step is not None:
                    s += f"+{fd.modifier.step}"
            elif fd.modifier.type == "default":
                s += f"|{value_to_dhoom(fd.modifier.default_value)}"
            elif fd.modifier.type == "nested":
                s += ">"
            elif fd.modifier.type == "delta":
                s += "^"
            elif fd.modifier.type == "morphism":
                s += f"->{fd.modifier.target}"
            elif fd.modifier.type == "interned":
                s += "&"
            elif fd.modifier.type == "computed":
                s += f"#{fd.modifier.expr}"
            elif fd.modifier.type == "constraint":
                s += f"!{fd.modifier.constraint}"
        parts.append(s)

    out = f"{prefix}{sparse_prefix}{name}{{{', '.join(parts)}}}:\n"

    for key, pool in interned_keys.items():
        out += f"{prefix}&{key}[{', '.join(pool)}]\n"

    rec_fields = [
        f
        for f in ordered_fields
        if not (f.modifier and f.modifier.type in ("arithmetic", "computed"))
    ]

    if use_sparse:
        for record in records:
            pairs: list[str] = []
            for rf in rec_fields:
                if rf.modifier and rf.modifier.type == "nested":
                    continue
                val = record.get(rf.name)
                if rf.modifier and rf.modifier.type == "interned":
                    pool = rf.modifier.pool
                    if pool and isinstance(val, str) and val in pool:
                        idx = pool.index(val)
                        pairs.append(f"{rf.name}:{idx}")
                        continue
                if val is not None and val != "":
                    pairs.append(f"{rf.name}:{value_to_dhoom(val)}")
            if not pairs:
                first_field = next(
                    (
                        f
                        for f in rec_fields
                        if not (f.modifier and f.modifier.type == "nested")
                    ),
                    None,
                )
                if first_field:
                    pairs.append(f"{first_field.name}:null")
            out += f"{prefix}{', '.join(pairs)}\n"
        return out

    record_idx = 0
    prev_delta: dict[str, int | float] = {}

    for record in records:
        values: list[str] = []
        nested_bundles: list[tuple[str, list[dict]]] = []

        for rf in rec_fields:
            if rf.modifier and rf.modifier.type == "nested":
                v = record[rf.name]
                if isinstance(v, list):
                    nested_bundles.append(("", v))
                continue

            val = record[rf.name]

            if rf.modifier and rf.modifier.type == "delta":
                num_val = (
                    val
                    if isinstance(val, (int, float)) and not isinstance(val, bool)
                    else 0
                )
                if record_idx == 0:
                    prev_delta[rf.name] = num_val
                    values.append(value_to_dhoom(num_val))
                else:
                    prev = prev_delta.get(rf.name, 0)
                    delta = num_val - prev
                    prev_delta[rf.name] = num_val
                    values.append(value_to_dhoom(delta))
            elif rf.modifier and rf.modifier.type == "interned":
                pool = rf.modifier.pool
                if pool and isinstance(val, str) and val in pool:
                    values.append(str(pool.index(val)))
                else:
                    values.append(value_to_dhoom(val))
            elif rf.modifier and rf.modifier.type == "default":
                if _json_equal(val, rf.modifier.default_value):
                    values.append("")
                else:
                    values.append(f":{value_to_dhoom(val)}")
            else:
                values.append(value_to_dhoom(val))

        while values and values[-1] == "":
            values.pop()

        out += f"{prefix}{', '.join(values)}"

        if nested_bundles:
            out += ",\n"
            for nb_name, nb_records in nested_bundles:
                out += _encode_bundle(nb_name, nb_records, indent + 2)
        else:
            out += "\n"

        record_idx += 1

    return out


def encode(value: JsonValue) -> str:
    """Encode a Python value into DHOOM format."""
    if isinstance(value, dict):
        keys = list(value.keys())
        if len(keys) == 1:
            arr = value[keys[0]]
            if isinstance(arr, list):
                return _encode_bundle(keys[0], arr, 0)
        raise DhoomError("Top-level object must have exactly one key (the bundle name)")
    if isinstance(value, list):
        return _encode_bundle("data", value, 0)
    raise DhoomError("Top-level value must be an object or array")
