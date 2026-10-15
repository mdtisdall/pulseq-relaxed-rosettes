"""TOML helpers for configs with arrays of tables (the phantom, the requirements).

Each table is a frozen dataclass whose fields are strings, integers, floats,
tuples of three floats, or string enums. The fields' annotations are read as
strings (the modules use ``from __future__ import annotations``).
"""

from __future__ import annotations

import json
from dataclasses import fields
from enum import StrEnum
from typing import Any

from relaxed_rosettes.config import ConfigError


def plain_table(item: Any) -> dict[str, Any]:
    """Plain dict of a table dataclass (enums as strings, tuples as lists)."""
    out: dict[str, Any] = {}
    for f in fields(item):
        value = getattr(item, f.name)
        if isinstance(value, StrEnum):
            value = value.value
        elif isinstance(value, tuple):
            value = list(value)
        out[f.name] = value
    return out


def toml_value(value: Any) -> str:
    """TOML text of a string, number, or list of them."""
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def check_value(name: str, value: Any, kind: type) -> Any:
    """``value`` if it has type ``kind`` (an integer is accepted as a float)."""
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, kind) or isinstance(value, bool):
        raise ConfigError(f"{name} must be {kind.__name__} (got {value!r})")
    return value


def float_list(name: str, value: Any, length: int | None) -> list[float]:
    """A list of numbers, of ``length`` if given."""
    if not isinstance(value, list) or (length is not None and len(value) != length):
        size = "" if length is None else f" of {length}"
        raise ConfigError(f"{name} must be a list{size} numbers")
    return [float(check_value(name, v, float)) for v in value]


def table_item(key: str, cls: type, table: Any, enums: dict[str, type[StrEnum]]) -> Any:
    """An instance of table dataclass ``cls`` from one ``[[key]]`` table.

    ``enums`` maps the annotation of each enum field to its class.
    """
    if not isinstance(table, dict):
        raise ConfigError(f"[[{key}]] must be an array of tables")
    table = dict(table)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in table:
            continue
        value = table.pop(f.name)
        label = f"{key}.{f.name}"
        kind = str(f.type)
        if kind == "str":
            kwargs[f.name] = check_value(label, value, str)
        elif kind == "int":
            kwargs[f.name] = check_value(label, value, int)
        elif kind == "float":
            kwargs[f.name] = float(check_value(label, value, float))
        elif kind.startswith("tuple"):
            kwargs[f.name] = tuple(float_list(label, value, 3))
        else:
            enum = enums[kind]
            try:
                kwargs[f.name] = enum(check_value(label, value, str))
            except ValueError as e:
                options = ", ".join(m.value for m in enum)
                raise ConfigError(f"{label} must be one of {options} (got {value!r})") from e
    if table:
        raise ConfigError(f"unknown key(s) in [[{key}]]: {', '.join(sorted(table))}")
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"[[{key}]] is missing a required key ({e})") from e
