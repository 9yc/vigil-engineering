"""Parse JSON responses and validate caller-supplied function argument schemas.

The supported descriptors are type, required and enum. This is a structural
validator, not a complete JSON Schema implementation or an execution engine.
"""

import json
import math


def _reject_constant(value):
    raise ValueError(f"nonstandard JSON constant: {value}")


def parse_output(raw):
    """Return (value, error, used_fallback); fallback finds the first JSON array."""
    try:
        value = json.loads(raw.strip(), parse_constant=_reject_constant)
        return value, None, False
    except (json.JSONDecodeError, ValueError) as first:
        start = raw.find("[")
        if start >= 0:
            try:
                decoder = json.JSONDecoder(parse_constant=_reject_constant)
                value, _ = decoder.raw_decode(raw[start:])
                return value, None, True
            except (json.JSONDecodeError, ValueError):
                pass
        return None, str(first), False


def _type_ok(value, declared):
    if declared in ("string", "string-enum"):
        return isinstance(value, str)
    if declared == "bool":
        return type(value) is bool
    if declared == "int":
        return type(value) is int
    if declared == "float":
        return type(value) is int or (type(value) is float and math.isfinite(value))
    if declared == "dictionary":
        return isinstance(value, dict)
    if declared == "list":
        return isinstance(value, list)
    return False


def validate_calls(value, specs):
    """Validate [{name, parameters}, ...] against specs[name][parameter] descriptors."""
    errors = []
    if not isinstance(value, list):
        return ["response is not a JSON array"]
    for index, call in enumerate(value):
        location = f"calls[{index}]"
        if not isinstance(call, dict) or set(call) != {"name", "parameters"}:
            errors.append(location + " must have exactly name and parameters")
            continue
        name, params = call["name"], call["parameters"]
        if not isinstance(name, str) or name not in specs:
            errors.append(location + ".name is unknown")
            continue
        if not isinstance(params, dict):
            errors.append(location + ".parameters is not an object")
            continue
        schema = specs[name]
        for key, descriptor in schema.items():
            if descriptor.get("required", False) and key not in params:
                errors.append(location + f".parameters.{key} required")
        for key, argument in params.items():
            if key not in schema:
                errors.append(location + f".parameters.{key} unknown")
                continue
            descriptor = schema[key]
            if not _type_ok(argument, descriptor["type"]):
                errors.append(location + f".parameters.{key} type mismatch")
            allowed = descriptor.get("enum")
            if allowed is not None and argument not in allowed:
                errors.append(location + f".parameters.{key} not in enum")
    return errors
