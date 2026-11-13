"""Replay the same allocation lifetimes at the same budget for three allocators."""

import argparse
import json
from pathlib import Path

from .pool import CapacityPool


def compare_case(case):
    pools = {
        "fixed_slots": CapacityPool(case["budget_units"], mode="fixed",
                                   block_size=case["slot_size"]),
        "block_pool": CapacityPool(case["budget_units"], mode="blocks",
                                  block_size=case["block_size"]),
        "first_fit": CapacityPool(case["budget_units"], mode="contiguous"),
    }
    active = {name: {} for name in pools}
    logical_live = set()
    for index, event in enumerate(case["events"]):
        op, key = event.get("op"), event.get("id")
        if not isinstance(key, str) or not key:
            raise ValueError(f"event {index}: id must be a nonempty string")
        if op == "allocate":
            if set(event) != {"op", "id", "units"}:
                raise ValueError(f"event {index}: unexpected allocation fields")
            if key in logical_live:
                raise ValueError(f"event {index}: duplicate live id")
            logical_live.add(key)
            for name, pool in pools.items():
                active[name][key] = pool.allocate(event["units"])
        elif op == "free":
            if set(event) != {"op", "id"}:
                raise ValueError(f"event {index}: unexpected free fields")
            if key not in logical_live:
                raise ValueError(f"event {index}: id has no active lifetime")
            logical_live.remove(key)
            for name, pool in pools.items():
                allocation = active[name].pop(key)
                # Every allocator sees the same lifetime end. A previously
                # rejected request has no allocation to return.
                if allocation is not None:
                    pool.free(allocation)
        else:
            raise ValueError(f"event {index}: unknown operation")
        for pool in pools.values():
            pool.check_invariants()
    return {
        "case": case["name"],
        "budget_units": case["budget_units"],
        "slot_size": case["slot_size"],
        "block_size": case["block_size"],
        "events": len(case["events"]),
        "allocators": {name: pool.stats() for name, pool in pools.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, nargs="?",
        default=Path(__file__).resolve().parents[1] / "examples/capacity_events.json",
    )
    parser.add_argument("--output", type=Path, help="new JSON file; default: stdout")
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        cases = data["cases"]
        names = [case["name"] for case in cases]
        if not cases or len(set(names)) != len(names):
            raise ValueError("expected nonempty cases with unique names")
        result = {
            "data_kind": data.get("data_kind", "user_supplied"),
            "scope": "Allocation accounting with identical budgets and request lifetimes; "
                     "no payload storage, metadata cost or execution timing is modeled.",
            "cases": [compare_case(case) for case in cases],
        }
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as target:
                target.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, f"replay error: {exc}\n")


if __name__ == "__main__":
    main()
