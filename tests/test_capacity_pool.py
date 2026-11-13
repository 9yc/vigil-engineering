import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from capacity_study import Allocation, CapacityPool
from capacity_study.replay import compare_case


ROOT = Path(__file__).resolve().parents[1]


class CapacityPoolTests(unittest.TestCase):
    def test_contiguous_coalesces_adjacent_ranges(self):
        pool = CapacityPool(12)
        a, b, c = [pool.allocate(4) for _ in range(3)]
        pool.free(a)
        pool.free(c)
        self.assertIsNone(pool.allocate(5))
        self.assertEqual(pool.stats()["failures_by_reason"]["fragmentation"], 1)
        pool.free(b)
        whole = pool.allocate(12)
        self.assertEqual(whole.segments, ((0, 12),))
        pool.check_invariants()

    def test_blocks_can_use_separated_ranges(self):
        pool = CapacityPool(12, mode="blocks", block_size=2)
        a, b, c = [pool.allocate(4) for _ in range(3)]
        pool.free(a)
        pool.free(c)
        result = pool.allocate(7)
        self.assertEqual(result.reserved_units, 8)
        self.assertEqual(result.segments, ((0, 2), (2, 2), (8, 2), (10, 2)))
        pool.check_invariants()
        pool.free(b)
        pool.free(result)
        self.assertEqual(pool.stats()["free_units"], 12)

    def test_fixed_slot_limit_and_tail_accounting(self):
        pool = CapacityPool(11, mode="fixed", block_size=4)
        self.assertEqual(pool.stats()["usable_units"], 8)
        self.assertEqual(pool.stats()["unallocatable_tail_units"], 3)
        self.assertIsNone(pool.allocate(5))
        self.assertEqual(pool.stats()["failures_by_reason"]["object_limit"], 1)
        self.assertEqual(pool.allocate(1).reserved_units, 4)
        pool.check_invariants()

    def test_rounding_failure_does_not_partially_allocate(self):
        pool = CapacityPool(12, mode="blocks", block_size=4)
        first = pool.allocate(5)
        self.assertIsNone(pool.allocate(5))
        self.assertEqual(pool.stats()["reserved_units"], 8)
        self.assertEqual(pool.stats()["failures_by_reason"]["rounding"], 1)
        pool.free(first)
        self.assertIsNotNone(pool.allocate(12))
        pool.check_invariants()

    def test_foreign_forged_and_repeated_frees_are_rejected(self):
        pool, other = CapacityPool(12), CapacityPool(12)
        allocation = pool.allocate(4)
        forged = Allocation(4, 4, ((0, 4),))
        for handle in (forged, other.allocate(4), None, []):
            with self.subTest(handle=handle), self.assertRaises(ValueError):
                pool.free(handle)
        pool.check_invariants()
        pool.free(allocation)
        with self.assertRaises(ValueError):
            pool.free(allocation)
        pool.check_invariants()

    def test_invalid_requests_do_not_change_counters(self):
        pool = CapacityPool(12)
        for units in (0, -1, True, 1.5, "2"):
            with self.subTest(units=units), self.assertRaises(ValueError):
                pool.allocate(units)
        self.assertEqual(pool.stats()["requests"], 0)
        for kwargs in ({"budget_units": 0}, {"budget_units": 5, "block_size": 0},
                       {"budget_units": 5, "mode": "unknown"},
                       {"budget_units": 5, "mode": "contiguous", "block_size": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CapacityPool(**kwargs)

    def test_stats_are_independent_snapshots(self):
        pool = CapacityPool(4)
        snapshot = pool.stats()
        snapshot["failures_by_reason"]["capacity"] = 99
        self.assertIsNone(pool.allocate(5))
        self.assertEqual(pool.stats()["failures_by_reason"]["capacity"], 1)
        pool.check_invariants()

    def test_randomized_contiguous_against_bitmap_reference(self):
        rng = random.Random(7)
        pool = CapacityPool(29)
        bitmap = [False] * 29
        active = []
        for _ in range(400):
            if active and rng.random() < 0.4:
                allocation = active.pop(rng.randrange(len(active)))
                for offset, length in allocation.segments:
                    bitmap[offset:offset + length] = [False] * length
                pool.free(allocation)
            else:
                units = rng.randint(1, 12)
                expected = next((offset for offset in range(30 - units)
                                 if not any(bitmap[offset:offset + units])), None)
                allocation = pool.allocate(units)
                if expected is None:
                    self.assertIsNone(allocation)
                else:
                    self.assertEqual(allocation.segments, ((expected, units),))
                    bitmap[expected:expected + units] = [True] * units
                    active.append(allocation)
            pool.check_invariants()
            self.assertEqual(pool.stats()["reserved_units"], sum(bitmap))


class CapacityReplayTests(unittest.TestCase):
    def test_shared_budget_cases_and_negative_control(self):
        data = json.loads((ROOT / "examples/capacity_events.json").read_text())
        expected = {
            "many_small_objects": (4, 6, 6),
            "objects_larger_than_one_slot": (0, 2, 2),
            "separated_free_ranges": (3, 4, 3),
            "rounding_negative_control": (4, 2, 4),
        }
        for case in data["cases"]:
            result = compare_case(case)
            allocators = result["allocators"]
            with self.subTest(case=case["name"]):
                self.assertEqual(tuple(item["accepted"] for item in allocators.values()),
                                 expected[case["name"]])
                self.assertTrue(all(item["budget_units"] == case["budget_units"]
                                    for item in allocators.values()))
                if case["name"] == "separated_free_ranges":
                    self.assertTrue(all(item["live_allocations"] == 0
                                        for item in allocators.values()))

    def test_invalid_lifetimes_are_rejected(self):
        base = {"name": "example", "budget_units": 8, "slot_size": 4, "block_size": 2}
        allocate = {"op": "allocate", "id": "a", "units": 1}
        for events in ([{"op": "free", "id": "missing"}], [allocate, allocate],
                       [{"op": "unknown", "id": "a"}],
                       [{"op": "allocate", "id": "", "units": 1}]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                compare_case(dict(base, events=events))

    def test_cli_creates_output_once(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            output = Path(directory) / "capacity-results.json"
            command = [sys.executable, "-B", "-m", "capacity_study.replay",
                       "--output", str(output)]
            first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                   timeout=10)
            self.assertEqual(first.returncode, 0, first.stderr)
            text = output.read_text()
            self.assertEqual(json.loads(text)["data_kind"], "synthetic")
            self.assertEqual(len(json.loads(text)["cases"]), 4)
            second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                    timeout=10)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_text(), text)
