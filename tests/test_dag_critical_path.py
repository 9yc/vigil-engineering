import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class TestDagCriticalPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.binary = artifact_dir / "dag_critical_path"
        if not cls.binary.is_file():
            raise RuntimeError(f"Missing compiled binary at {cls.binary}")
        cls.tmp_root = Path(os.environ["TMPDIR"])
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def _run_dag(self, tasks, workers=4, output_path=None):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td:
            dag_file = Path(td) / "dag.json"
            dag_file.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
            cmd = [
                str(self.binary),
                "--input",
                str(dag_file),
                "--workers",
                str(workers),
            ]
            if output_path is not None:
                cmd.extend(["--output", str(output_path)])
            return subprocess.run(cmd, capture_output=True, text=True, check=False)

    @staticmethod
    def _verify_schedule_timeline_invariants(tasks, schedule_entry, tc: unittest.TestCase) -> None:
        deps_map = {t["id"]: list(t.get("deps", [])) for t in tasks}
        cost_map = {t["id"]: float(t["cost"]) for t in tasks}
        intervals_by_task = {}
        worker_slices = {w: [] for w in range(schedule_entry["workers"])}

        for sl in schedule_entry["timeline"]:
            tid = sl["task_id"]
            wid = sl["worker_id"]
            st = sl["start_time"]
            fn = sl["finish_time"]
            tc.assertAlmostEqual(fn - st, cost_map[tid], places=6)
            intervals_by_task[tid] = (st, fn)
            if fn > st:
                worker_slices[wid].append((st, fn))

        tc.assertEqual(set(intervals_by_task.keys()), set(cost_map.keys()))

        # Verify precedence constraints: start(u) >= finish(dep) for every edge dep -> u.
        for tid, deps in deps_map.items():
            st, _ = intervals_by_task[tid]
            for dep in deps:
                _, dep_fn = intervals_by_task[dep]
                tc.assertGreaterEqual(st + 1e-9, dep_fn)

        # Verify worker exclusivity: no two positive-duration tasks overlap on the same lane.
        for wid, slices in worker_slices.items():
            slices.sort()
            for idx in range(1, len(slices)):
                tc.assertLessEqual(
                    slices[idx - 1][1], slices[idx][0] + 1e-9, f"Overlap on worker {wid}"
                )

    def test_diamond_dag_cpm_slack_and_critical_path(self) -> None:
        tasks = [
            {"id": "A", "cost": 10, "deps": []},
            {"id": "B", "cost": 25, "deps": ["A"]},
            {"id": "C", "cost": 8, "deps": ["A"]},
            {"id": "D", "cost": 5, "deps": ["C"]},
            {"id": "E", "cost": 12, "deps": ["B", "D"]},
        ]
        res = self._run_dag(tasks, workers=3)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        rep = json.loads(res.stdout)

        self.assertAlmostEqual(rep["total_work"], 60.0, places=6)
        self.assertAlmostEqual(rep["critical_path_cost"], 47.0, places=6)
        self.assertEqual(rep["critical_path"], ["A", "B", "E"])

        slack_by_id = {row["id"]: row for row in rep["slack_table"]}
        for cid in ("A", "B", "E"):
            self.assertTrue(slack_by_id[cid]["critical"])
            self.assertAlmostEqual(slack_by_id[cid]["total_slack"], 0.0, places=6)
        for nid in ("C", "D"):
            self.assertFalse(slack_by_id[nid]["critical"])
            self.assertAlmostEqual(slack_by_id[nid]["total_slack"], 12.0, places=6)

        self.assertAlmostEqual(slack_by_id["C"]["earliest_start"], 10.0, places=6)
        self.assertAlmostEqual(slack_by_id["C"]["latest_start"], 22.0, places=6)
        self.assertAlmostEqual(slack_by_id["D"]["latest_finish"], 35.0, places=6)

        for sched in rep["schedules"]:
            self._verify_schedule_timeline_invariants(tasks, sched, self)

    def test_parallel_and_chain_worker_scaling_bounds(self) -> None:
        tasks = [{"id": "root", "cost": 4, "deps": []}]
        branch_ids = []
        for i in range(6):
            bid = f"p{i}"
            branch_ids.append(bid)
            tasks.append({"id": bid, "cost": 10, "deps": ["root"]})
        tasks.append({"id": "sink", "cost": 6, "deps": branch_ids})

        res = self._run_dag(tasks, workers=6)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        rep = json.loads(res.stdout)
        self.assertAlmostEqual(rep["critical_path_cost"], 20.0, places=6)
        self.assertAlmostEqual(rep["total_work"], 70.0, places=6)

        schedules = {s["workers"]: s for s in rep["schedules"]}
        self.assertAlmostEqual(schedules[1]["makespan"], 70.0, places=6)
        self.assertAlmostEqual(schedules[2]["makespan"], 40.0, places=6)
        self.assertAlmostEqual(schedules[3]["makespan"], 30.0, places=6)
        self.assertAlmostEqual(schedules[6]["makespan"], 20.0, places=6)

        for k, sched in schedules.items():
            lb = max(20.0, 70.0 / k)
            self.assertTrue(sched["bounds_satisfied"])
            self.assertGreaterEqual(sched["makespan"] + 1e-9, lb)
            self.assertGreater(sched["average_utilization"], 0.0)
            self.assertLessEqual(sched["average_utilization"], 1.0 + 1e-9)
            self._verify_schedule_timeline_invariants(tasks, sched, self)

        chain = [
            {"id": "s0", "cost": 7, "deps": []},
            {"id": "s1", "cost": 9, "deps": ["s0"]},
            {"id": "s2", "cost": 14, "deps": ["s1"]},
        ]
        chain_rep = json.loads(self._run_dag(chain, workers=4).stdout)
        for sched in chain_rep["schedules"]:
            self.assertAlmostEqual(sched["makespan"], 30.0, places=6)

    def test_zero_cost_barriers_and_equal_critical_path_tiebreak(self) -> None:
        # Two tied critical branches (b_fast_lex vs c_slow_lex) through a 0-cost barrier node.
        tasks = [
            {"id": "init", "cost": 5, "deps": []},
            {"id": "barrier", "cost": 0, "deps": ["init"]},
            {"id": "branch_b", "cost": 15, "deps": ["barrier"]},
            {"id": "branch_a", "cost": 15, "deps": ["barrier"]},
            {"id": "side", "cost": 6, "deps": ["barrier"]},
            {"id": "commit", "cost": 4, "deps": ["branch_a", "branch_b", "side"]},
        ]
        res = self._run_dag(tasks, workers=3)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        rep = json.loads(res.stdout)
        self.assertAlmostEqual(rep["critical_path_cost"], 24.0, places=6)
        # Lexicographical tie-break must deterministically select branch_a over branch_b.
        self.assertEqual(rep["critical_path"], ["init", "barrier", "branch_a", "commit"])
        for sched in rep["schedules"]:
            self._verify_schedule_timeline_invariants(tasks, sched, self)

    def test_minimal_cycle_witness_detection(self) -> None:
        cyclic_tasks = [
            {"id": "entry", "cost": 3, "deps": []},
            {"id": "n1", "cost": 5, "deps": ["entry", "n4"]},
            {"id": "n2", "cost": 5, "deps": ["n1"]},
            {"id": "n3", "cost": 5, "deps": ["n2"]},
            {"id": "n4", "cost": 5, "deps": ["n3"]},
            {"id": "x1", "cost": 2, "deps": ["x2"]},
            {"id": "x2", "cost": 2, "deps": ["x1"]},
        ]
        res = self._run_dag(cyclic_tasks, workers=2)
        self.assertEqual(res.returncode, 2)
        self.assertIn("CYCLE_DETECTED", res.stderr)
        self.assertIn("x1 -> x2 -> x1", res.stderr)

        self_loop = [{"id": "z0", "cost": 1, "deps": ["z0"]}]
        res_self = self._run_dag(self_loop, workers=1)
        self.assertEqual(res_self.returncode, 2)
        self.assertIn("z0 -> z0", res_self.stderr)

    def test_missing_dependency_and_exclusive_output_guard(self) -> None:
        bad_deps = [{"id": "alpha", "cost": 4, "deps": ["ghost_node"]}]
        res_missing = self._run_dag(bad_deps, workers=2)
        self.assertEqual(res_missing.returncode, 2)
        self.assertIn("MISSING_DEPENDENCY", res_missing.stderr)
        self.assertIn("ghost_node", res_missing.stderr)

        valid_tasks = [
            {"id": "t0", "cost": 5, "deps": []},
            {"id": "t1", "cost": 8, "deps": ["t0"]},
        ]
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td:
            out_file = Path(td) / "dag_out.json"
            first = self._run_dag(valid_tasks, workers=2, output_path=out_file)
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            self.assertTrue(json.loads(out_file.read_text(encoding="utf-8"))["valid"])

            second = self._run_dag(valid_tasks, workers=2, output_path=out_file)
            self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
