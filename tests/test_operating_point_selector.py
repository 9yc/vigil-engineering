"""Unit tests for Clopper-Pearson selector, betainc math, Pareto pruning, and splits."""

import json
import math
import os
import pathlib
import tempfile
import unittest

from tools.operating_point_selector import (
    MODE_EXP_DECAY,
    MODE_HARD_DEADLINE,
    SELECTOR_CLOPPER_PEARSON,
    SELECTOR_MIN_ENERGY,
    SELECTOR_MIN_EPT,
    SELECTOR_MIN_LATENCY,
    SELECTOR_POINT,
    CandidateStats,
    TrialObservation,
    betainc,
    betaincinv,
    build_pareto_frontier,
    clopper_pearson_lower_bound,
    compute_oracle_energy_gap_pct,
    evaluate_split_resampling,
    main,
    summarize_candidates,
    validate_trial_table,
)


def _exact_binomial_upper_tail(n: int, k: int, p: float) -> float:
    """Exact sum_{j=k}^n C(n, j) p^j (1-p)^(n-j) for cross-checking betainc."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for j in range(k, n + 1):
        total += math.comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j))
    return total


def _build_synthetic_trials(num_items: int = 60) -> list[TrialObservation]:
    """Construct 4 candidate configurations with distinct utility/energy trade-offs."""
    observations: list[TrialObservation] = []
    for idx in range(num_items):
        item_id = f"eval_{idx:03d}"
        # cfg_ultra: 100% timely pass, highest energy (12 J), fast latency (0.9 s).
        observations.append(
            TrialObservation(
                item_id=item_id,
                config_id="cfg_ultra",
                passed=True,
                latency_s=0.90,
                energy_j=12.0,
                output_tokens=20,
            )
        )
        # cfg_safe: 95% timely pass (57/60), moderate energy (7.5 J), timely (1.4 s <= 2.0 s).
        observations.append(
            TrialObservation(
                item_id=item_id,
                config_id="cfg_safe",
                passed=(idx >= 3),
                latency_s=1.40,
                energy_j=7.5,
                output_tokens=20,
            )
        )
        # cfg_borderline: ~81.7% timely pass (49/60), lower energy (5.0 J), occasionally over deadline.
        observations.append(
            TrialObservation(
                item_id=item_id,
                config_id="cfg_borderline",
                passed=(idx >= 6),
                latency_s=1.80 if idx < 55 else 2.50,
                energy_j=5.0,
                output_tokens=25,
            )
        )
        # cfg_dominated: worse energy AND worse latency AND worse utility than cfg_safe!
        observations.append(
            TrialObservation(
                item_id=item_id,
                config_id="cfg_dominated",
                passed=(idx >= 15),
                latency_s=1.90,
                energy_j=9.5,
                output_tokens=15,
            )
        )
    return observations


class TestOperatingPointSelector(unittest.TestCase):
    """Cross-check betainc, Clopper-Pearson bounds, Pareto frontier, and split oracle gaps."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.tmpdir = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_betainc_and_clopper_pearson_match_exact_binomial_tails(self) -> None:
        # Identity: I_p(k, n - k + 1) == sum_{j=k}^n C(n, j) p^j (1 - p)^(n - j).
        test_cases = [
            (10, 1, 0.15),
            (10, 5, 0.35),
            (20, 14, 0.70),
            (40, 36, 0.88),
            (50, 50, 0.92),
        ]
        for n, k, p in test_cases:
            exact_tail = _exact_binomial_upper_tail(n, k, p)
            beta_val = betainc(float(k), float(n - k + 1), p)
            self.assertAlmostEqual(beta_val, exact_tail, places=11)

        # Symmetry check: I_x(a, b) + I_{1-x}(b, a) == 1.
        for a, b, x in [(2.5, 7.3, 0.22), (8.1, 3.4, 0.79), (0.5, 0.5, 0.5)]:
            self.assertAlmostEqual(betainc(a, b, x) + betainc(b, a, 1.0 - x), 1.0, places=12)

        # Boundary checks.
        self.assertEqual(betainc(2.0, 3.0, 0.0), 0.0)
        self.assertEqual(betainc(2.0, 3.0, 1.0), 1.0)
        self.assertEqual(betaincinv(2.0, 3.0, 0.0), 0.0)
        self.assertEqual(betaincinv(2.0, 3.0, 1.0), 1.0)

        # Clopper-Pearson exact inversion: when p_L = clopper_pearson_lower_bound(k, n, alpha),
        # exact binomial upper tail P_{Bin(n, p_L)}(X >= k) MUST equal alpha!
        for n, k, alpha in [(25, 20, 0.05), (50, 45, 0.0125), (30, 30, 0.025), (15, 1, 0.01)]:
            p_lower = clopper_pearson_lower_bound(float(k), n, alpha)
            tail_at_bound = _exact_binomial_upper_tail(n, k, p_lower)
            self.assertAlmostEqual(tail_at_bound, alpha, places=10)
            self.assertLessEqual(p_lower, float(k) / float(n))

        # k = 0 yields 0.0 lower bound; k = n yields closed-form alpha^(1/n).
        self.assertEqual(clopper_pearson_lower_bound(0.0, 30, 0.05), 0.0)
        self.assertAlmostEqual(
            clopper_pearson_lower_bound(20.0, 20, 0.05), 0.05 ** (1.0 / 20.0), places=11
        )

        # Bonferroni simultaneous penalty (delta / M) is strictly more conservative than delta.
        lb_single = clopper_pearson_lower_bound(46.0, 50, 0.05)
        lb_simul = clopper_pearson_lower_bound(46.0, 50, 0.05 / 4.0)
        self.assertLess(lb_simul, lb_single)

    def test_pareto_dominance_pruning_2d_and_3d(self) -> None:
        observations = _build_synthetic_trials(60)
        item_ids, config_ids, by_config = validate_trial_table(observations)
        stats = summarize_candidates(
            by_config=by_config,
            item_ids=item_ids,
            config_ids=config_ids,
            utility_mode=MODE_HARD_DEADLINE,
            delta=0.05,
            deadline_s=2.0,
        )
        frontier_3d = build_pareto_frontier(stats, use_cp_bound=False, include_latency=True)
        frontier_ids = [c.config_id for c in frontier_3d]
        self.assertNotIn("cfg_dominated", frontier_ids)
        self.assertEqual(frontier_ids, ["cfg_borderline", "cfg_safe", "cfg_ultra"])

        # Verify 2D (energy, utility) pruning when a candidate has faster latency
        # yet strictly higher energy and strictly lower utility.
        custom_candidates = [
            CandidateStats("c_cheap_high", 50, 48.0, 0.96, 0.85, 0.025, 5.0, 1.8, 0.25, 0.96),
            CandidateStats("c_fast_low", 50, 40.0, 0.80, 0.65, 0.025, 8.0, 0.7, 0.40, 0.80),
        ]
        # In 3D (energy, latency, utility), c_fast_low is non-dominated due to latency 0.7 < 1.8.
        self.assertEqual(len(build_pareto_frontier(custom_candidates, include_latency=True)), 2)
        # In 2D (energy, utility), c_cheap_high strictly dominates c_fast_low.
        frontier_2d = build_pareto_frontier(custom_candidates, include_latency=False)
        self.assertEqual([c.config_id for c in frontier_2d], ["c_cheap_high"])

    def test_hard_deadline_vs_exp_decay_utility_modes(self) -> None:
        obs = [
            TrialObservation("i0", "c1", passed=True, latency_s=1.0, energy_j=4.0, output_tokens=10),
            TrialObservation("i1", "c1", passed=True, latency_s=3.0, energy_j=4.0, output_tokens=10),
            TrialObservation("i2", "c1", passed=False, latency_s=0.5, energy_j=4.0, output_tokens=10),
        ]
        items, configs, by_cfg = validate_trial_table(obs)
        hd_stats = summarize_candidates(
            by_cfg, items, configs, utility_mode=MODE_HARD_DEADLINE, delta=0.05, deadline_s=2.0
        )[0]
        # Only i0 passes within 2.0s -> 1 / 3.
        self.assertAlmostEqual(hd_stats.mean_utility, 1.0 / 3.0, places=12)

        exp_stats = summarize_candidates(
            by_cfg, items, configs, utility_mode=MODE_EXP_DECAY, delta=0.05, tau_s=2.0
        )[0]
        expected_u = (math.exp(-1.0 / 2.0) + math.exp(-3.0 / 2.0) + 0.0) / 3.0
        self.assertAlmostEqual(exp_stats.mean_utility, expected_u, places=12)
        self.assertLess(exp_stats.cp_lower_bound, exp_stats.mean_utility)

    def test_split_resampling_disjointness_and_oracle_gap_math(self) -> None:
        # Exact oracle gap formula check: 100 * (E_selected / E_oracle - 1).
        self.assertAlmostEqual(compute_oracle_energy_gap_pct(7.5, 5.0), 50.0, places=12)
        self.assertAlmostEqual(compute_oracle_energy_gap_pct(5.0, 5.0), 0.0, places=12)
        self.assertAlmostEqual(compute_oracle_energy_gap_pct(4.0, 5.0), -20.0, places=12)
        with self.assertRaises(ValueError):
            compute_oracle_energy_gap_pct(5.0, 0.0)

        observations = _build_synthetic_trials(60)
        # Target 0.85:
        # - cfg_borderline has true timely pass rate 49/60 = 0.8167 (< 0.85), energy 5.0 J.
        #   Point screening on small calibration splits occasionally overestimates cfg_borderline >= 0.85!
        # - Simultaneous Clopper-Pearson (delta=0.05, M=4 -> alpha=0.0125) conservatively rejects
        #   cfg_borderline and selects cfg_safe / cfg_ultra, achieving higher held-out attainment!
        report = evaluate_split_resampling(
            observations=observations,
            calibration_size=30,
            utility_mode=MODE_HARD_DEADLINE,
            target_utility=0.82,
            delta=0.05,
            deadline_s=2.0,
            num_splits=120,
            seed=17,
        )
        self.assertTrue(report.all_splits_disjoint)
        self.assertEqual(report.total_items, 60)
        self.assertEqual(report.calibration_size, 30)
        self.assertEqual(report.evaluation_size, 30)

        by_method = {m.selector_name: m for m in report.method_summaries}
        self.assertEqual(
            set(by_method.keys()),
            {
                SELECTOR_CLOPPER_PEARSON,
                SELECTOR_POINT,
                SELECTOR_MIN_LATENCY,
                SELECTOR_MIN_ENERGY,
                SELECTOR_MIN_EPT,
            },
        )
        cp_summary = by_method[SELECTOR_CLOPPER_PEARSON]
        pt_summary = by_method[SELECTOR_POINT]
        min_e_summary = by_method[SELECTOR_MIN_ENERGY]

        # Simultaneous Clopper-Pearson achieves 100% held-out target attainment, beating naive point screening.
        self.assertAlmostEqual(cp_summary.held_out_attainment_rate, 1.0, places=12)
        self.assertGreater(cp_summary.held_out_attainment_rate, pt_summary.held_out_attainment_rate)
        self.assertGreater(cp_summary.held_out_attainment_rate, min_e_summary.held_out_attainment_rate)

        # Invalid calibration_size values must raise ValueError.
        with self.assertRaises(ValueError):
            evaluate_split_resampling(observations, 0, MODE_HARD_DEADLINE, 0.8, deadline_s=2.0)
        with self.assertRaises(ValueError):
            evaluate_split_resampling(observations, 60, MODE_HARD_DEADLINE, 0.8, deadline_s=2.0)

    def test_cli_execution_and_exclusive_output(self) -> None:
        observations = _build_synthetic_trials(40)
        in_path = self.tmpdir / "trials.jsonl"
        out_json = self.tmpdir / "selector_out.json"
        out_md = self.tmpdir / "selector_out.md"

        with open(in_path, "w", encoding="utf-8") as handle:
            for obs in observations:
                handle.write(
                    json.dumps(
                        {
                            "item_id": obs.item_id,
                            "config_id": obs.config_id,
                            "passed": obs.passed,
                            "latency_s": obs.latency_s,
                            "energy_j": obs.energy_j,
                            "output_tokens": obs.output_tokens,
                        }
                    )
                    + "\n"
                )

        rc = main(
            [
                "--input",
                str(in_path),
                "--mode",
                MODE_EXP_DECAY,
                "--tau-s",
                "4.0",
                "--target-utility",
                "0.65",
                "--delta",
                "0.05",
                "--calibration-size",
                "20",
                "--num-splits",
                "50",
                "--seed",
                "9",
                "--output",
                str(out_json),
                "--markdown-output",
                str(out_md),
            ]
        )
        self.assertEqual(rc, 0)
        parsed = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(len(parsed["full_dataset_candidates"]), 4)
        self.assertTrue(parsed["split_evaluation"]["all_splits_disjoint"])
        self.assertTrue(out_md.exists())

        # Exclusive --output collision must return exit code 1.
        rc_collision = main(
            [
                "--input",
                str(in_path),
                "--calibration-size",
                "20",
                "--output",
                str(out_json),
            ]
        )
        self.assertEqual(rc_collision, 1)


if __name__ == "__main__":
    unittest.main()
