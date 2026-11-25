#!/usr/bin/env python3
"""Cross-calibrate fuel-gauge power traces, coulomb counters and battery impedance.

Provides ordinary least-squares gain/offset calibration (`p_ref = gain * p_raw + offset`)
between uncalibrated telemetry rails and reference wattmeters, cross-checks integrated
`V(t) * I(t)` trace energy against start/end coulomb-counter deltas, and estimates
effective DC internal resistance (`R_int = -dV / dI`) across discharge load steps
stratified by temperature bands.
"""

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class LinearCalibration:
    gain: float
    offset_w: float
    r_squared: float
    rmse_w: float
    max_abs_error_w: float
    sample_count: int

    def apply(self, raw_power_w: float) -> float:
        if isinstance(raw_power_w, bool) or not isinstance(raw_power_w, (int, float)) or not math.isfinite(raw_power_w):
            raise ValueError("raw_power_w must be a finite number")
        return self.gain * float(raw_power_w) + self.offset_w

    def to_dict(self) -> dict:
        return {
            "gain": self.gain,
            "offset_w": self.offset_w,
            "r_squared": self.r_squared,
            "rmse_w": self.rmse_w,
            "max_abs_error_w": self.max_abs_error_w,
            "sample_count": self.sample_count,
        }


def _finite_number(val, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        raise ValueError(f"{name} must be a finite number")
    return float(val)


def fit_linear_power_calibration(pairs) -> LinearCalibration:
    """Fit `p_ref = gain * p_raw + offset_w` over `(p_raw_w, p_ref_w)` pairs."""
    cleaned: list[tuple[float, float]] = []
    for idx, item in enumerate(pairs):
        if len(item) != 2:
            raise ValueError(f"pair {idx} must have length 2")
        x = _finite_number(item[0], f"pairs[{idx}].raw_w")
        y = _finite_number(item[1], f"pairs[{idx}].ref_w")
        if x < 0.0 or y < 0.0:
            raise ValueError(f"pair {idx} power values must be nonnegative")
        cleaned.append((x, y))

    n = len(cleaned)
    if n < 2:
        raise ValueError("at least two calibration pairs are required")

    mean_x = math.fsum(x for x, _ in cleaned) / n
    mean_y = math.fsum(y for _, y in cleaned) / n
    s_xx = math.fsum((x - mean_x) ** 2 for x, _ in cleaned)
    if s_xx <= 1e-18:
        raise ValueError("raw power values have zero variance; cannot fit gain")
    s_xy = math.fsum((x - mean_x) * (y - mean_y) for x, y in cleaned)
    s_yy = math.fsum((y - mean_y) ** 2 for _, y in cleaned)

    gain = s_xy / s_xx
    offset_w = mean_y - gain * mean_x

    sq_errors = []
    max_abs = 0.0
    for x, y in cleaned:
        pred = gain * x + offset_w
        err = abs(y - pred)
        if err > max_abs:
            max_abs = err
        sq_errors.append(err * err)

    sse = math.fsum(sq_errors)
    rmse_w = math.sqrt(sse / n)
    r_squared = 1.0 - (sse / s_yy) if s_yy > 1e-18 else (1.0 if sse <= 1e-18 else 0.0)
    return LinearCalibration(
        gain=gain,
        offset_w=offset_w,
        r_squared=r_squared,
        rmse_w=rmse_w,
        max_abs_error_w=max_abs,
        sample_count=n,
    )


def reconcile_vi_and_coulomb_window(
    vi_samples,
    start_charge_uah: float,
    end_charge_uah: float,
    *,
    max_relative_discrepancy: float = 0.08,
) -> dict:
    """Compare trapezoidal `V(t)*|I(t)|` trace integrals against coulomb-counter delta."""
    q_start = _finite_number(start_charge_uah, "start_charge_uah")
    q_end = _finite_number(end_charge_uah, "end_charge_uah")
    tol = _finite_number(max_relative_discrepancy, "max_relative_discrepancy")
    if tol <= 0.0:
        raise ValueError("max_relative_discrepancy must be positive")
    if q_end > q_start:
        raise ValueError("end_charge_uah exceeds start_charge_uah during discharge")

    points: list[tuple[float, float, float]] = []
    for idx, row in enumerate(vi_samples):
        if len(row) != 3:
            raise ValueError(f"sample {idx} must be (time_s, voltage_v, current_a)")
        t = _finite_number(row[0], "time_s")
        v = _finite_number(row[1], "voltage_v")
        i = abs(_finite_number(row[2], "current_a"))
        if v <= 0.0:
            raise ValueError(f"sample {idx} voltage_v must be positive")
        if points and t <= points[-1][0]:
            raise ValueError(f"sample {idx} timestamp must be strictly increasing")
        points.append((t, v, i))

    if len(points) < 2:
        raise ValueError("at least two V/I samples are required")

    duration_s = points[-1][0] - points[0][0]
    charge_trapezoids = []
    energy_trapezoids = []
    voltage_trapezoids = []

    for (t0, v0, i0), (t1, v1, i1) in zip(points, points[1:]):
        dt = t1 - t0
        charge_trapezoids.append(0.5 * (i0 + i1) * dt)
        energy_trapezoids.append(0.5 * (v0 * i0 + v1 * i1) * dt)
        voltage_trapezoids.append(0.5 * (v0 + v1) * dt)

    vi_charge_c = math.fsum(charge_trapezoids)
    vi_energy_j = math.fsum(energy_trapezoids)
    time_mean_voltage_v = math.fsum(voltage_trapezoids) / duration_s

    counter_uah = q_start - q_end
    counter_charge_c = counter_uah * 0.0036
    counter_energy_j = counter_charge_c * time_mean_voltage_v

    denom_c = max(counter_charge_c, vi_charge_c, 1e-12)
    rel_charge_diff = abs(vi_charge_c - counter_charge_c) / denom_c
    denom_e = max(counter_energy_j, vi_energy_j, 1e-12)
    rel_energy_diff = abs(vi_energy_j - counter_energy_j) / denom_e

    return {
        "consistent": rel_charge_diff <= tol and rel_energy_diff <= tol,
        "duration_s": duration_s,
        "vi_charge_c": vi_charge_c,
        "counter_charge_c": counter_charge_c,
        "relative_charge_diff": rel_charge_diff,
        "vi_energy_j": vi_energy_j,
        "counter_energy_j": counter_energy_j,
        "relative_energy_diff": rel_energy_diff,
        "time_mean_voltage_v": time_mean_voltage_v,
    }


def estimate_dc_internal_resistance(
    samples,
    *,
    min_step_current_a: float = 0.25,
    max_step_dt_s: float = 1.5,
    temp_bin_width_c: float = 5.0,
) -> dict:
    """Estimate `R_int = -dV / dI` across adjacent discharge load transitions."""
    min_di = _finite_number(min_step_current_a, "min_step_current_a")
    max_dt = _finite_number(max_step_dt_s, "max_step_dt_s")
    bin_w = _finite_number(temp_bin_width_c, "temp_bin_width_c")
    if min_di <= 0.0 or max_dt <= 0.0 or bin_w <= 0.0:
        raise ValueError("step thresholds and bin width must be positive")

    parsed: list[tuple[float, float, float, float]] = []
    for idx, row in enumerate(samples):
        if len(row) != 4:
            raise ValueError(f"sample {idx} must be (time_s, voltage_v, current_a, temp_c)")
        t = _finite_number(row[0], "time_s")
        v = _finite_number(row[1], "voltage_v")
        i = abs(_finite_number(row[2], "current_a"))
        temp = _finite_number(row[3], "temp_c")
        if v <= 0.0:
            raise ValueError("voltage_v must be positive")
        if parsed and t <= parsed[-1][0]:
            raise ValueError("timestamps must be strictly increasing")
        parsed.append((t, v, i, temp))

    if len(parsed) < 2:
        raise ValueError("at least two samples are required")

    all_r_ohm: list[float] = []
    bins: dict[str, list[float]] = {}

    for (t0, v0, i0, temp0), (t1, v1, i1, temp1) in zip(parsed, parsed[1:]):
        dt = t1 - t0
        if dt > max_dt:
            continue
        di = i1 - i0
        dv = v1 - v0
        if abs(di) < min_di:
            continue
        # Under ohmic sag V_term = V_ocv - I * R_int, so -dV / dI > 0.
        r_est = -dv / di
        if r_est <= 0.0:
            continue
        all_r_ohm.append(r_est)
        mean_temp = 0.5 * (temp0 + temp1)
        lo = math.floor(mean_temp / bin_w) * bin_w
        label = f"{lo:.1f}_{lo + bin_w:.1f}C"
        bins.setdefault(label, []).append(r_est)

    if not all_r_ohm:
        raise ValueError("no qualifying current steps detected for impedance estimation")

    ordered = sorted(all_r_ohm)
    mid = len(ordered) // 2
    median_ohm = (
        ordered[mid]
        if len(ordered) % 2 == 1
        else 0.5 * (ordered[mid - 1] + ordered[mid])
    )

    by_temp = {}
    for label in sorted(bins):
        vals = sorted(bins[label])
        m = len(vals) // 2
        med = vals[m] if len(vals) % 2 == 1 else 0.5 * (vals[m - 1] + vals[m])
        by_temp[label] = {
            "steps": len(vals),
            "mean_mohm": (math.fsum(vals) / len(vals)) * 1000.0,
            "median_mohm": med * 1000.0,
        }

    return {
        "qualifying_steps": len(all_r_ohm),
        "mean_mohm": (math.fsum(all_r_ohm) / len(all_r_ohm)) * 1000.0,
        "median_mohm": median_ohm * 1000.0,
        "min_mohm": ordered[0] * 1000.0,
        "max_mohm": ordered[-1] * 1000.0,
        "temperature_bins": by_temp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    fit_p = sub.add_parser("fit", help="Fit linear calibration from raw_w,ref_w CSV")
    fit_p.add_argument("input", type=Path)
    fit_p.add_argument("--output", type=Path)

    imp_p = sub.add_parser("impedance", help="Estimate DC R_int from time_s,voltage_v,current_a,temp_c CSV")
    imp_p.add_argument("input", type=Path)
    imp_p.add_argument("--min-step-a", type=float, default=0.25)
    imp_p.add_argument("--max-dt-s", type=float, default=1.5)
    imp_p.add_argument("--output", type=Path)

    args = parser.parse_args()
    try:
        with args.input.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, strict=True))
        if not rows:
            raise ValueError("empty input CSV")

        if args.cmd == "fit":
            pairs = [(float(r["raw_w"]), float(r["ref_w"])) for r in rows]
            payload = fit_linear_power_calibration(pairs).to_dict()
        else:
            samples = [
                (float(r["time_s"]), float(r["voltage_v"]), float(r["current_a"]), float(r["temp_c"]))
                for r in rows
            ]
            payload = estimate_dc_internal_resistance(
                samples,
                min_step_current_a=args.min_step_a,
                max_step_dt_s=args.max_dt_s,
            )

        encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as out_f:
                out_f.write(encoded)
        else:
            print(encoded, end="")
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        parser.exit(2, f"battery_calibration error: {exc}\n")


if __name__ == "__main__":
    main()
