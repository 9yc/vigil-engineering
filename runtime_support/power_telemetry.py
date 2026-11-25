"""Telemetry parsers and energy integrators for battery, memory and V/I traces.

Parses standard Android/Linux text nodes (`dumpsys battery`, `/proc/meminfo`,
`/proc/uptime`) and computes gross and baseline-subtracted energy from coulomb
counters (`dQ * V_mean`) or piecewise-linear `V(t) * I(t)` trapezoidal samples.
"""

from dataclasses import dataclass
import math


_VOLTAGE_SCALES = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}
_CURRENT_SCALES = {"a": 1.0, "ma": 1e-3, "ua": 1e-6}


@dataclass(frozen=True)
class BatterySnapshot:
    level: int
    scale: int
    temperature_c: float
    voltage_v: float
    charge_uah: int | None
    status: int | None
    ac_powered: bool
    usb_powered: bool
    wireless_powered: bool

    @property
    def state_of_charge(self):
        return self.level / self.scale


def _parse_bool_token(raw, key):
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"invalid boolean value for {key}: {raw!r}")


def _parse_int_token(raw, key):
    token = raw.strip()
    try:
        return int(token, 10)
    except ValueError as exc:
        raise ValueError(f"invalid integer value for {key}: {raw!r}") from exc


def parse_dumpsys_battery(text):
    """Parse `dumpsys battery` output into a validated BatterySnapshot."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("dumpsys battery text must be a non-empty string")
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        normalized_key = key.strip().lower()
        stripped_val = value.strip()
        if normalized_key and stripped_val and normalized_key not in fields:
            fields[normalized_key] = stripped_val

    for required in ("level", "scale", "temperature", "voltage"):
        if required not in fields:
            raise ValueError(f"missing required battery field: {required}")

    level = _parse_int_token(fields["level"], "level")
    scale = _parse_int_token(fields["scale"], "scale")
    temp_tenths = _parse_int_token(fields["temperature"], "temperature")
    voltage_mv = _parse_int_token(fields["voltage"], "voltage")
    if scale <= 0 or not (0 <= level <= scale):
        raise ValueError("battery level/scale out of range")
    if voltage_mv <= 0:
        raise ValueError("battery voltage must be positive")

    charge_uah = (
        _parse_int_token(fields["charge counter"], "charge counter")
        if "charge counter" in fields else None
    )
    status = _parse_int_token(fields["status"], "status") if "status" in fields else None
    return BatterySnapshot(
        level=level,
        scale=scale,
        temperature_c=temp_tenths / 10.0,
        voltage_v=voltage_mv / 1000.0,
        charge_uah=charge_uah,
        status=status,
        ac_powered=_parse_bool_token(fields.get("ac powered", "false"), "ac powered"),
        usb_powered=_parse_bool_token(fields.get("usb powered", "false"), "usb powered"),
        wireless_powered=_parse_bool_token(
            fields.get("wireless powered", "false"), "wireless powered"
        ),
    )


def parse_proc_meminfo(text):
    """Parse `/proc/meminfo` kB counters and return derived memory pressure metrics."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("meminfo text must be a non-empty string")
    values_kb = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"malformed meminfo line: {raw_line!r}")
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            raise ValueError(f"missing value for meminfo field {key!r}")
        count = _parse_int_token(parts[0], key)
        if count < 0:
            raise ValueError(f"negative value for meminfo field {key!r}")
        if len(parts) > 1 and parts[1].lower() != "kb":
            raise ValueError(f"unsupported unit {parts[1]!r} for meminfo field {key!r}")
        if len(parts) > 2:
            raise ValueError(f"unexpected extra tokens in meminfo line: {raw_line!r}")
        values_kb[key.strip()] = count

    if "MemTotal" not in values_kb or values_kb["MemTotal"] <= 0:
        raise ValueError("MemTotal must be present and positive")
    total_kb = values_kb["MemTotal"]
    if "MemAvailable" in values_kb:
        available_kb = values_kb["MemAvailable"]
    elif "MemFree" in values_kb:
        available_kb = (
            values_kb["MemFree"]
            + values_kb.get("Buffers", 0)
            + values_kb.get("Cached", 0)
        )
    else:
        raise ValueError("MemAvailable or MemFree is required")
    if available_kb > total_kb:
        raise ValueError("available memory exceeds MemTotal")

    return {
        "mem_total_kb": total_kb,
        "mem_available_kb": available_kb,
        "mem_used_kb": total_kb - available_kb,
        "available_ratio": available_kb / total_kb,
        "swap_total_kb": values_kb.get("SwapTotal", 0),
        "swap_free_kb": values_kb.get("SwapFree", 0),
        "anon_pages_kb": values_kb.get("AnonPages", 0),
    }


def parse_proc_uptime(text):
    """Parse `/proc/uptime` into `(uptime_s, idle_s)` floats."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("uptime text must be a non-empty string")
    parts = text.strip().split()
    if len(parts) != 2:
        raise ValueError("expected exactly two fields in /proc/uptime")
    uptime_s = float(parts[0])
    idle_s = float(parts[1])
    if not (math.isfinite(uptime_s) and math.isfinite(idle_s)):
        raise ValueError("uptime values must be finite")
    if uptime_s < 0 or idle_s < 0:
        raise ValueError("uptime values must be nonnegative")
    return {"uptime_s": uptime_s, "idle_s": idle_s}


def integrate_coulomb_energy(
    start_charge_uah,
    end_charge_uah,
    start_voltage_v,
    end_voltage_v,
    duration_s,
):
    """Estimate discharge energy via `dQ * V_mean` from uAh coulomb counters."""
    for label, val in (
        ("start_charge_uah", start_charge_uah),
        ("end_charge_uah", end_charge_uah),
        ("start_voltage_v", start_voltage_v),
        ("end_voltage_v", end_voltage_v),
        ("duration_s", duration_s),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{label} must be a finite number")
    if start_voltage_v <= 0 or end_voltage_v <= 0:
        raise ValueError("voltages must be positive")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    discharged_uah = float(start_charge_uah - end_charge_uah)
    if discharged_uah < 0:
        raise ValueError("charge counter increased during discharge window")
    # 1 uAh = 1e-6 A * 3600 s = 0.0036 Coulombs.
    discharged_c = discharged_uah * 0.0036
    mean_voltage_v = 0.5 * (start_voltage_v + end_voltage_v)
    energy_j = discharged_c * mean_voltage_v
    return {
        "discharged_uah": discharged_uah,
        "discharged_c": discharged_c,
        "mean_voltage_v": mean_voltage_v,
        "duration_s": float(duration_s),
        "energy_j": energy_j,
        "mean_power_w": energy_j / duration_s,
    }


def _interp(t, t0, p0, t1, p1):
    if t == t0:
        return p0
    if t == t1:
        return p1
    alpha = (t - t0) / (t1 - t0)
    return p0 + alpha * (p1 - p0)


def integrate_vi_trapezoidal(
    samples,
    *,
    start_s=None,
    end_s=None,
    max_gap_s=None,
    voltage_unit="v",
    current_unit="a",
):
    """Integrate `V(t) * |I(t)|` over `(time_s, voltage, current)` via trapezoids."""
    if voltage_unit not in _VOLTAGE_SCALES:
        raise ValueError(f"unsupported voltage_unit: {voltage_unit!r}")
    if current_unit not in _CURRENT_SCALES:
        raise ValueError(f"unsupported current_unit: {current_unit!r}")
    v_scale = _VOLTAGE_SCALES[voltage_unit]
    i_scale = _CURRENT_SCALES[current_unit]

    points = []
    for row in samples:
        if len(row) != 3:
            raise ValueError("each sample must be a (time_s, voltage, current) triple")
        t_raw, v_raw, i_raw = row
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in (t_raw, v_raw, i_raw)):
            raise ValueError("sample values must be finite numbers")
        if v_raw < 0:
            raise ValueError("voltage must be nonnegative")
        points.append((float(t_raw), float(v_raw) * v_scale * abs(float(i_raw)) * i_scale))

    if len(points) < 2:
        raise ValueError("at least two samples are required")
    gaps = [b[0] - a[0] for a, b in zip(points, points[1:])]
    if any(not math.isfinite(g) or g <= 0 for g in gaps):
        raise ValueError("sample timestamps must be strictly increasing")
    if max_gap_s is not None:
        if isinstance(max_gap_s, bool) or not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("max_gap_s must be finite and positive")
        if max(gaps) > max_gap_s:
            raise ValueError("sample gap exceeds max_gap_s")

    win_start = points[0][0] if start_s is None else float(start_s)
    win_end = points[-1][0] if end_s is None else float(end_s)
    if not (math.isfinite(win_start) and math.isfinite(win_end)) or win_start >= win_end:
        raise ValueError("window bounds must be finite with start_s < end_s")
    if win_start < points[0][0] or win_end > points[-1][0]:
        raise ValueError("samples do not cover requested window")

    trapezoids = []
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        left = max(t0, win_start)
        right = min(t1, win_end)
        if right <= left:
            continue
        p_left = _interp(left, t0, p0, t1, p1)
        p_right = _interp(right, t0, p0, t1, p1)
        trapezoids.append(0.5 * (p_left + p_right) * (right - left))

    energy_j = math.fsum(trapezoids)
    duration_s = win_end - win_start
    return {
        "samples": len(points),
        "start_s": win_start,
        "end_s": win_end,
        "duration_s": duration_s,
        "energy_j": energy_j,
        "mean_power_w": energy_j / duration_s,
        "max_gap_s": max(gaps),
        "method": "piecewise_trapezoidal_vi",
    }


def subtract_standby_baseline(gross_energy_j, duration_s, baseline_power_w, *, clamp_zero=False):
    """Subtract standby baseline energy `baseline_power_w * duration_s` from gross energy."""
    for label, val in (
        ("gross_energy_j", gross_energy_j),
        ("duration_s", duration_s),
        ("baseline_power_w", baseline_power_w),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{label} must be a finite number")
    if gross_energy_j < 0 or baseline_power_w < 0:
        raise ValueError("gross_energy_j and baseline_power_w must be nonnegative")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    standby_energy_j = float(baseline_power_w) * float(duration_s)
    net_energy_j = float(gross_energy_j) - standby_energy_j
    if clamp_zero and net_energy_j < 0.0:
        net_energy_j = 0.0
    return {
        "gross_energy_j": float(gross_energy_j),
        "standby_energy_j": standby_energy_j,
        "net_energy_j": net_energy_j,
        "duration_s": float(duration_s),
        "gross_power_w": float(gross_energy_j) / float(duration_s),
        "net_power_w": net_energy_j / float(duration_s),
    }
