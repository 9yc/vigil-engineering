"""Pre-run thermal/battery gate and active/cooldown duty-cycle controller.

Tracks monotonic active and paused intervals without wall-clock sleeps during
tests when a synthetic clock and sleep hook are injected.
"""

from dataclasses import dataclass
import math
import time


class ThermalGateError(RuntimeError):
    """Base exception for thermal or battery gate violations."""


class ThermalTimeoutError(ThermalGateError):
    """Raised when cooldown or entry gating exceeds max_cooldown_wait_s."""


class BatteryLowError(ThermalGateError):
    """Raised when battery state of charge drops below min_soc without power."""


@dataclass(frozen=True)
class ProbeReading:
    temperature_c: float
    state_of_charge: float
    external_power: bool = False

    def __post_init__(self):
        if (isinstance(self.temperature_c, bool)
                or not isinstance(self.temperature_c, (int, float))
                or not math.isfinite(self.temperature_c)):
            raise ValueError("temperature_c must be a finite number")
        if (isinstance(self.state_of_charge, bool)
                or not isinstance(self.state_of_charge, (int, float))
                or not math.isfinite(self.state_of_charge)
                or not (0.0 <= self.state_of_charge <= 1.0)):
            raise ValueError("state_of_charge must be a finite number in [0.0, 1.0]")
        if not isinstance(self.external_power, bool):
            raise ValueError("external_power must be a bool")


@dataclass(frozen=True)
class ThermalGateConfig:
    max_entry_temp_c: float = 38.0
    pause_temp_c: float = 42.0
    resume_temp_c: float = 37.0
    min_soc: float = 0.20
    max_active_window_s: float | None = None
    min_cooldown_s: float = 5.0
    max_cooldown_wait_s: float = 600.0
    poll_interval_s: float = 2.0
    keepalive_interval_s: float = 15.0

    def __post_init__(self):
        for name in (
            "max_entry_temp_c",
            "pause_temp_c",
            "resume_temp_c",
            "min_soc",
            "min_cooldown_s",
            "max_cooldown_wait_s",
            "poll_interval_s",
            "keepalive_interval_s",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number")
        if self.resume_temp_c >= self.pause_temp_c:
            raise ValueError("resume_temp_c must be strictly below pause_temp_c")
        if self.max_entry_temp_c >= self.pause_temp_c:
            raise ValueError("max_entry_temp_c must be strictly below pause_temp_c")
        if not (0.0 <= self.min_soc <= 1.0):
            raise ValueError("min_soc must be in [0.0, 1.0]")
        if self.min_cooldown_s < 0:
            raise ValueError("min_cooldown_s must be nonnegative")
        if self.poll_interval_s <= 0 or self.keepalive_interval_s <= 0:
            raise ValueError("poll_interval_s and keepalive_interval_s must be positive")
        if self.max_cooldown_wait_s <= 0 or self.max_cooldown_wait_s < self.min_cooldown_s:
            raise ValueError("max_cooldown_wait_s must be positive and >= min_cooldown_s")
        if self.max_active_window_s is not None:
            if (isinstance(self.max_active_window_s, bool)
                    or not isinstance(self.max_active_window_s, (int, float))
                    or not math.isfinite(self.max_active_window_s)
                    or self.max_active_window_s <= 0):
                raise ValueError("max_active_window_s must be finite and positive when set")


class DutyCycleController:
    """Enforce pre-run gating and mid-run active/cooldown duty cycling."""

    def __init__(
        self,
        config,
        probe_fn,
        *,
        clock_fn=time.monotonic,
        sleep_fn=time.sleep,
        keepalive_fn=None,
    ):
        if not isinstance(config, ThermalGateConfig):
            raise ValueError("config must be a ThermalGateConfig instance")
        if not callable(probe_fn):
            raise ValueError("probe_fn must be callable")
        if keepalive_fn is not None and not callable(keepalive_fn):
            raise ValueError("keepalive_fn must be callable when provided")
        self.config = config
        self._probe = probe_fn
        self._clock = clock_fn
        self._sleep = sleep_fn
        self._keepalive = keepalive_fn

        self.state = "idle"
        self._entry_wait_s = 0.0
        self._active_elapsed_s = 0.0
        self._paused_elapsed_s = 0.0
        self._active_slice_start = None
        self._run_start = None
        self._last_keepalive = None
        self._cooldown_episodes = 0
        self._keepalive_calls = 0
        self._peak_temp_c = None

    def _read_probe(self):
        reading = self._probe()
        if not isinstance(reading, ProbeReading):
            raise ValueError("probe_fn must return a ProbeReading")
        if self._peak_temp_c is None or reading.temperature_c > self._peak_temp_c:
            self._peak_temp_c = float(reading.temperature_c)
        if reading.state_of_charge < self.config.min_soc and not reading.external_power:
            raise BatteryLowError(
                f"state_of_charge {reading.state_of_charge:.3f} below "
                f"minimum {self.config.min_soc:.3f}"
            )
        return reading

    def _maybe_keepalive(self, now):
        if self._keepalive is None:
            return
        if self._last_keepalive is None:
            self._last_keepalive = now
            return
        if now - self._last_keepalive >= self.config.keepalive_interval_s:
            self._keepalive()
            self._keepalive_calls += 1
            self._last_keepalive = self._clock()

    def await_entry_gate(self):
        """Wait for initial thermal/battery readiness and start active execution."""
        if self.state != "idle":
            raise RuntimeError(f"cannot enter from state {self.state!r}")
        gate_start = self._clock()
        self._last_keepalive = gate_start
        while True:
            reading = self._read_probe()
            now = self._clock()
            if reading.temperature_c <= self.config.max_entry_temp_c:
                self._entry_wait_s = max(0.0, now - gate_start)
                self._run_start = now
                self._active_slice_start = now
                self._last_keepalive = now
                self.state = "active"
                return reading
            if now - gate_start >= self.config.max_cooldown_wait_s:
                raise ThermalTimeoutError(
                    f"entry temperature {reading.temperature_c:.1f}C exceeded "
                    f"{self.config.max_entry_temp_c:.1f}C beyond timeout"
                )
            self._sleep(self.config.poll_interval_s)
            self._maybe_keepalive(self._clock())

    def checkpoint(self):
        """Inspect thermal/slice state between active steps; pause until cooled if needed."""
        if self.state != "active":
            raise RuntimeError(f"checkpoint requires active state, got {self.state!r}")
        reading = self._read_probe()
        now = self._clock()
        slice_elapsed = max(0.0, now - self._active_slice_start)
        over_temp = reading.temperature_c >= self.config.pause_temp_c
        over_slice = (
            self.config.max_active_window_s is not None
            and slice_elapsed >= self.config.max_active_window_s
        )
        if not over_temp and not over_slice:
            self._maybe_keepalive(now)
            return {"paused": False, "pause_duration_s": 0.0, "reading": reading}

        self._active_elapsed_s += slice_elapsed
        self.state = "cooldown"
        self._cooldown_episodes += 1
        pause_start = now
        self._last_keepalive = pause_start

        while True:
            self._sleep(self.config.poll_interval_s)
            after_sleep = self._clock()
            self._maybe_keepalive(after_sleep)
            reading = self._read_probe()
            now = self._clock()
            paused_for = max(0.0, now - pause_start)
            if (paused_for >= self.config.min_cooldown_s
                    and reading.temperature_c <= self.config.resume_temp_c):
                self._paused_elapsed_s += paused_for
                self._active_slice_start = now
                self.state = "active"
                return {
                    "paused": True,
                    "pause_duration_s": paused_for,
                    "reading": reading,
                }
            if paused_for >= self.config.max_cooldown_wait_s:
                self._paused_elapsed_s += paused_for
                self.state = "aborted"
                raise ThermalTimeoutError(
                    f"cooldown exceeded {self.config.max_cooldown_wait_s:.1f}s "
                    f"at {reading.temperature_c:.1f}C"
                )

    def finish(self):
        """Close the active slice and return deterministic duty-cycle accounting."""
        if self.state == "finished":
            return self.summary()
        if self.state != "active":
            raise RuntimeError(f"cannot finish from state {self.state!r}")
        now = self._clock()
        self._active_elapsed_s += max(0.0, now - self._active_slice_start)
        self._active_slice_start = None
        self.state = "finished"
        return self.summary()

    def summary(self):
        active_s = self._active_elapsed_s
        if self.state == "active" and self._active_slice_start is not None:
            active_s += max(0.0, self._clock() - self._active_slice_start)
        run_elapsed_s = active_s + self._paused_elapsed_s
        duty_factor = (active_s / run_elapsed_s) if run_elapsed_s > 0 else 1.0
        return {
            "state": self.state,
            "entry_wait_s": self._entry_wait_s,
            "active_elapsed_s": active_s,
            "paused_elapsed_s": self._paused_elapsed_s,
            "run_elapsed_s": run_elapsed_s,
            "duty_factor": duty_factor,
            "cooldown_episodes": self._cooldown_episodes,
            "keepalive_calls": self._keepalive_calls,
            "peak_temperature_c": self._peak_temp_c,
        }
