"""Independently usable host-side runtime support utilities."""

from .adb_session import (
    AdbError,
    AdbSession,
    CommandResult,
    RetryPolicy,
    TcpStreamForwarder,
    validate_serial,
)
from .power_telemetry import (
    BatterySnapshot,
    integrate_coulomb_energy,
    integrate_vi_trapezoidal,
    parse_dumpsys_battery,
    parse_proc_meminfo,
    parse_proc_uptime,
    subtract_standby_baseline,
)

__all__ = [
    "AdbError",
    "AdbSession",
    "BatterySnapshot",
    "CommandResult",
    "RetryPolicy",
    "TcpStreamForwarder",
    "integrate_coulomb_energy",
    "integrate_vi_trapezoidal",
    "parse_dumpsys_battery",
    "parse_proc_meminfo",
    "parse_proc_uptime",
    "subtract_standby_baseline",
    "validate_serial",
]
